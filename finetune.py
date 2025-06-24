import os
import re
import random
import time
import glob
import argparse
import pandas as pd
import json
import numpy as np
import torch
import torch.nn as nn
import transformers
import lightning as pl
import swanlab

from copy import deepcopy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import wrap, enable_wrap
from typing import Optional, Dict, Tuple, List, Union, Unpack, Sequence, Any
from itertools import chain
from datasets import Dataset, load_dataset
from lightning import Trainer, LightningDataModule, LightningModule
from lightning.pytorch.strategies import FSDPStrategy, SingleDeviceStrategy, DeepSpeedStrategy, DDPStrategy
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput, SequenceClassifierOutput
from torchmetrics import MeanMetric, SumMetric, MatthewsCorrCoef, F1Score, Accuracy, AUROC
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics import Metric
from swanlab.integration.pytorch_lightning import SwanLabLogger

from model.vq.configuration_vq import VQConfig
from model.vq.modeling_vq import VQModel, VQOutput
from model.configuration_transformer import TransformerConfig
from model.modeling_transformer import TransformerForCausalLM, TransformerForSequenceClassification
from model.utils import CheckpointCallback, CosineLRSchedule, clean_metrics

torch.set_float32_matmul_precision("medium")

# HF_CACHE = os.environ["HF_HOME"]
HF_CACHE = "/root/autodl-tmp/hf_cache"
os.environ["HF_HOME"] = HF_CACHE
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class FinetuneDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer: transformers.AutoTokenizer, 
        train_config: argparse.Namespace, 
        split: str="train",
        num_seqs: int=1,
        label_offset: int=0,
        **kwargs
    ):
        super().__init__()

        self.data_path = train_config.data_path
        self.tokenizer = tokenizer
        self.num_seqs = num_seqs
        self.num_class = 0
        self.label_offset = label_offset

        self.train_config = train_config

        # loading
        if split != "":
            candidate = glob.glob(self.data_path + f"/{split}.*", recursive=True)
            tgt = [_ for _ in candidate if _.endswith((".csv", ".parquet"))][0]
            raw_data = pd.read_csv(tgt) if tgt.endswith(".csv") else pd.read_parquet(tgt)
        else:
            raw_data = pd.read_csv(self.data_path) if self.data_path.endswith(".csv") else pd.read_parquet(self.data_path)

        data = []
        label = []
        for row in raw_data.itertuples(index=False):
            seqs = []
            for _ in row[:self.num_seqs]:
                seq = re.sub(r"[^ATCG]", "", _.strip().upper())
                seqs.append(seq)
            data.append(tuple([*seqs, row[-1]]))
            label.append(row[-1])
        
        self.num_class = max(self.num_class, len(set(label)))
        self.valid_labels = list(set(label)) if isinstance(label[0], int) else None
        self.data = np.array(data, dtype=object)
    
    def process(self, datas: Sequence[Tuple]):
        texts, labels = tuple(tuple(_[:-1]) for _ in datas), tuple(_[-1] for _ in datas)

        pair_texts = None
        if len(texts[0]) > 1:
            texts, pair_texts = tuple([_[i] for _ in texts] for i in range(0, 2))
        else:
            texts = tuple(_[0] for _ in texts)
        
        output = self.tokenizer(
            texts, pair_texts,
            return_tensors="pt",
            padding="longest",
            max_length=int(1e6),
            truncation=False,
            padding_side="right",
            add_special_tokens=False
        )
        return dict(
            input_ids=output["input_ids"],
            labels=list(labels) if isinstance(labels[0], str) else torch.tensor([_ + self.label_offset for _ in labels], dtype=torch.int64 if isinstance(labels[0], int) else torch.float32)
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    

class FinetuneDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        tokenizer: transformers.AutoTokenizer, 
        train_config: argparse.Namespace,
        num_seqs: Optional[int]=1, 
        label_offset: Optional[int]=0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.data_path = train_config.data_path
        self.tokenizer = tokenizer
        self.num_seqs = num_seqs
        self.label_offset = label_offset
        self.train_config = train_config

        self.data = {}
        self.data.update(
            train=FinetuneDataset(self.tokenizer, train_config, "train", self.num_seqs, self.label_offset),
            eval=FinetuneDataset(self.tokenizer, train_config, "dev", self.num_seqs, self.label_offset),
            test=FinetuneDataset(self.tokenizer, train_config, "test", self.num_seqs, self.label_offset),
        )
        self.num_class = self.data["train"].num_class
        if self.data["train"].valid_labels is not None:
            print(f"Valid labels: {",".join([str(_) for _ in self.data["train"].valid_labels])}")
    
    def setup(self, stage):
        pass
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data["train"], shuffle=True,
            batch_size=self.train_config.batch_size,
            collate_fn=self.data["train"].process,
            num_workers=self.train_config.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data["eval"],
            batch_size=self.train_config.eval_batch_size,
            collate_fn=self.data["eval"].process,
            num_workers=self.train_config.num_workers
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data["test"],
            batch_size=self.train_config.eval_batch_size,
            collate_fn=self.data["test"].process,
            num_workers=self.train_config.num_workers
        )

#########################################################
#                  --- model ---
#########################################################
class PretrainModule(LightningModule):
    def __init__(self, model: TransformerForCausalLM, vq: VQModel, model_config: TransformerConfig, vq_config: VQConfig, train_config: argparse.Namespace, **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.vq = vq
        self.model_config = model_config
        self.vq_config = vq_config
        self.train_config = train_config

        self.vq.requires_grad_(False)
        self.vq = self.vq.eval()

        self._date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self._train_tokens = 0

        self.save_hyperparameters(train_config)
    
    # @rank_zero_only
    def on_fit_start(self):
        if self.global_rank == 0:
            print(f"--------------- Settings ---------------")
            for (k, v) in self.hparams.items(): print(f"{k}:\t {v}")
            print(f"------------- Architecture -------------")
            print(self.model)
            print(f"Total Param: {sum([p.numel() for p in self.model.parameters()])}")
            print(f"------------ Start Training ------------")
            self._start = time.perf_counter()
        
        # metrics
        self.train_metrics = {
            "loss": MeanMetric().to(self.trainer.strategy.root_device),
            "codebook_usage": MeanMetric().to(self.trainer.strategy.root_device)
        }

        self.eval_metrics = {
            "steps": 0,
            "wall_clock": 0,
            "MCC": MatthewsCorrCoef(task="multiclass", num_classes=self.model_config.num_class).to(self.trainer.strategy.root_device),
            "F1": F1Score(task="multiclass", num_classes=self.model_config.num_class, average="macro").to(self.trainer.strategy.root_device),
            "Accuracy": Accuracy(task="multiclass", num_classes=self.model_config.num_class, average="macro").to(self.trainer.strategy.root_device),
            "AUROC": AUROC(task="multiclass", num_classes=self.model_config.num_class, average="macro").to(self.trainer.strategy.root_device)
        }
    
    # @rank_zero_only
    def on_fit_end(self):
        if self.trainer.global_rank == 0:
            wall_clock = time.perf_counter() - self._start
            print(f"Total wall-clock time: {wall_clock:.4f}")
            print("------------ End Training ------------")

            swanlab.finish()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.max_lr,
            eps=1e-5,
            weight_decay=0.1,
            betas=(0.9, 0.98)
        )
        schedule = CosineLRSchedule(
            optimizer,
            warmup=int(self.hparams.warmup_ratio * self.hparams.max_steps),
            max_lr=self.hparams.max_lr,
            min_lr=self.hparams.min_lr,
            max_steps=self.hparams.max_steps
        )
        return [optimizer], [{"scheduler": schedule, "interval": "step", "frequency": 1}]
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()
    
    def forward(self, data: Dict):
        with torch.no_grad():
            input_ids = data['input_ids']
            res_dict = self.vq.encode(input_ids)
            inputs_embeds, input_ids = res_dict['quant'], res_dict['indices']

            codebook_usage = (torch.bincount(input_ids.flatten(), minlength=self.vq_config.codebook_size) > 0).to(torch.int64)
            codebook_usage = codebook_usage.sum() / self.vq_config.codebook_size

        self.train_metrics['codebook_usage'].update(codebook_usage)

        bos_ids = torch.full((input_ids.size(0), 1), self.model_config.bos_token_id, dtype=torch.int64, device=self.trainer.strategy.root_device)
        eos_ids = torch.full((input_ids.size(0), 1), self.model_config.eos_token_id, dtype=torch.int64, device=self.trainer.strategy.root_device)

        inputs_embeds = torch.cat([self.vq.base_vocab(bos_ids), inputs_embeds, self.vq.base_vocab(eos_ids)], dim=1)
        inputs_embeds = self.model.model.latent_project(inputs_embeds)

        outputs: SequenceClassifierOutput = self.model(
            inputs_embeds=inputs_embeds,
            labels=data["labels"]
        )
        return outputs
    
    def training_step(self, batch: Dict, batch_idx):
        outputs: SequenceClassifierOutput = self(batch)

        self.train_metrics["loss"].update(outputs.loss)
        
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            loss = self.train_metrics["loss"].compute()
            lr = self.optimizers().optimizer.param_groups[0]["lr"]
            steps = self.trainer.global_step
            codebook_usage = self.train_metrics["codebook_usage"].compute()

            self.log_dict(
                dict(loss=loss, lr=lr, steps=steps),
                prog_bar=True,
                logger=False
            )
            self.log_dict({"finetune/loss": loss, "finetune/lr": lr, "finetune/codebook_usage": codebook_usage}, prog_bar=False, logger=True)

            self.train_metrics["loss"].reset()
            self.train_metrics["codebook_usage"].reset()

        return outputs.loss
    
    def on_validation_epoch_start(self):
        clean_metrics(self.eval_metrics)
        self.eval_start = time.perf_counter()
    
    def validation_step(self, batch: Dict, batch_idx):
        outputs: SequenceClassifierOutput = self(batch)
        preds = outputs.logits.softmax(-1)
        target = batch["labels"].flatten()

        self.eval_metrics["steps"] = self.global_step
        for v in self.eval_metrics.values():
            if isinstance(v, Metric): v.update(preds, target)
    
    def on_validation_epoch_end(self):
        self.eval_metrics["wall_clock"] = time.perf_counter() - self.eval_start
        for k, v in self.eval_metrics.items():
            if isinstance(v, Metric):
                self.log(f"eval/{k}", v.compute(), sync_dist=True)
    
    def on_test_epoch_start(self):
        self.on_validation_epoch_start()
    
    def test_step(self, batch: Dict, batch_idx):
        self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        self.eval_metrics["wall_clock"] = time.perf_counter() - self.eval_start
        for k, v in self.eval_metrics.items():
            if isinstance(v, Metric):
                self.log(f"test/{k}", v.compute(), sync_dist=True)

def main(train_config: argparse.Namespace):
    pl.seed_everything(train_config.seed)

    base_path = os.path.dirname(os.path.abspath(__file__))
    train_config.data_path = os.path.join(base_path, train_config.data_path, train_config.data_name)
    train_config.ckpt_path = os.path.join(base_path, train_config.ckpt_path)
    logger_path = train_config.ckpt_path if not os.path.exists("/root/tf-logs") else "/root/tf-logs"
    logger_name = f"VQTransformer|CLM|Finetune|{train_config.data_name}"

    if logger_path == "/root/tf-logs":
        logger_path = os.path.join(logger_path, *logger_name.split('|'))

    train_config.ckpt_path = os.path.join(train_config.ckpt_path, *logger_name.split('|'))
    if not os.path.exists(train_config.ckpt_path): os.makedirs(train_config.ckpt_path)

    model_config = TransformerConfig()
    vq_config = VQConfig(
        latent_size=train_config.latent_size,
        codebook_size=train_config.codebook_size,
        downsample_rate=train_config.downsample_rate,
        num_heads=train_config.num_heads,
        vq_type=train_config.vq_type
    )

    assert (train_config.max_epochs != -1 or train_config.max_steps != -1)

    logger = SwanLabLogger(
        project=train_config.logger_project,
        experiment_name=logger_name,
        save_dir=logger_path
    )
    trainer = Trainer(
        precision=train_config.precision,
        devices=[0],
        strategy="auto",
        max_epochs=train_config.max_epochs if train_config.max_steps == -1 else None,
        max_steps=train_config.max_steps if train_config.max_steps != -1 else -1,
        default_root_dir=train_config.ckpt_path,
        accumulate_grad_batches=train_config.accumulate_grad_batches,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        logger=logger,
        callbacks=CheckpointCallback(train_config),
        log_every_n_steps=20,
        check_val_every_n_epoch=65536,
        enable_checkpointing=False
    )

    raw_batch_size = train_config.batch_size
    if trainer.num_devices > 1:
        train_config.batch_size = train_config.batch_size // trainer.num_devices
        train_config.eval_batch_size = train_config.eval_batch_size // trainer.num_devices
    if trainer.accumulate_grad_batches > 1: 
        train_config.batch_size = train_config.batch_size // trainer.accumulate_grad_batches

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(os.path.join(base_path, "tokenizer/base"), use_fast=True, trust_remote_code=True, local_files_only=True)
    model_config.vocab_size = tokenizer.vocab_size
    model_config.bos_token_id = tokenizer.vocab[tokenizer.bos_token]
    model_config.eos_token_id = tokenizer.vocab[tokenizer.eos_token]
    model_config.pad_token_id = tokenizer.vocab[tokenizer.pad_token]

    vq = VQModel(vq_config)
    vq_state_dict = torch.load(os.path.join(train_config.vq_path, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    vq.load_state_dict(vq_state_dict)

    datamodule = FinetuneDataModule(tokenizer, train_config)

    model_config.num_class = datamodule.data["train"].num_class
    model = TransformerForSequenceClassification(model_config, tokenizer.vocab_size + vq_config.codebook_size, vq_config.latent_size)
    
    if train_config.max_steps == -1: train_config.max_steps = (len(datamodule.data) + raw_batch_size - 1) // raw_batch_size

    if train_config.pretrained_ckpt_path != "":
        pretrained_ckpt = torch.load(os.path.join(train_config.pretrained_ckpt_path, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        try:
            model.load_state_dict(pretrained_ckpt, strict=True)
        except Exception as e:
            print(e)
            print("Try flexible loading...")
            model.load_state_dict(pretrained_ckpt, strict=False)

    plmodel = PretrainModule(model, vq, model_config, vq_config, train_config)
    trainer.fit(plmodel, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset Args:
    finetune_parser = parser.add_argument_group("finetune")
    finetune_parser.add_argument("--seed", type=int, default=17)
    finetune_parser.add_argument("--data_path", type=str, help="Path to the dataset dir", default="data/finetune")
    finetune_parser.add_argument("--data_name", type=str, help="Name of the dataset", default="tf/0")
    finetune_parser.add_argument("--dataset_name", type=str, default="")
    finetune_parser.add_argument("--ckpt_path", type=str, help="Path to the checkpoint", default="result/finetune")
    finetune_parser.add_argument("--pretrained_ckpt_path", type=str, default="", help="Path to the pretrained checkpoint")
    finetune_parser.add_argument("--vq_path", type=str, default="")

    finetune_parser.add_argument("--batch_size", type=int, default=32)
    finetune_parser.add_argument("--eval_batch_size", type=int, default=32)
    finetune_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients for every n batches")
    finetune_parser.add_argument("--num_workers", type=int, default=4)
    finetune_parser.add_argument("--num_preprocess_workers", type=int, default=32)

    finetune_parser.add_argument("--max_steps", type=int, default=6000)
    finetune_parser.add_argument("--max_epochs", type=int, default=-1)
    finetune_parser.add_argument("--max_lr", type=float, default=1e-4, help="Maximum learning rate")
    finetune_parser.add_argument("--min_lr", type=float, default=0, help="Minimum learning rate")
    finetune_parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of steps to warm up learning rate")
    finetune_parser.add_argument("--precision", type=str, default="bf16-mixed")
    finetune_parser.add_argument("--dropout", type=float, default=0.1)

    finetune_parser.add_argument("--logger_project", type=str)

    # vq
    finetune_parser.add_argument("--latent_size", type=int, default=512)
    finetune_parser.add_argument("--codebook_size", type=int, default=8192)
    finetune_parser.add_argument("--downsample_rate", type=int, default=16)
    finetune_parser.add_argument("--num_heads", type=int, default=1)
    finetune_parser.add_argument("--vq_type", type=str, default="optvq")

    # for ckpt callback
    finetune_parser.add_argument("--eval_start", type=int, default=200)
    finetune_parser.add_argument("--eval_step", type=int, default=200, help="Gap between evaluations")
    finetune_parser.add_argument("--top_k", type=int, default=5, help="Top k checkpoints to save")
    finetune_parser.add_argument("--core_metric", type=str, default="MCC")
    finetune_parser.add_argument("--ascending", type=bool, default=False)
    finetune_parser.add_argument("--ensemble", type=bool, default=True)
    finetune_parser.add_argument("--ensemble_only", type=bool, default=True)
    finetune_parser.add_argument("--test_in_end", type=bool, default=True)
    finetune_parser.add_argument("--save_best", type=bool, default=False)
    finetune_parser.add_argument("--save_by_steps", type=bool, default=False)

    args = parser.parse_args()
    # args.data_path="data/finetune/GUE"
    # args.ckpt_path="result/finetune"
    # args.pretrain_objective="MLM"
    # args.model_size="middle"
    # args.batch_size=64
    # args.max_lr=1e-4
    # args.min_lr=0
    # args.tokenizer="ordered_kmer"

    main(args)