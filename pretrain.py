import os
import random
import time
import glob
import argparse
import pandas
import json
import numpy as np
import torch
import torch.distributed.fsdp
import torch.distributed.tensor
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
from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy, ModelParallelStrategy, SingleDeviceStrategy
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput, SequenceClassifierOutput
from torchmetrics import MeanMetric, SumMetric, Metric
from swanlab.integration.pytorch_lightning import SwanLabLogger

from model.vq.configuration_vq import VQConfig
from model.vq.modeling_vq import VQModel, VQOutput
from model.configuration_transformer import TransformerConfig
from model.modeling_transformer import TransformerForCausalLM
from model.utils import CosineLRSchedule
from model.parallelism_transformer import parallelize_setting
from model.vq.parallelism_vq import parallelize_setting as vq_parallelize_setting

torch.set_float32_matmul_precision("medium")

# HF_CACHE = os.environ["HF_HOME"]
HF_CACHE = "/root/autodl-tmp/hf_cache"
os.environ["HF_HOME"] = HF_CACHE
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# API = d9f9f42bdc91cdc59144b5ed1dce53b098f17fd6

class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: AutoTokenizer, model_config: VQConfig, train_config: argparse.Namespace, **kwargs):
        super().__init__(**kwargs)

        self.model_config = model_config
        self.train_config = train_config

        data_files = glob.glob(self.train_config.data_path + "/**/*.parquet", recursive=True)
        
        # static results about the dataset
        species_stats = {}
        for _data_dir in data_files:
            _dir = _data_dir[len(self.train_config.data_path)+1:]
            species = _dir.split("/")[0]
            data_size = os.path.getsize(_data_dir) / (1024 * 1024)

            if species not in species_stats: species_stats[species] = data_size
            else: species_stats[species] += data_size
        
        print("Species:")
        for (k, v) in species_stats.items(): print(f"{k}:\t {v:.2f} MB")
        print(f"\nTotal:\t {sum(species_stats.values()):.2f} MB\n")

        datas: Dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=False, trust_remote_code=True, columns=["sequence"], cache_dir=HF_CACHE, num_proc=train_config.num_preprocess_workers)
        datas = datas.shuffle(seed=self.train_config.seed)

        # pre-chunk
        self.datas = datas.map(
            self._preprocess,
            batched=True,
            batch_size=100,
            writer_batch_size=100,
            num_proc=train_config.num_preprocess_workers,
            remove_columns=datas.column_names,
            load_from_cache_file=True,
            cache_file_name=os.path.join(HF_CACHE, f"parquet/pretrain/pretrain_map.cache")
        )
        self.tokenizer = tokenizer
    
    def _preprocess(self, raw):
        texts = [_ for _ in raw['sequence']]

        # chunk for the full-chr
        chunk_size = self.train_config.max_chunk_size
        for i in range(len(texts)):
            chunk = []
            for pos in range(0, len(texts[i]), chunk_size): chunk.append(texts[i][pos:pos+chunk_size])
            texts[i] = chunk
        texts = list(chain(*texts))
        return {"input_ids": texts}
    
    def process(self, indices: list[int]):
        data = self.datas.select(indices)["input_ids"]
        data = self.tokenizer(data, return_tensors="pt", add_special_tokens=False, padding="longest", padding_side="right")['input_ids']
        return dict(
            input_ids=data,
            labels=data
        )

    def __getitem__(self, idx: int):
        return idx
    
    def __len__(self):
        return len(self.datas)

class PretrainDataModule(LightningDataModule):
    def __init__(self, tokenizer: AutoTokenizer, model_config: VQConfig, train_config: argparse.Namespace, **kwargs):
        super().__init__(**kwargs)

        self.model_config = model_config
        self.train_config = train_config

        self.data = PretrainDataset(tokenizer, model_config, train_config, **kwargs)
    
    def setup(self, stage):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=self.train_config.num_workers,
            collate_fn=self.data.process,
            pin_memory=True,
            persistent_workers=True
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
            # remove exists ckpt:
            prev_ckpt = glob.glob(os.path.join(self.train_config.ckpt_path, "*.ckpt"))
            for ckpt in prev_ckpt: os.remove(ckpt)

            print(f"--------------- Settings ---------------")
            for (k, v) in self.hparams.items(): print(f"{k}:\t {v}")
            print(f"------------- Architecture -------------")
            print(self.model)
            print(f"Total Param: {sum([p.numel() for p in self.model.parameters()])}")
            print(f"------------ Start Training ------------")
            self._start = time.perf_counter()
        
        self.logger.log_hyperparams(self.train_config)
        
        # metrics
        self.train_metrics = {
            "loss": MeanMetric().to(self.trainer.strategy.root_device),
            "batch_size": SumMetric().to(self.trainer.strategy.root_device),
            "train_tokens": SumMetric().to(self.trainer.strategy.root_device),
            "codebook_usage": MeanMetric().to(self.trainer.strategy.root_device),
        }
    
    # @rank_zero_only
    def on_fit_end(self):
        self._train_tokens = self.train_metrics['train_tokens'].compute().item()
        if self.trainer.global_rank == 0:
            wall_clock = time.perf_counter() - self._start
            print(f"Total wall-clock time: {wall_clock:.4f}")
            print(f"Total train tokens: {self._train_tokens:.4f}")
            print("------------ End Training ------------")
            swanlab.finish()
        
        self.trainer.save_checkpoint(os.path.join(self.train_config.ckpt_path, self._date, "pytorch_model.bin"), weights_only=True)
        if self.trainer.is_global_zero:
            ckpt = torch.load(os.path.join(self.train_config.ckpt_path, self._date, "pytorch_model.bin"), map_location='cpu', weights_only=True)["state_dict"]
            cache = {}
            for k, v in ckpt.items():
                head = k.split(".")[0]
                if head == "vq": continue
                new_k = k.split(".")[1:]
                new_k = ".".join(new_k)
                cache[new_k] = v
            ckpt = cache
            torch.save(ckpt, os.path.join(self.train_config.ckpt_path, self._date, "pytorch_model.bin"))
        

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.max_lr,
            eps=self.model_config.eps,
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
    
    def configure_model(self):
        if isinstance(self.trainer.strategy, ModelParallelStrategy):
            self.model = parallelize_setting(self.model, self.device_mesh)
            self.vq = vq_parallelize_setting(self.vq, self.device_mesh)
    
    def forward(self, data: Dict):
        with torch.no_grad():
            input_ids = data['input_ids']
            inputs_embeds, input_ids, _ = self.vq.encode(input_ids)
            codebook_usage = (torch.bincount(input_ids.flatten(), minlength=self.vq_config.codebook_size) > 0).to(torch.int64)
            codebook_usage = codebook_usage.sum() / self.vq_config.codebook_size

        self.train_metrics['codebook_usage'].update(codebook_usage)

        bos_ids = torch.full((input_ids.size(0), 1), self.model_config.bos_token_id, dtype=torch.int64, device=self.trainer.strategy.root_device)
        eos_ids = torch.full((input_ids.size(0), 1), self.model_config.eos_token_id, dtype=torch.int64, device=self.trainer.strategy.root_device)
        input_ids = torch.cat([bos_ids, input_ids + self.vq.base_vocab.num_embeddings, eos_ids], dim=-1)

        inputs_embeds = torch.cat([self.vq.base_vocab(bos_ids), inputs_embeds, self.vq.base_vocab(eos_ids)], dim=1)
        inputs_embeds = self.model.model.latent_project(inputs_embeds)

        data['input_ids'] = input_ids
        outputs: SequenceClassifierOutput = self.model(
            inputs_embeds=inputs_embeds,
            labels=data['input_ids']
        )
        return outputs
    
    def training_step(self, batch: Dict, batch_idx):
        outputs: VQOutput = self(batch)

        self.train_metrics["loss"].update(outputs.loss)
        self.train_metrics["batch_size"].update(batch["input_ids"].size(0))
        self.train_metrics["train_tokens"].update(batch["input_ids"].size(0) * batch["input_ids"].size(1))
        
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            for k, v in self.train_metrics.items():
                if isinstance(v, Metric):
                    self.log(f"pretrain/{k}", v.compute(), prog_bar=False)

            lr = self.optimizers().optimizer.param_groups[0]["lr"]
            steps = self.trainer.global_step

            self.log_dict(
                dict(loss=self.train_metrics['loss'].compute(), lr=lr, steps=steps, batch_size=self.train_metrics['batch_size'].compute()),
                prog_bar=True,
                logger=False
            )
            self.log("pretrain/lr", lr, prog_bar=False)

            self.train_metrics["loss"].reset()
            self.train_metrics["batch_size"].reset()

        return outputs.loss

def main(train_config: argparse.Namespace):
    pl.seed_everything(train_config.seed)

    base_path = os.path.dirname(os.path.abspath(__file__))
    train_config.data_path = os.path.join(base_path, train_config.data_path)
    train_config.ckpt_path = os.path.join(base_path, train_config.ckpt_path)
    logger_path = train_config.ckpt_path if not os.path.exists("/root/tf-logs") else "/root/tf-logs"
    logger_name = f"VQTransformer|CLM"

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

    if torch.cuda.device_count() > 1:
        # parallelism_strategy = ModelParallelStrategy(data_parallel_size=torch.cuda.device_count(), tensor_parallel_size=1, save_distributed_checkpoint=False)
        parallelism_strategy = DDPStrategy()
    else:
        parallelism_strategy = SingleDeviceStrategy(device="cuda:0")

    logger = SwanLabLogger(
        project=train_config.logger_project,
        experiment_name=logger_name,
        save_dir=logger_path
    )
    trainer = Trainer(
        precision=train_config.precision,
        strategy=parallelism_strategy,
        max_epochs=train_config.max_epochs,
        max_steps=train_config.max_steps,
        default_root_dir=train_config.ckpt_path,
        accumulate_grad_batches=train_config.accumulate_grad_batches,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        logger=logger,
        log_every_n_steps=20,
        enable_checkpointing=False
    )
    if train_config.batch_size == -1: train_config.batch_size = train_config.max_token_per_batch // train_config.max_seqlen

    raw_batch_size = train_config.batch_size
    if trainer.num_devices > 1:
        train_config.batch_size = train_config.batch_size // trainer.num_devices
        train_config.max_token_per_batch = train_config.max_token_per_batch // trainer.num_devices
    if trainer.accumulate_grad_batches > 1: 
        train_config.batch_size = train_config.batch_size // trainer.accumulate_grad_batches
        train_config.max_token_per_batch = train_config.max_token_per_batch // trainer.accumulate_grad_batches

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(os.path.join(base_path, "tokenizer/base"), use_fast=True, trust_remote_code=True, local_files_only=True)
    model_config.vocab_size = tokenizer.vocab_size
    model_config.bos_token_id = tokenizer.vocab[tokenizer.bos_token]
    model_config.eos_token_id = tokenizer.vocab[tokenizer.eos_token]
    model_config.pad_token_id = tokenizer.vocab[tokenizer.pad_token]

    vq = VQModel(vq_config)
    vq_state_dict = torch.load(os.path.join(train_config.vq_path, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    vq.load_state_dict(vq_state_dict)

    model = TransformerForCausalLM(model_config, tokenizer.vocab_size + vq_config.codebook_size, vq_config.latent_size)
    datamodule = PretrainDataModule(tokenizer, model_config, train_config)

    if train_config.max_epochs != -1:
        epoch_steps = (len(datamodule.data) * train_config.max_epochs + raw_batch_size - 1) // raw_batch_size
        train_config.max_steps = min(train_config.max_steps, epoch_steps) if train_config.max_steps > 0 else epoch_steps

    plmodel = PretrainModule(model, vq, model_config, vq_config, train_config)
    trainer.fit(plmodel, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset Args:
    pretrain_parser = parser.add_argument_group("pretrain")
    pretrain_parser.add_argument("--seed", type=int, default=17)
    pretrain_parser.add_argument("--data_path", type=str, help="Path to the dataset dir", default="data/pretrain")
    pretrain_parser.add_argument("--ckpt_path", type=str, help="Path to the checkpoint", default="result/pretrain")
    pretrain_parser.add_argument("--vq_path", type=str, default="")
    pretrain_parser.add_argument("--logger_project", type=str, default="OrderedKmer-Pretrain")

    pretrain_parser.add_argument("--max_chunk_size", type=int, default=16384)
    pretrain_parser.add_argument("--max_seqlen", type=int, default=2048)
    pretrain_parser.add_argument("--max_token_per_batch", type=int, default=524288)
    pretrain_parser.add_argument("--batch_size", type=int, default=-1)
    pretrain_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients for every n batches")
    pretrain_parser.add_argument("--num_workers", type=int, default=4)
    pretrain_parser.add_argument("--num_preprocess_workers", type=int, default=32)

    pretrain_parser.add_argument("--max_steps", type=int, default=-1)
    pretrain_parser.add_argument("--max_epochs", type=int, default=1)
    pretrain_parser.add_argument("--save_steps", type=int, default=-1)
    pretrain_parser.add_argument("--max_lr", type=float, default=5e-4, help="Maximum learning rate")
    pretrain_parser.add_argument("--min_lr", type=float, default=0, help="Minimum learning rate")
    pretrain_parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of steps to warm up learning rate")
    pretrain_parser.add_argument("--precision", type=str, default="bf16-mixed")

    # vq
    pretrain_parser.add_argument("--latent_size", type=int, default=512)
    pretrain_parser.add_argument("--codebook_size", type=int, default=8192)
    pretrain_parser.add_argument("--downsample_rate", type=int, default=16)
    pretrain_parser.add_argument("--num_heads", type=int, default=1)
    pretrain_parser.add_argument("--vq_type", type=str, default="optvq")

    args = parser.parse_args()
    # args.data_path="data/pretrain"
    # args.ckpt_path="result/pretrain"
    # args.vq_path="/root/autodl-tmp/vq/result/pretrain/Pretrain/VQ/8192/16/2025-05-28 20:29:30/pytorch_model.bin"
    # args.max_seqlen=16384
    # args.max_token_per_batch=1048576 * 2
    # args.accumulate_grad_batches=1
    # args.num_preprocess_workers=48
    # args.latent_size=256
    # args.codebook_size=8192
    # args.downsample_rate=16
    # args.vq_type="optvq"
    # args.max_steps=100
    main(args)