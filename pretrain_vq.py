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
import torch.nn as nn
import transformers
import lightning as pl

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

from model.vq.configuration_vq import VQConfig
from model.vq.modeling_vq import VQModel, VQOutput
from model.utils import CosineLRSchedule

torch.set_float32_matmul_precision("medium")

# HF_CACHE = os.environ["HF_HOME"]
HF_CACHE = "/root/autodl-tmp/hf_cache"
os.environ["HF_HOME"] = HF_CACHE
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# API = d9f9f42bdc91cdc59144b5ed1dce53b098f17fd6

class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: AutoTokenizer, model_config: VQConfig, train_config: argparse.Namespace, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
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
    
    def _preprocess(self, raw):
        pad_token_id = self.tokenizer.vocab[self.tokenizer.pad_token]
        texts = [_ for _ in raw['sequence']]

        # chunk for the full-chr
        chunk_size = int(5e4)
        for i in range(len(texts)):
            chunk = []
            for pos in range(0, len(texts[i]), chunk_size): chunk.append(texts[i][pos:pos+chunk_size])
            texts[i] = chunk
        texts = list(chain(*texts))
        outputs = self.tokenizer(texts, add_special_tokens=False)['input_ids']
        
        max_seqlen = self.train_config.max_seqlen

        texts = list(chain(*outputs))
        texts = [texts[i:i+max_seqlen] if i + max_seqlen <= len(texts) else texts[i:i+max_seqlen] + [pad_token_id for j in range(i + self.train_config.max_seqlen - len(texts))] for i in range(0, len(texts), max_seqlen)]

        return {"input_ids": texts}
    
    def process(self, indices: list[int]):
        data = self.datas.select(indices)["input_ids"]
        input_ids, labels = data, data
            
        input_ids, labels = list(map(lambda x: torch.tensor(np.stack(x, axis=0), dtype=torch.int64), [input_ids, labels]))
        
        return dict(
            input_ids=input_ids,
            labels=labels
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
    def __init__(self, tokenizer: AutoTokenizer, model: PreTrainedModel, model_config: VQConfig, train_config: argparse.Namespace, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.model = model
        self.model_config = model_config
        self.train_config = train_config

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
            "reconstruct_loss": MeanMetric().to(self.trainer.strategy.root_device),
            "reconstruct_acc": MeanMetric().to(self.trainer.strategy.root_device),
            "commit_loss": MeanMetric().to(self.trainer.strategy.root_device),
            "codebook_usage": MeanMetric().to(self.trainer.strategy.root_device),
            "batch_size": SumMetric().to(self.trainer.strategy.root_device),
            
        }
    
    # @rank_zero_only
    def on_fit_end(self):
        self._train_tokens = self.train_metrics['train_tokens'].compute().item()
        if self.trainer.global_rank == 0:
            wall_clock = time.perf_counter() - self._start
            print(f"Total wall-clock time: {wall_clock:.4f}")
            print(f"Total train tokens: {self._train_tokens:.4f}")
            print("------------ End Training ------------")
        
        self.trainer.save_checkpoint(os.path.join(self.train_config.ckpt_path, self._date, "pytorch_model.bin"), weights_only=True)

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
    
    def forward(self, data: Dict):
        outputs: VQOutput = self.model(
            input_ids=data["input_ids"]
        )
        return outputs
    
    def training_step(self, batch: Dict, batch_idx):
        outputs: VQOutput = self(batch)
        self._train_tokens += batch["input_ids"].size(0) * batch["input_ids"].size(1)

        self.train_metrics["loss"].update(outputs.loss)
        self.train_metrics["reconstruct_loss"].update(outputs.reconstruct_loss)
        self.train_metrics["reconstruct_acc"].update(outputs.reconstruct_acc)
        self.train_metrics["commit_loss"].update(outputs.commit_loss)
        self.train_metrics["codebook_usage"].update(outputs.codebook_usage)
        self.train_metrics["batch_size"].update(batch["input_ids"].size(0))
        
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

            for k, v in self.train_metrics.items():
                if isinstance(v, Metric): v.reset()

        return outputs.loss

def main(train_config: argparse.Namespace):
    pl.seed_everything(train_config.seed)

    base_path = os.path.dirname(os.path.abspath(__file__))
    train_config.data_path = os.path.join(base_path, train_config.data_path)
    train_config.ckpt_path = os.path.join(base_path, train_config.ckpt_path)
    logger_path = train_config.ckpt_path if not os.path.exists("/root/tf-logs") else "/root/tf-logs"
    logger_name = f"Pretrain|VQ|{train_config.codebook_size}|{train_config.downsample_rate}"

    if logger_path == "/root/tf-logs":
        logger_path = os.path.join(logger_path, *logger_name.split('|'))

    train_config.ckpt_path = os.path.join(train_config.ckpt_path, *logger_name.split('|'))

    if not os.path.exists(train_config.ckpt_path): os.makedirs(train_config.ckpt_path)

    model_config = VQConfig(codebook_size=train_config.codebook_size, downsample_rate=train_config.downsample_rate)

    assert (train_config.max_epochs != -1 or train_config.max_steps != -1)

    if torch.cuda.device_count() > 1:
        # parallelism_strategy = ModelParallelStrategy(data_parallel_size=torch.cuda.device_count(), tensor_parallel_size=1, save_distributed_checkpoint=False)
        parallelism_strategy = FSDPStrategy(
            sharding_strategy="SHARD_GRAD_OP",
            state_dict_type="full",
            cpu_offload=False
        )
    else:
        parallelism_strategy = SingleDeviceStrategy(device="cuda:0")
    trainer = Trainer(
        precision=train_config.precision,
        strategy=parallelism_strategy,
        max_epochs=train_config.max_epochs,
        max_steps=train_config.max_steps,
        default_root_dir=train_config.ckpt_path,
        accumulate_grad_batches=train_config.accumulate_grad_batches,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        logger=WandbLogger(
            project="VQ|Pretrain_VQ",
            name=logger_name,
            save_dir=logger_path
        ),
        log_every_n_steps=20,
        enable_checkpointing=False
    )
    if train_config.max_seqlen > 0: train_config.batch_size = train_config.max_token_per_batch // train_config.max_seqlen

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

    model = VQModel(model_config)
    datamodule = PretrainDataModule(tokenizer, model_config, train_config)

    if train_config.max_epochs != -1:
        epoch_steps = (len(datamodule.data) * train_config.max_epochs + raw_batch_size - 1) // raw_batch_size
        train_config.max_steps = min(train_config.max_steps, epoch_steps) if train_config.max_steps > 0 else epoch_steps

    plmodel = PretrainModule(tokenizer, model, model_config, train_config)
    trainer.fit(plmodel, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset Args:
    pretrain_parser = parser.add_argument_group("pretrain")
    pretrain_parser.add_argument("--seed", type=int, default=17)
    pretrain_parser.add_argument("--data_path", type=str, help="Path to the dataset dir", default="data/pretrain")
    pretrain_parser.add_argument("--ckpt_path", type=str, help="Path to the checkpoint", default="result/pretrain")

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

    pretrain_parser.add_argument("--codebook_size", type=int, default=4096)
    pretrain_parser.add_argument("--downsample_rate", type=int, default=16)
    pretrain_parser.add_argument("--vq_type", type=str, default="")

    args = parser.parse_args()
    args.data_path="data/pretrain/fungi"
    args.ckpt_path="result/pretrain"
    args.max_seqlen=16384
    args.max_token_per_batch=262144
    args.accumulate_grad_batches=1
    args.num_preprocess_workers=12
    args.codebook_size=4096
    args.downsample_rate=16
    args.vq_type=""
    main(args)