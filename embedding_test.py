import os
import sys
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
from einops import rearrange
from tqdm import tqdm
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
            test=FinetuneDataset(self.tokenizer, train_config, "test", self.num_seqs, self.label_offset),
        )
        self.num_class = self.data["test"].num_class
        if self.data["test"].valid_labels is not None:
            print(f"Valid labels: {",".join([str(_) for _ in self.data["test"].valid_labels])}")
    
    def setup(self, stage):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data["train"],
            batch_size=self.train_config.batch_size,
            collate_fn=self.data["train"].process,
            num_workers=self.train_config.num_workers
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data["test"],
            batch_size=self.train_config.batch_size,
            collate_fn=self.data["test"].process,
            num_workers=self.train_config.num_workers
        )

def main(train_config: argparse.Namespace):
    pl.seed_everything(train_config.seed)

    base_path = os.path.dirname(os.path.abspath(__file__))
    train_config.data_path = os.path.join(base_path, train_config.data_path, train_config.data_name)
    logger_path = train_config.ckpt_path if not os.path.exists("/root/tf-logs") else "/root/tf-logs"
    logger_name = f"VQ|Embedding_test|{train_config.data_name}|{train_config.latent_size}|{train_config.codebook_size}|{train_config.downsample_rate}|{train_config.num_heads}|{train_config.vq_type}"

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

    logger = swanlab.init(
        project="VQ-Embedding_test",
        experiment_name=logger_name,
        logdir=logger_path
    )

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(os.path.join(base_path, "tokenizer/base"), use_fast=True, trust_remote_code=True, local_files_only=True)
    model_config.vocab_size = tokenizer.vocab_size
    model_config.bos_token_id = tokenizer.vocab[tokenizer.bos_token]
    model_config.eos_token_id = tokenizer.vocab[tokenizer.eos_token]
    model_config.pad_token_id = tokenizer.vocab[tokenizer.pad_token]

    vq = VQModel(vq_config).eval()
    vq.requires_grad_(False)

    vq_state_dict = torch.load(os.path.join(train_config.vq_path, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    vq.load_state_dict(vq_state_dict)
    vq_codebook = vq.get_codebook()
    vq_codebook.requires_grad_(False)

    datamodule = FinetuneDataModule(tokenizer, train_config)

    device = "cuda:0"
    vq = vq.to(device)

    all_embs = {
        "train": {"predict": None, "label": None, "indices": None},
        "test": {"predict": None, "label": None, "indices": None}
    }

    def maybe_concat(src: np.ndarray, split_name: str, query: str, axis: int=0):
        if all_embs[split_name][query] is None: all_embs[split_name][query] = src
        else: all_embs[split_name][query] = np.concatenate([all_embs[split_name][query], src], axis=axis)

    with torch.no_grad():
        for split_name, loader in zip(['train', 'test'], [datamodule.train_dataloader(), datamodule.test_dataloader()]):
            for data in tqdm(loader):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    res = vq.encode(data['input_ids'])
                    emb: torch.Tensor = res['quant']
                    idx: torch.Tensor = res['indices']
                
                emb = emb.mean(1).to(device="cpu", dtype=torch.float).numpy()
                idx = idx.to(device="cpu", dtype=torch.long).numpy()

                maybe_concat(emb, split_name, "predict")
                maybe_concat(idx, split_name, "indices")
                maybe_concat(data['labels'].to(device="cpu", dtype=torch.long).numpy(), split_name, "label")
    
    import xgboost
    from matplotlib import pyplot as plt
    plt.rcParams["font.size"] = 20

    train_predict, train_label, train_indices = all_embs["train"]["predict"], all_embs["train"]["label"], all_embs["train"]["indices"]
    test_predict, test_label, test_indices = all_embs["test"]["predict"], all_embs["test"]["label"], all_embs["test"]["indices"]

    train_indices: np.ndarray = train_indices
    indices_bincount = np.bincount(train_indices.reshape(-1), minlength=vq_config.codebook_size)
    indices_bincount = np.sort(indices_bincount)[::-1]
    codebook_usage = (indices_bincount > 0).sum().item() / vq_config.codebook_size
    logger.log({"codebook_usage": codebook_usage})

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(indices_bincount, linewidth=2)
    ax.set_yscale('log')
    ax.set_ylabel("Usage Frequency")
    img = swanlab.Image(fig, caption=f"{train_config.data_name}")
    logger.log({"codebook_usage": img})
    plt.close()
    plt.clf()

    classifier = xgboost.XGBClassifier()
    classifier.fit(train_predict, train_label)
    preds = classifier.predict(test_predict)

    from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
    f1 = f1_score(test_label, preds, average="macro")
    mcc = matthews_corrcoef(test_label, preds)
    accuracy = accuracy_score(test_label, preds)
    logger.log({"test/f1": f1, "test/mcc": mcc, "test/accuracy": accuracy}, print_to_console=True)
    
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=train_config.seed, n_jobs=-1)
    x_tsne = tsne.fit_transform(train_predict)
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(x_tsne[:, 0], x_tsne[:, 1], c=train_label, cmap=plt.get_cmap("tab10"), s=10)

    # remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # remove top and right spines
    fig.gca().spines["top"].set_visible(False)
    fig.gca().spines["bottom"].set_visible(False)
    fig.gca().spines["right"].set_visible(False)
    fig.gca().spines["left"].set_visible(False)

    label_legend = None
    if train_config.data_name == "NT_biotype":
        label_legend = ["Histone Markers", "Enhancers", "Promoters", "Splice Site"] 
    elif train_config.data_name in ["species_1024_10000_1000_1000", "species_16384_10000_1000_1000"]:
        label_legend = ["Human", "Lemur", "Mouse", "Pig", "Hippo"]
    elif train_config.data_name == "gene_type":
        label_legend = {"control", "tRNA", "cds", "pseudo", "rRNA", "ncRNA", "misc_RNA"}
    elif train_config.data_name == "species_type":
        label_legend = {"vertebrate_other", "invertebrate", "plant", "fungi", "vertebrate_mammalian", "protozoa"}
    
    if label_legend is not None:
        ax.legend(handles=scatter.legend_elements()[0], labels=label_legend, loc="upper right")

    img = swanlab.Image(fig, caption=f"{train_config.data_name}")
    logger.log({"summary_embedding": img})
    swanlab.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset Args:
    finetune_parser = parser.add_argument_group("finetune")
    finetune_parser.add_argument("--seed", type=int, default=17)
    finetune_parser.add_argument("--data_path", type=str, help="Path to the dataset dir", default="data/finetune")
    finetune_parser.add_argument("--data_name", type=str, help="Name of the dataset", default="tf/0")
    finetune_parser.add_argument("--dataset_name", type=str, default="")
    finetune_parser.add_argument("--ckpt_path", type=str, help="Path to the checkpoint", default="result/finetune")
    finetune_parser.add_argument("--vq_path", type=str, default="")

    finetune_parser.add_argument("--batch_size", type=int, default=32)
    finetune_parser.add_argument("--num_workers", type=int, default=4)
    finetune_parser.add_argument("--num_preprocess_workers", type=int, default=32)

    finetune_parser.add_argument("--logger_project", type=str)

    # vq
    finetune_parser.add_argument("--latent_size", type=int, default=512)
    finetune_parser.add_argument("--codebook_size", type=int, default=8192)
    finetune_parser.add_argument("--downsample_rate", type=int, default=16)
    finetune_parser.add_argument("--num_heads", type=int, default=1)
    finetune_parser.add_argument("--vq_type", type=str, default="optvq")

    args = parser.parse_args()

    main(args)