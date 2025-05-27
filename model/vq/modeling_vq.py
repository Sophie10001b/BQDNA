import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import transformers

from typing import Optional, Dict, Tuple, List, Union, Unpack, Sequence, Any
from collections import OrderedDict
from flash_attn.ops.triton.layer_norm import RMSNorm
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from einops import rearrange
from itertools import chain
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

from .configuration_vq import VQConfig
from transformers.modeling_utils import PreTrainedModel

@torch.no_grad()
def linear_init(
    linear: nn.Linear,
    distribution: Optional[str]='normal',
    zero_bias: Optional[bool]=False,
    gain: Optional[float]=1.0
) ->None:
    if distribution == 'normal':
        nn.init.xavier_normal_(linear.weight, gain=gain)
    elif distribution == 'uniform':
        nn.init.xavier_uniform_(linear.weight, gain=gain)
    if linear.bias is not None:
        if zero_bias:
            nn.init.zeros_(linear.bias)
        else:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(linear.bias, -bound, bound)

@torch.no_grad()
def embedding_init(embedding: nn.Embedding, distribution: str='normal') ->None:
    if distribution == 'normal':
        fan_out = embedding.weight.size(1)
        std = 1.0 * math.sqrt(1.0 / float(fan_out))
        nn.init.normal_(embedding.weight, 0., std)
    elif distribution == 'uniform':
        nn.init.uniform_(embedding.weight, -1 / embedding.num_embeddings, 1 / embedding.num_embeddings)
    else: raise ValueError(f"Distribution {distribution} not support for embedding initialization")

    if embedding.padding_idx is not None:
        embedding.weight[embedding.padding_idx].fill_(0)

@dataclass
class VQOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    reconstruct_loss: Optional[torch.Tensor] = None
    reconstruct_acc: Optional[torch.Tensor] = None
    commit_loss: Optional[torch.Tensor] = None
    codebook_usage: Optional[torch.Tensor] = None

    codebook_emb: Optional[torch.Tensor] = None
    codebook_idx: Optional[torch.LongTensor] = None


class Downsample(nn.Module):
    def __init__(self, config: VQConfig):
        super().__init__()

        self.config = config
        self.pre_linear = nn.Linear(config.latent_size, config.latent_size, bias=False)
        self.norm1 = RMSNorm(hidden_size=config.latent_size, eps=config.eps)
        self.activation = nn.SiLU()
        self.conv = nn.Conv1d(config.latent_size, config.latent_size, config.kernel_size, 2, bias=False)
        self.norm2 = RMSNorm(hidden_size=config.latent_size, eps=config.eps)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): linear_init(m, zero_bias=True)
    
    def forward(self, x: torch.Tensor):
        x = self.activation(self.norm1(self.pre_linear(x) + x))

        output_length = math.ceil(x.size(1) / 2)
        padding_length = 2 * (output_length - 1) + self.config.kernel_size - x.size(1)
        x = torch.nn.functional.pad(x, (0, 0, 0, padding_length), "constant", 0)

        x = self.conv(rearrange(x, "B L D -> B D L"))
        x = rearrange(x, "B D L -> B L D")
        return self.norm2(x)

class VQEncoder(nn.Module):
    def __init__(self, config: VQConfig):
        super().__init__()

        self.config = config
        self.downsample = nn.ModuleList([
            Downsample(config) for _ in range(int(math.log2(config.downsample_rate)))
        ])
        self.post_linear = nn.Linear(config.latent_size, config.latent_size, bias=False)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): linear_init(m, zero_bias=True)
    
    def forward(self, x: torch.Tensor):
        for downsample in self.downsample:
            x = downsample(x)
        return self.post_linear(x)

class Upsample(nn.Module):
    def __init__(self, config: VQConfig):
        super().__init__()

        self.config = config
        self.pre_linear = nn.Linear(config.latent_size, config.latent_size, bias=False)
        self.norm1 = RMSNorm(hidden_size=config.latent_size, eps=config.eps)
        self.activation = nn.SiLU()
        self.conv = nn.ConvTranspose1d(config.latent_size, config.latent_size, config.kernel_size, 2, bias=False)
        self.norm2 = RMSNorm(hidden_size=config.latent_size, eps=config.eps)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): linear_init(m, zero_bias=True)

    def forward(self, x: torch.Tensor):
        x = self.activation(self.norm1(self.pre_linear(x) + x))

        output_length = x.size(1) * 2
        x = self.conv(rearrange(x, "B L D -> B D L"))
        x = rearrange(x, "B D L -> B L D")[:, :output_length]
        return self.norm2(x)

class VQDecoder(nn.Module):
    def __init__(self, config: VQConfig):
        super().__init__()

        self.config = config
        self.downsample = nn.ModuleList([
            Upsample(config) for _ in range(int(math.log2(config.downsample_rate)))
        ])
        self.post_linear = nn.Linear(config.latent_size, config.latent_size, bias=False)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): linear_init(m, zero_bias=True)
    
    def forward(self, x: torch.Tensor):
        for downsample in self.downsample:
            x = downsample(x)
        return self.post_linear(x)


class EMACache(nn.Module):
    def __init__(self, init_value: torch.Tensor, decay: float):
        super().__init__()
        self.decay = decay
        self.count = 0
        self.register_buffer("value", init_value)
    
    def forward(self, x: torch.Tensor):
        self.count += 1
        self.value = self.decay * self.value + (1 - self.decay) * x
        return self.value / (1 - self.decay ** self.count)

class VQModel(PreTrainedModel):
    def __init__(self, config: VQConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.pre_linear = nn.Linear(config.hidden_size, config.latent_size, bias=False)
        self.post_linear = nn.Linear(config.latent_size, config.hidden_size, bias=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.base_vocab = nn.Embedding(config.vocab_size, config.hidden_size)
        self.codebook = nn.Embedding(config.codebook_size, config.latent_size)

        self.encoder = VQEncoder(config)
        self.decoder = VQDecoder(config)

        # EMA settings
        self.beta = config.beta
        self.eps = config.eps
        self.ema_cluster_size = EMACache(torch.zeros(config.codebook_size), decay=config.gamma)
        self.ema_weight = EMACache(self.codebook.weight.clone(), decay=config.gamma)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): linear_init(m, zero_bias=True)
        
        embedding_init(self.base_vocab)
        embedding_init(self.codebook, distribution="uniform")

    def forward(self, input_ids: torch.LongTensor):
        x = self.pre_linear(self.base_vocab(input_ids))
        x = self.encoder(x)

        # codebook lookup
        dist = torch.cdist(x, self.codebook.weight, p=2)
        encode_idx = torch.argmin(dist, dim=-1)
        x_encoded = self.codebook(encode_idx)

        if self.training:
            with torch.no_grad():
                encode_idx_onehot = torch.nn.functional.one_hot(encode_idx, num_classes=self.config.codebook_size).flatten(0, 1)
                cluster_size = encode_idx_onehot.sum(0)
                updated_ema_cluster_size: torch.Tensor = self.ema_cluster_size(cluster_size)

                total_cluster_size = updated_ema_cluster_size.sum(dim=-1, keepdim=True)
                updated_ema_cluster_size = (updated_ema_cluster_size + self.eps) * total_cluster_size / (total_cluster_size + self.config.codebook_size * self.eps)
                new_ema_weight = encode_idx_onehot.float().transpose(0, 1) @ x.flatten(0, 1)
                updated_ema_weight = self.ema_weight(new_ema_weight)

                self.codebook.weight.data = updated_ema_weight / updated_ema_cluster_size.unsqueeze(-1)

            x_decoded = self.decoder(x + (x_encoded - x).detach()) # STE
            x_decoded = self.lm_head(self.post_linear(x_decoded))

            # compute loss
            reconstruct_ce = CrossEntropyLoss(ignore_index=self.config.pad_token_id, reduction='mean')
            commit_mse = torch.nn.MSELoss()

            l_reconstruct = reconstruct_ce(x_decoded.flatten(0, 1), input_ids.flatten())
            l_commit = commit_mse(x.flatten(0, 1), x_encoded.flatten(0, 1).detach())

            codebook_usage = (torch.bincount(encode_idx.flatten(), minlength=self.codebook.num_embeddings) > 0).sum().item() / self.codebook.num_embeddings
            reconstruct_acc = (torch.softmax(x_decoded, dim=-1).argmax(-1) == input_ids).sum(-1) / (input_ids != self.config.pad_token_id).sum(-1)
            reconstruct_acc = reconstruct_acc.mean()

            return VQOutput(
                loss=l_reconstruct + l_commit * self.beta,
                reconstruct_loss=l_reconstruct,
                reconstruct_acc=reconstruct_acc,
                commit_loss=l_commit,
                codebook_usage=codebook_usage
            )

        else:
            return VQOutput(
                codebook_emb=x_encoded,
                codebook_idx=encode_idx
            )