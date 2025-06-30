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
from mamba_ssm import Mamba2
from causal_conv1d import causal_conv1d_fn
from einops import rearrange
from itertools import chain
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from fla.modules import ShortConvolution

from .configuration_vq import VQConfig
from .VQ import OptVQ, LFQ, VanillaVQ, IBQ
# from .lfq import LFQ
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

@torch.no_grad()
def factorize(num: int):
    res = []
    while num % 2 == 0:
        num //= 2
        res.append(2)
    
    while num % 3 == 0:
        num //= 3
        res.append(3)
    
    if num != 1: res = [num]
    
    res.reverse()
    return res

@dataclass
class VQOutput(ModelOutput):
    quant: Optional[torch.Tensor] = None
    indices: Optional[torch.LongTensor] = None
    loss: Optional[torch.Tensor] = None
    reconstruct_loss: Optional[torch.Tensor] = None
    reconstruct_acc: Optional[torch.Tensor] = None
    commit_loss: Optional[torch.Tensor] = None
    diversity_loss: Optional[torch.Tensor] = None
    codebook_usage: Optional[torch.Tensor] = None
    codebook_ppl: Optional[torch.Tensor] = None

    codebook_emb: Optional[torch.Tensor] = None
    codebook_idx: Optional[torch.LongTensor] = None


class Downsample(nn.Module):
    def __init__(self, config: VQConfig, stride: int):
        super().__init__()

        self.config = config
        self.stride = stride
        self.pre_linear = nn.Linear(config.latent_size, config.latent_size, bias=False)
        self.norm1 = RMSNorm(hidden_size=config.latent_size, eps=config.eps)
        self.activation = nn.SiLU()
        self.conv = nn.Conv1d(config.latent_size, config.latent_size, max(config.kernel_size, stride), stride, bias=False)
        self.norm2 = RMSNorm(hidden_size=config.latent_size, eps=config.eps)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): linear_init(m, zero_bias=True)
    
    def forward(self, x: torch.Tensor):
        x = self.activation(self.norm1(self.pre_linear(x) + x))

        output_length = math.ceil(x.size(1) / self.stride)
        padding_length = self.stride * (output_length - 1) + self.conv.kernel_size[0] - x.size(1)
        x = torch.nn.functional.pad(x, (0, 0, 0, padding_length), "constant", 0)

        x = self.conv(rearrange(x, "B L D -> B D L"))
        x = rearrange(x, "B D L -> B L D")
        return self.norm2(x)

class VQEncoder(nn.Module):
    def __init__(self, config: VQConfig):
        super().__init__()

        self.config = config
        downsample_list = factorize(config.downsample_rate)
        self.downsample = nn.ModuleList([
            Downsample(config, _) for _ in downsample_list
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
    def __init__(self, config: VQConfig, stride: int):
        super().__init__()

        self.config = config
        self.stride = stride
        self.pre_linear = nn.Linear(config.latent_size, config.latent_size, bias=False)
        self.norm1 = RMSNorm(hidden_size=config.latent_size, eps=config.eps)
        self.activation = nn.SiLU()
        self.conv = nn.ConvTranspose1d(config.latent_size, config.latent_size, max(config.kernel_size, stride), stride, bias=False)
        self.norm2 = RMSNorm(hidden_size=config.latent_size, eps=config.eps)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): linear_init(m, zero_bias=True)

    def forward(self, x: torch.Tensor):
        x = self.activation(self.norm1(self.pre_linear(x) + x))

        output_length = x.size(1) * self.stride
        x = self.conv(rearrange(x, "B L D -> B D L"))
        x = rearrange(x, "B D L -> B L D")[:, :output_length]
        return self.norm2(x)

class VQDecoder(nn.Module):
    def __init__(self, config: VQConfig):
        super().__init__()

        self.config = config
        upsample_list = factorize(config.downsample_rate)
        self.upsample = nn.ModuleList([
            Upsample(config, _) for _ in upsample_list
        ])

        self.post_linear = nn.Linear(config.latent_size, config.latent_size, bias=False)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): linear_init(m, zero_bias=True)
    
    def forward(self, x: torch.Tensor):
        for upsample in self.upsample:
            x = upsample(x)
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

class PreEncoder(nn.Module):
    def __init__(self, config: VQConfig):
        super().__init__()

        self.config = config
        self.encoder_tower = nn.ModuleList([
            ShortConvolution(config.latent_size, config.kernel_size) for _ in range(4)
        ])
        self.norm_tower = nn.ModuleList([
            RMSNorm(hidden_size=config.latent_size, eps=config.eps) for _ in range(4)
        ])
    
    def forward(self, x: torch.Tensor):
        for encoder, norm in zip(self.encoder_tower, self.norm_tower):
            x_new, _ = encoder(x)
            x_new = torch.nn.functional.dropout(x_new, p=self.config.dropout, training=self.training)
            x = norm(x_new + x)
        return x

class VQModel(PreTrainedModel):
    def __init__(self, config: VQConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.config = config
        self.num_heads = config.num_heads
        self.lm_head = nn.Linear(config.latent_size, config.vocab_size, bias=False)
        self.base_vocab = nn.Embedding(config.vocab_size, config.latent_size, padding_idx=config.pad_token_id)

        # self.project_in = nn.Linear(config.hidden_size, config.latent_size)
        # self.project_out = nn.Linear(config.latent_size, config.hidden_size)

        self.encoder = VQEncoder(config)
        self.decoder = VQDecoder(config)

        # Pre-encoder
        self.pre_encoder = PreEncoder(config)

        if config.vq_type == 'optvq':
            self.vq = OptVQ(config)
        elif config.vq_type == 'lfq':
            self.vq = LFQ(config)
            # self.vq = LFQ(
            #     dim=config.latent_size,
            #     codebook_size=config.codebook_size,
            #     entropy_loss_weight=config.diversity_loss_factor,
            #     num_codebooks=config.num_heads
            # )
        elif config.vq_type == 'vanillavq':
            self.vq = VanillaVQ(config)
        elif config.vq_type == "ibq":
            self.vq = IBQ(config)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): linear_init(m, zero_bias=True)
        
        embedding_init(self.base_vocab, distribution="uniform")
    
    def get_codebook(self) -> torch.Tensor:
        return self.vq.get_codebook()
    
    def encode(self, input_ids: torch.LongTensor) -> Dict:
        x = self.base_vocab(input_ids)
        # x = self.project_in(x)
        x = self.pre_encoder(x)
        x = self.encoder(x)

        res = self.vq(x)

        return dict(
            quant=res[0],
            indices=res[1],
            aux_loss=res[2]
        )
    
    def decode(self, input_ids: torch.LongTensor):
        x = self.vq.index_to_code(input_ids)
        x = self.decoder(x)
        # x = self.project_out(x)
        return self.lm_head(x)

    def forward(self, input_ids: torch.LongTensor):
        res = self.encode(input_ids)
        x_quant, indices, aux_loss = res['quant'], res['indices'], res['aux_loss']

        # compute metric
        indices_onthot = torch.nn.functional.one_hot(indices, num_classes=self.vq.codebook_size)
        codebook_ppl = indices_onthot.flatten(0, -2).float().mean(0)
        codebook_ppl = codebook_ppl * torch.log(codebook_ppl + self.config.eps)
        codebook_ppl = torch.exp(-codebook_ppl.sum())

        codebook_usage = (torch.bincount(indices.flatten(), minlength=self.vq.codebook_size) > 0).to(torch.int64)

        if self.training:
            x_decoded = self.decoder(x_quant)
            # x_decoded = self.project_out(x_decoded)
            x_decoded = self.lm_head(x_decoded)

            # compute loss
            reconstruct_ce = CrossEntropyLoss(ignore_index=self.config.pad_token_id, reduction='mean')
            x_decoded = x_decoded[:, :input_ids.size(-1)]
            reconstruct_loss = reconstruct_ce(x_decoded.flatten(0, 1), input_ids.flatten())

            reconstruct_acc = (torch.softmax(x_decoded, dim=-1).argmax(-1) == input_ids).logical_and(input_ids != self.config.pad_token_id).sum(-1)
            reconstruct_acc = reconstruct_acc / (input_ids != self.config.pad_token_id).sum(-1)
            reconstruct_acc = reconstruct_acc.mean()
            loss = reconstruct_loss + aux_loss
        else:
            reconstruct_loss = self.vq.zero
            reconstruct_acc = self.vq.zero
            loss = self.vq.zero

        return VQOutput(
            quant=x_quant,
            indices=indices,
            loss=loss,
            reconstruct_loss=reconstruct_loss,
            reconstruct_acc=reconstruct_acc,
            commit_loss=aux_loss,
            diversity_loss=self.vq.zero,
            codebook_usage=codebook_usage,
            codebook_ppl=codebook_ppl
        )