from typing import Dict, List, Tuple, Union, Optional, Any
from transformers.configuration_utils import PretrainedConfig

class VQConfig(PretrainedConfig):

    model_type = 'vqvae'

    def __init__(
        self,
        hidden_size: int = 512,
        latent_size: int = 256,
        kernel_size: int = 4,
        downsample_rate: int = 16,
        vocab_size: int = 10,
        codebook_size: int = 8192,
        num_heads: int = 1,
        pad_token_id: int = 4,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        dropout: float = 0.0,
        eps: int = 1e-5,
        diversity_gamma: float = 1.0,
        distance_temperature: float = 100,
        diversity_loss_factor: float = 0.25,
        commit_loss_factor: float = 0.25,
        ema_gamma: float = 0.99,
        optvq_eps: int = 10,
        optvq_niters: int = 3,
        vq_type: str = "",
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.kernel_size = kernel_size
        self.downsample_rate = downsample_rate
        self.num_heads = num_heads

        self.eps = eps
        self.diversity_gamma = diversity_gamma
        self.distance_temperature = distance_temperature
        self.diversity_loss_factor = diversity_loss_factor
        self.commit_loss_factor = commit_loss_factor
        self.ema_gamma = ema_gamma
        self.optvq_eps = optvq_eps
        self.optvq_niters = optvq_niters

        self.vq_type = vq_type

        self.vocab_size = vocab_size
        self.codebook_size = codebook_size
        self.dropout = dropout