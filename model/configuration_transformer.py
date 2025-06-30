from typing import Dict, List, Tuple, Union, Optional, Any
from transformers.configuration_utils import PretrainedConfig

class TransformerConfig(PretrainedConfig):

    model_type = 'transformer'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 512,
        num_hidden_layers: int = 24,
        num_heads: int = 8,
        num_kv_heads: int = 8,
        window_size: Optional[int] = None,
        rope_base: Optional[int] = int(1e6),
        intermediate_size: Optional[int] = 2048,
        hidden_act: str = "swish",
        eps: float = 1e-5,
        use_cache: bool = False,
        pad_token_id: int = 4,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        tie_word_embeddings: bool = False,
        vocab_size: int = 10,
        dropout: float = 0.0,
        num_labels: int = 1,
        num_class: int = 2,
        problem_type: str = "single_label_classification",
        use_mamba: bool = False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size
        self.rope_base = rope_base

        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.eps = eps
        self.use_cache = use_cache

        self.vocab_size = vocab_size
        self.dropout = dropout
        self.num_labels = num_labels
        self.num_class = num_class
        self.problem_type = problem_type
        self.use_mamba = use_mamba