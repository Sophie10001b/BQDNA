import os
import math
import re
import json
import itertools

from typing import Optional, Dict, List, Tuple, Union, Any
from transformers import PreTrainedTokenizer

BASE = ['A', 'T', 'C', 'G']

class BaseTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids"]

    def __init__(
        self,
        model_max_length: int=int(1e9),
        bos_token="<s>",
        eos_token="</s>",
        sep_token="<sep>",
        cls_token="</s>",
        pad_token="<pad>",
        mask_token="<mask>",
        unk_token="<unk>",
        **kwargs
    ):
        
        self.model_max_length = model_max_length
        self.vocab = {}
        self.ids_to_tokens = {}
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            model_max_length=model_max_length,
            padding_side=kwargs.pop("padding_side", "right"),
            **kwargs
        )

        # get vocab
        token_list = self.all_special_tokens + BASE
        
        self.vocab: Dict[str:int] = {k:i for i, k in enumerate(token_list)}
        self.ids_to_tokens: Dict[int:str] = {v:k for k, v in self.vocab.items()}
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        return self.vocab
    
    def _convert_token_to_id(self, token) -> int:
        return self.vocab.get(token, self.vocab[self.unk_token])
    
    def _convert_id_to_token(self, index) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)
    
    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)
    
    def _tokenize(self, text: str, **kwargs) -> list[str]:
        token_max_length = max([len(_) for _ in self.vocab.keys()])
        result = []

        pos = 0
        while pos < len(text):
            is_matched = False
            for j in range(min(len(text) - pos, token_max_length), 0, -1):
                sub_str = text[pos:pos+j]
                if sub_str in self.vocab:
                    result.append(sub_str)
                    pos += j
                    is_matched = True
                    break
            if not is_matched: pos += 1
        
        return result
    
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if self.bos_token_id is not None:
            token_ids_0 = [self.bos_token_id] + token_ids_0
        if self.eos_token_id is not None:
            token_ids_0 = token_ids_0 + [self.eos_token_id]
        
        if token_ids_1 is not None:
            if self.bos_token_id is not None:
                token_ids_1 = [self.bos_token_id] + token_ids_1
            if self.eos_token_id is not None:
                token_ids_1 = token_ids_1 + [self.eos_token_id]
            
            result = token_ids_0 + token_ids_1
        else: result = token_ids_0

        return result
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple:
        return ()