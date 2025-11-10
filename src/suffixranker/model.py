# -*- coding: utf-8 -*-
"""Model wrapper: extracts last-token embeddings per candidate and scores with a linear head."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class ModelConfig:
    model_name: str
    max_length: int = 4096
    use_bitsandbytes: bool = False

class SuffixRanker(nn.Module):
    """LLM + linear scorer for suffix classification.

    Given concatenated [prefix ++ suffix0 ++ suffix1 ++ ...], we compute the hidden states, index the
    last token of each suffix, and pass through a linear head to obtain logits.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        quant_args = {}
        if cfg.use_bitsandbytes:
            # Lazy import to keep optional
            from transformers import BitsAndBytesConfig
            quant_args["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            quant_args["device_map"] = "auto"

        self.backbone = AutoModelForCausalLM.from_pretrained(cfg.model_name, **quant_args)
        hidden = self.backbone.config.hidden_size
        self.scorer = nn.Linear(hidden, 1)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        suffix_last_token_indices: List[int],
    ) -> torch.Tensor:
        """Compute logits for each suffix via last-token pooling.

        Args:
            inputs: Tokenized input suitable for the backbone (input_ids, attention_mask, etc.).
            suffix_last_token_indices: Positions (indices) of the last token for each suffix candidate.

        Returns:
            logits: Tensor of shape (num_suffixes,) for the current item.
        """
        outputs = self.backbone(**inputs, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states[-1]  # (1, T, H)
        suffix_vecs = hidden_states[0, suffix_last_token_indices, :]  # (K, H)
        logits = self.scorer(suffix_vecs).squeeze(-1)  # (K,)
        return logits
