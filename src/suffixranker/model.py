"""Model wrapper: extracts last-token embeddings per candidate and scores with a linear head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelConfig:
    """Configuration for SuffixRanker backbone."""

    model_name: str
    max_length: int = 2048
    use_bitsandbytes: bool = False


class SuffixRanker(nn.Module):
    """LLM + linear scorer for suffix classification.

    Given concatenated `[prefix ++ suffix0 ++ suffix1 ++ ...]`, we compute the
    hidden states, index the last token of each suffix, and pass through a
    linear head to obtain logits.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        quant_kwargs: dict | None = None
        if cfg.use_bitsandbytes:
            quant_kwargs = {
                "load_in_8bit": True,
            }

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            **(quant_kwargs or {}),
        )
        hidden = self.backbone.config.hidden_size
        self.scorer = nn.Linear(hidden, 1)

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        suffix_last_token_indices: list[int],
    ) -> torch.Tensor:
        """Compute logits for each suffix via last-token pooling.

        Args
        ----
        inputs:
            Tokenized input suitable for the backbone (input_ids,
            attention_mask, etc.).
        suffix_last_token_indices:
            Positions (indices) of the last token for each suffix candidate.

        Returns
        -------
        torch.Tensor
            Logits of shape (num_suffixes,).
        """
        outputs = self.backbone(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[-1]  # (B, T, H)
        # We assume batch_size == 1 for concatenated prefix+suffix representation.
        last_vecs = hidden_states[0, suffix_last_token_indices, :]  # (K, H)
        logits = self.scorer(last_vecs).squeeze(-1)  # (K,)
        return logits
