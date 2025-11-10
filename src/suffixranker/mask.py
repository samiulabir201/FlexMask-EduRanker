# -*- coding: utf-8 -*-
"""FlexMask: Build custom attention masks to isolate suffixes while allowing prefix visibility."""

from __future__ import annotations
from typing import Optional, Tuple
import torch

def build_suffix_mask(suffix_ids: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
    """Create a boolean attention mask over a sequence with multiple suffix segments.

    This mask implements the following logic per (query, key) token pair:
        causal = q_idx >= kv_idx
        is_prefix = suffix_ids[kv_idx] == -1
        same_suffix = (suffix_ids[q_idx] == suffix_ids[kv_idx])
        same_doc = doc_ids[q_idx] == doc_ids[kv_idx]
        allow = causal & (same_suffix | is_prefix) & same_doc

    Args:
        suffix_ids: (T,) tensor. -1 for prefix tokens, otherwise the suffix index (0..N-1).
        doc_ids: (T,) tensor. Identifies which tokens belong to the same document (question).
                 Useful when packing multiple questions per batch element.

    Returns:
        attn_mask: (T, T) boolean tensor where True means **allowed to attend**.
    """
    T = suffix_ids.size(0)
    q_idx = torch.arange(T, device=suffix_ids.device).unsqueeze(1)  # (T,1)
    kv_idx = torch.arange(T, device=suffix_ids.device).unsqueeze(0)  # (1,T)

    causal = q_idx >= kv_idx  # (T,T)
    is_prefix = suffix_ids[kv_idx] == -1  # broadcast (1,T) -> (T,T)
    same_suffix = suffix_ids[q_idx] == suffix_ids[kv_idx]
    same_doc = doc_ids[q_idx] == doc_ids[kv_idx]

    allow = causal & (same_doc & (is_prefix | same_suffix))
    return allow
