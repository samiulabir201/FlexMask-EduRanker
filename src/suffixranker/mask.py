"""FlexMask: build custom attention masks to isolate suffixes while allowing prefix visibility."""

from __future__ import annotations

import torch


def build_suffix_mask(
    suffix_ids: torch.Tensor,
    doc_ids: torch.Tensor,
) -> torch.Tensor:
    """Build a boolean attention mask enforcing:

    - causal attention;
    - all tokens can see **prefix** tokens (suffix_ids == -1);
    - suffix tokens can only see tokens from the same suffix or prefix;
    - tokens from different documents cannot see each other.

    Parameters
    ----------
    suffix_ids:
        Tensor of shape (T,) or (B, T) assigning each token to a suffix index,
        or -1 for prefix tokens.
    doc_ids:
        Tensor of same shape giving a document id per token.

    Returns
    -------
    torch.Tensor
        Boolean mask of shape (T, T) or (B, T, T) where True means attention is allowed.
    """
    if suffix_ids.dim() == 1:
        suffix_ids = suffix_ids.unsqueeze(0)
        doc_ids = doc_ids.unsqueeze(0)
    bsz, seqlen = suffix_ids.shape

    q_idx = torch.arange(seqlen, device=suffix_ids.device).view(1, -1, 1)
    kv_idx = torch.arange(seqlen, device=suffix_ids.device).view(1, 1, -1)

    causal = q_idx >= kv_idx
    is_prefix = suffix_ids.eq(-1).unsqueeze(1)
    same_suffix = suffix_ids.unsqueeze(1).eq(suffix_ids.unsqueeze(2))
    same_doc = doc_ids.unsqueeze(1).eq(doc_ids.unsqueeze(2))

    allow = causal & (is_prefix | same_suffix) & same_doc
    return allow
