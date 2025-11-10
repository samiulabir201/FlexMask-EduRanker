# -*- coding: utf-8 -*-
"""Utilities: seeding, metrics, and helpers."""

from __future__ import annotations
import random
from typing import Iterable, List, Sequence
import numpy as np
import torch

def seed_everything(seed: int) -> None:
    """Fix seeds for reproducibility across libraries and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mapk(y_true: Sequence[int], y_pred_ranks: Sequence[Sequence[int]], k: int = 3) -> float:
    """Compute mean average precision at k.

    Args:
        y_true: Iterable of true target indices per sample (integer id within candidates).
        y_pred_ranks: Iterable of predicted ranking lists per sample (indices in descending score order).
        k: Cutoff.

    Returns:
        MAP@k value in [0,1].
    """
    assert len(y_true) == len(y_pred_ranks), "Mismatched lengths"
    ap_list = []
    for t, ranks in zip(y_true, y_pred_ranks):
        ranks = list(ranks)[:k]
        if t in ranks:
            pos = ranks.index(t)
            ap_list.append(1.0 / (pos + 1))
        else:
            ap_list.append(0.0)
    return float(np.mean(ap_list)) if ap_list else 0.0
