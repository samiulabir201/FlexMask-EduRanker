"""Utilities: seeding, metrics, and helpers."""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and (optionally) PyTorch for reproducibility.

    This function is safe to call even when PyTorch is not installed
    or cannot be imported (e.g., minimal CI / CPU-only environments).
    """
    random.seed(seed)
    np.random.seed(seed)

    # Optional, lazy PyTorch seeding
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        # If torch is unavailable or fails to import (e.g. missing DLLs),
        # we still have deterministic behavior for Python and NumPy.
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Make CUDA deterministic as much as possible
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def mapk(
    y_true: Sequence[int],
    y_pred_ranks: Sequence[Sequence[int]],
    k: int = 3,
) -> float:
    """Compute Mean Average Precision at k (MAP@k) for ranked predictions.

    Args:
        y_true:
            Iterable of true target indices per sample (integer id within
            the candidate set).
        y_pred_ranks:
            Iterable of predicted ranking lists per sample (indices in
            descending score order).
        k:
            Cutoff for MAP@k.

    Returns:
        MAP@k score as a float in [0.0, 1.0].
    """
    ap_values: list[float] = []

    for yt, ranks in zip(y_true, y_pred_ranks):
        score = 0.0
        for j, r in enumerate(ranks[:k], start=1):
            if r == yt:
                score = 1.0 / float(j)
                break
        ap_values.append(score)

    return float(np.mean(ap_values)) if ap_values else 0.0
