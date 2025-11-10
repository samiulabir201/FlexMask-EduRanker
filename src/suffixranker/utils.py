from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mapk(
    y_true: Sequence[int],
    y_pred_ranks: Sequence[Sequence[int]],
    k: int = 3,
) -> float:
    """Compute mean average precision at k (MAP@k).

    Args:
        y_true: Sequence of true label indices per sample.
        y_pred_ranks: Sequence of predicted ranking lists per sample.
        k: Cutoff for the ranking.

    Returns:
        MAP@k score in the range [0.0, 1.0].
    """
    ap_values: list[float] = []

    for true_label, ranks in zip(y_true, y_pred_ranks):
        score = 0.0
        for j, pred in enumerate(ranks[:k], start=1):
            if pred == true_label:
                score = 1.0 / j
                break
        ap_values.append(score)

    return float(np.mean(ap_values)) if ap_values else 0.0
