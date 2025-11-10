#!/usr/bin/env python
"""
Metrics and plotting utilities for MAP@3 and confusion matrix.

Functions
---------
- mapk(y_true, y_pred_ranks, k): Mean Average Precision at k.
- make_confusion(true_labels, pred_labels, labels): Confusion matrix (counts).
- plot_confusion(cm, labels, out_png, normalize): Save a confusion matrix heatmap.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np


def mapk(
    y_true: Sequence[int],
    y_pred_ranks: Sequence[Sequence[int]],
    k: int = 3,
) -> float:
    """Compute MAP@k for ranked indices.

    Parameters
    ----------
    y_true:
        Sequence of true target indices per sample (integer id within candidates).
    y_pred_ranks:
        Sequence of predicted ranking lists per sample (indices in descending
        score order).
    k:
        Cutoff.

    Returns
    -------
    float
        Mean Average Precision at k.
    """
    ap = []
    for yt, ranks in zip(y_true, y_pred_ranks, strict=False):
        if yt in ranks[:k]:
            pos = ranks.index(yt)
            ap.append(1.0 / (pos + 1))
        else:
            ap.append(0.0)
    return float(np.mean(ap)) if ap else 0.0


def make_confusion(
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
    labels: list[str],
) -> np.ndarray:
    """Build a confusion matrix (counts).

    Parameters
    ----------
    true_labels:
        Sequence of true label indices.
    pred_labels:
        Sequence of predicted label indices.
    labels:
        List of label names in index order.

    Returns
    -------
    np.ndarray
        Confusion matrix of shape (n_labels, n_labels).
    """
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true_labels, pred_labels, strict=False):
        cm[t, p] += 1
    return cm


def plot_confusion(
    cm: np.ndarray,
    labels: list[str],
    out_png: str,
    normalize: bool = False,
) -> None:
    """Plot (and save) a confusion matrix heatmap.

    Parameters
    ----------
    cm:
        Confusion matrix counts.
    labels:
        List of label names.
    out_png:
        Path to save PNG.
    normalize:
        Whether to normalize rows to probabilities.
    """
    if normalize:
        m = cm.astype(float)
        row_sum = m.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        m = m / row_sum
    else:
        m = cm.astype(float)

    plt.figure(figsize=(8, 6))
    plt.imshow(m, interpolation="nearest", aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix" + (" (normalized)" if normalize else ""))

    # Add numbers
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            val = m[i, j] if normalize else int(cm[i, j])
            plt.text(
                j,
                i,
                f"{val:.2f}" if normalize else f"{val}",
                ha="center",
                va="center",
                fontsize=7,
            )
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()
