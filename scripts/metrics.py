#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metrics and plotting utilities for MAP@3 and confusion matrix.

Goal & Intuition:
- Provide reliable, competition-aligned MAP@3 and a standard confusion matrix.
- Accuracy maps clarify which categories the model confuses.

Functions:
- mapk(y_true, y_pred_ranks, k): Mean Average Precision @ k.
- make_confusion(true_labels, pred_labels, labels): Dense confusion matrix (counts).
- plot_confusion(cm, labels, out_png, normalize): Save a confusion matrix heatmap.
"""
from __future__ import annotations
from typing import Sequence, List
import numpy as np
import matplotlib.pyplot as plt

def mapk(y_true: Sequence[int], y_pred_ranks: Sequence[Sequence[int]], k: int = 3) -> float:
    """
    Compute mean average precision at k (MAP@k).

    Intuition:
        Rewards placing the correct label as high as possible within top-k.
        For each item, AP = 1/(rank of correct label) if within top-k else 0.

    Args:
        y_true: Iterable of true label indices per item.
        y_pred_ranks: Iterable of ranked label indices per item (best → worst).
        k: Cutoff (default 3 per competition).

    Returns:
        Mean of per-item AP@k in [0,1].
    """
    assert len(y_true) == len(y_pred_ranks), "Lengths mismatch"
    ap = []
    for t, ranks in zip(y_true, y_pred_ranks):
        r = list(ranks)[:k]
        ap.append(1.0/(r.index(t)+1) if t in r else 0.0)
    return float(np.mean(ap)) if ap else 0.0

def make_confusion(true_labels: Sequence[int], pred_labels: Sequence[int], labels: List[str]) -> np.ndarray:
    """
    Build a confusion matrix (counts).

    Args:
        true_labels: True label indices.
        pred_labels: Predicted label indices (argmax or top-1).
        labels: Ordered list of label names (defines matrix size & order).

    Returns:
        cm: (L, L) ndarray where rows = true, cols = predicted.
    """
    L = len(labels)
    cm = np.zeros((L, L), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        if 0 <= t < L and 0 <= p < L:
            cm[t, p] += 1
    return cm

def plot_confusion(cm: np.ndarray, labels: List[str], out_png: str, normalize: bool = False) -> None:
    """
    Plot (and save) confusion matrix with or without normalization.

    Args:
        cm: Raw counts matrix.
        labels: Class labels (row/col order).
        out_png: Path to save the PNG figure.
        normalize: If True, row-normalize to show per-class accuracies.

    Output:
        Saves a PNG heatmap with numeric annotations.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    m = cm.astype(float)
    if normalize:
        row_sums = m.sum(axis=1, keepdims=True) + 1e-12
        m = m / row_sums

    plt.figure()
    plt.imshow(m, aspect="auto")
    plt.title("Accuracy Map (Confusion Matrix)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    plt.colorbar()
    # annotate
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            val = m[i, j] if normalize else int(cm[i, j])
            plt.text(j, i, f"{val:.2f}" if normalize else f"{val}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()
