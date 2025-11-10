from __future__ import annotations

from suffixranker.utils import mapk


def test_mapk_basic() -> None:
    """Basic sanity checks for MAP@3."""
    y_true = [0, 1, 2]

    # Perfect top-1
    preds = [[0, 1, 2], [1, 0, 2], [2, 1, 0]]
    assert mapk(y_true, preds, k=3) == 1.0

    # Correct labels appear at positions 1, 2, 3
    preds = [[0, 1, 2], [2, 1, 0], [0, 1, 2]]
    score = mapk(y_true, preds, k=3)
    assert 0.0 < score < 1.0
