# -*- coding: utf-8 -*-
"""Ensembling: average logits/ranks from multiple runs into a final ranking."""

from __future__ import annotations

import argparse
import glob
import json
from typing import Dict, List, Sequence
import numpy as np

def load_runs(paths: Sequence[str]) -> List[List[int]]:
    """Load rank lists from multiple JSONL files and align by line order.

    Returns:
        A list of lists: per-item list of ranks per run.
    """
    per_run = []
    for p in paths:
        ranks = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                ranks.append(json.loads(line)["ranks"])
        per_run.append(ranks)
    # Transpose: items × runs
    return list(map(list, zip(*per_run)))

def ensemble_ranks(rank_lists: List[List[int]]) -> List[int]:
    """Rank aggregation by Borda count (simple, works without logits)."""
    K = len(rank_lists[0])
    scores = np.zeros(K, dtype=float)
    for ranks in rank_lists:
        for pos, idx in enumerate(ranks):
            scores[idx] += (K - pos)
    order = list(np.argsort(-scores))
    return order

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Glob or list of JSONL rank files.")
    parser.add_argument("--output", required=True, help="Output JSONL.")
    args = parser.parse_args()

    expanded = []
    for p in args.inputs:
        expanded.extend(glob.glob(p))
    items = load_runs(expanded)

    with open(args.output, "w", encoding="utf-8") as f:
        for rank_lists in items:
            final = ensemble_ranks(rank_lists)
            f.write(json.dumps({"ranks": final}) + "\\n")

if __name__ == "__main__":
    main()
