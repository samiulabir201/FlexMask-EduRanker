"""Ensembling: average logits/ranks from multiple runs into a final ranking."""

from __future__ import annotations

import argparse
import glob
import json
from collections.abc import Sequence

import numpy as np


def load_runs(paths: Sequence[str]) -> list[list[int]]:
    """Load rank lists from multiple JSONL files and align by line order."""
    all_paths: list[str] = []
    for p in paths:
        expanded = glob.glob(p)
        if expanded:
            all_paths.extend(expanded)
        else:
            all_paths.append(p)

    per_run: list[list[list[int]]] = []
    for p in all_paths:
        ranks: list[list[int]] = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                ranks.append(json.loads(line)["ranks"])
        per_run.append(ranks)

    # Transpose: num_runs × num_items → num_items × num_runs
    return list(map(list, zip(*per_run)))


def ensemble_ranks(rank_lists: list[list[int]]) -> list[int]:
    """Rank aggregation by Borda count (simple, works without logits)."""
    K = len(rank_lists[0])
    scores = np.zeros(K, dtype=float)
    for ranks in rank_lists:
        for pos, idx in enumerate(ranks):
            scores[idx] += K - pos
    final_order = np.argsort(scores)[::-1].tolist()
    return final_order


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Glob or list of JSONL rank files.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL.")
    args = parser.parse_args()

    per_item_runs = load_runs(args.inputs)
    with open(args.output, "w", encoding="utf-8") as f:
        for run_ranks in per_item_runs:
            agg = ensemble_ranks(run_ranks)
            f.write(json.dumps({"ranks": agg}) + "\n")


if __name__ == "__main__":
    main()
