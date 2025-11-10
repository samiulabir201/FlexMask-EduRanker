from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.metrics import make_confusion, mapk, plot_confusion


def main() -> None:
    """Evaluate predictions: compute MAP@3 and confusion matrix.

    Expected usage:
        python scripts/evaluate.py \
            --ground_truth artifacts/data/val.csv \
            --pred_jsonl artifacts/preds/run_001.jsonl \
            --labels_json artifacts/data/labels.json \
            --out_dir artifacts/eval
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground_truth",
        required=True,
        help="CSV with true 'LabelIndex' column.",
    )
    parser.add_argument(
        "--pred_jsonl",
        required=True,
        help=(
            "JSONL with {'y_true', 'ranks'} per line or "
            "{'ranks'} if ground truth provided separately."
        ),
    )
    parser.add_argument(
        "--labels_json",
        required=True,
        help="JSON with 'labels': list[str] in correct index order.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Where to write metrics and figures.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    df = pd.read_csv(args.ground_truth)
    y_true = df["LabelIndex"].tolist()

    # Load predicted ranks
    ranks: list[list[int]] = []
    with open(args.pred_jsonl, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ranks.append(obj["ranks"])

    # Load label names
    with open(args.labels_json, encoding="utf-8") as f:
        labels = json.load(f)["labels"]

    # MAP@3
    map3 = mapk(y_true, ranks, k=3)
    (out_dir / "metrics.txt").write_text(f"MAP@3: {map3:.6f}\n", encoding="utf-8")

    # Confusion matrix (using top-1 prediction)
    top1 = [r[0] for r in ranks]
    cm = make_confusion(y_true, top1, labels)
    np.save(out_dir / "confusion.npy", cm)

    plot_confusion(cm, labels, str(out_dir / "confusion.png"), normalize=False)
    plot_confusion(cm, labels, str(out_dir / "confusion_norm.png"), normalize=True)


if __name__ == "__main__":
    main()
