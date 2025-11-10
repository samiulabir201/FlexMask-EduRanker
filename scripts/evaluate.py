#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate predictions: computes MAP@3 and confusion matrix PNG.

Inputs:
- Ground truth CSV (with 'Category' and optionally remapped indices).
- Predictions JSONL/CSV containing a list of ranked indices per row OR top-1 label.

Usage:
    python scripts/evaluate.py \
        --ground_truth artifacts/data/val.csv \
        --pred_jsonl artifacts/preds/run_0/val_logits.jsonl \
        --labels_json artifacts/data/labels.json \
        --out_dir artifacts/eval
"""
from __future__ import annotations

import os, json, argparse
import numpy as np
import pandas as pd
from metrics import mapk, make_confusion, plot_confusion

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground_truth", required=True, help="CSV with true 'LabelIndex' column")
    ap.add_argument("--pred_jsonl", required=True, help="JSONL with {'y_true', 'ranks'} per line or {'ranks'}")
    ap.add_argument("--labels_json", required=True, help="JSON with 'labels': list[str] in correct index order")
    ap.add_argument("--out_dir", required=True, help="Where to write metrics and figs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gt = pd.read_csv(args.ground_truth)
    y_true = gt["LabelIndex"].astype(int).tolist()

    ranks = []
    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ranks.append(obj["ranks"])

    with open(args.labels_json, "r", encoding="utf-8") as f:
        labels = json.load(f)["labels"]

    score = mapk(y_true, ranks, k=3)
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"MAP@3": score}, f, indent=2)

    # Top-1 confusion
    y_pred_top1 = [r[0] for r in ranks]
    cm = make_confusion(y_true, y_pred_top1, labels)

    raw_png = os.path.join(args.out_dir, "confusion_raw.png")
    norm_png = os.path.join(args.out_dir, "confusion_row_normalized.png")
    plot_confusion(cm, labels, raw_png, normalize=False)
    plot_confusion(cm, labels, norm_png, normalize=True)
    print(f"MAP@3: {score:.4f}. Saved confusion maps to {args.out_dir}")

if __name__ == "__main__":
    main()
