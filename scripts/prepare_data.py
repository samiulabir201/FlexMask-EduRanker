#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare dataset: deduplicate, stratified folds, and export splits.

This script expects an input CSV with the following columns:
- QuestionText, MC_Choices, Answer, MisconceptionCandidates,
  MC_Answer, StudentExplanation, Category, QuestionId

Outputs:
- artifacts/data/train.csv (cleaned)
- artifacts/data/folds.json (list of fold assignments)
"""

import argparse
import json
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def deduplicate(df):
    """Remove exact-duplicate rows to stabilize training.

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Deduplicated dataframe.
    """
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    print(f"Deduplicated: {before} -> {after}")
    return df

def build_folds(df, n_splits, seed, label_col="Category"):
    """Create stratified K-fold assignments by a label column.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
        n_splits (int): Number of folds.
        seed (int): Random seed.
        label_col (str): Column for stratification.

    Returns:
        list[int]: Fold index (0..n_splits-1) per row.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [-1] * len(df)
    for fi, (_, val_idx) in enumerate(skf.split(df, df[label_col])):
        for i in val_idx:
            folds[i] = fi
    assert all(f >= 0 for f in folds), "Invalid fold assignment"
    return folds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input)
    df = deduplicate(df)

    # Save cleaned CSV
    out_csv = os.path.join(args.output_dir, "train.csv")
    df.to_csv(out_csv, index=False)

    # Build folds
    folds = build_folds(df, n_splits=args.folds, seed=args.seed, label_col="Category")
    with open(os.path.join(args.output_dir, "folds.json"), "w", encoding="utf-8") as f:
        json.dump(folds, f)

    print("Done. Cleaned data and folds saved:", args.output_dir)

if __name__ == "__main__":
    main()
