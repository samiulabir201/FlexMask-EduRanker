#!/usr/bin/env python
"""
Prepare MAP@3 training data:

- Load raw train.csv
- Drop duplicates
- Create stratified k-folds by Category
- Build a numeric label index and suffix candidate strings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate QuestionId + StudentExplanation rows."""
    return df.drop_duplicates(
        subset=["QuestionId", "MC_Answer", "StudentExplanation"],
    ).reset_index(drop=True)


def build_suffix_column(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Build canonical 'Category:Misconception' string and label index.

    Returns
    -------
    df:
        DataFrame with columns 'CategoryMis', 'LabelIndex' and 'SuffixCandidates'.
    labels:
        List of all unique label strings in index order.
    """
    df = df.copy()
    df["CategoryMis"] = df["Category"].fillna("NA") + ":" + df["Misconception"].fillna("NA")
    labels = sorted(df["CategoryMis"].unique().tolist())
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    df["LabelIndex"] = df["CategoryMis"].map(label_to_idx)

    # For this repo, we assume candidates per item are provided externally or equal
    # to all labels (for demo); in competition, you'd limit to question-specific.
    df["SuffixCandidates"] = [labels] * len(df)
    return df, labels


def make_folds(df: pd.DataFrame, n_splits: int) -> list[int]:
    """Create stratified folds by CategoryMis."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = [-1] * len(df)
    for fold, (_, val_idx) in enumerate(skf.split(df, df["CategoryMis"])):
        for i in val_idx:
            folds[i] = fold
    return folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw train.csv from Kaggle.")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write cleaned CSV + folds + labels.",
    )
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df = deduplicate(df)

    df, labels = build_suffix_column(df)
    folds = make_folds(df, n_splits=args.folds)
    df = df.assign(fold=folds)

    clean_csv = out_dir / "train_clean.csv"
    labels_json = out_dir / "labels.json"
    folds_json = out_dir / "folds.json"

    df.to_csv(clean_csv, index=False)
    labels_json.write_text(json.dumps({"labels": labels}, ensure_ascii=False, indent=2))
    folds_json.write_text(json.dumps(folds))

    print(f"Wrote cleaned data to {clean_csv}")
    print(f"Wrote labels to {labels_json}")
    print(f"Wrote folds to {folds_json}")


if __name__ == "__main__":
    main()
