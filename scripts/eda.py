#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reproducible EDA for MAP dataset (Eedi Diagnostic Questions).
Generates figures under docs/figs/ and summary tables under docs/tables/.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def safe_len(s):
    try:
        return len(str(s))
    except Exception:
        return 0

def word_count(s):
    try:
        return len(str(s).split())
    except Exception:
        return 0

def parse_category(cat):
    cat = str(cat)
    if "_" in cat:
        a, b = cat.split("_", 1)
        return a, b
    return "Unknown", "Unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True, help="Path to train.csv")
    parser.add_argument("--repo_root", default=".", help="Repository root to place figs/tables")
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)

    df["AnsCorrect"], df["ExplAssess"] = zip(*df["Category"].map(parse_category))
    df["StudentExplanation_len_char"] = df["StudentExplanation"].map(safe_len)
    df["StudentExplanation_len_words"] = df["StudentExplanation"].map(word_count)

    if "QuestionText" in df.columns:
        df["QuestionText_len_char"] = df["QuestionText"].map(safe_len)
        df["QuestionText_len_words"] = df["QuestionText"].map(word_count)

    if "Misconception" in df.columns:
        df["HasMisconception"] = df["Misconception"].fillna("NA").astype(str) != "NA"
    else:
        df["HasMisconception"] = df["ExplAssess"] == "Misconception"

    fig_dir = os.path.join(args.repo_root, "docs", "figs")
    tab_dir = os.path.join(args.repo_root, "docs", "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    # Aggregations
    cat_counts = df["Category"].value_counts().sort_values(ascending=False)
    cross = pd.crosstab(df["AnsCorrect"], df["ExplAssess"]).sort_index()
    length_by_assess = df.groupby("ExplAssess")["StudentExplanation_len_words"].describe()

    top_mis = None
    if "Misconception" in df.columns:
        top_mis = df.loc[df["HasMisconception"], "Misconception"].value_counts().head(20)

    df["len_decile"] = pd.qcut(df["StudentExplanation_len_words"].rank(method="first"), 10, labels=False)
    mis_rate_by_len = df.groupby("len_decile")["HasMisconception"].mean()

    # Plots
    plt.figure()
    cat_counts.plot(kind="bar")
    plt.title("Distribution of Category")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "category_distribution.png"), dpi=160)
    plt.close()

    plt.figure()
    plt.imshow(cross.values, aspect="auto")
    plt.title("Answer Correctness vs Explanation Assessment")
    plt.xlabel("ExplAssess")
    plt.ylabel("AnsCorrect")
    plt.xticks(ticks=np.arange(cross.shape[1]), labels=list(cross.columns), rotation=45, ha="right")
    plt.yticks(ticks=np.arange(cross.shape[0]), labels=list(cross.index))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "answer_vs_expl_heatmap.png"), dpi=160)
    plt.close()

    plt.figure()
    plt.hist(df["StudentExplanation_len_words"].dropna(), bins=40)
    plt.title("Student Explanation Length (words)")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "explanation_length_hist.png"), dpi=160)
    plt.close()

    plt.figure()
    groups = [g.dropna().values for _, g in df.groupby("ExplAssess")["StudentExplanation_len_words"]]
    labels = list(df.groupby("ExplAssess").groups.keys())
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.title("Explanation Length by Explanation Assessment")
    plt.xlabel("Explanation Assessment")
    plt.ylabel("Words")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "explanation_length_by_assessment_box.png"), dpi=160)
    plt.close()

    if top_mis is not None and len(top_mis) > 0:
        plt.figure()
        top_mis.sort_values(ascending=True).plot(kind="barh")
        plt.title("Top 20 Misconceptions")
        plt.xlabel("Count")
        plt.ylabel("Misconception")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "top_misconceptions.png"), dpi=160)
        plt.close()

    plt.figure()
    mis_rate_by_len.plot(kind="line", marker="o")
    plt.title("Misconception Rate vs Explanation Length (Deciles)")
    plt.xlabel("Explanation Length Decile (short → long)")
    plt.ylabel("Misconception Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "mis_rate_vs_len_deciles.png"), dpi=160)
    plt.close()

    # Tables
    cat_counts.to_csv(os.path.join(tab_dir, "category_counts.csv"))
    cross.to_csv(os.path.join(tab_dir, "ans_vs_expl_crosstab.csv"))
    length_by_assess.to_csv(os.path.join(tab_dir, "length_by_assessment.csv"))
    if top_mis is not None:
        top_mis.to_csv(os.path.join(tab_dir, "top_misconceptions.csv"))

    print("EDA complete. Figures at docs/figs, tables at docs/tables.")

if __name__ == "__main__":
    main()
