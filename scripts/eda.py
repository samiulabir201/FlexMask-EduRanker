from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def safe_len(text: str | float) -> int:
    """Safely compute token length of a possibly-missing explanation.

    Args:
        text: Explanation text or NaN/None.

    Returns:
        Number of whitespace-separated tokens, or 0 if missing.
    """
    if not isinstance(text, str):
        return 0
    return len(text.split())
    

def run_eda(input_csv: str, out_dir: str) -> None:
    """Run exploratory data analysis and save figures/tables.

    Args:
        input_csv: Path to the training CSV file.
        out_dir: Directory where figures and tables will be written.
    """
    out_path = Path(out_dir)
    figs_path = out_path / "figs"
    tables_path = out_path / "tables"
    figs_path.mkdir(parents=True, exist_ok=True)
    tables_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    # Basic derived columns
    df["StudentExplanation_len_words"] = df["StudentExplanation"].map(safe_len)
    df["AnsCorrect"] = df["Category"].str.startswith("True")
    df["ExplAssess"] = df["Category"].str.split("_", n=1).str[1]
    df["HasMisconception"] = df["ExplAssess"] == "Misconception"

    # Category distribution
    cat_counts = df["Category"].value_counts().sort_index()
    cat_counts.to_csv(tables_path / "category_counts.csv")
    plt.figure()
    cat_counts.plot(kind="bar")
    plt.title("Category distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figs_path / "category_distribution.png", dpi=170)
    plt.close()

    # Responses per QuestionId
    resp_per_q = df["QuestionId"].value_counts()
    resp_per_q.to_csv(tables_path / "responses_per_question_counts.csv")
    plt.figure()
    resp_per_q.hist(bins=30)
    plt.title("Responses per QuestionId")
    plt.xlabel("Num responses")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figs_path / "responses_per_question_hist.png", dpi=170)
    plt.close()

    # Explanation length histogram
    plt.figure()
    df["StudentExplanation_len_words"].hist(bins=40)
    plt.title("Explanation length (words)")
    plt.xlabel("Words")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figs_path / "explanation_length_hist.png", dpi=170)
    plt.close()

    # Length by Category
    plt.figure()
    df.boxplot(
        column="StudentExplanation_len_words",
        by="Category",
        showfliers=False,
        rot=90,
    )
    plt.suptitle("")
    plt.title("Explanation length by Category")
    plt.xlabel("Category")
    plt.ylabel("Words")
    plt.tight_layout()
    plt.savefig(figs_path / "explanation_length_by_category_box.png", dpi=170)
    plt.close()

    # Length by assessment (Correct / Misconception / Neither)
    groups = [
        g.dropna().values
        for _, g in df.groupby("ExplAssess")["StudentExplanation_len_words"]
    ]
    labels = list(df.groupby("ExplAssess").groups.keys())
    plt.figure()
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.title("Explanation length by assessment")
    plt.ylabel("Words")
    plt.tight_layout()
    plt.savefig(figs_path / "explanation_length_by_assessment_box.png", dpi=170)
    plt.close()

    # Misconception rate by length deciles
    df["len_decile"] = pd.qcut(
        df["StudentExplanation_len_words"].rank(method="first"),
        10,
        labels=False,
    )
    mis_rate_by_len = df.groupby("len_decile")["HasMisconception"].mean()
    mis_rate_by_len.to_csv(tables_path / "mis_rate_vs_len_deciles.csv")
    plt.figure()
    mis_rate_by_len.plot(marker="o")
    plt.title("Misconception rate vs explanation-length decile")
    plt.xlabel("Length decile (0 = shortest)")
    plt.ylabel("Misconception rate")
    plt.tight_layout()
    plt.savefig(figs_path / "mis_rate_vs_len_deciles.png", dpi=170)
    plt.close()

    # Answer correctness × explanation assessment crosstab
    ans_vs_expl = pd.crosstab(df["AnsCorrect"], df["ExplAssess"])
    ans_vs_expl.to_csv(tables_path / "ans_vs_expl_crosstab.csv")
    norm = ans_vs_expl.div(ans_vs_expl.sum(axis=1), axis=0)
    norm.to_csv(tables_path / "ans_correct_vs_explassess_normalized.csv")

    plt.figure()
    norm.plot(kind="bar", stacked=True)
    plt.title("Answer correctness × explanation assessment (row-normalized)")
    plt.xlabel("Answer correct?")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(figs_path / "ans_correct_vs_explassess_stacked.png", dpi=170)
    plt.close()

    # Top misconceptions
    top_mis = (
        df.loc[df["HasMisconception"], "Misconception"]
        .value_counts()
        .head(20)
        .sort_values(ascending=False)
    )
    top_mis.to_csv(tables_path / "top_misconceptions.csv")
    plt.figure()
    top_mis.plot(kind="barh")
    plt.title("Top misconceptions (frequency)")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(figs_path / "top_misconceptions.png", dpi=170)
    plt.close()


def main() -> None:
    """CLI entrypoint for reproducible EDA."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to train.csv.")
    parser.add_argument("--out_dir", required=True, help="Directory for EDA outputs.")
    args = parser.parse_args()
    run_eda(args.input_csv, args.out_dir)


if __name__ == "__main__":
    main()
