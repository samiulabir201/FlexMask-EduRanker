"""Data utilities for reading, cleaning, and batching the dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
from torch.utils.data import Dataset


@dataclass
class TextColumns:
    """Names of text columns used for prompt construction."""

    question_text: str
    choices: str
    correct_answer: str
    misconception_candidates: str
    student_answer: str
    student_explanation: str
    suffix_column: str
    label_column: str


class SuffixDataset(Dataset):
    """Dataset that returns prefix bundle + suffix candidates.

    Each item contains:
    - a "bundle" dict with all fields needed to build the prefix, and
    - a list of suffix strings (Category:Misconception candidates) and
    - a numeric label index (optional on test).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cols: TextColumns,
        labels: list[int] | None = None,
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        df:
            DataFrame with raw MAP samples and label indices.
        cols:
            Column names wrapper.
        labels:
            Optional list of int label indices; if None, read from df[cols.label_column].
        """
        self.df = df.reset_index(drop=True)
        self.cols = cols
        if labels is None and cols.label_column in df.columns:
            self.labels = df[cols.label_column].tolist()
        else:
            self.labels = labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """Return a raw item containing text fields and optional label.

        Returns
        -------
        dict
            Dictionary containing:
            - "bundle": question bundle (question, choices, answer,
              misconceptions, student_answer, student_explanation)
            - "suffixes": list[str] suffix candidates
            - "label": int label index if available
        """
        row = self.df.iloc[idx]
        bundle = {
            "QuestionText": row[self.cols.question_text],
            "MC_Choices": row[self.cols.choices],
            "Answer": row[self.cols.correct_answer],
            "MisconceptionCandidates": row[self.cols.misconception_candidates],
            "MC_Answer": row[self.cols.student_answer],
            "StudentExplanation": row[self.cols.student_explanation],
        }

        suffixes_raw = row[self.cols.suffix_column]
        if isinstance(suffixes_raw, str):
            try:
                suffixes = json.loads(suffixes_raw)
            except json.JSONDecodeError:
                suffixes = suffixes_raw.split(" ")
        else:
            suffixes = list(suffixes_raw)

        item: dict = {"bundle": bundle, "suffixes": suffixes}
        if self.labels is not None:
            item["label"] = int(self.labels[idx])
        return item
