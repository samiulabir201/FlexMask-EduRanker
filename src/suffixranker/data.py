# -*- coding: utf-8 -*-
"""Data utilities for reading, cleaning, and batching the dataset."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

@dataclass
class TextColumns:
    """Column names used in the dataset."""
    question: str
    choices: str
    answer: str
    misconceptions: str
    student_answer: str
    student_explanation: str
    category: str
    question_id: str

class SuffixDataset(Dataset):
    """PyTorch dataset encapsulating prefix-shared packaging.

    Each item corresponds to one **question** with multiple **suffix** candidates.
    The dataset returns the raw fields for later tokenization/packing.
    """

    def __init__(self, df: pd.DataFrame, cols: TextColumns, labels: List[int] | None = None):
        """Initialize the dataset.

        Args:
            df: Dataframe of examples.
            cols: Column schema.
            labels: Optional index of correct suffix per row (for training).
        """
        self.df = df.reset_index(drop=True)
        self.cols = cols
        self.labels = labels

    def __len__(self) -> int:
        """Number of rows in the dataframe."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        """Return a raw item containing text fields and optional label.

        Returns:
            Dict containing:
                - question bundle (question, choices, answer, misconceptions, student_answer, student_explanation)
                - suffix candidates (list[str])
                - label (int) if available
                - question_id for grouping
        """
        row = self.df.iloc[idx]
        bundle = {
            "question": str(row[self.cols.question]),
            "choices": str(row[self.cols.choices]),
            "answer": str(row[self.cols.answer]),
            "misconceptions": str(row[self.cols.misconceptions]),
            "student_answer": str(row[self.cols.student_answer]),
            "student_explanation": str(row[self.cols.student_explanation]),
        }
        # Expect a JSON list or | separated suffix candidates depending on upstream formatting.
        suffix_raw = row.get("SuffixCandidates", "")
        if isinstance(suffix_raw, str):
            try:
                suffixes = json.loads(suffix_raw)
            except Exception:
                suffixes = [s.strip() for s in suffix_raw.split("|") if s.strip()]
        else:
            suffixes = list(suffix_raw)

        item = {
            "bundle": bundle,
            "suffixes": suffixes,
            "question_id": row[self.cols.question_id],
        }
        if self.labels is not None:
            item["label"] = int(self.labels[idx])
        return item
