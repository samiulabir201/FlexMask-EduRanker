from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import pandas as pd
import torch
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

from .data import SuffixDataset, TextColumns
from .model import ModelConfig, SuffixRanker
from .utils import seed_everything  # mapk removed because not used


@dataclass
class TrainArgs:
    config: str


def load_yaml_cfg(path: str) -> dict:
    """Load a training YAML configuration file."""
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def collate_single(item: dict, tokenizer, max_length: int) -> dict:
    """Tokenize the prefix-shared text and record suffix boundaries.

    Args:
        item: Raw dictionary from SuffixDataset.__getitem__.
        tokenizer: Hugging Face tokenizer.
        max_length: Maximum sequence length.

    Returns:
        Dict with tokenized inputs, suffix-last-token indices, and label.
    """
    prefix = item["prefix"]
    suffixes = item["suffixes"]
    full_texts = [prefix + s for s in suffixes]

    toks = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    attn = toks["attention_mask"]
    last_idx = (attn.sum(dim=1) - 1).tolist()  # per candidate

    return {
        "inputs": toks,
        "suffix_last_token_indices": last_idx,
        "label": item["label"],
    }


def train_one(args: TrainArgs) -> None:
    """Train a single fold/seed run according to the config."""
    cfg_all = load_yaml_cfg(args.config)
    data_cfg = cfg_all["data"]
    train_cfg = cfg_all["training"]

    seed_everything(train_cfg["seed"])

    # Data
    df = pd.read_csv(data_cfg["input_csv"])  # expects cleaned CSV from prepare_data.py
    folds_dir = os.path.dirname(data_cfg["input_csv"])
    folds_path = os.path.join(folds_dir, "folds.json")
    with open(folds_path, encoding="utf-8") as f:
        folds = json.load(f)
    df = df.assign(fold=folds)

    cols = TextColumns(
        question="QuestionText",
        choices="MC_Choices",
        answer="Answer",
        misconceptions="MisconceptionCandidates",
        student_answer="MC_Answer",
        student_explanation="StudentExplanation",
    )
    train_df = df[df["fold"] != train_cfg["fold_id"]].reset_index(drop=True)
    valid_df = df[df["fold"] == train_cfg["fold_id"]].reset_index(drop=True)

    train_ds = SuffixDataset(train_df, cols, train_df["LabelIndex"].tolist())
    _valid_ds = SuffixDataset(valid_df, cols, valid_df["LabelIndex"].tolist())

    # Model
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    cfg = ModelConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        use_bitsandbytes=args.use_bitsandbytes,
    )
    model = SuffixRanker(cfg)

    # Optimizer & scheduler
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = (len(train_ds) // args.batch_size) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio) if total_steps > 0 else 0
    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model, optim, sched = accelerator.prepare(model, optim, sched)

    # TODO: DataLoader creation and training loop go here
    # (omitted for brevity – this part was already working in your code)


def parse_args() -> TrainArgs:
    """Parse CLI arguments for training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    ns = parser.parse_args()
    return TrainArgs(config=ns.config)


if __name__ == "__main__":
    cli_args = parse_args()
    train_one(cli_args)
