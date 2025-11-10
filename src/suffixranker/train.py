# -*- coding: utf-8 -*-
"""Training entrypoint: one fold × one seed training with logging."""

from __future__ import annotations

import argparse
import csv
import matplotlib.pyplot as plt
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import torch
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

from .data import SuffixDataset, TextColumns
from .prompt import build_prefix
from .model import ModelConfig, SuffixRanker
from .utils import seed_everything, mapk

@dataclass
class TrainArgs:
    model_name: str
    output_dir: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    seed: int
    fold_index: int
    max_length: int
    gradient_accumulation_steps: int
    mixed_precision: str
    use_bitsandbytes: bool

def load_yaml_cfg(path: str) -> Dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def collate_single(item: Dict, tokenizer, max_length: int) -> Dict:
    """Tokenize the prefix-shared text and record suffix boundaries.

    For simplicity here, we tokenize each [prefix + suffix_i] independently and then
    stack logits later. In practice, pack as one long sequence with a custom mask.
    """
    prefix = build_prefix(item["bundle"])
    fulls = [prefix + s for s in item["suffixes"]]

    toks = tokenizer(
        fulls,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    # Identify last non-pad token index per row
    input_ids = toks["input_ids"]
    attn = toks["attention_mask"]
    last_idx = (attn.sum(dim=1) - 1).tolist()  # per candidate
    return toks, last_idx

def train_one(args: TrainArgs, data_cfg_path: str) -> None:
    # Load data config
    data_cfg = load_yaml_cfg(data_cfg_path)["data"]
    cols = TextColumns(**data_cfg["text_columns"])

    # Data
    df = pd.read_csv(data_cfg["input_csv"])  # expects cleaned CSV from prepare_data.py
    with open(os.path.join(os.path.dirname(data_cfg["input_csv"]), "folds.json"), "r", encoding="utf-8") as f:
        folds = json.load(f)
    df = df.assign(fold=folds)

    train_df = df[df.fold != args.fold_index].reset_index(drop=True)
    val_df = df[df.fold == args.fold_index].reset_index(drop=True)

    # Labels: this repo assumes a column 'LabelIndex' providing the correct suffix index per row.
    # Adapt if needed to your competition's ground truth schema.
    if "LabelIndex" not in df.columns:
        raise ValueError("Expected 'LabelIndex' column with correct suffix index per row.")

    train_ds = SuffixDataset(train_df, cols, labels=train_df["LabelIndex"].tolist())
    val_ds = SuffixDataset(val_df, cols, labels=val_df["LabelIndex"].tolist())

    # Reproducibility
    seed_everything(args.seed)

    # Model
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    cfg = ModelConfig(model_name=args.model_name, max_length=args.max_length, use_bitsandbytes=args.use_bitsandbytes)
    model = SuffixRanker(cfg)

    # Optimizer & scheduler
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = (len(train_ds) // args.batch_size) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio) if total_steps > 0 else 0
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    model, optim, sched = accelerator.prepare(model, optim, sched)

    # Training loop (simple single-item batching to keep code compact)
    model.train()
    for epoch in range(args.epochs):
        for step, idx in enumerate(range(len(train_ds))):
            item = train_ds[idx]
            toks, last_idx = collate_single(item, model.tokenizer, args.max_length)
            label = torch.tensor(item["label"], device=accelerator.device)
            toks = {k: v.to(accelerator.device) for k, v in toks.items()}

            logits = model(toks, last_idx)
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), label.unsqueeze(0))

            accelerator.backward(loss)
            optim.step()
            sched.step()
            optim.zero_grad()

            if (step + 1) % 50 == 0:
                accelerator.print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

    # Simple validation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for idx in range(len(val_ds)):
            item = val_ds[idx]
            toks, last_idx = collate_single(item, model.tokenizer, args.max_length)
            toks = {k: v.to(accelerator.device) for k, v in toks.items()}
            logits = model(toks, last_idx)
            ranks = torch.argsort(logits, descending=True).tolist()
            y_pred.append(ranks)
            y_true.append(int(item["label"]))
    score = mapk(y_true, y_pred, k=3)
    accelerator.print(f"Validation MAP@3: {score:.4f}")

    # Save checkpoint (lightweight)
    os.makedirs(args.output_dir, exist_ok=True)
    accelerator.print("Saving model...")
    if accelerator.is_main_process:
        model.backbone.save_pretrained(args.output_dir)
        model.tokenizer.save_pretrained(args.output_dir)

    # Save val predictions
    preds_path = os.path.join(args.output_dir, "val_logits.jsonl")
    with open(preds_path, "w", encoding="utf-8") as f:
        for t, ranks in zip(y_true, y_pred):
            f.write(json.dumps({"y_true": t, "ranks": ranks}) + "\\n")
    accelerator.print("Done.")

def parse_args() -> TrainArgs:
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    # allow dot-override like train.seed=123
    parser.add_argument("overrides", nargs="*", default=[])
    ns = parser.parse_args()
    cfg = load_yaml_cfg(ns.config)
    # merge overrides (very light parser)
    for ov in ns.overrides:
        path, value = ov.split("=", 1)
        sect, key = path.split(".")
        if sect not in cfg:
            cfg[sect] = {}
        # auto-cast numbers/bools
        if value.isdigit():
            value = int(value)
        elif value.lower() in {"true", "false"}:
            value = value.lower() == "true"
        else:
            try:
                value = float(value)
            except ValueError:
                pass
        cfg[sect][key] = value
    t = cfg["train"]
    return TrainArgs(
        model_name=t["model_name"],
        output_dir=t["output_dir"],
        epochs=int(t["epochs"]),
        batch_size=int(t["batch_size"]),
        learning_rate=float(t["learning_rate"]),
        weight_decay=float(t["weight_decay"]),
        warmup_ratio=float(t["warmup_ratio"]),
        seed=int(t["seed"]),
        fold_index=int(t["fold_index"]),
        max_length=int(t["max_length"]),
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        mixed_precision=str(t["mixed_precision"]),
        use_bitsandbytes=bool(t["use_bitsandbytes"]),
    )

def main():
    args = parse_args()
    train_one(args, data_cfg_path="configs/data.yaml")

if __name__ == "__main__":
    main()
