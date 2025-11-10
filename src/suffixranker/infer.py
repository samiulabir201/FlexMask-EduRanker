# -*- coding: utf-8 -*-
"""Inference entrypoint: produces per-item ranked suffix predictions."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from .model import ModelConfig, SuffixRanker
from .prompt import build_prefix

@dataclass
class InferArgs:
    model_name_or_path: str
    input_jsonl: str
    output_jsonl: str
    batch_size: int
    use_bitsandbytes: bool
    max_length: int

def load_yaml_cfg(path: str) -> Dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def infer_item(model: SuffixRanker, item: Dict, max_length: int) -> Dict:
    """Score a single item and return ranked suffix indices."""
    prefix = build_prefix(item["bundle"])
    fulls = [prefix + s for s in item["suffixes"]]
    toks = model.tokenizer(fulls, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    last_idx = (toks["attention_mask"].sum(dim=1) - 1).tolist()
    toks = {k: v.to(next(model.parameters()).device) for k, v in toks.items()}
    with torch.no_grad():
        logits = model(toks, last_idx)
        ranks = torch.argsort(logits, descending=True).tolist()
    return {"ranks": ranks}

def run(args: InferArgs) -> None:
    cfg = ModelConfig(model_name=args.model_name_or_path, max_length=args.max_length, use_bitsandbytes=args.use_bitsandbytes)
    model = SuffixRanker(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            pred = infer_item(model, item, args.max_length)
            fout.write(json.dumps(pred) + "\\n")

def parse_args() -> InferArgs:
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/infer.yaml")
    ns = parser.parse_args()
    cfg = load_yaml_cfg(ns.config)["infer"]
    return InferArgs(
        model_name_or_path=cfg["model_name_or_path"],
        input_jsonl=cfg["input_jsonl"],
        output_jsonl=cfg["output_jsonl"],
        batch_size=int(cfg["batch_size"]),
        use_bitsandbytes=bool(cfg["use_bitsandbytes"]),
        max_length=int(cfg["max_length"]),
    )

def main():
    run(parse_args())

if __name__ == "__main__":
    main()
