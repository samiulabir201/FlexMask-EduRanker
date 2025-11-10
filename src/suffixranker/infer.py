from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import torch

from .model import ModelConfig, SuffixRanker
from .prompt import build_prefix


@dataclass
class InferArgs:
    model_name_or_path: str
    input_jsonl: str
    output_jsonl: str
    max_length: int
    use_bitsandbytes: bool


def load_yaml_cfg(path: str) -> dict:
    """Load a small YAML configuration file."""
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_item(model: SuffixRanker, item: dict, max_length: int) -> dict:
    """Score a single item and return ranked suffix indices.

    Args:
        model: Trained SuffixRanker model.
        item: Raw JSON-serializable dict with 'bundle' and 'suffixes'.
        max_length: Maximum sequence length for tokenization.

    Returns:
        Dict with 'ranks': list[int] giving candidate indices sorted by score descending.
    """
    prefix = build_prefix(item["bundle"])
    fulls = [prefix + s for s in item["suffixes"]]

    toks = model.tokenizer(
        fulls,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    last_idx = (toks["attention_mask"].sum(dim=1) - 1).tolist()
    toks = {k: v.to(next(model.parameters()).device) for k, v in toks.items()}

    with torch.no_grad():
        logits = model(toks, last_idx).cpu().numpy().tolist()

    ranks = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)
    return {"ranks": ranks}


def run(args: InferArgs) -> None:
    """Run inference over a JSONL file of items."""
    cfg = ModelConfig(
        model_name=args.model_name_or_path,
        max_length=args.max_length,
        use_bitsandbytes=args.use_bitsandbytes,
    )
    model = SuffixRanker(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    with open(args.input_jsonl, encoding="utf-8") as fin, open(
        args.output_jsonl,
        "w",
        encoding="utf-8",
    ) as fout:
        for line in fin:
            item = json.loads(line)
            out = infer_item(model, item, args.max_length)
            fout.write(json.dumps(out) + "\n")


def parse_args() -> InferArgs:
    """CLI argument parser for inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/infer.yaml")
    args_ns = parser.parse_args()

    cfg = load_yaml_cfg(args_ns.config)
    model_cfg = cfg["model"]
    infer_cfg = cfg["infer"]

    return InferArgs(
        model_name_or_path=model_cfg["name_or_path"],
        input_jsonl=infer_cfg["input_jsonl"],
        output_jsonl=infer_cfg["output_jsonl"],
        max_length=model_cfg.get("max_length", 2048),
        use_bitsandbytes=model_cfg.get("use_bitsandbytes", False),
    )


if __name__ == "__main__":
    cli_args = parse_args()
    run(cli_args)
