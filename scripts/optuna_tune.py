#!/usr/bin/env python
"""
Optuna hyperparameter tuning for suffix ranker.

This script shells out to `python -m suffixranker.train` with overrides and reads
the validation MAP@3 from a log file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
)


def objective(
    trial: optuna.Trial,
    base_config: str,
    data_config: str,
    out_root: str,
) -> float:
    """Optuna objective: run one training job and return validation MAP@3."""
    lr = trial.suggest_float("learning_rate", 5e-6, 3e-5, log=True)
    wd = trial.suggest_float("weight_decay", 0.0, 0.1)
    seed = trial.suggest_int("seed", 1, 5)

    run_dir = Path(out_root) / f"trial_{trial.number}"
    run_dir.mkdir(parents=True, exist_ok=True)

    overrides = [
        f"--training.learning_rate={lr}",
        f"--training.weight_decay={wd}",
        f"--training.seed={seed}",
        f"--logging.out_dir={run_dir}",
        f"--data.config={data_config}",
    ]

    cmd = ["python", "-m", "suffixranker.train", "--config", base_config] + overrides
    env = os.environ.copy()
    # You could add CUDA_VISIBLE_DEVICES, etc., here.
    subprocess.run(cmd, env=env, check=False)

    val_file = run_dir / "logs" / "val_map3.txt"
    if val_file.exists():
        score = float(val_file.read_text(encoding="utf-8").strip())
    else:
        # Penalize failed runs
        score = 0.0
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--data_config", default="configs/data.yaml")
    parser.add_argument("--out_root", default="artifacts/optuna")
    parser.add_argument("--n_trials", type=int, default=20)
    args = parser.parse_args()

    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    def obj_wrapper(trial: optuna.Trial) -> float:
        return objective(trial, args.config, args.data_config, args.out_root)

    study = optuna.create_study(direction="maximize")
    study.optimize(obj_wrapper, n_trials=args.n_trials)

    # Save study summary
    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
    }
    Path(args.out_root, "best.json").write_text(
        json.dumps(best, indent=2),
        encoding="utf-8",
    )

    # Plots
    plt.figure()
    plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(Path(args.out_root, "opt_history.png"), dpi=170)
    plt.close()

    plt.figure()
    plot_param_importances(study)
    plt.tight_layout()
    plt.savefig(Path(args.out_root, "opt_param_importances.png"), dpi=170)
    plt.close()

    print(f"Best MAP@3: {study.best_value:.5f}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
