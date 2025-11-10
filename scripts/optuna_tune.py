#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optuna hyperparameter tuning for FlexMask EduRanker.

This script launches multiple trials, calling the training entrypoint per trial with overrides.
It logs MAP@3 per trial and saves optimization plots.

Note: For speed/cost, you may limit to one fold and fewer steps/epochs.
"""
from __future__ import annotations

import os, json, argparse, subprocess, tempfile
import optuna
import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

def objective(trial: optuna.Trial, base_config: str, data_config: str, out_root: str) -> float:
    lr = trial.suggest_float("learning_rate", 5e-6, 5e-5, log=True)
    wd = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup = trial.suggest_float("warmup_ratio", 0.0, 0.1)
    seed = trial.suggest_int("seed", 1, 999)
    batch = trial.suggest_categorical("batch_size", [8, 16, 32])

    run_dir = os.path.join(out_root, f"trial_{trial.number}")
    os.makedirs(run_dir, exist_ok=True)

    overrides = [
        f"train.learning_rate={lr}",
        f"train.weight_decay={wd}",
        f"train.warmup_ratio={warmup}",
        f"train.seed={seed}",
        f"train.batch_size={batch}",
        f"train.output_dir={run_dir}",
    ]
    cmd = ["python", "-m", "suffixranker.train", "--config", base_config] + overrides
    env = os.environ.copy()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    # read val score
    val_file = os.path.join(run_dir, "logs", "val_map3.txt")
    if os.path.exists(val_file):
        with open(val_file, "r", encoding="utf-8") as f:
            score = float(f.read().strip())
    else:
        score = 0.0  # failed trial
    return score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", default="configs/train.yaml")
    ap.add_argument("--data_config", default="configs/data.yaml")
    ap.add_argument("--out_dir", default="artifacts/optuna")
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    study = optuna.create_study(direction="maximize", study_name="eduranker-tuning")
    study.optimize(lambda t: objective(t, args.base_config, args.data_config, args.out_dir), n_trials=args.trials)

    # Save study
    with open(os.path.join(args.out_dir, "study_best.json"), "w", encoding="utf-8") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=2)

    # Plots
    fig1 = plot_optimization_history(study)
    fig1.figure.savefig(os.path.join(args.out_dir, "opt_history.png"), dpi=170)
    plt.close(fig1.figure)

    fig2 = plot_param_importances(study)
    fig2.figure.savefig(os.path.join(args.out_dir, "opt_param_importance.png"), dpi=170)
    plt.close(fig2.figure)

    print("Optuna tuning complete. Plots saved under:", args.out_dir)

if __name__ == "__main__":
    main()
