#!/usr/bin/env bash
set -euo pipefail

# Example CV runner (customize model / seeds / folds)
SEEDS=(101 202 303)
FOLDS=(0 1 2 3 4)

for S in "${SEEDS[@]}"; do
  for F in "${FOLDS[@]}"; do
    echo "=== Train seed=$S fold=$F ==="
    python -m suffixranker.train --config configs/train.yaml train.seed=$S train.fold_index=$F train.output_dir=artifacts/checkpoints/run_seed${S}_fold${F}
  done
done
