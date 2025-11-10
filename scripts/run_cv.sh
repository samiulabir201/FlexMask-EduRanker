#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/train.yaml}

for FOLD in 0 1 2 3 4; do
  echo "=== Training fold ${FOLD} ==="
  python -m suffixranker.train --config "${CONFIG}" --training.fold="${FOLD}"
done
