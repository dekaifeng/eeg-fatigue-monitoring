#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-data/raw/figshare_14273687_v3.mat}"
test -s "$dataset"
python -c 'from eeg_fatigue.datasets import FIGSHARE_SHA256, sha256_file; import sys; assert sha256_file(sys.argv[1]) == FIGSHARE_SHA256' "$dataset"
eeg-fatigue-real-study \
  --dataset "$dataset" \
  --config configs/figshare.yaml \
  --output results/figshare
test -s results/figshare/summary.json
test -s results/figshare/cross_subject_performance.png
