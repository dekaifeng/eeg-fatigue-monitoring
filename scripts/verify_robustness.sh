#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-data/raw/figshare_14273687_v3.mat}"
test -s "$dataset"
eeg-fatigue-robustness \
  --dataset "$dataset" \
  --study-config configs/figshare.yaml \
  --robustness-config configs/robustness.yaml \
  --output results/robustness
test -s results/robustness/robustness_summary.json
test -s results/robustness/threshold_sensitivity.png
test -s results/robustness/calibration_curve.png
