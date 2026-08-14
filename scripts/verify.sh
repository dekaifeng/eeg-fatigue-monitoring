#!/usr/bin/env bash
set -euo pipefail

python -m ruff check .
python -m pytest
tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "$tmp_dir"' EXIT
eeg-fatigue-study --config configs/default.yaml --output "$tmp_dir/results" >/dev/null
test -s "$tmp_dir/results/summary.json"
test -s "$tmp_dir/results/cross_subject_performance.png"
