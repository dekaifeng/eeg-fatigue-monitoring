#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-data/raw/figshare_14273687_v3.mat}"
test -s "$dataset"
eeg-fatigue-replay \
  --dataset "$dataset" \
  --study-config configs/figshare.yaml \
  --streaming-config configs/streaming.yaml \
  --output results/streaming
test -s results/streaming/streaming_summary.json
test -s results/streaming/replay_events.csv
test -s results/streaming/epoch_replay_demo.gif
