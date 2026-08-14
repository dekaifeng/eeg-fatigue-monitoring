"""Run the Phase 4 accelerated EEG epoch-replay demonstration."""

from __future__ import annotations

import argparse
import json

from eeg_fatigue.datasets import FIGSHARE_FILENAME
from eeg_fatigue.models import StudyConfig
from eeg_fatigue.streaming_experiment import StreamingConfig, run_streaming_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=f"data/raw/{FIGSHARE_FILENAME}")
    parser.add_argument("--study-config", default="configs/figshare.yaml")
    parser.add_argument("--streaming-config", default="configs/streaming.yaml")
    parser.add_argument("--output", default="results/streaming")
    arguments = parser.parse_args()
    summary = run_streaming_study(
        arguments.dataset,
        StudyConfig.from_yaml(arguments.study_config),
        StreamingConfig.from_yaml(arguments.streaming_config),
        arguments.output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
