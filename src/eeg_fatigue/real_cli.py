"""Evaluate fixed baselines on the Phase 2 public EEG dataset."""

from __future__ import annotations

import argparse
import json

from eeg_fatigue.datasets import FIGSHARE_FILENAME
from eeg_fatigue.models import StudyConfig
from eeg_fatigue.real_experiment import run_real_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=f"data/raw/{FIGSHARE_FILENAME}")
    parser.add_argument("--config", default="configs/figshare.yaml")
    parser.add_argument("--output", default="results/figshare")
    arguments = parser.parse_args()
    summary = run_real_study(
        arguments.dataset, StudyConfig.from_yaml(arguments.config), arguments.output
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
