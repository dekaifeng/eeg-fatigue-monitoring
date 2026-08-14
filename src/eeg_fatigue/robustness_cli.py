"""Run Phase 3 robustness, calibration, and nested-validation experiments."""

from __future__ import annotations

import argparse
import json

from eeg_fatigue.datasets import FIGSHARE_FILENAME
from eeg_fatigue.models import StudyConfig
from eeg_fatigue.robustness import RobustnessConfig, run_robustness_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=f"data/raw/{FIGSHARE_FILENAME}")
    parser.add_argument("--study-config", default="configs/figshare.yaml")
    parser.add_argument("--robustness-config", default="configs/robustness.yaml")
    parser.add_argument("--output", default="results/robustness")
    arguments = parser.parse_args()
    summary = run_robustness_study(
        arguments.dataset,
        StudyConfig.from_yaml(arguments.study_config),
        RobustnessConfig.from_yaml(arguments.robustness_config),
        arguments.output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
