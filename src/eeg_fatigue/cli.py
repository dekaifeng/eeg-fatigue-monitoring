"""Command-line entry point."""

from __future__ import annotations

import argparse
import json

from eeg_fatigue.experiment import run_study
from eeg_fatigue.models import StudyConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="results")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    summary = run_study(StudyConfig.from_yaml(arguments.config), arguments.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
