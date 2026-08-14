"""Run the handcrafted-feature baselines on the licensed public dataset."""

from __future__ import annotations

from pathlib import Path

from eeg_fatigue.datasets import FIGSHARE_DOI, load_figshare_dataset
from eeg_fatigue.experiment import run_dataset_study
from eeg_fatigue.models import StudyConfig


def run_real_study(
    dataset_path: str | Path, config: StudyConfig, output_dir: str | Path
) -> dict[str, object]:
    dataset = load_figshare_dataset(dataset_path)
    return run_dataset_study(
        dataset,
        config,
        output_dir,
        data_source=f"Figshare EEG driver drowsiness dataset v3, DOI {FIGSHARE_DOI}",
    )
