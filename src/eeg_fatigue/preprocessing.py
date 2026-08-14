"""Filtering and transparent amplitude-based artifact rejection."""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
from numpy.typing import NDArray

from eeg_fatigue.models import PreprocessingConfig
from eeg_fatigue.synthetic import EpochDataset


@dataclass(frozen=True)
class PreprocessingReport:
    total_epochs: int
    retained_epochs: int
    rejected_epochs: int
    rejection_rate: float


def preprocess_dataset(
    dataset: EpochDataset, config: PreprocessingConfig
) -> tuple[EpochDataset, NDArray[np.bool_], PreprocessingReport]:
    if dataset.data_v.ndim != 3 or not np.isfinite(dataset.data_v).all():
        raise ValueError("EEG data must be a finite epochs-by-channels-by-samples array")
    threshold_v = config.peak_to_peak_threshold_uv * 1e-6
    if config.reject_peak_to_peak:
        peak_to_peak = np.ptp(dataset.data_v, axis=-1)
        keep = np.max(peak_to_peak, axis=1) <= threshold_v
    else:
        keep = np.ones(len(dataset.data_v), dtype=np.bool_)
    if not np.any(keep):
        raise ValueError("artifact rejection removed every epoch")

    cleaned = dataset.subset(keep)
    if config.apply_filter:
        filtered = mne.filter.filter_data(
            cleaned.data_v,
            sfreq=cleaned.sampling_rate_hz,
            l_freq=config.low_cut_hz,
            h_freq=config.high_cut_hz,
            method="iir",
            iir_params={"order": 4, "ftype": "butter"},
            verbose=False,
        )
    else:
        filtered = cleaned.data_v.copy()
    cleaned = EpochDataset(
        data_v=np.asarray(filtered, dtype=np.float64),
        labels=cleaned.labels,
        subjects=cleaned.subjects,
        epoch_ids=cleaned.epoch_ids,
        artifact_truth=cleaned.artifact_truth,
        sampling_rate_hz=cleaned.sampling_rate_hz,
        channel_names=cleaned.channel_names,
    )
    rejected = int((~keep).sum())
    report = PreprocessingReport(
        total_epochs=len(keep),
        retained_epochs=int(keep.sum()),
        rejected_epochs=rejected,
        rejection_rate=rejected / len(keep),
    )
    return cleaned, keep, report
