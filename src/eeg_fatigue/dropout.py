"""Deterministic missing-channel injection and explicit imputation policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from eeg_fatigue.synthetic import EpochDataset


@dataclass(frozen=True)
class DropoutResult:
    dataset: EpochDataset | None
    missing_mask: NDArray[np.bool_]
    valid: bool
    reason: str


def inject_channel_dropout(
    dataset: EpochDataset,
    n_missing: int,
    max_missing_channels: int,
    random_seed: int = 2026,
) -> DropoutResult:
    n_epochs, n_channels, _ = dataset.data_v.shape
    if n_missing < 0 or n_missing > n_channels:
        raise ValueError("n_missing is outside the channel range")
    missing_mask = np.zeros((n_epochs, n_channels), dtype=np.bool_)
    if n_missing > max_missing_channels or n_missing == n_channels:
        return DropoutResult(None, missing_mask, False, "missing-channel limit exceeded")
    if n_missing == 0:
        return DropoutResult(dataset, missing_mask, True, "no dropout")

    rng = np.random.default_rng(random_seed + n_missing)
    imputed = dataset.data_v.copy()
    for epoch in range(n_epochs):
        missing = rng.choice(n_channels, size=n_missing, replace=False)
        missing_mask[epoch, missing] = True
        available = ~missing_mask[epoch]
        replacement = np.median(imputed[epoch, available], axis=0)
        imputed[epoch, missing] = replacement
    return DropoutResult(
        EpochDataset(
            data_v=imputed,
            labels=dataset.labels,
            subjects=dataset.subjects,
            epoch_ids=dataset.epoch_ids,
            artifact_truth=dataset.artifact_truth,
            sampling_rate_hz=dataset.sampling_rate_hz,
            channel_names=dataset.channel_names,
        ),
        missing_mask,
        True,
        "median across available channels",
    )
