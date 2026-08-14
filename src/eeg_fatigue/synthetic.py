"""Deterministic synthetic EEG used to exercise the complete pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from eeg_fatigue.models import SyntheticConfig

CHANNEL_NAMES = ("F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2")


@dataclass(frozen=True)
class EpochDataset:
    data_v: NDArray[np.float64]
    labels: NDArray[np.int64]
    subjects: NDArray[np.int64]
    epoch_ids: NDArray[np.int64]
    artifact_truth: NDArray[np.bool_]
    sampling_rate_hz: float
    channel_names: tuple[str, ...] = CHANNEL_NAMES

    def subset(self, keep: NDArray[np.bool_]) -> EpochDataset:
        return EpochDataset(
            data_v=self.data_v[keep],
            labels=self.labels[keep],
            subjects=self.subjects[keep],
            epoch_ids=self.epoch_ids[keep],
            artifact_truth=self.artifact_truth[keep],
            sampling_rate_hz=self.sampling_rate_hz,
            channel_names=self.channel_names,
        )


def _oscillation(
    rng: np.random.Generator,
    time_s: NDArray[np.float64],
    center_hz: float,
    amplitude_uv: float,
    channel_weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    frequency = center_hz + rng.normal(0.0, 0.25)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(len(channel_weights), 1))
    carrier = np.sin(2.0 * np.pi * frequency * time_s[None, :] + phases)
    return amplitude_uv * 1e-6 * channel_weights[:, None] * carrier


def generate_synthetic_dataset(config: SyntheticConfig) -> EpochDataset:
    """Generate alert/fatigued epochs with subject variability and sparse artifacts.

    Label 0 denotes alert and label 1 denotes fatigued. The spectral changes are
    intentionally explicit, so these data validate software behavior rather than
    estimate clinical performance.
    """
    rng = np.random.default_rng(config.random_seed)
    n_samples = int(round(config.sampling_rate_hz * config.epoch_duration_s))
    time_s = np.arange(n_samples, dtype=float) / config.sampling_rate_hz
    frontal = np.array([1.25, 1.25, 1.0, 1.0, 0.8, 0.8, 0.65, 0.65])
    posterior = frontal[::-1]
    uniform = np.ones(len(CHANNEL_NAMES))

    epochs: list[NDArray[np.float64]] = []
    labels: list[int] = []
    subjects: list[int] = []
    epoch_ids: list[int] = []
    artifacts: list[bool] = []
    epoch_id = 0

    for subject in range(config.n_subjects):
        subject_gain = rng.lognormal(mean=0.0, sigma=0.12)
        subject_offset = rng.normal(0.0, 0.25)
        for label in (0, 1):
            for _ in range(config.epochs_per_class):
                if label == 0:
                    theta_uv, alpha_uv, beta_uv = 3.5, 8.0, 6.0
                else:
                    theta_uv, alpha_uv, beta_uv = 8.5, 10.0, 3.0
                signal = _oscillation(
                    rng, time_s, 6.0 + subject_offset, theta_uv, frontal
                )
                signal += _oscillation(
                    rng, time_s, 10.0 + 0.3 * subject_offset, alpha_uv, posterior
                )
                signal += _oscillation(rng, time_s, 20.0, beta_uv, uniform)
                signal += rng.normal(0.0, 2.5e-6, size=signal.shape)
                signal *= subject_gain

                has_artifact = rng.random() < config.artifact_probability
                if has_artifact:
                    channel = int(rng.integers(0, len(CHANNEL_NAMES)))
                    center = int(rng.integers(n_samples // 4, 3 * n_samples // 4))
                    width = max(2, n_samples // 64)
                    pulse = np.exp(-0.5 * ((np.arange(n_samples) - center) / width) ** 2)
                    signal[channel] += rng.choice((-1.0, 1.0)) * 350e-6 * pulse

                epochs.append(signal)
                labels.append(label)
                subjects.append(subject)
                epoch_ids.append(epoch_id)
                artifacts.append(has_artifact)
                epoch_id += 1

    return EpochDataset(
        data_v=np.stack(epochs),
        labels=np.asarray(labels, dtype=np.int64),
        subjects=np.asarray(subjects, dtype=np.int64),
        epoch_ids=np.asarray(epoch_ids, dtype=np.int64),
        artifact_truth=np.asarray(artifacts, dtype=np.bool_),
        sampling_rate_hz=config.sampling_rate_hz,
    )
