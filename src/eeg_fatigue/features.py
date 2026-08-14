"""Frequency-domain features with explicit units and band definitions."""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd

from eeg_fatigue.synthetic import EpochDataset

BANDS_HZ = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def _integrate_band(psd: np.ndarray, frequencies: np.ndarray, bounds: tuple[float, float]):
    low, high = bounds
    mask = (frequencies >= low) & (frequencies < high)
    return np.trapezoid(psd[..., mask], frequencies[mask], axis=-1)


def extract_features(dataset: EpochDataset) -> pd.DataFrame:
    n_times = dataset.data_v.shape[-1]
    psd, frequencies = mne.time_frequency.psd_array_welch(
        dataset.data_v,
        sfreq=dataset.sampling_rate_hz,
        fmin=1.0,
        fmax=40.0,
        n_fft=n_times,
        n_per_seg=n_times,
        average="mean",
        verbose=False,
    )
    band_power = {
        name: _integrate_band(psd, frequencies, limits)
        for name, limits in BANDS_HZ.items()
    }
    total = _integrate_band(psd, frequencies, (1.0, 40.01))
    eps = np.finfo(float).eps

    features: dict[str, np.ndarray] = {
        "epoch_id": dataset.epoch_ids,
        "subject": dataset.subjects,
        "label": dataset.labels,
    }
    for name, values in band_power.items():
        features[f"{name}_relative"] = np.mean(values / (total + eps), axis=1)

    theta = band_power["theta"].mean(axis=1)
    alpha = band_power["alpha"].mean(axis=1)
    beta = band_power["beta"].mean(axis=1)
    features["theta_beta_ratio"] = theta / (beta + eps)
    features["theta_alpha_beta_ratio"] = (theta + alpha) / (beta + eps)
    features["theta_spatial_std"] = np.std(band_power["theta"] / (total + eps), axis=1)
    features["alpha_spatial_std"] = np.std(band_power["alpha"] / (total + eps), axis=1)

    probability = psd / (psd.sum(axis=-1, keepdims=True) + eps)
    entropy = -np.sum(probability * np.log(probability + eps), axis=-1)
    entropy /= np.log(psd.shape[-1])
    features["spectral_entropy"] = entropy.mean(axis=1)
    features["rms_uv"] = np.sqrt(np.mean(dataset.data_v**2, axis=(1, 2))) * 1e6

    frame = pd.DataFrame(features)
    if not np.isfinite(frame.select_dtypes(include=["number"]).to_numpy()).all():
        raise ValueError("feature extraction produced a non-finite value")
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in {"epoch_id", "subject", "label"}]
