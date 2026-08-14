"""Licensed public-dataset download and loading utilities."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from eeg_fatigue.synthetic import EpochDataset

FIGSHARE_ARTICLE_ID = 14273687
FIGSHARE_VERSION = 3
FIGSHARE_DOI = "10.6084/m9.figshare.14273687.v3"
FIGSHARE_DOWNLOAD_URL = "https://figshare.com/ndownloader/files/30707285"
FIGSHARE_SHA256 = "53cc4ef14b1343f7f3fb5322dd2b541c031c6c2297bddf1817aed04dc687a6a4"
FIGSHARE_FILENAME = "figshare_14273687_v3.mat"
FIGSHARE_SAMPLING_RATE_HZ = 128.0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_figshare_dataset(destination: str | Path, overwrite: bool = False) -> Path:
    """Download the CC BY 4.0 Figshare v3 artifact and verify its checksum."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        if sha256_file(destination) != FIGSHARE_SHA256:
            raise ValueError("existing dataset checksum does not match Figshare v3")
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(FIGSHARE_DOWNLOAD_URL, temporary)  # noqa: S310
        actual = sha256_file(temporary)
        if actual != FIGSHARE_SHA256:
            raise ValueError(f"dataset checksum mismatch: {actual}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_figshare_dataset(path: str | Path) -> EpochDataset:
    """Load the 30-channel, three-second samples and convert microvolts to volts."""
    payload = loadmat(Path(path), variable_names=["EEGsample", "substate", "subindex"])
    required = {"EEGsample", "substate", "subindex"}
    if not required.issubset(payload):
        raise ValueError(f"dataset is missing variables: {sorted(required - payload.keys())}")

    eeg_uv = np.asarray(payload["EEGsample"], dtype=np.float64)
    labels = np.asarray(payload["substate"]).reshape(-1).astype(np.int64)
    subjects = np.asarray(payload["subindex"]).reshape(-1).astype(np.int64)
    if eeg_uv.ndim != 3 or eeg_uv.shape[1:] != (30, 384):
        raise ValueError(f"expected EEG shape (epochs, 30, 384), received {eeg_uv.shape}")
    if len(labels) != len(eeg_uv) or len(subjects) != len(eeg_uv):
        raise ValueError("EEG, label, and subject arrays are not aligned")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("labels must contain alert=0 and drowsy=1")
    if len(np.unique(subjects)) < 2:
        raise ValueError("at least two subjects are required")
    if not np.isfinite(eeg_uv).all():
        raise ValueError("dataset contains non-finite EEG values")

    channel_names = tuple(f"EEG{index:02d}" for index in range(1, 31))
    return EpochDataset(
        data_v=eeg_uv * 1e-6,
        labels=labels,
        subjects=subjects,
        epoch_ids=np.arange(len(eeg_uv), dtype=np.int64),
        artifact_truth=np.zeros(len(eeg_uv), dtype=np.bool_),
        sampling_rate_hz=FIGSHARE_SAMPLING_RATE_HZ,
        channel_names=channel_names,
    )
