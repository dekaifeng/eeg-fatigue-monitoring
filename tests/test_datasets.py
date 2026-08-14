import numpy as np
import pytest
from scipy.io import savemat

from eeg_fatigue.datasets import load_figshare_dataset, sha256_file


def _write_fixture(path, shape=(4, 30, 384)):
    eeg = np.arange(np.prod(shape), dtype=float).reshape(shape) / 100.0
    savemat(
        path,
        {
            "EEGsample": eeg,
            "substate": np.array([[0], [1], [0], [1]], dtype=np.uint8),
            "subindex": np.array([[1], [1], [2], [2]], dtype=np.uint8),
        },
    )


def test_figshare_loader_converts_microvolts_and_preserves_groups(tmp_path):
    path = tmp_path / "fixture.mat"
    _write_fixture(path)
    dataset = load_figshare_dataset(path)
    assert dataset.data_v.shape == (4, 30, 384)
    assert dataset.data_v[0, 0, 1] == pytest.approx(0.01e-6)
    assert dataset.channel_names[0] == "EEG01"
    assert dataset.channel_names[-1] == "EEG30"
    assert dataset.subjects.tolist() == [1, 1, 2, 2]


def test_figshare_loader_rejects_wrong_shape(tmp_path):
    path = tmp_path / "fixture.mat"
    _write_fixture(path, shape=(4, 29, 384))
    with pytest.raises(ValueError, match="expected EEG shape"):
        load_figshare_dataset(path)


def test_sha256_file(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
