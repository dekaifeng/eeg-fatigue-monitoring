import numpy as np
import pytest

from eeg_fatigue.dropout import inject_channel_dropout
from eeg_fatigue.synthetic import EpochDataset


def _dataset() -> EpochDataset:
    data = np.arange(4 * 5 * 8, dtype=float).reshape(4, 5, 8)
    return EpochDataset(
        data_v=data,
        labels=np.array([0, 1, 0, 1]),
        subjects=np.array([1, 1, 2, 2]),
        epoch_ids=np.arange(4),
        artifact_truth=np.zeros(4, dtype=bool),
        sampling_rate_hz=128.0,
        channel_names=tuple(f"C{index}" for index in range(5)),
    )


def test_zero_dropout_returns_unchanged_dataset() -> None:
    dataset = _dataset()
    result = inject_channel_dropout(dataset, 0, max_missing_channels=2)

    assert result.valid
    assert result.dataset is dataset
    assert not result.missing_mask.any()


def test_dropout_imputes_a_finite_epoch_median() -> None:
    result = inject_channel_dropout(_dataset(), 2, max_missing_channels=2)

    assert result.valid
    assert result.dataset is not None
    assert result.missing_mask.sum(axis=1).tolist() == [2, 2, 2, 2]
    assert np.isfinite(result.dataset.data_v).all()


def test_dropout_above_policy_limit_returns_fault() -> None:
    result = inject_channel_dropout(_dataset(), 3, max_missing_channels=2)

    assert not result.valid
    assert result.dataset is None
    assert result.reason == "missing-channel limit exceeded"


@pytest.mark.parametrize("n_missing", [-1, 6])
def test_invalid_dropout_count_is_rejected(n_missing: int) -> None:
    with pytest.raises(ValueError, match="channel range"):
        inject_channel_dropout(_dataset(), n_missing, max_missing_channels=2)
