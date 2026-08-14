import numpy as np

from eeg_fatigue.models import SyntheticConfig
from eeg_fatigue.synthetic import generate_synthetic_dataset


def test_synthetic_dataset_shape_balance_and_determinism():
    config = SyntheticConfig(n_subjects=3, epochs_per_class=2, artifact_probability=0.0)
    first = generate_synthetic_dataset(config)
    second = generate_synthetic_dataset(config)
    assert first.data_v.shape == (12, 8, 512)
    assert np.bincount(first.labels).tolist() == [6, 6]
    assert np.array_equal(first.data_v, second.data_v)
    assert set(first.subjects) == {0, 1, 2}


def test_subset_keeps_metadata_aligned():
    dataset = generate_synthetic_dataset(
        SyntheticConfig(n_subjects=2, epochs_per_class=2, artifact_probability=0.0)
    )
    keep = dataset.labels == 1
    subset = dataset.subset(keep)
    assert len(subset.data_v) == 4
    assert np.all(subset.labels == 1)
    assert np.array_equal(subset.epoch_ids, dataset.epoch_ids[keep])
