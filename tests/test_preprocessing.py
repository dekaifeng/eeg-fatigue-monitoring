import numpy as np

from eeg_fatigue.models import PreprocessingConfig, SyntheticConfig
from eeg_fatigue.preprocessing import preprocess_dataset
from eeg_fatigue.synthetic import EpochDataset, generate_synthetic_dataset


def test_preprocessing_retains_clean_finite_epochs():
    dataset = generate_synthetic_dataset(
        SyntheticConfig(n_subjects=2, epochs_per_class=2, artifact_probability=0.0)
    )
    cleaned, keep, report = preprocess_dataset(dataset, PreprocessingConfig())
    assert keep.all()
    assert report.rejected_epochs == 0
    assert cleaned.data_v.shape == dataset.data_v.shape
    assert np.isfinite(cleaned.data_v).all()


def test_amplitude_artifact_is_rejected():
    dataset = generate_synthetic_dataset(
        SyntheticConfig(n_subjects=2, epochs_per_class=2, artifact_probability=0.0)
    )
    corrupted = dataset.data_v.copy()
    corrupted[0, 0, 100] = 500e-6
    modified = EpochDataset(
        corrupted,
        dataset.labels,
        dataset.subjects,
        dataset.epoch_ids,
        dataset.artifact_truth,
        dataset.sampling_rate_hz,
    )
    cleaned, keep, report = preprocess_dataset(modified, PreprocessingConfig())
    assert not keep[0]
    assert report.rejected_epochs == 1
    assert len(cleaned.data_v) == len(dataset.data_v) - 1
