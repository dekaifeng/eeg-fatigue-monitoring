import numpy as np

from eeg_fatigue.features import extract_features, feature_columns
from eeg_fatigue.models import PreprocessingConfig, SyntheticConfig
from eeg_fatigue.preprocessing import preprocess_dataset
from eeg_fatigue.synthetic import generate_synthetic_dataset


def test_features_are_finite_and_show_designed_fatigue_shift():
    dataset = generate_synthetic_dataset(
        SyntheticConfig(n_subjects=4, epochs_per_class=5, artifact_probability=0.0)
    )
    cleaned, _, _ = preprocess_dataset(dataset, PreprocessingConfig())
    frame = extract_features(cleaned)
    assert len(feature_columns(frame)) == 10
    assert np.isfinite(frame[feature_columns(frame)].to_numpy()).all()
    ratios = frame.groupby("label")["theta_beta_ratio"].mean()
    assert ratios[1] > 2.0 * ratios[0]
