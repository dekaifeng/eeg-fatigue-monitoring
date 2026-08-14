from eeg_fatigue.evaluation import evaluate_models
from eeg_fatigue.features import extract_features
from eeg_fatigue.models import EvaluationConfig, PreprocessingConfig, SyntheticConfig
from eeg_fatigue.preprocessing import preprocess_dataset
from eeg_fatigue.synthetic import generate_synthetic_dataset


def test_grouped_evaluation_predicts_each_retained_epoch_once_per_model():
    dataset = generate_synthetic_dataset(
        SyntheticConfig(n_subjects=5, epochs_per_class=4, artifact_probability=0.0)
    )
    cleaned, _, _ = preprocess_dataset(dataset, PreprocessingConfig())
    features = extract_features(cleaned)
    result = evaluate_models(features, EvaluationConfig(n_splits=5))
    for _, rows in result.predictions.groupby("model"):
        assert len(rows) == len(features)
        assert rows["epoch_id"].nunique() == len(features)
    assert result.summary["logistic_regression"]["balanced_accuracy_mean"] > 0.9
