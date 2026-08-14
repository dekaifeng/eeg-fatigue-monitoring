import json

from eeg_fatigue.experiment import run_study
from eeg_fatigue.models import EvaluationConfig, StudyConfig, SyntheticConfig


def test_end_to_end_study_writes_expected_outputs(tmp_path):
    config = StudyConfig(
        synthetic=SyntheticConfig(
            n_subjects=4, epochs_per_class=3, artifact_probability=0.0
        ),
        evaluation=EvaluationConfig(n_splits=2),
    )
    summary = run_study(config, tmp_path)
    expected = {
        "features.csv",
        "fold_metrics.csv",
        "predictions.csv",
        "summary.json",
        "feature_separation.png",
        "cross_subject_performance.png",
        "subject_performance.png",
        "confusion_matrix.png",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
