import numpy as np
from scipy.io import savemat

from eeg_fatigue.models import EvaluationConfig, PreprocessingConfig, StudyConfig
from eeg_fatigue.real_experiment import run_real_study


def test_real_study_runs_on_structurally_valid_fixture(tmp_path):
    rng = np.random.default_rng(7)
    path = tmp_path / "fixture.mat"
    labels = np.tile([0, 1], 4).reshape(-1, 1)
    subjects = np.repeat([1, 2], 4).reshape(-1, 1)
    savemat(
        path,
        {
            "EEGsample": rng.normal(0.0, 10.0, size=(8, 30, 384)),
            "substate": labels,
            "subindex": subjects,
        },
    )
    config = StudyConfig(
        preprocessing=PreprocessingConfig(
            peak_to_peak_threshold_uv=500.0,
            apply_filter=False,
            reject_peak_to_peak=False,
        ),
        evaluation=EvaluationConfig(n_splits=2, strategy="leave_one_subject_out"),
    )
    summary = run_real_study(path, config, tmp_path / "results")
    assert summary["n_subjects"] == 2
    assert summary["n_splits"] == 2
    assert summary["evaluation"] == "leave_one_subject_out"
