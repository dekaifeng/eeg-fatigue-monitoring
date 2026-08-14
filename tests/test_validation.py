import numpy as np
import pandas as pd

from eeg_fatigue.validation import (
    cross_fitted_random_forest_calibration,
    nested_logistic_evaluation,
)


def _feature_frame() -> pd.DataFrame:
    rng = np.random.default_rng(12)
    rows = []
    epoch_id = 0
    for subject in range(6):
        for label in (0, 1):
            for _ in range(5):
                rows.append(
                    {
                        "epoch_id": epoch_id,
                        "subject": subject,
                        "label": label,
                        "feature_a": label + rng.normal(0.0, 0.25),
                        "feature_b": 0.2 * subject + rng.normal(0.0, 0.3),
                    }
                )
                epoch_id += 1
    return pd.DataFrame(rows)


def test_cross_fitted_calibration_covers_every_outer_sample_twice():
    frame = _feature_frame()
    result = cross_fitted_random_forest_calibration(
        frame, inner_splits=3, n_estimators=20
    )
    assert set(result.fold_metrics["mode"]) == {"raw", "platt_oof"}
    assert len(result.fold_metrics) == 12
    assert len(result.predictions) == 2 * len(frame)
    assert result.predictions.groupby("mode")["epoch_id"].nunique().eq(len(frame)).all()
    assert result.predictions["probability_drowsy"].between(0.0, 1.0).all()


def test_nested_logistic_selects_inside_each_outer_fold():
    frame = _feature_frame()
    result = nested_logistic_evaluation(
        frame, c_values=(0.1, 1.0), inner_splits=3
    )
    assert len(result.fold_metrics) == 6
    assert set(result.fold_metrics["selected_c"]).issubset({0.1, 1.0})
    assert result.predictions["epoch_id"].nunique() == len(frame)
    assert result.predictions["probability_drowsy"].between(0.0, 1.0).all()
