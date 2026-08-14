import numpy as np
import pytest

from eeg_fatigue.statistics import (
    bootstrap_mean_ci,
    exact_sign_flip_pvalue,
    expected_calibration_error,
)


def test_bootstrap_mean_interval_is_deterministic_and_contains_mean():
    first = bootstrap_mean_ci([0.55, 0.65, 0.75, 0.85], n_resamples=1000)
    second = bootstrap_mean_ci([0.55, 0.65, 0.75, 0.85], n_resamples=1000)
    assert first == second
    assert first[1] <= first[0] <= first[2]


def test_exact_sign_flip_pvalue_for_uniform_improvement():
    assert exact_sign_flip_pvalue([0.7, 0.7, 0.7, 0.7]) == pytest.approx(0.125)


def test_expected_calibration_error_for_confident_correct_predictions():
    value = expected_calibration_error([0, 0, 1, 1], [0.1, 0.1, 0.9, 0.9], n_bins=5)
    assert value == pytest.approx(0.1)
    with pytest.raises(ValueError):
        expected_calibration_error([0], [0.1, 0.2])


def test_bootstrap_rejects_non_finite_input():
    with pytest.raises(ValueError):
        bootstrap_mean_ci([0.5, np.nan])
