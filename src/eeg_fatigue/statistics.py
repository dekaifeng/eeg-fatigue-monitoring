"""Subject-level uncertainty and chance-comparison utilities."""

from __future__ import annotations

from itertools import product

import numpy as np
from numpy.typing import ArrayLike


def bootstrap_mean_ci(
    values: ArrayLike,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    random_seed: int = 2026,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be a finite one-dimensional array")
    if not 0 < confidence < 1 or n_resamples < 100:
        raise ValueError("invalid bootstrap configuration")
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, len(array), size=(n_resamples, len(array)))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return float(array.mean()), float(lower), float(upper)


def exact_sign_flip_pvalue(values: ArrayLike, reference: float = 0.5) -> float:
    """Two-sided exact sign-flip test of a subject-level mean against a reference."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2 or len(array) > 20:
        raise ValueError("sign-flip input must contain between 2 and 20 values")
    differences = array - reference
    observed = abs(float(differences.mean()))
    permutations = np.asarray(list(product((-1.0, 1.0), repeat=len(array))))
    permuted = np.abs((permutations * differences).mean(axis=1))
    return float(np.mean(permuted >= observed - 1e-15))


def expected_calibration_error(
    labels: ArrayLike, probabilities: ArrayLike, n_bins: int = 10
) -> float:
    y = np.asarray(labels, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    if y.shape != probability.shape or y.ndim != 1:
        raise ValueError("labels and probabilities must be aligned vectors")
    if n_bins < 2 or not np.isfinite(probability).all():
        raise ValueError("invalid calibration input")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.minimum(np.digitize(probability, edges[1:-1]), n_bins - 1)
    error = 0.0
    for index in range(n_bins):
        mask = bins == index
        if np.any(mask):
            error += mask.mean() * abs(y[mask].mean() - probability[mask].mean())
    return float(error)
