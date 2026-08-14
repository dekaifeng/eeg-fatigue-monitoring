"""Nested subject-grouped tuning and cross-fitted probability calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_fatigue.features import feature_columns
from eeg_fatigue.statistics import expected_calibration_error


@dataclass(frozen=True)
class ValidationResult:
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame


def _random_forest(random_seed: int, n_estimators: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_seed,
    )


def _probability_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "balanced_accuracy": balanced_accuracy_score(labels, prediction),
        "f1": f1_score(labels, prediction, zero_division=0),
        "roc_auc": roc_auc_score(labels, probability),
        "brier": brier_score_loss(labels, probability),
        "log_loss": log_loss(labels, probability, labels=[0, 1]),
        "ece": expected_calibration_error(labels, probability),
    }


def cross_fitted_random_forest_calibration(
    frame: pd.DataFrame,
    random_seed: int = 2026,
    inner_splits: int = 5,
    n_estimators: int = 200,
) -> ValidationResult:
    columns = feature_columns(frame)
    x = frame[columns].to_numpy()
    y = frame["label"].to_numpy(dtype=int)
    groups = frame["subject"].to_numpy(dtype=int)
    rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[dict[str, float | int | str]] = []

    for fold, (outer_train, outer_test) in enumerate(
        LeaveOneGroupOut().split(x, y, groups), start=1
    ):
        train_x, train_y, train_groups = x[outer_train], y[outer_train], groups[outer_train]
        n_inner = min(inner_splits, len(np.unique(train_groups)))
        oof_probability = np.full(len(outer_train), np.nan)
        for inner_train, inner_validation in GroupKFold(n_splits=n_inner).split(
            train_x, train_y, train_groups
        ):
            estimator = _random_forest(random_seed + fold, n_estimators)
            estimator.fit(train_x[inner_train], train_y[inner_train])
            oof_probability[inner_validation] = estimator.predict_proba(
                train_x[inner_validation]
            )[:, 1]
        if not np.isfinite(oof_probability).all():
            raise RuntimeError("inner grouped calibration did not cover every training sample")

        epsilon = 1e-6
        oof_logit = np.log(
            np.clip(oof_probability, epsilon, 1.0 - epsilon)
            / np.clip(1.0 - oof_probability, epsilon, 1.0 - epsilon)
        )
        calibrator = LogisticRegression(C=1.0, max_iter=2000, random_state=random_seed)
        calibrator.fit(oof_logit.reshape(-1, 1), train_y)

        estimator = _random_forest(random_seed + fold, n_estimators)
        estimator.fit(train_x, train_y)
        raw_probability = estimator.predict_proba(x[outer_test])[:, 1]
        raw_logit = np.log(
            np.clip(raw_probability, epsilon, 1.0 - epsilon)
            / np.clip(1.0 - raw_probability, epsilon, 1.0 - epsilon)
        )
        calibrated_probability = calibrator.predict_proba(raw_logit.reshape(-1, 1))[:, 1]
        held_out_subject = int(np.unique(groups[outer_test]).item())

        for mode, probability in (
            ("raw", raw_probability),
            ("platt_oof", calibrated_probability),
        ):
            metrics = _probability_metrics(y[outer_test], probability)
            rows.append(
                {
                    "mode": mode,
                    "fold": fold,
                    "held_out_subject": held_out_subject,
                    **metrics,
                }
            )
            for local_index, row_index in enumerate(outer_test):
                prediction_rows.append(
                    {
                        "mode": mode,
                        "fold": fold,
                        "epoch_id": int(frame.iloc[row_index]["epoch_id"]),
                        "subject": int(groups[row_index]),
                        "label": int(y[row_index]),
                        "probability_drowsy": float(probability[local_index]),
                    }
                )
    return ValidationResult(pd.DataFrame(rows), pd.DataFrame(prediction_rows))


def nested_logistic_evaluation(
    frame: pd.DataFrame,
    c_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
    random_seed: int = 2026,
    inner_splits: int = 5,
) -> ValidationResult:
    columns = feature_columns(frame)
    x = frame[columns].to_numpy()
    y = frame["label"].to_numpy(dtype=int)
    groups = frame["subject"].to_numpy(dtype=int)
    rows: list[dict[str, float | int]] = []
    prediction_rows: list[dict[str, float | int]] = []

    for fold, (outer_train, outer_test) in enumerate(
        LeaveOneGroupOut().split(x, y, groups), start=1
    ):
        n_inner = min(inner_splits, len(np.unique(groups[outer_train])))
        candidate_scores: dict[float, float] = {}
        for c_value in c_values:
            scores: list[float] = []
            for inner_train_local, inner_validation_local in GroupKFold(
                n_splits=n_inner
            ).split(x[outer_train], y[outer_train], groups[outer_train]):
                inner_train = outer_train[inner_train_local]
                inner_validation = outer_train[inner_validation_local]
                estimator = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=c_value, max_iter=2000, random_state=random_seed
                    ),
                )
                estimator.fit(x[inner_train], y[inner_train])
                prediction = estimator.predict(x[inner_validation])
                scores.append(balanced_accuracy_score(y[inner_validation], prediction))
            candidate_scores[c_value] = float(np.mean(scores))
        selected_c = min(
            c_values, key=lambda value: (-candidate_scores[value], value)
        )
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=selected_c, max_iter=2000, random_state=random_seed),
        )
        estimator.fit(x[outer_train], y[outer_train])
        probability = estimator.predict_proba(x[outer_test])[:, 1]
        metrics = _probability_metrics(y[outer_test], probability)
        held_out_subject = int(np.unique(groups[outer_test]).item())
        rows.append(
            {
                "fold": fold,
                "held_out_subject": held_out_subject,
                "selected_c": selected_c,
                "inner_balanced_accuracy": candidate_scores[selected_c],
                **metrics,
            }
        )
        for local_index, row_index in enumerate(outer_test):
            prediction_rows.append(
                {
                    "fold": fold,
                    "epoch_id": int(frame.iloc[row_index]["epoch_id"]),
                    "subject": int(groups[row_index]),
                    "label": int(y[row_index]),
                    "probability_drowsy": float(probability[local_index]),
                }
            )
    return ValidationResult(pd.DataFrame(rows), pd.DataFrame(prediction_rows))
