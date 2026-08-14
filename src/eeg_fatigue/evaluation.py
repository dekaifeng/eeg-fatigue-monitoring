"""Leakage-resistant subject-grouped model evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_fatigue.features import feature_columns
from eeg_fatigue.models import EvaluationConfig


@dataclass(frozen=True)
class EvaluationResult:
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    summary: dict[str, dict[str, float]]


def evaluate_models(frame: pd.DataFrame, config: EvaluationConfig) -> EvaluationResult:
    columns = feature_columns(frame)
    x = frame[columns].to_numpy()
    y = frame["label"].to_numpy(dtype=int)
    groups = frame["subject"].to_numpy(dtype=int)
    models = {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=config.random_seed),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.random_seed,
        ),
    }
    unique_groups = np.unique(groups)
    if config.strategy == "leave_one_subject_out":
        splitter = LeaveOneGroupOut()
    else:
        if config.n_splits > len(unique_groups):
            raise ValueError("n_splits exceeds the number of subjects in the feature table")
        splitter = GroupKFold(n_splits=config.n_splits)
    metric_rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[dict[str, float | int | str]] = []

    for model_name, estimator in models.items():
        for fold, (train, test) in enumerate(splitter.split(x, y, groups), start=1):
            train_groups = set(groups[train])
            test_groups = set(groups[test])
            if train_groups & test_groups:
                raise RuntimeError("subject leakage detected")
            fitted = clone(estimator).fit(x[train], y[train])
            predicted = fitted.predict(x[test])
            probability = fitted.predict_proba(x[test])[:, 1]
            metrics = {
                "model": model_name,
                "fold": fold,
                "n_train": len(train),
                "n_test": len(test),
                "held_out_subjects": ",".join(str(value) for value in sorted(test_groups)),
                "balanced_accuracy": balanced_accuracy_score(y[test], predicted),
                "f1": f1_score(y[test], predicted, zero_division=0),
                "roc_auc": roc_auc_score(y[test], probability),
            }
            metric_rows.append(metrics)
            for index, row_index in enumerate(test):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "epoch_id": int(frame.iloc[row_index]["epoch_id"]),
                        "subject": int(groups[row_index]),
                        "label": int(y[row_index]),
                        "prediction": int(predicted[index]),
                        "probability_fatigued": float(probability[index]),
                    }
                )

    fold_metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary: dict[str, dict[str, float]] = {}
    for model_name, group in fold_metrics.groupby("model", sort=True):
        summary[str(model_name)] = {}
        for metric in ("balanced_accuracy", "f1", "roc_auc"):
            summary[str(model_name)][f"{metric}_mean"] = float(group[metric].mean())
            summary[str(model_name)][f"{metric}_std"] = float(group[metric].std(ddof=0))
    return EvaluationResult(fold_metrics, predictions, summary)
