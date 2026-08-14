"""Phase 3 sensitivity, ablation, uncertainty, calibration, and nested tuning."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve

from eeg_fatigue.datasets import FIGSHARE_DOI, load_figshare_dataset
from eeg_fatigue.evaluation import evaluate_models
from eeg_fatigue.features import extract_features, feature_columns
from eeg_fatigue.models import StudyConfig
from eeg_fatigue.preprocessing import preprocess_dataset
from eeg_fatigue.statistics import bootstrap_mean_ci, exact_sign_flip_pvalue
from eeg_fatigue.validation import (
    cross_fitted_random_forest_calibration,
    nested_logistic_evaluation,
)


@dataclass(frozen=True)
class RobustnessConfig:
    artifact_thresholds_uv: tuple[float | None, ...]
    bootstrap_resamples: int
    inner_splits: int
    calibration_trees: int
    logistic_c_values: tuple[float, ...]
    random_seed: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> RobustnessConfig:
        with Path(path).open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
        return cls(
            artifact_thresholds_uv=tuple(payload["artifact_thresholds_uv"]),
            bootstrap_resamples=int(payload["bootstrap_resamples"]),
            inner_splits=int(payload["inner_splits"]),
            calibration_trees=int(payload["calibration_trees"]),
            logistic_c_values=tuple(float(value) for value in payload["logistic_c_values"]),
            random_seed=int(payload["random_seed"]),
        )

    def validate(self) -> None:
        if not self.artifact_thresholds_uv or self.artifact_thresholds_uv[0] is not None:
            raise ValueError("the first artifact condition must retain all finite samples")
        finite_thresholds = [v for v in self.artifact_thresholds_uv if v is not None]
        if not finite_thresholds or any(value <= 0 for value in finite_thresholds):
            raise ValueError("artifact thresholds must be positive")
        if self.bootstrap_resamples < 100 or self.inner_splits < 2:
            raise ValueError("insufficient resampling configuration")
        if self.calibration_trees < 10 or not self.logistic_c_values:
            raise ValueError("invalid model configuration")


def _plot_threshold_sensitivity(frame: pd.DataFrame, output: Path) -> None:
    selected = frame[frame["model"].isin(["logistic_regression", "random_forest"])]
    labels = list(dict.fromkeys(selected["threshold_label"]))
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(7.8, 4.3))
    for model, group in selected.groupby("model", sort=True):
        ordered = group.set_index("threshold_label").loc[labels]
        axis.plot(x, ordered["balanced_accuracy_mean"], marker="o", label=model)
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(x, labels)
    axis.set_xlabel("Peak-to-peak rejection threshold (µV)")
    axis.set_ylabel("LOSO balanced accuracy")
    axis.set_ylim(0.45, 0.8)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_ablation(frame: pd.DataFrame, output: Path) -> None:
    selected = frame[frame["model"].isin(["logistic_regression", "random_forest"])]
    groups = list(dict.fromkeys(selected["feature_group"]))
    models = sorted(selected["model"].unique())
    x = np.arange(len(groups))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.6, 4.5))
    for index, model in enumerate(models):
        ordered = selected[selected["model"] == model].set_index("feature_group").loc[groups]
        axis.bar(
            x + (index - 0.5) * width,
            ordered["balanced_accuracy_mean"],
            width,
            label=model,
        )
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(x, [value.replace("_", "\n") for value in groups])
    axis.set_ylim(0.45, 0.8)
    axis.set_ylabel("LOSO balanced accuracy")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_calibration(predictions: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.5, 5.0))
    for mode, rows in predictions.groupby("mode", sort=True):
        observed, predicted = calibration_curve(
            rows["label"], rows["probability_drowsy"], n_bins=8, strategy="quantile"
        )
        axis.plot(predicted, observed, marker="o", label=mode)
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", label="ideal")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed drowsy fraction")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_nested_comparison(
    baseline_metrics: pd.DataFrame, nested_metrics: pd.DataFrame, output: Path
) -> None:
    baseline = baseline_metrics[baseline_metrics["model"] == "logistic_regression"].copy()
    baseline["held_out_subject"] = baseline["held_out_subjects"].astype(int)
    merged = baseline.merge(nested_metrics, on="held_out_subject", suffixes=("_fixed", "_nested"))
    fig, axis = plt.subplots(figsize=(8.0, 4.3))
    axis.plot(
        merged["held_out_subject"],
        merged["balanced_accuracy_fixed"],
        marker="o",
        label="fixed C=1",
    )
    axis.plot(
        merged["held_out_subject"],
        merged["balanced_accuracy_nested"],
        marker="o",
        label="nested selected C",
    )
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(sorted(merged["held_out_subject"]))
    axis.set_xlabel("Held-out subject")
    axis.set_ylabel("Balanced accuracy")
    axis.set_ylim(0.35, 0.9)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run_robustness_study(
    dataset_path: str | Path,
    study_config: StudyConfig,
    robustness_config: RobustnessConfig,
    output_dir: str | Path,
) -> dict[str, object]:
    study_config.validate()
    robustness_config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_figshare_dataset(dataset_path)

    threshold_rows: list[dict[str, object]] = []
    baseline_features: pd.DataFrame | None = None
    baseline_evaluation = None
    for threshold in robustness_config.artifact_thresholds_uv:
        preprocessing_config = replace(
            study_config.preprocessing,
            apply_filter=False,
            reject_peak_to_peak=threshold is not None,
            peak_to_peak_threshold_uv=(
                study_config.preprocessing.peak_to_peak_threshold_uv
                if threshold is None
                else threshold
            ),
        )
        cleaned, _, report = preprocess_dataset(dataset, preprocessing_config)
        features = extract_features(cleaned)
        evaluation = evaluate_models(features, study_config.evaluation)
        threshold_label = "none" if threshold is None else f"{threshold:g}"
        for model, metrics in evaluation.summary.items():
            threshold_rows.append(
                {
                    "threshold_label": threshold_label,
                    "threshold_uv": threshold,
                    "model": model,
                    "epochs_retained": report.retained_epochs,
                    "alert_retained": int(np.sum(cleaned.labels == 0)),
                    "drowsy_retained": int(np.sum(cleaned.labels == 1)),
                    **metrics,
                }
            )
        if threshold is None:
            baseline_features = features
            baseline_evaluation = evaluation
    if baseline_features is None or baseline_evaluation is None:
        raise RuntimeError("missing no-rejection baseline")

    metadata = ["epoch_id", "subject", "label"]
    all_columns = feature_columns(baseline_features)
    ablations = {
        "band_powers": [f"{name}_relative" for name in ("delta", "theta", "alpha", "beta")],
        "fatigue_ratios": ["theta_beta_ratio", "theta_alpha_beta_ratio"],
        "spectral_core": [
            "delta_relative",
            "theta_relative",
            "alpha_relative",
            "beta_relative",
            "theta_beta_ratio",
            "theta_alpha_beta_ratio",
            "spectral_entropy",
        ],
        "all_features": all_columns,
    }
    ablation_rows: list[dict[str, object]] = []
    for group_name, columns in ablations.items():
        evaluation = evaluate_models(
            baseline_features[metadata + columns], study_config.evaluation
        )
        for model, metrics in evaluation.summary.items():
            ablation_rows.append(
                {
                    "feature_group": group_name,
                    "n_features": len(columns),
                    "model": model,
                    **metrics,
                }
            )

    statistics_rows: list[dict[str, float | str]] = []
    for model in ("logistic_regression", "random_forest"):
        values = baseline_evaluation.fold_metrics.loc[
            baseline_evaluation.fold_metrics["model"] == model, "balanced_accuracy"
        ].to_numpy()
        mean, lower, upper = bootstrap_mean_ci(
            values,
            n_resamples=robustness_config.bootstrap_resamples,
            random_seed=robustness_config.random_seed,
        )
        statistics_rows.append(
            {
                "model": model,
                "balanced_accuracy_mean": mean,
                "bootstrap_95_lower": lower,
                "bootstrap_95_upper": upper,
                "exact_sign_flip_p_vs_0p5": exact_sign_flip_pvalue(values),
            }
        )

    calibration = cross_fitted_random_forest_calibration(
        baseline_features,
        random_seed=robustness_config.random_seed,
        inner_splits=robustness_config.inner_splits,
        n_estimators=robustness_config.calibration_trees,
    )
    nested = nested_logistic_evaluation(
        baseline_features,
        c_values=robustness_config.logistic_c_values,
        random_seed=robustness_config.random_seed,
        inner_splits=robustness_config.inner_splits,
    )

    threshold_frame = pd.DataFrame(threshold_rows)
    ablation_frame = pd.DataFrame(ablation_rows)
    statistics_frame = pd.DataFrame(statistics_rows)
    threshold_frame.to_csv(output / "threshold_sensitivity.csv", index=False, lineterminator="\n")
    ablation_frame.to_csv(output / "feature_ablation.csv", index=False, lineterminator="\n")
    statistics_frame.to_csv(output / "subject_statistics.csv", index=False, lineterminator="\n")
    calibration.fold_metrics.to_csv(
        output / "calibration_fold_metrics.csv", index=False, lineterminator="\n"
    )
    calibration.predictions.to_csv(
        output / "calibration_predictions.csv", index=False, lineterminator="\n"
    )
    nested.fold_metrics.to_csv(
        output / "nested_logistic_folds.csv", index=False, lineterminator="\n"
    )
    nested.predictions.to_csv(
        output / "nested_logistic_predictions.csv", index=False, lineterminator="\n"
    )

    _plot_threshold_sensitivity(threshold_frame, output / "threshold_sensitivity.png")
    _plot_ablation(ablation_frame, output / "feature_ablation.png")
    _plot_calibration(calibration.predictions, output / "calibration_curve.png")
    _plot_nested_comparison(
        baseline_evaluation.fold_metrics,
        nested.fold_metrics,
        output / "nested_tuning.png",
    )

    calibration_summary = (
        calibration.fold_metrics.groupby("mode")[["brier", "log_loss", "ece"]]
        .mean()
        .to_dict(orient="index")
    )
    nested_summary = {
        "balanced_accuracy_mean": float(nested.fold_metrics["balanced_accuracy"].mean()),
        "balanced_accuracy_std": float(nested.fold_metrics["balanced_accuracy"].std(ddof=0)),
        "selected_c_counts": {
            str(key): int(value)
            for key, value in nested.fold_metrics["selected_c"].value_counts().sort_index().items()
        },
    }
    summary: dict[str, object] = {
        "data_source": f"Figshare DOI {FIGSHARE_DOI}",
        "subjects": int(np.unique(dataset.subjects).size),
        "epochs": len(dataset.labels),
        "threshold_conditions": len(robustness_config.artifact_thresholds_uv),
        "feature_ablation_conditions": len(ablations),
        "subject_statistics": statistics_rows,
        "calibration": calibration_summary,
        "nested_logistic": nested_summary,
    }
    with (output / "robustness_summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return summary
