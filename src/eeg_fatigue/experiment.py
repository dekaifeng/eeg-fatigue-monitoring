"""End-to-end experiment orchestration and publication-quality outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from eeg_fatigue.evaluation import EvaluationResult, evaluate_models
from eeg_fatigue.features import extract_features
from eeg_fatigue.models import StudyConfig
from eeg_fatigue.preprocessing import preprocess_dataset
from eeg_fatigue.synthetic import EpochDataset, generate_synthetic_dataset


def _plot_feature_separation(features, output: Path, class_names: tuple[str, str]) -> None:
    columns = ["theta_relative", "beta_relative", "theta_beta_ratio", "spectral_entropy"]
    fig, axes = plt.subplots(1, len(columns), figsize=(12, 3.4))
    for axis, column in zip(axes, columns, strict=True):
        values = [features.loc[features["label"] == label, column] for label in (0, 1)]
        axis.boxplot(values, tick_labels=class_names, showfliers=False)
        axis.set_title(column.replace("_", " "))
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("EEG feature separation")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_performance(result: EvaluationResult, output: Path) -> None:
    metrics = ("balanced_accuracy", "f1", "roc_auc")
    models = list(result.summary)
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    for index, model in enumerate(models):
        means = [result.summary[model][f"{metric}_mean"] for metric in metrics]
        stds = [result.summary[model][f"{metric}_std"] for metric in metrics]
        offset = (index - (len(models) - 1) / 2) * width
        axis.bar(x + offset, means, width, yerr=stds, capsize=3, label=model)
    axis.set_xticks(x, [name.replace("_", " ") for name in metrics])
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Grouped cross-validation score")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_subject_performance(result: EvaluationResult, output: Path) -> None:
    rows = result.fold_metrics[result.fold_metrics["model"] != "dummy_prior"].copy()
    is_single_subject = not rows["held_out_subjects"].str.contains(",", regex=False).any()
    x_column = "held_out_subjects" if is_single_subject else "fold"
    if is_single_subject:
        rows[x_column] = rows[x_column].astype(int)
    fig, axis = plt.subplots(figsize=(8.0, 4.2))
    for model, group in rows.groupby("model", sort=True):
        group = group.sort_values(x_column)
        axis.plot(
            group[x_column],
            group["balanced_accuracy"],
            marker="o",
            label=model,
        )
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance baseline")
    axis.set_xticks(sorted(rows[x_column].unique()))
    axis.set_ylim(0.35, 1.0)
    axis.set_xlabel("Held-out subject" if is_single_subject else "Cross-validation fold")
    axis.set_ylabel("Balanced accuracy")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_confusion(
    result: EvaluationResult, output: Path, class_names: tuple[str, str]
) -> None:
    rows = result.predictions[result.predictions["model"] == "logistic_regression"]
    matrix = confusion_matrix(rows["label"], rows["prediction"], labels=[0, 1])
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    display.plot(cmap="Blues", colorbar=False)
    display.ax_.set_title("Out-of-subject predictions")
    display.figure_.tight_layout()
    display.figure_.savefig(output, dpi=180)
    plt.close(display.figure_)


def run_dataset_study(
    raw: EpochDataset,
    config: StudyConfig,
    output_dir: str | Path,
    data_source: str,
    class_names: tuple[str, str] = ("Alert", "Drowsy"),
) -> dict[str, object]:
    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    cleaned, _, preprocessing = preprocess_dataset(raw, config.preprocessing)
    features = extract_features(cleaned)
    evaluation = evaluate_models(features, config.evaluation)

    features.to_csv(output / "features.csv", index=False, lineterminator="\n")
    evaluation.fold_metrics.to_csv(output / "fold_metrics.csv", index=False, lineterminator="\n")
    evaluation.predictions.to_csv(output / "predictions.csv", index=False, lineterminator="\n")
    _plot_feature_separation(features, output / "feature_separation.png", class_names)
    _plot_performance(evaluation, output / "cross_subject_performance.png")
    _plot_subject_performance(evaluation, output / "subject_performance.png")
    _plot_confusion(evaluation, output / "confusion_matrix.png", class_names)

    primary_model = "logistic_regression"
    primary = evaluation.summary[primary_model]
    candidates = {
        name: values
        for name, values in evaluation.summary.items()
        if name != "dummy_prior"
    }
    best_observed_model = max(
        candidates, key=lambda name: candidates[name]["balanced_accuracy_mean"]
    )
    summary: dict[str, object] = {
        "data_source": data_source,
        "n_subjects": int(np.unique(raw.subjects).size),
        "epochs_generated": preprocessing.total_epochs,
        "epochs_retained": preprocessing.retained_epochs,
        "epochs_rejected": preprocessing.rejected_epochs,
        "rejection_rate": preprocessing.rejection_rate,
        "evaluation": config.evaluation.strategy,
        "n_splits": int(evaluation.fold_metrics["fold"].nunique()),
        "class_counts_generated": {
            str(label): int(np.sum(raw.labels == label)) for label in (0, 1)
        },
        "class_counts_retained": {
            str(label): int(np.sum(cleaned.labels == label)) for label in (0, 1)
        },
        "models": evaluation.summary,
        "primary_model": primary_model,
        "primary_balanced_accuracy": primary["balanced_accuracy_mean"],
        "best_observed_model": best_observed_model,
    }
    with (output / "summary.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return summary


def run_study(config: StudyConfig, output_dir: str | Path) -> dict[str, object]:
    raw = generate_synthetic_dataset(config.synthetic)
    return run_dataset_study(
        raw,
        config,
        output_dir,
        data_source="deterministic synthetic EEG",
        class_names=("Alert", "Fatigued"),
    )
