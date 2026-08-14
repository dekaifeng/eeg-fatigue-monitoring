"""Phase 4 accelerated replay, dropout robustness, latency, and demo outputs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

from eeg_fatigue.datasets import FIGSHARE_DOI, load_figshare_dataset
from eeg_fatigue.dropout import inject_channel_dropout
from eeg_fatigue.evaluation import EvaluationResult, evaluate_models
from eeg_fatigue.features import extract_features, feature_columns
from eeg_fatigue.models import StudyConfig
from eeg_fatigue.preprocessing import preprocess_dataset
from eeg_fatigue.streaming import (
    DrowsinessMonitor,
    HysteresisConfig,
    MonitorState,
    replay_predictions,
)
from eeg_fatigue.synthetic import EpochDataset


@dataclass(frozen=True)
class StreamingConfig:
    enter_probability: float
    exit_probability: float
    enter_windows: int
    exit_windows: int
    max_missing_channels: int
    dropout_scenarios: tuple[int, ...]
    benchmark_subject: int
    benchmark_windows: int
    gif_frames: int
    random_seed: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> StreamingConfig:
        with Path(path).open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
        return cls(
            enter_probability=float(payload["enter_probability"]),
            exit_probability=float(payload["exit_probability"]),
            enter_windows=int(payload["enter_windows"]),
            exit_windows=int(payload["exit_windows"]),
            max_missing_channels=int(payload["max_missing_channels"]),
            dropout_scenarios=tuple(int(value) for value in payload["dropout_scenarios"]),
            benchmark_subject=int(payload["benchmark_subject"]),
            benchmark_windows=int(payload["benchmark_windows"]),
            gif_frames=int(payload["gif_frames"]),
            random_seed=int(payload["random_seed"]),
        )

    @property
    def hysteresis(self) -> HysteresisConfig:
        return HysteresisConfig(
            enter_probability=self.enter_probability,
            exit_probability=self.exit_probability,
            enter_windows=self.enter_windows,
            exit_windows=self.exit_windows,
        )

    def validate(self, n_channels: int = 30) -> None:
        self.hysteresis.validate()
        if not 0 <= self.max_missing_channels < n_channels:
            raise ValueError("invalid missing-channel limit")
        if not self.dropout_scenarios or any(
            value < 0 or value > n_channels for value in self.dropout_scenarios
        ):
            raise ValueError("invalid dropout scenarios")
        if self.benchmark_subject < 1 or self.benchmark_windows < 1 or self.gif_frames < 2:
            raise ValueError("invalid replay configuration")


def _random_forest(random_seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_seed,
    )


def _single_epoch(dataset: EpochDataset, index: int) -> EpochDataset:
    return EpochDataset(
        data_v=dataset.data_v[index : index + 1],
        labels=dataset.labels[index : index + 1],
        subjects=dataset.subjects[index : index + 1],
        epoch_ids=dataset.epoch_ids[index : index + 1],
        artifact_truth=dataset.artifact_truth[index : index + 1],
        sampling_rate_hz=dataset.sampling_rate_hz,
        channel_names=dataset.channel_names,
    )


def _benchmark_latency(
    dataset: EpochDataset,
    features: pd.DataFrame,
    config: StreamingConfig,
) -> pd.DataFrame:
    columns = feature_columns(features)
    train = features["subject"] != config.benchmark_subject
    estimator = _random_forest(config.random_seed)
    estimator.fit(features.loc[train, columns], features.loc[train, "label"])
    indices = np.flatnonzero(dataset.subjects == config.benchmark_subject)
    indices = indices[: config.benchmark_windows]
    if len(indices) == 0:
        raise ValueError("benchmark subject is absent")

    for index in indices[: min(5, len(indices))]:
        epoch_features = extract_features(_single_epoch(dataset, int(index)))
        estimator.predict_proba(epoch_features[columns])

    rows: list[dict[str, float | int]] = []
    for sequence, index in enumerate(indices, start=1):
        start = time.perf_counter_ns()
        epoch_features = extract_features(_single_epoch(dataset, int(index)))
        probability = estimator.predict_proba(epoch_features[columns])[0, 1]
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        rows.append(
            {
                "sequence": sequence,
                "epoch_id": int(dataset.epoch_ids[index]),
                "latency_ms": elapsed_ms,
                "probability_drowsy": float(probability),
            }
        )
    return pd.DataFrame(rows)


def _plot_timeline(
    events: pd.DataFrame,
    subject: int,
    config: StreamingConfig,
    output: Path,
) -> None:
    rows = events[events["subject"] == subject].sort_values("sequence")
    if rows.empty:
        raise ValueError("demo subject has no replay events")
    fig, axis = plt.subplots(figsize=(10.0, 4.5))
    axis.plot(rows["sequence"], rows["probability_drowsy"], label="RF probability", linewidth=1.5)
    axis.axhline(config.enter_probability, color="tab:red", linestyle="--", label="enter threshold")
    axis.axhline(config.exit_probability, color="tab:green", linestyle="--", label="exit threshold")
    drowsy = rows["monitor_state"] == MonitorState.DROWSY.value
    axis.fill_between(
        rows["sequence"], 0.0, 1.0, where=drowsy, color="tab:red", alpha=0.12, label="alarm active"
    )
    axis.scatter(
        rows["sequence"],
        np.where(rows["label"] == 1, 0.03, 0.0),
        c=rows["label"],
        cmap="coolwarm",
        s=10,
        label="published label",
    )
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel("Published epoch order (not continuous time)")
    axis.set_ylabel("Drowsy probability")
    axis.set_title(f"Accelerated epoch replay — subject {subject}")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_latency(latency: pd.DataFrame, output: Path) -> None:
    p95 = float(np.percentile(latency["latency_ms"], 95))
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.hist(latency["latency_ms"], bins=24, color="tab:blue", alpha=0.8)
    axis.axvline(p95, color="tab:red", linestyle="--", label=f"p95 = {p95:.2f} ms")
    axis.set_xlabel("Feature extraction + inference latency (ms)")
    axis.set_ylabel("Epoch count")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_dropout(frame: pd.DataFrame, output: Path) -> None:
    valid = frame[frame["status"] == "evaluated"]
    fault = frame[frame["status"] == "data_fault"]
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(
        valid["missing_channels"],
        valid["balanced_accuracy"],
        marker="o",
        label="RF balanced accuracy",
    )
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    if not fault.empty:
        boundary = float(valid["missing_channels"].max()) + 0.5
        upper = float(frame["missing_channels"].max()) + 0.5
        axis.axvspan(
            boundary,
            upper,
            color="tab:red",
            alpha=0.08,
            label="DATA_FAULT region (>8 missing)",
        )
        axis.axvline(boundary, color="tab:red", linestyle=":")
    axis.set_xticks(frame["missing_channels"])
    axis.set_ylim(0.45, 0.8)
    axis.set_xlabel("Random missing channels per epoch")
    axis.set_ylabel("LOSO balanced accuracy")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _create_demo_gif(
    events: pd.DataFrame, subject: int, config: StreamingConfig, output: Path
) -> None:
    rows = events[events["subject"] == subject].sort_values("sequence").reset_index(drop=True)
    frame_ends = np.unique(
        np.linspace(1, len(rows), min(config.gif_frames, len(rows))).round().astype(int)
    )
    fig, axis = plt.subplots(figsize=(8.0, 4.2))

    def draw(frame_index: int):
        end = int(frame_ends[frame_index])
        visible = rows.iloc[:end]
        current = visible.iloc[-1]
        axis.clear()
        axis.plot(visible["sequence"], visible["probability_drowsy"], color="tab:blue")
        axis.axhline(config.enter_probability, color="tab:red", linestyle="--")
        axis.axhline(config.exit_probability, color="tab:green", linestyle="--")
        axis.scatter(
            [current["sequence"]],
            [current["probability_drowsy"]],
            color="tab:orange",
            s=45,
        )
        axis.set_xlim(1, len(rows))
        axis.set_ylim(0, 1)
        axis.set_xlabel("Published epoch order")
        axis.set_ylabel("Drowsy probability")
        axis.set_title(
            f"Subject {subject} accelerated replay | state={current['monitor_state']}"
        )
        axis.grid(alpha=0.2)
        return axis.lines

    movie = animation.FuncAnimation(fig, draw, frames=len(frame_ends), interval=125, blit=False)
    movie.save(output, writer=animation.PillowWriter(fps=8), dpi=110)
    plt.close(fig)


def _fault_demo(config: StreamingConfig) -> list[str]:
    monitor = DrowsinessMonitor(config.hysteresis)
    states = [monitor.update(0.8) for _ in range(config.enter_windows)]
    states.extend(monitor.update(None, valid=False) for _ in range(2))
    states.extend(monitor.update(0.2) for _ in range(config.exit_windows))
    return [state.value for state in states]


def run_streaming_study(
    dataset_path: str | Path,
    study_config: StudyConfig,
    streaming_config: StreamingConfig,
    output_dir: str | Path,
) -> dict[str, object]:
    study_config.validate()
    streaming_config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    raw = load_figshare_dataset(dataset_path)
    clean, _, _ = preprocess_dataset(raw, study_config.preprocessing)
    clean_features = extract_features(clean)
    baseline: EvaluationResult = evaluate_models(clean_features, study_config.evaluation)
    rf_predictions = baseline.predictions[
        baseline.predictions["model"] == "random_forest"
    ].copy()
    replay = replay_predictions(rf_predictions, streaming_config.hysteresis)

    baseline_probability = rf_predictions.set_index("epoch_id")["probability_fatigued"]
    dropout_rows: list[dict[str, object]] = []
    for n_missing in streaming_config.dropout_scenarios:
        result = inject_channel_dropout(
            clean,
            n_missing=n_missing,
            max_missing_channels=streaming_config.max_missing_channels,
            random_seed=streaming_config.random_seed,
        )
        if not result.valid or result.dataset is None:
            dropout_rows.append(
                {
                    "missing_channels": n_missing,
                    "status": "data_fault",
                    "imputation": "not attempted",
                    "balanced_accuracy": None,
                    "roc_auc": None,
                    "mean_probability_shift": None,
                }
            )
            continue
        features = clean_features if n_missing == 0 else extract_features(result.dataset)
        evaluation = (
            baseline
            if n_missing == 0
            else evaluate_models(features, study_config.evaluation)
        )
        metrics = evaluation.summary["random_forest"]
        probabilities = evaluation.predictions[
            evaluation.predictions["model"] == "random_forest"
        ].set_index("epoch_id")["probability_fatigued"]
        aligned = probabilities.reindex(baseline_probability.index)
        dropout_rows.append(
            {
                "missing_channels": n_missing,
                "status": "evaluated",
                "imputation": result.reason,
                "balanced_accuracy": metrics["balanced_accuracy_mean"],
                "roc_auc": metrics["roc_auc_mean"],
                "mean_probability_shift": float(
                    np.mean(np.abs(aligned.to_numpy() - baseline_probability.to_numpy()))
                ),
            }
        )
    dropout_frame = pd.DataFrame(dropout_rows)
    latency = _benchmark_latency(clean, clean_features, streaming_config)

    replay.to_csv(output / "replay_events.csv", index=False, lineterminator="\n")
    dropout_frame.to_csv(output / "dropout_robustness.csv", index=False, lineterminator="\n")
    latency.to_csv(output / "latency_samples.csv", index=False, lineterminator="\n")
    _plot_timeline(
        replay,
        streaming_config.benchmark_subject,
        streaming_config,
        output / "replay_timeline.png",
    )
    _plot_latency(latency, output / "latency_distribution.png")
    _plot_dropout(dropout_frame, output / "missing_channel_robustness.png")
    _create_demo_gif(
        replay,
        streaming_config.benchmark_subject,
        streaming_config,
        output / "epoch_replay_demo.gif",
    )

    latency_values = latency["latency_ms"].to_numpy()
    summary: dict[str, object] = {
        "data_source": f"Figshare DOI {FIGSHARE_DOI}",
        "replay_mode": "accelerated replay of independently published three-second epochs",
        "continuous_time_claim": False,
        "subjects": int(np.unique(clean.subjects).size),
        "epochs": len(clean.labels),
        "baseline_random_forest_balanced_accuracy": baseline.summary["random_forest"][
            "balanced_accuracy_mean"
        ],
        "alarm_transitions": int(replay["transition"].sum()),
        "alarm_active_epochs": int(replay["alarm_active"].sum()),
        "fault_demo_states": _fault_demo(streaming_config),
        "missing_channel_policy": (
            f"median imputation through {streaming_config.max_missing_channels}; "
            "DATA_FAULT above the limit"
        ),
        "latency_scope": (
            "single epoch PSD/features plus RF predict_proba; "
            "excludes acquisition and I/O"
        ),
        "latency_ms": {
            "n": len(latency_values),
            "median": float(np.median(latency_values)),
            "p95": float(np.percentile(latency_values, 95)),
            "p99": float(np.percentile(latency_values, 99)),
            "max": float(np.max(latency_values)),
        },
        "hard_realtime_claim": False,
    }
    with (output / "streaming_summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return summary
