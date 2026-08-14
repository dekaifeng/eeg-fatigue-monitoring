"""Deterministic alarm hysteresis and accelerated epoch replay."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class MonitorState(str, Enum):
    NORMAL = "NORMAL"
    DROWSY = "DROWSY"
    DATA_FAULT = "DATA_FAULT"


@dataclass(frozen=True)
class HysteresisConfig:
    enter_probability: float = 0.65
    exit_probability: float = 0.35
    enter_windows: int = 3
    exit_windows: int = 2

    def validate(self) -> None:
        if not 0.0 < self.exit_probability < self.enter_probability < 1.0:
            raise ValueError("hysteresis probabilities must satisfy 0 < exit < enter < 1")
        if self.enter_windows < 1 or self.exit_windows < 1:
            raise ValueError("hysteresis window counts must be positive")


class DrowsinessMonitor:
    def __init__(self, config: HysteresisConfig):
        config.validate()
        self.config = config
        self.state = MonitorState.NORMAL
        self._above_count = 0
        self._below_count = 0

    def reset(self) -> None:
        self.state = MonitorState.NORMAL
        self._above_count = 0
        self._below_count = 0

    def update(self, probability: float | None, valid: bool = True) -> MonitorState:
        if not valid or probability is None or not np.isfinite(probability):
            self.state = MonitorState.DATA_FAULT
            self._above_count = 0
            self._below_count = 0
            return self.state
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        if self.state == MonitorState.DATA_FAULT:
            self.state = MonitorState.NORMAL

        if self.state == MonitorState.NORMAL:
            self._above_count = (
                self._above_count + 1
                if probability >= self.config.enter_probability
                else 0
            )
            self._below_count = 0
            if self._above_count >= self.config.enter_windows:
                self.state = MonitorState.DROWSY
                self._above_count = 0
        elif self.state == MonitorState.DROWSY:
            self._below_count = (
                self._below_count + 1
                if probability <= self.config.exit_probability
                else 0
            )
            self._above_count = 0
            if self._below_count >= self.config.exit_windows:
                self.state = MonitorState.NORMAL
                self._below_count = 0
        return self.state


def replay_predictions(
    predictions: pd.DataFrame, config: HysteresisConfig
) -> pd.DataFrame:
    required = {"epoch_id", "subject", "label", "probability_fatigued"}
    if not required.issubset(predictions):
        raise ValueError(f"predictions are missing columns: {sorted(required - set(predictions))}")
    monitor = DrowsinessMonitor(config)
    rows: list[dict[str, object]] = []
    for subject, subject_rows in predictions.groupby("subject", sort=True):
        monitor.reset()
        previous_state = monitor.state
        for sequence, (_, row) in enumerate(
            subject_rows.sort_values("epoch_id").iterrows(), start=1
        ):
            state = monitor.update(float(row["probability_fatigued"]), valid=True)
            rows.append(
                {
                    "epoch_id": int(row["epoch_id"]),
                    "subject": int(subject),
                    "sequence": sequence,
                    "label": int(row["label"]),
                    "probability_drowsy": float(row["probability_fatigued"]),
                    "monitor_state": state.value,
                    "alarm_active": state == MonitorState.DROWSY,
                    "transition": state != previous_state,
                }
            )
            previous_state = state
    return pd.DataFrame(rows)
