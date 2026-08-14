"""Configuration and data structures used by the study."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SyntheticConfig:
    n_subjects: int = 10
    epochs_per_class: int = 12
    sampling_rate_hz: float = 128.0
    epoch_duration_s: float = 4.0
    artifact_probability: float = 0.05
    random_seed: int = 2026


@dataclass(frozen=True)
class PreprocessingConfig:
    low_cut_hz: float = 1.0
    high_cut_hz: float = 40.0
    peak_to_peak_threshold_uv: float = 200.0
    apply_filter: bool = True
    reject_peak_to_peak: bool = True


@dataclass(frozen=True)
class EvaluationConfig:
    n_splits: int = 5
    random_seed: int = 2026
    strategy: str = "group_kfold"


@dataclass(frozen=True)
class StudyConfig:
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> StudyConfig:
        with Path(path).open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream) or {}
        return cls(
            synthetic=SyntheticConfig(**payload.get("synthetic", {})),
            preprocessing=PreprocessingConfig(**payload.get("preprocessing", {})),
            evaluation=EvaluationConfig(**payload.get("evaluation", {})),
        )

    def validate(self) -> None:
        s = self.synthetic
        p = self.preprocessing
        e = self.evaluation
        if s.n_subjects < 2:
            raise ValueError("n_subjects must be at least 2")
        if s.epochs_per_class < 1:
            raise ValueError("epochs_per_class must be positive")
        if s.sampling_rate_hz <= 2 * p.high_cut_hz:
            raise ValueError("sampling rate must exceed twice the high-cut frequency")
        if s.epoch_duration_s <= 0:
            raise ValueError("epoch duration must be positive")
        if not 0 <= s.artifact_probability < 1:
            raise ValueError("artifact_probability must be in [0, 1)")
        if not 0 < p.low_cut_hz < p.high_cut_hz:
            raise ValueError("invalid band-pass frequencies")
        if p.peak_to_peak_threshold_uv <= 0:
            raise ValueError("artifact threshold must be positive")
        if not 2 <= e.n_splits <= s.n_subjects:
            raise ValueError("n_splits must be between 2 and n_subjects")
        if e.strategy not in {"group_kfold", "leave_one_subject_out"}:
            raise ValueError("unsupported evaluation strategy")
