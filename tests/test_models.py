from pathlib import Path

import pytest

from eeg_fatigue.models import StudyConfig


def test_default_configuration_is_valid():
    StudyConfig().validate()


def test_yaml_configuration_loads():
    config = StudyConfig.from_yaml(Path("configs/default.yaml"))
    assert config.synthetic.n_subjects == 10
    assert config.evaluation.n_splits == 5


def test_invalid_sampling_rate_is_rejected():
    config = StudyConfig.from_yaml(Path("configs/default.yaml"))
    object.__setattr__(config.synthetic, "sampling_rate_hz", 64.0)
    with pytest.raises(ValueError, match="sampling rate"):
        config.validate()
