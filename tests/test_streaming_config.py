from pathlib import Path

import pytest

from eeg_fatigue.streaming_experiment import StreamingConfig


def test_streaming_config_loads_and_validates() -> None:
    path = Path(__file__).parents[1] / "configs" / "streaming.yaml"
    config = StreamingConfig.from_yaml(path)

    config.validate()
    assert config.dropout_scenarios == (0, 1, 4, 8, 16)
    assert config.hysteresis.enter_windows == 3


def test_streaming_config_rejects_invalid_channel_limit() -> None:
    path = Path(__file__).parents[1] / "configs" / "streaming.yaml"
    config = StreamingConfig.from_yaml(path)

    with pytest.raises(ValueError, match="missing-channel limit"):
        config.validate(n_channels=8)
