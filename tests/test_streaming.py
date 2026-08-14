import pandas as pd

from eeg_fatigue.streaming import (
    DrowsinessMonitor,
    HysteresisConfig,
    MonitorState,
    replay_predictions,
)


def test_hysteresis_requires_consecutive_windows() -> None:
    monitor = DrowsinessMonitor(
        HysteresisConfig(
            enter_probability=0.65,
            exit_probability=0.35,
            enter_windows=3,
            exit_windows=2,
        )
    )

    assert monitor.update(0.8) == MonitorState.NORMAL
    assert monitor.update(0.8) == MonitorState.NORMAL
    assert monitor.update(0.8) == MonitorState.DROWSY
    assert monitor.update(0.2) == MonitorState.DROWSY
    assert monitor.update(0.2) == MonitorState.NORMAL


def test_data_fault_is_explicit_and_valid_data_recovers() -> None:
    monitor = DrowsinessMonitor(HysteresisConfig())

    assert monitor.update(None, valid=False) == MonitorState.DATA_FAULT
    assert monitor.update(float("nan")) == MonitorState.DATA_FAULT
    assert monitor.update(0.2) == MonitorState.NORMAL


def test_replay_sorts_epochs_and_resets_between_subjects() -> None:
    predictions = pd.DataFrame(
        {
            "epoch_id": [3, 1, 2, 4],
            "subject": [1, 1, 1, 2],
            "label": [1, 1, 1, 1],
            "probability_fatigued": [0.9, 0.9, 0.9, 0.9],
        }
    )

    replay = replay_predictions(predictions, HysteresisConfig())

    assert replay["epoch_id"].tolist() == [1, 2, 3, 4]
    assert replay["sequence"].tolist() == [1, 2, 3, 1]
    assert replay["monitor_state"].tolist() == [
        "NORMAL",
        "NORMAL",
        "DROWSY",
        "NORMAL",
    ]
