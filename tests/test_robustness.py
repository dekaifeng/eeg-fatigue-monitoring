from eeg_fatigue.robustness import RobustnessConfig


def test_robustness_configuration_loads_and_validates():
    config = RobustnessConfig.from_yaml("configs/robustness.yaml")
    config.validate()
    assert config.artifact_thresholds_uv[0] is None
    assert config.logistic_c_values == (0.01, 0.1, 1.0, 10.0, 100.0)
