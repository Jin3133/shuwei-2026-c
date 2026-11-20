from __future__ import annotations

from .gru import GRUForecaster
from .kinematic import ConstantVelocityBaseline, FilterPyKalmanBaseline, LastObservationBaseline
from .quality_gru import QualityAwareGRUForecaster
from .tcn import TCNForecaster


def build_model(config: dict, data_config: dict):
    name = config["name"]
    common = {"horizon": data_config["horizon"], "target_dim": data_config.get("target_dim", 3)}
    if name == "last_observation": return LastObservationBaseline(**common)
    if name == "constant_velocity": return ConstantVelocityBaseline(common["horizon"], config["sample_period_s"], tuple(config.get("velocity_indices", (3, 4, 5))))
    if name == "kalman": return FilterPyKalmanBaseline(common["horizon"], config["sample_period_s"], config.get("quality_index", 6), config.get("process_variance", 1e-3), config.get("measurement_variance", 1e-3))
    if name == "gru": return GRUForecaster(input_dim=data_config["input_dim"], **common, **{k: v for k, v in config.items() if k != "name"})
    if name == "quality_gru": return QualityAwareGRUForecaster(input_dim=data_config["input_dim"], **common, **{k: v for k, v in config.items() if k != "name"})
    if name == "tcn": return TCNForecaster(input_dim=data_config["input_dim"], **common, **{k: v for k, v in config.items() if k != "name"})
    raise ValueError(f"Unknown model: {name}")

