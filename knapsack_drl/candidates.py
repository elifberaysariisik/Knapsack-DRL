
import numpy as np

from knapsack_drl.config import ExperimentConfig
from knapsack_drl.physics import (
    allocation_cost,
    optimal_sampling_count,
    optimal_transmit_power,
    scheduling_load,
)


def build_candidate_table(
    design_channel_power: np.ndarray,
    config: ExperimentConfig,
) -> dict[str, np.ndarray]:
    gain = np.asarray(design_channel_power, dtype=np.float64).reshape(config.nodes, 1)
    blocklength = config.blocklengths.astype(np.float64).reshape(1, config.action_count)
    count = optimal_sampling_count(
        gain,
        blocklength,
        config.max_tx_power_w,
        config.noise_power_w,
        config.packet_bits,
        config.reliability,
        config.maximum_k,
    )
    transmit_power = optimal_transmit_power(
        gain,
        blocklength,
        count,
        config.noise_power_w,
        config.packet_bits,
        config.reliability,
    )
    load = scheduling_load(
        blocklength,
        count,
        config.bandwidth_hz,
        config.paoi_threshold_s,
    )
    cost = allocation_cost(transmit_power, config.circuit_power_w, load)
    feasible = (
        (count < config.maximum_k)
        & np.isfinite(transmit_power)
        & (transmit_power <= config.max_tx_power_w * (1.0 + config.numerical_tolerance))
        & np.isfinite(load)
        & (load > 0.0)
    )
    return {
        "sampling_count": count,
        "transmit_power_w": transmit_power,
        "load": load,
        "cost": cost,
        "feasible": feasible,
    }


def minimum_feasible_loads(table: dict[str, np.ndarray]) -> np.ndarray:
    return np.where(table["feasible"], table["load"], np.inf).min(axis=1)


def frame_is_feasible(
    table: dict[str, np.ndarray],
    config: ExperimentConfig,
) -> bool:
    if not np.all(table["feasible"].any(axis=1)):
        return False
    minimum_loads = minimum_feasible_loads(table)
    return bool(
        np.isfinite(minimum_loads).all()
        and minimum_loads.sum() <= config.scheduling_budget + config.numerical_tolerance
    )

