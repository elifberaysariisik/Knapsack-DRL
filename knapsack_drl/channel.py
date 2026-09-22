
import math

import numpy as np

from knapsack_drl.config import ExperimentConfig


def complex_gaussian(
    generator: np.random.Generator,
    variance: float,
    size: int,
) -> np.ndarray:
    standard_deviation = math.sqrt(variance / 2.0)
    return standard_deviation * (
        generator.standard_normal(size) + 1j * generator.standard_normal(size)
    )


def generate_topology(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    seed_sequence = np.random.SeedSequence(config.topology_seed)
    distance_seed, shadowing_seed = seed_sequence.spawn(2)
    distance_generator = np.random.default_rng(distance_seed)
    shadowing_generator = np.random.default_rng(shadowing_seed)
    distances = distance_generator.uniform(
        config.distance_min_m,
        config.distance_max_m,
        size=config.nodes,
    )
    shadowing = shadowing_generator.normal(
        0.0,
        config.shadowing_std_db,
        size=config.nodes,
    )
    pathloss_db = (
        config.pathloss_reference_db
        + 10.0
        * config.pathloss_exponent
        * np.log10(distances / config.reference_distance_m)
        + shadowing
    )
    return distances, 10.0 ** (-pathloss_db / 10.0)


def bernstein_channel_power(
    estimated_channel: np.ndarray,
    error_variance: np.ndarray,
    outage_tolerance: float,
) -> np.ndarray:
    estimated_power = np.abs(estimated_channel) ** 2
    factor = math.sqrt(-2.0 * math.log(outage_tolerance))
    uncertainty = np.sqrt(error_variance**2 + 2.0 * error_variance * estimated_power)
    return estimated_power + error_variance - factor * uncertainty


class ChannelGenerator:

    def __init__(self, config: ExperimentConfig, path_gain: np.ndarray):
        self.config = config
        self.path_gain = np.asarray(path_gain, dtype=np.float64)
        self.relative_error_variance = np.full(
            config.nodes,
            config.initial_csi_error_variance,
            dtype=np.float64,
        )

    def update_error_variance(
        self,
        first_estimate: np.ndarray,
        second_estimate: np.ndarray,
    ) -> None:
        reference = np.maximum(np.abs(first_estimate) ** 2, np.finfo(float).tiny)
        residual = 0.5 * np.abs(first_estimate - second_estimate) ** 2 / reference
        smoothing = self.config.ewma_smoothing
        self.relative_error_variance = (
            (1.0 - smoothing) * self.relative_error_variance + smoothing * residual
        )

    def sample(self, generator: np.random.Generator) -> dict[str, np.ndarray]:
        config = self.config
        fading = complex_gaussian(generator, 1.0, config.nodes)
        estimated_channel = np.sqrt(self.path_gain) * fading
        absolute_error_variance = (
            self.relative_error_variance * np.abs(estimated_channel) ** 2
        )
        error = np.sqrt(
            config.csi_error_variance * np.maximum(np.abs(estimated_channel) ** 2, 0.0)
        ) * complex_gaussian(generator, 1.0, config.nodes)
        true_channel = estimated_channel + error
        independent_error = np.sqrt(
            config.csi_error_variance * np.maximum(np.abs(estimated_channel) ** 2, 0.0)
        ) * complex_gaussian(generator, 1.0, config.nodes)
        second_estimate = estimated_channel + error - independent_error
        bound = bernstein_channel_power(
            estimated_channel,
            absolute_error_variance,
            config.csi_outage_tolerance,
        )
        frame = {
            "estimated_channel": estimated_channel,
            "true_channel": true_channel,
            "estimated_power": np.abs(estimated_channel) ** 2,
            "true_power": np.abs(true_channel) ** 2,
            "error_variance": absolute_error_variance,
            "relative_error_variance": self.relative_error_variance.copy(),
            "bernstein_power_raw": bound,
            "bernstein_power": np.maximum(bound, np.finfo(float).tiny),
            "second_estimate": second_estimate,
        }
        return frame

    def sample_feasible(
        self,
        generator: np.random.Generator,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
        from knapsack_drl.candidates import build_candidate_table, frame_is_feasible

        for attempt in range(1, self.config.maximum_frame_attempts + 1):
            frame = self.sample(generator)
            if np.any(frame["bernstein_power_raw"] <= 0.0):
                continue
            table = build_candidate_table(frame["bernstein_power"], self.config)
            if frame_is_feasible(table, self.config):
                self.update_error_variance(
                    frame["estimated_channel"],
                    frame["second_estimate"],
                )
                return frame, table, attempt
        raise RuntimeError("no robustly feasible frame was sampled")
