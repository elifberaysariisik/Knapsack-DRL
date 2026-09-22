
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ExperimentConfig:

    nodes: int = 30
    bandwidth_hz: float = 100_000.0
    blocklength_min: int = 1
    blocklength_max: int = 200
    packet_bits: int = 100
    reliability: float = 0.99
    paoi_threshold_s: float = 0.101
    max_tx_power_w: float = 0.250
    circuit_power_w: float = 0.005
    noise_density_dbm_hz: float = -174.0
    scheduling_budget: float = 0.20
    csi_error_variance: float = 0.002
    initial_csi_error_variance: float = 0.01
    csi_outage_tolerance: float = 0.05
    ewma_smoothing: float = 0.1
    pathloss_reference_db: float = 35.3
    reference_distance_m: float = 1.0
    pathloss_exponent: float = 3.76
    shadowing_std_db: float = 4.0
    distance_min_m: float = 5.0
    distance_max_m: float = 20.0
    channel_correlation: float = 0.6
    train_episodes: int = 10_000
    test_episodes: int = 500
    topology_seed: int = 2026
    train_seed: int = 31
    test_seed: int = 1031
    hidden_sizes: tuple[int, int, int] = (32, 64, 300)
    batch_size: int = 64
    replay_capacity: int = 150_000
    replay_alpha: float = 0.6
    replay_beta_start: float = 0.4
    replay_beta_end: float = 1.0
    replay_epsilon: float = 1e-6
    discount: float = 0.666
    soft_update_rate: float = 1e-3
    learning_rate: float = 3e-4
    learning_rate_decay: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 1e-4
    scheduling_penalty: float = 1e-3
    reward_scale: float = 1_000.0
    gradient_clip: float = 1.0
    warmup_transitions: int = 500
    learn_every_steps: int = 4
    maximum_k: int = 10_000
    numerical_tolerance: float = 1e-9
    maximum_frame_attempts: int = 500

    @property
    def noise_power_w(self) -> float:
        return 10.0 ** ((self.noise_density_dbm_hz - 30.0) / 10.0) * self.bandwidth_hz

    @property
    def blocklengths(self) -> np.ndarray:
        return np.arange(self.blocklength_min, self.blocklength_max + 1, dtype=np.int64)

    @property
    def action_count(self) -> int:
        return self.blocklength_max - self.blocklength_min + 1

    @property
    def state_dimension(self) -> int:
        return 6

    def validate(self) -> None:
        if self.nodes < 1:
            raise ValueError("nodes must be positive")
        if not 0.0 < self.scheduling_budget <= 1.0:
            raise ValueError("scheduling_budget must be in (0, 1]")
        if not 0.0 < self.reliability < 1.0:
            raise ValueError("reliability must be in (0, 1)")
        if not 0.0 < self.csi_outage_tolerance < 1.0:
            raise ValueError("csi_outage_tolerance must be in (0, 1)")
        if self.blocklength_min < 1 or self.blocklength_min > self.blocklength_max:
            raise ValueError("invalid blocklength interval")
        if self.blocklength_max >= self.bandwidth_hz * self.paoi_threshold_s:
            raise ValueError("blocklengths must satisfy m < B alpha")

    def with_overrides(self, **values: object) -> "ExperimentConfig":
        updated = replace(self, **values)
        updated.validate()
        return updated

    def to_dict(self) -> dict:
        return asdict(self)

    def output_directory(self, root: Path) -> Path:
        variance = format(self.csi_error_variance, ".6g")
        return root / f"N{self.nodes}_sigma{variance}"
