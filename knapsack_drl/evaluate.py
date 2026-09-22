from pathlib import Path

import numpy as np
import pandas as pd

from knapsack_drl.agent import D3QNAgent
from knapsack_drl.channel import ChannelGenerator, generate_topology
from knapsack_drl.config import ExperimentConfig
from knapsack_drl.environment import allocate_frame, evaluate_allocation


def evaluate(
    config: ExperimentConfig,
    checkpoint_path: Path,
    output_root: Path,
) -> tuple[Path, dict[str, float]]:
    config.validate()
    scenario_directory = config.output_directory(output_root)
    scenario_directory.mkdir(parents=True, exist_ok=True)
    result_path = scenario_directory / "evaluation.csv"
    generator = np.random.default_rng(config.test_seed)
    _, path_gain = generate_topology(config)
    channel = ChannelGenerator(config, path_gain)
    agent = D3QNAgent(config)
    agent.load(checkpoint_path)
    records = []
    for realization in range(config.test_episodes):
        frame, table, attempts = channel.sample_feasible(generator)
        allocation = allocate_frame(
            agent,
            frame,
            table,
            config,
            generator,
            training=False,
        )
        metrics = evaluate_allocation(allocation, frame, table, config)
        records.append(
            {
                "realization": realization + 1,
                **metrics,
                "frame_sampling_attempts": attempts,
            }
        )
    results = pd.DataFrame.from_records(records)
    results.to_csv(result_path, index=False)
    summary = {
        "mean_total_power_w": float(results["total_power_w"].mean()),
        "mean_normalized_paoi_max": float(results["normalized_paoi_max"].mean()),
        "mean_realized_normalized_paoi": float(
            results["realized_normalized_paoi_mean"].mean()
        ),
        "mean_normalized_scheduling_load": float(
            results["normalized_scheduling_load"].mean()
        ),
        "paoi_violation_rate": float(results["paoi_violation"].mean()),
        "realized_paoi_violation_rate": float(
            results["realized_paoi_violation"].mean()
        ),
        "scheduling_violation_rate": float(results["scheduling_violation"].mean()),
        "transmit_power_violation_rate": float(
            results["transmit_power_violation"].mean()
        ),
        "mean_teacher_intervention_rate": float(
            results["teacher_intervention_rate"].mean()
        ),
        "mean_decision_time_ms": float(results["decision_time_ms"].mean()),
    }
    return result_path, summary
