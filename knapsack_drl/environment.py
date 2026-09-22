import math
import time

import numpy as np

from knapsack_drl.agent import D3QNAgent
from knapsack_drl.config import ExperimentConfig
from knapsack_drl.physics import normalized_paoi_metric, packet_error_probability
from knapsack_drl.safety import sequential_feasible_mask, verify_action


def logarithmic_gain_feature(value: float) -> float:
    return float(
        np.clip((math.log10(max(float(value), 1e-16)) + 16.0) / 12.0, 0.0, 1.0)
    )


def build_state(
    node_index: int,
    remaining_budget: float,
    frame: dict[str, np.ndarray],
    config: ExperimentConfig,
) -> np.ndarray:
    remaining_nodes = max(config.nodes - node_index, 1)
    estimated_power = float(frame["estimated_power"][node_index])
    error_variance = float(frame["error_variance"][node_index])
    uncertainty_ratio = error_variance / max(
        estimated_power + error_variance,
        np.finfo(float).tiny,
    )
    return np.asarray(
        [
            (node_index + 1) / config.nodes,
            remaining_budget / config.scheduling_budget,
            remaining_budget / remaining_nodes,
            logarithmic_gain_feature(estimated_power),
            uncertainty_ratio,
            logarithmic_gain_feature(frame["bernstein_power"][node_index]),
        ],
        dtype=np.float32,
    )


def allocate_frame(
    agent: D3QNAgent,
    frame: dict[str, np.ndarray],
    table: dict[str, np.ndarray],
    config: ExperimentConfig,
    generator: np.random.Generator,
    training: bool,
) -> dict[str, object]:
    start_time = time.perf_counter()
    remaining_budget = config.scheduling_budget
    proposed_actions = np.zeros(config.nodes, dtype=np.int64)
    executed_actions = np.zeros(config.nodes, dtype=np.int64)
    transitions = []
    interventions = 0
    for node_index in range(config.nodes):
        state = build_state(node_index, remaining_budget, frame, config)
        proposed_action = agent.act(
            state,
            table["feasible"][node_index],
            generator,
            training,
        )
        verification = verify_action(
            node_index,
            proposed_action,
            remaining_budget,
            table,
            config,
        )
        executed_action = int(verification["executed_action"])
        remaining_budget = float(verification["remaining_budget"])
        proposed_actions[node_index] = proposed_action
        executed_actions[node_index] = executed_action
        interventions += int(verification["intervened"])
        done = node_index == config.nodes - 1
        if done:
            next_state = np.zeros(config.state_dimension, dtype=np.float32)
            next_mask = np.ones(config.action_count, dtype=bool)
        else:
            next_state = build_state(node_index + 1, remaining_budget, frame, config)
            next_mask, _ = sequential_feasible_mask(
                node_index + 1,
                remaining_budget,
                table,
                config,
            )
        transitions.append(
            {
                "state": state,
                "proposed_action": proposed_action,
                "executed_action": executed_action,
                "next_state": next_state,
                "done": done,
                "next_mask": next_mask,
            }
        )
    elapsed_ms = 1_000.0 * (time.perf_counter() - start_time)
    return {
        "proposed_actions": proposed_actions,
        "executed_actions": executed_actions,
        "transitions": transitions,
        "intervention_rate": interventions / config.nodes,
        "remaining_budget": remaining_budget,
        "decision_time_ms": elapsed_ms,
    }


def frame_reward(
    executed_actions: np.ndarray,
    table: dict[str, np.ndarray],
    config: ExperimentConfig,
) -> float:
    nodes = np.arange(config.nodes)
    costs = table["cost"][nodes, executed_actions]
    loads = table["load"][nodes, executed_actions]
    scheduling_violation = float(
        loads.sum() > config.scheduling_budget + config.numerical_tolerance
    )
    return -float(costs.sum()) - config.scheduling_penalty * scheduling_violation


def evaluate_allocation(
    allocation: dict[str, object],
    frame: dict[str, np.ndarray],
    table: dict[str, np.ndarray],
    config: ExperimentConfig,
) -> dict[str, float]:
    actions = np.asarray(allocation["executed_actions"], dtype=np.int64)
    nodes = np.arange(config.nodes)
    sampling_count = table["sampling_count"][nodes, actions]
    required_power = table["transmit_power_w"][nodes, actions]
    applied_power = np.minimum(required_power, config.max_tx_power_w)
    load = table["load"][nodes, actions]
    packet_error = packet_error_probability(
        frame["true_power"],
        config.blocklengths[actions],
        applied_power,
        config.noise_power_w,
        config.packet_bits,
    )
    paoi = normalized_paoi_metric(
        packet_error,
        sampling_count,
        config.reliability,
    )
    total_power = float(((applied_power + config.circuit_power_w) * load).sum())
    normalized_load = float(load.sum() / config.scheduling_budget)
    return {
        "reward": frame_reward(actions, table, config),
        "total_power_w": total_power,
        "normalized_paoi_max": float(paoi.max()),
        "normalized_scheduling_load": normalized_load,
        "paoi_violation": float(np.any(paoi > 1.0 + config.numerical_tolerance)),
        "scheduling_violation": float(
            normalized_load > 1.0 + config.numerical_tolerance
        ),
        "transmit_power_violation": float(
            np.any(
                required_power
                > config.max_tx_power_w * (1.0 + config.numerical_tolerance)
            )
        ),
        "teacher_intervention_rate": float(allocation["intervention_rate"]),
        "decision_time_ms": float(allocation["decision_time_ms"]),
    }

