
import numpy as np

from knapsack_drl.candidates import minimum_feasible_loads
from knapsack_drl.config import ExperimentConfig


def future_minimum_load(
    node_index: int,
    table: dict[str, np.ndarray],
) -> float:
    if node_index >= table["load"].shape[0] - 1:
        return 0.0
    remaining = {
        "feasible": table["feasible"][node_index + 1 :],
        "load": table["load"][node_index + 1 :],
    }
    loads = minimum_feasible_loads(remaining)
    return float(loads.sum()) if np.isfinite(loads).all() else float("inf")


def sequential_feasible_mask(
    node_index: int,
    remaining_budget: float,
    table: dict[str, np.ndarray],
    config: ExperimentConfig,
) -> tuple[np.ndarray, float]:
    reserve = future_minimum_load(node_index, table)
    node_load = table["load"][node_index]
    tolerance = config.numerical_tolerance
    mask = (
        table["feasible"][node_index]
        & (node_load <= remaining_budget + tolerance)
        & (remaining_budget - node_load >= reserve - tolerance)
    )
    return mask, reserve


def verify_action(
    node_index: int,
    proposed_action: int,
    remaining_budget: float,
    table: dict[str, np.ndarray],
    config: ExperimentConfig,
) -> dict[str, object]:
    mask, reserve = sequential_feasible_mask(
        node_index,
        remaining_budget,
        table,
        config,
    )
    candidates = np.flatnonzero(mask)
    if candidates.size == 0:
        raise RuntimeError("future-budget guard found no sequentially feasible action")
    if mask[proposed_action]:
        executed_action = int(proposed_action)
    else:
        proposed_blocklength = config.blocklengths[proposed_action]
        distances = np.abs(config.blocklengths[candidates] - proposed_blocklength)
        costs = table["cost"][node_index, candidates]
        executed_action = int(candidates[np.lexsort((costs, distances))[0]])
    load = float(table["load"][node_index, executed_action])
    return {
        "executed_action": executed_action,
        "intervened": executed_action != proposed_action,
        "feasible_mask": mask,
        "future_minimum_load": reserve,
        "remaining_budget": remaining_budget - load,
    }

