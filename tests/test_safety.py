import numpy as np

from knapsack_drl.config import ExperimentConfig
from knapsack_drl.safety import (
    future_minimum_load,
    sequential_feasible_mask,
    verify_action,
)


def make_table() -> dict[str, np.ndarray]:
    return {
        "feasible": np.ones((3, 3), dtype=bool),
        "load": np.asarray(
            [
                [0.02, 0.05, 0.08],
                [0.03, 0.06, 0.09],
                [0.04, 0.07, 0.10],
            ]
        ),
        "cost": np.asarray(
            [
                [0.8, 0.5, 0.4],
                [0.8, 0.5, 0.4],
                [0.8, 0.5, 0.4],
            ]
        ),
    }


def test_future_guard_reserves_minimum_remaining_load() -> None:
    assert np.isclose(future_minimum_load(0, make_table()), 0.07)


def test_sequential_mask_rejects_budget_exhaustion() -> None:
    config = ExperimentConfig(nodes=3, blocklength_min=25, blocklength_max=27)
    mask, reserve = sequential_feasible_mask(0, 0.12, make_table(), config)
    assert np.isclose(reserve, 0.07)
    np.testing.assert_array_equal(mask, np.asarray([True, True, False]))


def test_teacher_projects_to_closest_guarded_action() -> None:
    config = ExperimentConfig(nodes=3, blocklength_min=25, blocklength_max=27)
    result = verify_action(0, 2, 0.12, make_table(), config)
    assert result["executed_action"] == 1
    assert result["intervened"]

