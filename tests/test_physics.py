import numpy as np

from knapsack_drl.physics import (
    gaussian_q,
    inverse_gaussian_q,
    normalized_paoi_metric,
    optimal_sampling_count,
    optimal_transmit_power,
    packet_error_probability,
)


def test_q_function_inverse() -> None:
    probabilities = np.asarray([0.001, 0.01, 0.1, 0.5, 0.9])
    np.testing.assert_allclose(
        gaussian_q(inverse_gaussian_q(probabilities)),
        probabilities,
        atol=1e-10,
    )


def test_packet_error_decreases_with_power() -> None:
    powers = np.logspace(-4, 0, 20)
    errors = packet_error_probability(1e-7, 100, powers, 4e-16, 100)
    assert np.all(np.diff(errors) <= 0.0)


def test_optimal_power_satisfies_paoi_boundary() -> None:
    gain = np.asarray([[1e-7]])
    blocklength = np.asarray([[100.0]])
    count = optimal_sampling_count(
        gain,
        blocklength,
        0.25,
        4e-16,
        100,
        0.99,
        10_000,
    )
    power = optimal_transmit_power(
        gain,
        blocklength,
        count,
        4e-16,
        100,
        0.99,
    )
    error = packet_error_probability(gain, blocklength, power, 4e-16, 100)
    metric = normalized_paoi_metric(error, count, 0.99)
    assert float(metric.item()) <= 1.0 + 1e-8

