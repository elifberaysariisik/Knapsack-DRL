
import numpy as np
from scipy import special


def gaussian_q(value: np.ndarray | float) -> np.ndarray:
    return special.ndtr(-np.asarray(value, dtype=np.float64))


def inverse_gaussian_q(probability: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-14, 1.0 - 1e-14)
    return special.ndtri(1.0 - clipped)


def packet_error_probability(
    channel_power: np.ndarray | float,
    blocklength: np.ndarray | float,
    transmit_power_w: np.ndarray | float,
    noise_power_w: float,
    packet_bits: int,
) -> np.ndarray:
    gain = np.maximum(np.asarray(channel_power, dtype=np.float64), 0.0)
    length = np.asarray(blocklength, dtype=np.float64)
    power = np.maximum(np.asarray(transmit_power_w, dtype=np.float64), 0.0)
    argument = np.sqrt(length) * (
        np.log1p(power * gain / noise_power_w) - np.log(2.0) * packet_bits / length
    )
    return np.clip(gaussian_q(argument), 1e-14, 1.0 - 1e-14)


def optimal_sampling_count(
    channel_power: np.ndarray,
    blocklength: np.ndarray,
    maximum_transmit_power_w: float,
    noise_power_w: float,
    packet_bits: int,
    reliability: float,
    maximum_k: int,
) -> np.ndarray:
    probability = packet_error_probability(
        channel_power,
        blocklength,
        maximum_transmit_power_w,
        noise_power_w,
        packet_bits,
    )
    ratio = np.log1p(-reliability) / np.log(probability)
    finite = np.isfinite(ratio) & (ratio > 0.0)
    result = np.where(finite, np.maximum(1.0, np.ceil(ratio)), maximum_k)
    return np.minimum(result, maximum_k).astype(np.int64)


def optimal_transmit_power(
    channel_power: np.ndarray,
    blocklength: np.ndarray,
    sampling_count: np.ndarray,
    noise_power_w: float,
    packet_bits: int,
    reliability: float,
) -> np.ndarray:
    gain = np.asarray(channel_power, dtype=np.float64)
    length = np.asarray(blocklength, dtype=np.float64)
    count = np.asarray(sampling_count, dtype=np.float64)
    target_error = np.exp(np.log1p(-reliability) / count)
    exponent = inverse_gaussian_q(target_error) / np.sqrt(length)
    exponent += np.log(2.0) * packet_bits / length
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        power = noise_power_w * np.expm1(exponent) / gain
    return np.where((gain > 0.0) & np.isfinite(power), np.maximum(power, 0.0), np.inf)


def scheduling_load(
    blocklength: np.ndarray,
    sampling_count: np.ndarray,
    bandwidth_hz: float,
    paoi_threshold_s: float,
) -> np.ndarray:
    length = np.asarray(blocklength, dtype=np.float64)
    denominator = bandwidth_hz * paoi_threshold_s - length
    with np.errstate(divide="ignore", invalid="ignore"):
        load = length * np.asarray(sampling_count, dtype=np.float64) / denominator
    return np.where(denominator > 0.0, load, np.inf)


def allocation_cost(
    transmit_power_w: np.ndarray,
    circuit_power_w: float,
    load: np.ndarray,
) -> np.ndarray:
    return (np.asarray(transmit_power_w) + circuit_power_w) * np.asarray(load)


def normalized_paoi_metric(
    packet_error: np.ndarray,
    sampling_count: np.ndarray,
    reliability: float,
) -> np.ndarray:
    numerator = np.power(np.asarray(packet_error), np.asarray(sampling_count))
    return numerator / (1.0 - reliability)

