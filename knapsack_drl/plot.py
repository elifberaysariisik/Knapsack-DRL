import os
from pathlib import Path

cache_root = Path(__file__).resolve().parents[1] / "work"
(cache_root / "matplotlib-cache").mkdir(parents=True, exist_ok=True)
(cache_root / "cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    probabilities = np.arange(1, ordered.size + 1) / ordered.size
    return ordered, probabilities


def create_paper_figure(scenario_directory: Path) -> Path:
    training = pd.read_csv(scenario_directory / "training.csv")
    evaluation = pd.read_csv(scenario_directory / "evaluation.csv")
    blue = "#0072BD"
    figure, axes = plt.subplots(2, 2, figsize=(7.16, 5.4), constrained_layout=True)
    smoothed_reward = training["reward"].rolling(100, min_periods=1).mean()
    axes[0, 0].plot(training["episode"], smoothed_reward, color=blue, linewidth=1.4)
    axes[0, 0].set_xlabel("Training episode")
    axes[0, 0].set_ylabel("Reward")
    power, cumulative = empirical_cdf(1_000.0 * evaluation["total_power_w"].to_numpy())
    axes[0, 1].plot(power, cumulative, color=blue, linewidth=1.4)
    axes[0, 1].set_xlabel("Total power (mW)")
    axes[0, 1].set_ylabel("ECDF")
    axes[1, 0].plot(
        evaluation["realization"],
        evaluation["normalized_paoi_max"],
        color=blue,
        linewidth=1.0,
    )
    axes[1, 0].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[1, 0].set_xlabel("Test realization")
    axes[1, 0].set_ylabel("Normalized PAoI")
    axes[1, 1].plot(
        evaluation["realization"],
        evaluation["normalized_scheduling_load"],
        color=blue,
        linewidth=1.0,
    )
    axes[1, 1].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[1, 1].set_xlabel("Test realization")
    axes[1, 1].set_ylabel("Normalized scheduling load")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    output_path = scenario_directory / "knapsack_drl_paper_figure.pdf"
    figure.savefig(output_path, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path
