import argparse
import json
from pathlib import Path

from knapsack_drl.config import ExperimentConfig
from knapsack_drl.evaluate import evaluate
from knapsack_drl.plot import create_paper_figure
from knapsack_drl.train import train


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "evaluate", "plot", "all"))
    parser.add_argument("--nodes", type=int, default=30)
    parser.add_argument("--sigma-e2", type=float, default=0.002)
    parser.add_argument("--train-episodes", type=int, default=10_000)
    parser.add_argument("--test-episodes", type=int, default=500)
    parser.add_argument("--train-seed", type=int, default=31)
    parser.add_argument("--test-seed", type=int, default=1031)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def build_config(arguments: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        nodes=arguments.nodes,
        csi_error_variance=arguments.sigma_e2,
        train_episodes=arguments.train_episodes,
        test_episodes=arguments.test_episodes,
        train_seed=arguments.train_seed,
        test_seed=arguments.test_seed,
    )


def resolve_checkpoint(
    arguments: argparse.Namespace,
    config: ExperimentConfig,
) -> Path:
    if arguments.checkpoint is not None:
        return arguments.checkpoint
    return config.output_directory(arguments.output) / "knapsack_drl.pt"


def main() -> None:
    arguments = parse_arguments()
    config = build_config(arguments)
    checkpoint = resolve_checkpoint(arguments, config)
    if arguments.command in ("train", "all"):
        checkpoint, history_path = train(config, arguments.output)
        print(history_path)
    if arguments.command in ("evaluate", "all"):
        result_path, summary = evaluate(
            config,
            checkpoint,
            arguments.output,
        )
        print(result_path)
        print(json.dumps(summary, indent=2))
    if arguments.command in ("plot", "all"):
        figure_path = create_paper_figure(
            config.output_directory(arguments.output)
        )
        print(figure_path)


if __name__ == "__main__":
    main()

