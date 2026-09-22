import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from knapsack_drl.agent import D3QNAgent
from knapsack_drl.channel import ChannelGenerator, generate_topology
from knapsack_drl.config import ExperimentConfig
from knapsack_drl.environment import allocate_frame, frame_reward
from knapsack_drl.replay import PrioritizedReplay


def replay_beta(episode: int, config: ExperimentConfig) -> float:
    progress = episode / max(config.train_episodes - 1, 1)
    return config.replay_beta_start + progress * (
        config.replay_beta_end - config.replay_beta_start
    )


def initialize_randomness(config: ExperimentConfig) -> np.random.Generator:
    random.seed(config.train_seed)
    np.random.seed(config.train_seed)
    torch.manual_seed(config.train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.train_seed)
    return np.random.default_rng(config.train_seed)


def store_transitions(
    replay: PrioritizedReplay,
    transitions: list[dict[str, object]],
    reward: float,
    config: ExperimentConfig,
) -> None:
    scaled_reward = config.reward_scale * reward
    for transition in transitions:
        replay.add(
            np.asarray(transition["state"]),
            int(transition["proposed_action"]),
            scaled_reward,
            np.asarray(transition["next_state"]),
            bool(transition["done"]),
            np.asarray(transition["next_mask"]),
        )


def train(
    config: ExperimentConfig,
    output_root: Path,
) -> tuple[Path, Path]:
    config.validate()
    scenario_directory = config.output_directory(output_root)
    scenario_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = scenario_directory / "knapsack_drl.pt"
    history_path = scenario_directory / "training.csv"
    config_path = scenario_directory / "config.json"
    config_path.write_text(
        json.dumps(config.to_dict(), indent=2),
        encoding="utf-8",
    )
    generator = initialize_randomness(config)
    _, path_gain = generate_topology(config)
    channel = ChannelGenerator(config, path_gain)
    agent = D3QNAgent(config)
    replay = PrioritizedReplay(
        config.replay_capacity,
        config.state_dimension,
        config.action_count,
        config.replay_alpha,
        config.replay_epsilon,
    )
    records = []
    global_step = 0
    for episode in range(config.train_episodes):
        frame, table, attempts = channel.sample_feasible(generator)
        allocation = allocate_frame(
            agent,
            frame,
            table,
            config,
            generator,
            training=True,
        )
        reward = frame_reward(
            np.asarray(allocation["executed_actions"]),
            table,
            config,
        )
        store_transitions(
            replay,
            list(allocation["transitions"]),
            reward,
            config,
        )
        losses = []
        for _ in allocation["transitions"]:
            global_step += 1
            if (
                len(replay) >= max(config.warmup_transitions, config.batch_size)
                and global_step % config.learn_every_steps == 0
            ):
                losses.append(
                    agent.learn(
                        replay,
                        replay_beta(episode, config),
                        generator,
                    )
                )
        learning_rate = agent.advance_schedules()
        records.append(
            {
                "episode": episode + 1,
                "reward": reward,
                "epsilon": agent.epsilon,
                "learning_rate": learning_rate,
                "loss": float(np.mean(losses)) if losses else np.nan,
                "teacher_intervention_rate": allocation["intervention_rate"],
                "remaining_budget": allocation["remaining_budget"],
                "frame_sampling_attempts": attempts,
            }
        )
        if (episode + 1) % max(1, min(500, config.train_episodes)) == 0:
            recent = records[-min(500, len(records)) :]
            mean_reward = np.mean([record["reward"] for record in recent])
            print(
                f"episode={episode + 1} reward={mean_reward:.8f} "
                f"epsilon={agent.epsilon:.4f}"
            )
    agent.save(checkpoint_path)
    pd.DataFrame.from_records(records).to_csv(history_path, index=False)
    return checkpoint_path, history_path

