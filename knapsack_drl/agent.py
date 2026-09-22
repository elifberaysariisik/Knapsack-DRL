from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from knapsack_drl.config import ExperimentConfig
from knapsack_drl.network import DuelingQNetwork
from knapsack_drl.replay import PrioritizedReplay


class D3QNAgent:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online = DuelingQNetwork(
            config.state_dimension,
            config.action_count,
            config.hidden_sizes,
        ).to(self.device)
        self.target = DuelingQNetwork(
            config.state_dimension,
            config.action_count,
            config.hidden_sizes,
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.RMSprop(
            self.online.parameters(),
            lr=config.learning_rate,
        )
        self.epsilon = config.epsilon_start
        self.schedule_step = 0

    def act(
        self,
        state: np.ndarray,
        feasible_mask: np.ndarray,
        generator: np.random.Generator,
        training: bool,
    ) -> int:
        candidates = np.flatnonzero(feasible_mask)
        if candidates.size == 0:
            raise RuntimeError("agent received an empty robust action set")
        if training and generator.random() < self.epsilon:
            return int(generator.choice(candidates))
        self.online.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            values = self.online(state_tensor).squeeze(0).cpu().numpy()
        if training:
            self.online.train()
        return int(np.argmax(np.where(feasible_mask, values, -np.inf)))

    def learn(
        self,
        replay: PrioritizedReplay,
        beta: float,
        generator: np.random.Generator,
    ) -> float:
        batch = replay.sample(self.config.batch_size, beta, generator)
        states, actions, rewards, next_states, dones, weights, indices, masks = batch
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_state_tensor = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )
        done_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        predicted = self.online(state_tensor).gather(
            1,
            action_tensor.unsqueeze(1),
        ).squeeze(1)
        with torch.no_grad():
            online_next = self.online(next_state_tensor).masked_fill(~mask_tensor, -torch.inf)
            next_actions = online_next.argmax(dim=1, keepdim=True)
            target_next = self.target(next_state_tensor).gather(1, next_actions).squeeze(1)
            expected = reward_tensor + self.config.discount * (1.0 - done_tensor) * target_next
        errors = (expected - predicted).detach().cpu().numpy()
        element_losses = functional.smooth_l1_loss(
            predicted,
            expected,
            reduction="none",
        )
        loss = (weight_tensor * element_losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online.parameters(),
            self.config.gradient_clip,
        )
        self.optimizer.step()
        with torch.no_grad():
            for target_parameter, online_parameter in zip(
                self.target.parameters(),
                self.online.parameters(),
            ):
                target_parameter.mul_(1.0 - self.config.soft_update_rate)
                target_parameter.add_(
                    self.config.soft_update_rate * online_parameter
                )
        replay.update_priorities(indices, errors)
        return float(loss.item())

    def advance_schedules(self) -> float:
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon - self.config.epsilon_decay,
        )
        self.schedule_step += 1
        learning_rate = self.config.learning_rate / (
            1.0 + self.config.learning_rate_decay * self.schedule_step
        )
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "epsilon": self.epsilon,
                "schedule_step": self.schedule_step,
                "config": self.config.to_dict(),
            },
            path,
        )

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint["online"])
        self.epsilon = float(checkpoint.get("epsilon", self.config.epsilon_end))
        self.schedule_step = int(checkpoint.get("schedule_step", 0))

