import torch
from torch import nn


class DuelingQNetwork(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        action_count: int,
        hidden_sizes: tuple[int, int, int],
    ):
        super().__init__()
        first, second, third = hidden_sizes
        self.features = nn.Sequential(
            nn.Linear(input_dimension, first),
            nn.ReLU(),
            nn.Linear(first, second),
            nn.ReLU(),
            nn.Linear(second, third),
            nn.ReLU(),
        )
        self.value = nn.Linear(third, 1)
        self.advantage = nn.Linear(third, action_count)
        nn.init.uniform_(self.value.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.value.bias)
        nn.init.uniform_(self.advantage.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.advantage.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.features(state)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

