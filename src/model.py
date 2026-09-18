"""Actor-critic network used by PPO."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(observation_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(observations).squeeze(-1)

    def get_action_and_value(
        self, observations: torch.Tensor, actions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor(observations)
        distribution = torch.distributions.Normal(mean, self.log_std.exp())
        if actions is None:
            actions = distribution.sample()
        return actions, distribution.log_prob(actions).sum(-1), distribution.entropy().sum(-1), self.get_value(observations)