"""PPO rollout processing and update logic."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.amp import GradScaler, autocast


@dataclass
class Rollout:
    observations: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(rewards.shape[1], device=rewards.device)
    for step in reversed(range(rewards.shape[0])):
        next_non_terminal = 1.0 - dones[step]
        next_values = next_value if step == rewards.shape[0] - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[step] = last_gae
    return advantages, advantages + values


class PPOAgent:
    def __init__(self, model: nn.Module, learning_rate: float, device: torch.device, amp_enabled: bool) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
        self.device = device
        self.amp_enabled = amp_enabled and device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.amp_enabled)

    def update(
        self, rollout: Rollout, advantages: torch.Tensor, returns: torch.Tensor,
        batch_size: int, n_epochs: int, clip_coef: float, ent_coef: float,
        vf_coef: float, max_grad_norm: float,
    ) -> dict[str, float]:
        observations = rollout.observations.reshape(-1, rollout.observations.shape[-1])
        actions = rollout.actions.reshape(-1, rollout.actions.shape[-1])
        old_logprobs = rollout.logprobs.reshape(-1)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)
        values = rollout.values.reshape(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total = observations.shape[0]
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        updates = 0
        for _ in range(n_epochs):
            indices = torch.randperm(total, device=self.device)
            for start in range(0, total, batch_size):
                batch = indices[start:start + batch_size]
                with autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    _, new_logprob, entropy, new_value = self.model.get_action_and_value(observations[batch], actions[batch])
                    ratio = (new_logprob - old_logprobs[batch]).exp()
                    pg_loss = torch.max(-advantages[batch] * ratio, -advantages[batch] * ratio.clamp(1 - clip_coef, 1 + clip_coef)).mean()
                    value_loss = 0.5 * ((new_value - returns[batch]) ** 2).mean()
                    loss = pg_loss + vf_coef * value_loss - ent_coef * entropy.mean()
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                metrics["policy_loss"] += pg_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.mean().item()
                updates += 1
        return {key: value / max(updates, 1) for key, value in metrics.items()}