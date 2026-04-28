"""Rollout storage for vectorized PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RolloutBatch:
    obs: Any
    actions: Any
    logprobs: Any
    advantages: Any
    returns: Any
    values: Any


class RolloutBuffer:
    def __init__(self, rollout_steps: int, num_envs: int, obs_shape: tuple[int, int, int], device: str | Any):
        import torch

        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.device = device
        self.obs = torch.zeros((rollout_steps, num_envs, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((rollout_steps, num_envs), dtype=torch.long, device=device)
        self.logprobs = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((rollout_steps, num_envs), dtype=torch.float32, device=device)

    def store(self, step: int, obs: Any, action: Any, logprob: Any, reward: Any, done: Any, value: Any) -> None:
        import torch

        self.obs[step].copy_(obs)
        self.actions[step].copy_(action)
        self.logprobs[step].copy_(logprob)
        self.rewards[step].copy_(torch.as_tensor(reward, dtype=torch.float32, device=self.device))
        self.dones[step].copy_(torch.as_tensor(done, dtype=torch.float32, device=self.device))
        self.values[step].copy_(value)

    def compute_returns_and_advantages(self, next_value: Any, next_done: Any, gamma: float, gae_lambda: float) -> None:
        import torch

        next_done = torch.as_tensor(next_done, dtype=torch.float32, device=self.device)
        last_gae_lam = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for step in reversed(range(self.rollout_steps)):
            if step == self.rollout_steps - 1:
                next_non_terminal = 1.0 - next_done
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def flatten(self) -> RolloutBatch:
        batch_size = self.rollout_steps * self.num_envs
        return RolloutBatch(
            obs=self.obs.reshape((batch_size, *self.obs.shape[2:])),
            actions=self.actions.reshape(batch_size),
            logprobs=self.logprobs.reshape(batch_size),
            advantages=self.advantages.reshape(batch_size),
            returns=self.returns.reshape(batch_size),
            values=self.values.reshape(batch_size),
        )
