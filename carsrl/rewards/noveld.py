"""Curiosity and novelty intrinsic rewards for the unified PPO pipeline.

This is a compact reimplementation for fair comparison inside the current
Gymnasium/MiniGrid PPO trainer. It follows the key reward idea used in the
provided NovelD code:

    max(error(s_next) - scale_fac * error(s), 0)

with an episodic first-visit gate. The original repository uses an
actor-learner/V-trace stack; this module intentionally exposes only the reward
module needed by the shared PPO implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _layer_init(layer: Any, std: float = 1.0) -> Any:
    import torch

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, 0.0)
    return layer


def _obs_hash(obs_item: Any) -> bytes:
    image = obs_item["image"] if isinstance(obs_item, dict) else obs_item
    array = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if not isinstance(obs_item, dict):
        return array.tobytes()
    parts = [array.tobytes()]
    if "direction" in obs_item:
        parts.append(np.asarray(obs_item["direction"], dtype=np.int64).tobytes())
    if "mission" in obs_item:
        parts.append(str(obs_item["mission"]).encode("utf-8"))
    return b"|".join(parts)


def _obs_item(obs_batch: Any, index: int) -> Any:
    if isinstance(obs_batch, dict):
        return {key: value[index] for key, value in obs_batch.items()}
    return obs_batch[index]


def _build_embedding_network(obs_shape: tuple[int, int, int], embedding_dim: int) -> Any:
    import torch
    from torch import nn

    channels, height, width = obs_shape
    return nn.Sequential(
        _layer_init(nn.Conv2d(channels, 32, kernel_size=3, padding=1)),
        nn.ReLU(),
        _layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
        nn.ReLU(),
        nn.Flatten(),
        _layer_init(nn.Linear(64 * height * width, 256)),
        nn.ReLU(),
        _layer_init(nn.Linear(256, embedding_dim), std=0.1),
    )


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> Any:
    from torch import nn

    return nn.Sequential(
        _layer_init(nn.Linear(input_dim, hidden_dim)),
        nn.ReLU(),
        _layer_init(nn.Linear(hidden_dim, output_dim), std=0.1),
    )


@dataclass(frozen=True)
class NovelDStepResult:
    intrinsic_rewards: np.ndarray
    raw_rewards: np.ndarray
    prediction_errors: np.ndarray
    first_visit: np.ndarray
    episode_metrics: dict[int, dict[str, Any]]


class NovelDReward:
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        num_envs: int,
        device: Any,
        coef: float = 0.05,
        learning_rate: float = 1.0e-4,
        embedding_dim: int = 128,
        scale_fac: float = 0.5,
        reward_clip_min: float = 0.0,
        reward_clip_max: float = 1.0,
        update_epochs: int = 1,
        train_batch_size: int = 256,
    ):
        import torch

        self.num_envs = num_envs
        self.device = device
        self.coef = coef
        self.scale_fac = scale_fac
        self.reward_clip_min = reward_clip_min
        self.reward_clip_max = reward_clip_max
        self.update_epochs = update_epochs
        self.train_batch_size = train_batch_size

        self.target = _build_embedding_network(obs_shape, embedding_dim).to(device)
        self.predictor = _build_embedding_network(obs_shape, embedding_dim).to(device)
        self.target.eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)

        self.episode_seen: list[set[bytes]] = [set() for _ in range(num_envs)]
        self.episode_intrinsic_returns = np.zeros(num_envs, dtype=np.float32)
        self.episode_unique_counts = np.zeros(num_envs, dtype=np.int64)
        self.train_obs: list[Any] = []
        self.last_update_stats: dict[str, float] = {}

    def reset(self, obs_batch: Any) -> None:
        self.episode_seen = [set() for _ in range(self.num_envs)]
        self.episode_intrinsic_returns.fill(0.0)
        self.episode_unique_counts.fill(0)
        for index in range(self.num_envs):
            key = _obs_hash(_obs_item(obs_batch, index))
            self.episode_seen[index].add(key)
            self.episode_unique_counts[index] += 1

    def step(
        self,
        current_obs_tensor: Any,
        next_obs_tensor: Any,
        next_obs_batch: Any,
        done: np.ndarray,
        actions: np.ndarray | None = None,
    ) -> NovelDStepResult:
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            target_current = self.target(current_obs_tensor)
            pred_current = self.predictor(current_obs_tensor)
            target_next = self.target(next_obs_tensor)
            pred_next = self.predictor(next_obs_tensor)
            err_current = torch.norm(pred_current - target_current, dim=1, p=2)
            err_next = torch.norm(pred_next - target_next, dim=1, p=2)
            raw = F.relu(err_next - self.scale_fac * err_current)

        raw_np = raw.detach().cpu().numpy().astype(np.float32)
        err_next_np = err_next.detach().cpu().numpy().astype(np.float32)
        first_visit = np.zeros(self.num_envs, dtype=np.float32)
        episode_metrics: dict[int, dict[str, Any]] = {}

        for index in range(self.num_envs):
            key = _obs_hash(_obs_item(next_obs_batch, index))
            if key not in self.episode_seen[index]:
                first_visit[index] = 1.0
                self.episode_seen[index].add(key)
                self.episode_unique_counts[index] += 1

        rewards = self.coef * raw_np * first_visit
        rewards = np.clip(rewards, self.reward_clip_min, self.reward_clip_max).astype(np.float32)
        self.episode_intrinsic_returns += rewards
        self.train_obs.append(next_obs_tensor.detach().cpu())

        for index, is_done in enumerate(done):
            if not bool(is_done):
                continue
            episode_metrics[index] = {
                "intrinsic_return": float(self.episode_intrinsic_returns[index]),
                "intrinsic_unique_states": int(self.episode_unique_counts[index]),
                "noveld_unique_states": int(self.episode_unique_counts[index]),
            }
            self.episode_seen[index] = set()
            key = _obs_hash(_obs_item(next_obs_batch, index))
            self.episode_seen[index].add(key)
            self.episode_intrinsic_returns[index] = 0.0
            self.episode_unique_counts[index] = 1

        return NovelDStepResult(
            intrinsic_rewards=rewards,
            raw_rewards=raw_np,
            prediction_errors=err_next_np,
            first_visit=first_visit,
            episode_metrics=episode_metrics,
        )

    def update(self) -> dict[str, float]:
        if not self.train_obs:
            self.last_update_stats = {"noveld_loss": 0.0}
            return self.last_update_stats

        import torch
        import torch.nn.functional as F

        obs = torch.cat(self.train_obs, dim=0).to(self.device)
        self.train_obs.clear()
        indices = torch.randperm(obs.shape[0], device=self.device)
        losses: list[float] = []
        self.predictor.train()

        for _ in range(self.update_epochs):
            for start in range(0, obs.shape[0], self.train_batch_size):
                idx = indices[start : start + self.train_batch_size]
                batch = obs[idx]
                with torch.no_grad():
                    target = self.target(batch)
                pred = self.predictor(batch)
                loss = F.mse_loss(pred, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.detach().cpu()))

        self.last_update_stats = {"noveld_loss": float(np.mean(losses)) if losses else 0.0}
        return self.last_update_stats

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "noveld_target_state_dict": self.target.state_dict(),
            "noveld_predictor_state_dict": self.predictor.state_dict(),
            "noveld_optimizer_state_dict": self.optimizer.state_dict(),
        }


class RNDReward(NovelDReward):
    """Random Network Distillation baseline without NovelD's difference or first-visit gate."""

    def step(
        self,
        current_obs_tensor: Any,
        next_obs_tensor: Any,
        next_obs_batch: Any,
        done: np.ndarray,
        actions: np.ndarray | None = None,
    ) -> NovelDStepResult:
        import torch

        del current_obs_tensor
        with torch.no_grad():
            target_next = self.target(next_obs_tensor)
            pred_next = self.predictor(next_obs_tensor)
            err_next = torch.norm(pred_next - target_next, dim=1, p=2)

        raw_np = err_next.detach().cpu().numpy().astype(np.float32)
        first_visit = np.zeros(self.num_envs, dtype=np.float32)
        episode_metrics: dict[int, dict[str, Any]] = {}

        for index in range(self.num_envs):
            key = _obs_hash(_obs_item(next_obs_batch, index))
            if key not in self.episode_seen[index]:
                first_visit[index] = 1.0
                self.episode_seen[index].add(key)
                self.episode_unique_counts[index] += 1

        rewards = self.coef * raw_np
        rewards = np.clip(rewards, self.reward_clip_min, self.reward_clip_max).astype(np.float32)
        self.episode_intrinsic_returns += rewards
        self.train_obs.append(next_obs_tensor.detach().cpu())

        for index, is_done in enumerate(done):
            if not bool(is_done):
                continue
            episode_metrics[index] = {
                "intrinsic_return": float(self.episode_intrinsic_returns[index]),
                "intrinsic_unique_states": int(self.episode_unique_counts[index]),
            }
            self.episode_seen[index] = set()
            key = _obs_hash(_obs_item(next_obs_batch, index))
            self.episode_seen[index].add(key)
            self.episode_intrinsic_returns[index] = 0.0
            self.episode_unique_counts[index] = 1

        return NovelDStepResult(
            intrinsic_rewards=rewards,
            raw_rewards=raw_np,
            prediction_errors=raw_np,
            first_visit=first_visit,
            episode_metrics=episode_metrics,
        )

    def update(self) -> dict[str, float]:
        stats = super().update()
        return {"rnd_loss": float(stats.get("noveld_loss", 0.0))}

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "rnd_target_state_dict": self.target.state_dict(),
            "rnd_predictor_state_dict": self.predictor.state_dict(),
            "rnd_optimizer_state_dict": self.optimizer.state_dict(),
        }


class ICMReward:
    """Intrinsic Curiosity Module baseline with inverse and forward dynamics losses."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        action_dim: int,
        num_envs: int,
        device: Any,
        coef: float = 0.05,
        learning_rate: float = 1.0e-4,
        embedding_dim: int = 128,
        reward_clip_min: float = 0.0,
        reward_clip_max: float = 1.0,
        update_epochs: int = 1,
        train_batch_size: int = 256,
    ):
        import torch

        self.num_envs = num_envs
        self.action_dim = action_dim
        self.device = device
        self.coef = coef
        self.reward_clip_min = reward_clip_min
        self.reward_clip_max = reward_clip_max
        self.update_epochs = update_epochs
        self.train_batch_size = train_batch_size

        self.encoder = _build_embedding_network(obs_shape, embedding_dim).to(device)
        self.inverse = _build_mlp(embedding_dim * 2, 256, action_dim).to(device)
        self.forward = _build_mlp(embedding_dim + action_dim, 256, embedding_dim).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.inverse.parameters()) + list(self.forward.parameters()),
            lr=learning_rate,
        )

        self.episode_seen: list[set[bytes]] = [set() for _ in range(num_envs)]
        self.episode_intrinsic_returns = np.zeros(num_envs, dtype=np.float32)
        self.episode_unique_counts = np.zeros(num_envs, dtype=np.int64)
        self.train_current_obs: list[Any] = []
        self.train_next_obs: list[Any] = []
        self.train_actions: list[Any] = []
        self.last_update_stats: dict[str, float] = {}

    def reset(self, obs_batch: Any) -> None:
        self.episode_seen = [set() for _ in range(self.num_envs)]
        self.episode_intrinsic_returns.fill(0.0)
        self.episode_unique_counts.fill(0)
        for index in range(self.num_envs):
            key = _obs_hash(_obs_item(obs_batch, index))
            self.episode_seen[index].add(key)
            self.episode_unique_counts[index] += 1

    def _actions_tensor(self, actions: np.ndarray | None) -> Any:
        import torch

        if actions is None:
            raise ValueError("ICM/RIDE intrinsic rewards require transition actions.")
        return torch.as_tensor(actions, dtype=torch.long, device=self.device)

    def _one_hot(self, actions: Any) -> Any:
        import torch.nn.functional as F

        return F.one_hot(actions, num_classes=self.action_dim).float()

    def step(
        self,
        current_obs_tensor: Any,
        next_obs_tensor: Any,
        next_obs_batch: Any,
        done: np.ndarray,
        actions: np.ndarray | None = None,
    ) -> NovelDStepResult:
        import torch

        actions_tensor = self._actions_tensor(actions)
        with torch.no_grad():
            current_features = self.encoder(current_obs_tensor)
            next_features = self.encoder(next_obs_tensor)
            pred_next_features = self.forward(torch.cat([current_features, self._one_hot(actions_tensor)], dim=1))
            forward_error = torch.norm(pred_next_features - next_features, dim=1, p=2)

        raw_np = forward_error.detach().cpu().numpy().astype(np.float32)
        first_visit = np.zeros(self.num_envs, dtype=np.float32)
        episode_metrics: dict[int, dict[str, Any]] = {}

        for index in range(self.num_envs):
            key = _obs_hash(_obs_item(next_obs_batch, index))
            if key not in self.episode_seen[index]:
                first_visit[index] = 1.0
                self.episode_seen[index].add(key)
                self.episode_unique_counts[index] += 1

        rewards = self.coef * raw_np
        rewards = np.clip(rewards, self.reward_clip_min, self.reward_clip_max).astype(np.float32)
        self.episode_intrinsic_returns += rewards
        self.train_current_obs.append(current_obs_tensor.detach().cpu())
        self.train_next_obs.append(next_obs_tensor.detach().cpu())
        self.train_actions.append(actions_tensor.detach().cpu())

        for index, is_done in enumerate(done):
            if not bool(is_done):
                continue
            episode_metrics[index] = {
                "intrinsic_return": float(self.episode_intrinsic_returns[index]),
                "intrinsic_unique_states": int(self.episode_unique_counts[index]),
            }
            self.episode_seen[index] = set()
            key = _obs_hash(_obs_item(next_obs_batch, index))
            self.episode_seen[index].add(key)
            self.episode_intrinsic_returns[index] = 0.0
            self.episode_unique_counts[index] = 1

        return NovelDStepResult(
            intrinsic_rewards=rewards,
            raw_rewards=raw_np,
            prediction_errors=raw_np,
            first_visit=first_visit,
            episode_metrics=episode_metrics,
        )

    def update(self) -> dict[str, float]:
        if not self.train_current_obs:
            self.last_update_stats = {"icm_loss": 0.0, "icm_forward_loss": 0.0, "icm_inverse_loss": 0.0}
            return self.last_update_stats

        import torch
        import torch.nn.functional as F

        current_obs = torch.cat(self.train_current_obs, dim=0).to(self.device)
        next_obs = torch.cat(self.train_next_obs, dim=0).to(self.device)
        actions = torch.cat(self.train_actions, dim=0).to(self.device)
        self.train_current_obs.clear()
        self.train_next_obs.clear()
        self.train_actions.clear()

        indices = torch.randperm(current_obs.shape[0], device=self.device)
        total_losses: list[float] = []
        forward_losses: list[float] = []
        inverse_losses: list[float] = []
        self.encoder.train()
        self.inverse.train()
        self.forward.train()

        for _ in range(self.update_epochs):
            for start in range(0, current_obs.shape[0], self.train_batch_size):
                idx = indices[start : start + self.train_batch_size]
                current_batch = current_obs[idx]
                next_batch = next_obs[idx]
                action_batch = actions[idx]

                current_features = self.encoder(current_batch)
                next_features = self.encoder(next_batch)
                inverse_logits = self.inverse(torch.cat([current_features, next_features], dim=1))
                pred_next_features = self.forward(torch.cat([current_features, self._one_hot(action_batch)], dim=1))

                inverse_loss = F.cross_entropy(inverse_logits, action_batch)
                forward_loss = F.mse_loss(pred_next_features, next_features.detach())
                loss = forward_loss + 0.2 * inverse_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_losses.append(float(loss.detach().cpu()))
                forward_losses.append(float(forward_loss.detach().cpu()))
                inverse_losses.append(float(inverse_loss.detach().cpu()))

        self.last_update_stats = {
            "icm_loss": float(np.mean(total_losses)) if total_losses else 0.0,
            "icm_forward_loss": float(np.mean(forward_losses)) if forward_losses else 0.0,
            "icm_inverse_loss": float(np.mean(inverse_losses)) if inverse_losses else 0.0,
        }
        return self.last_update_stats

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "icm_encoder_state_dict": self.encoder.state_dict(),
            "icm_inverse_state_dict": self.inverse.state_dict(),
            "icm_forward_state_dict": self.forward.state_dict(),
            "icm_optimizer_state_dict": self.optimizer.state_dict(),
        }


class RIDEReward(ICMReward):
    """RIDE-style reward from feature-space impact with episodic visit-count normalization."""

    def reset(self, obs_batch: Any) -> None:
        super().reset(obs_batch)
        self.episode_counts: list[dict[bytes, int]] = [dict() for _ in range(self.num_envs)]
        for index in range(self.num_envs):
            key = _obs_hash(_obs_item(obs_batch, index))
            self.episode_counts[index][key] = 1

    def step(
        self,
        current_obs_tensor: Any,
        next_obs_tensor: Any,
        next_obs_batch: Any,
        done: np.ndarray,
        actions: np.ndarray | None = None,
    ) -> NovelDStepResult:
        import torch

        actions_tensor = self._actions_tensor(actions)
        with torch.no_grad():
            current_features = self.encoder(current_obs_tensor)
            next_features = self.encoder(next_obs_tensor)
            impact = torch.norm(next_features - current_features, dim=1, p=2)

        raw_np = impact.detach().cpu().numpy().astype(np.float32)
        first_visit = np.zeros(self.num_envs, dtype=np.float32)
        counts = np.ones(self.num_envs, dtype=np.float32)
        episode_metrics: dict[int, dict[str, Any]] = {}

        for index in range(self.num_envs):
            key = _obs_hash(_obs_item(next_obs_batch, index))
            previous_count = self.episode_counts[index].get(key, 0)
            if previous_count == 0:
                first_visit[index] = 1.0
                self.episode_seen[index].add(key)
                self.episode_unique_counts[index] += 1
            current_count = previous_count + 1
            self.episode_counts[index][key] = current_count
            counts[index] = float(current_count)

        rewards = self.coef * raw_np / np.sqrt(counts)
        rewards = np.clip(rewards, self.reward_clip_min, self.reward_clip_max).astype(np.float32)
        self.episode_intrinsic_returns += rewards
        self.train_current_obs.append(current_obs_tensor.detach().cpu())
        self.train_next_obs.append(next_obs_tensor.detach().cpu())
        self.train_actions.append(actions_tensor.detach().cpu())

        for index, is_done in enumerate(done):
            if not bool(is_done):
                continue
            episode_metrics[index] = {
                "intrinsic_return": float(self.episode_intrinsic_returns[index]),
                "intrinsic_unique_states": int(self.episode_unique_counts[index]),
            }
            self.episode_seen[index] = set()
            self.episode_counts[index] = {}
            key = _obs_hash(_obs_item(next_obs_batch, index))
            self.episode_seen[index].add(key)
            self.episode_counts[index][key] = 1
            self.episode_intrinsic_returns[index] = 0.0
            self.episode_unique_counts[index] = 1

        return NovelDStepResult(
            intrinsic_rewards=rewards,
            raw_rewards=raw_np,
            prediction_errors=raw_np,
            first_visit=first_visit,
            episode_metrics=episode_metrics,
        )

    def update(self) -> dict[str, float]:
        stats = super().update()
        return {
            "ride_loss": float(stats.get("icm_loss", 0.0)),
            "ride_forward_loss": float(stats.get("icm_forward_loss", 0.0)),
            "ride_inverse_loss": float(stats.get("icm_inverse_loss", 0.0)),
        }

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "ride_encoder_state_dict": self.encoder.state_dict(),
            "ride_inverse_state_dict": self.inverse.state_dict(),
            "ride_forward_state_dict": self.forward.state_dict(),
            "ride_optimizer_state_dict": self.optimizer.state_dict(),
        }
