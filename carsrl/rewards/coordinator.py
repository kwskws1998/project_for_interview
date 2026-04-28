"""Intrinsic reward coordinator for PPO rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from carsrl.config import ExperimentConfig
from carsrl.rewards.noveld import ICMReward, NovelDReward, RIDEReward, RNDReward


@dataclass(frozen=True)
class IntrinsicStepResult:
    intrinsic_rewards: np.ndarray
    episode_metrics: dict[int, dict[str, Any]]
    stats: dict[str, float]


class IntrinsicRewardCoordinator:
    def __init__(self, module: Any, name: str):
        self.module = module
        self.name = name
        self.last_step_stats: dict[str, float] = {}

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig,
        obs_shape: tuple[int, int, int],
        action_dim: int,
        device: Any,
    ) -> "IntrinsicRewardCoordinator":
        if config.algo not in {
            "ppo_rnd",
            "ppo_rnd_cars",
            "ppo_icm",
            "ppo_icm_cars",
            "ppo_ride",
            "ppo_ride_cars",
            "ppo_noveld",
            "ppo_noveld_cars",
        }:
            raise ValueError(f"Unsupported intrinsic algorithm: {config.algo}")
        if config.algo in {"ppo_rnd", "ppo_rnd_cars"}:
            reward_cls = RNDReward
            name = "rnd"
        elif config.algo in {"ppo_icm", "ppo_icm_cars"}:
            reward_cls = ICMReward
            name = "icm"
        elif config.algo in {"ppo_ride", "ppo_ride_cars"}:
            reward_cls = RIDEReward
            name = "ride"
        else:
            reward_cls = NovelDReward
            name = "noveld"

        kwargs: dict[str, Any] = {
            "obs_shape": obs_shape,
            "num_envs": config.ppo.num_envs,
            "device": device,
            "coef": config.intrinsic.coef,
            "learning_rate": config.intrinsic.learning_rate,
            "embedding_dim": config.intrinsic.embedding_dim,
            "reward_clip_min": config.intrinsic.reward_clip_min,
            "reward_clip_max": config.intrinsic.reward_clip_max,
            "update_epochs": config.intrinsic.update_epochs,
            "train_batch_size": config.intrinsic.train_batch_size,
        }
        if reward_cls in {ICMReward, RIDEReward}:
            kwargs["action_dim"] = action_dim
        else:
            kwargs["scale_fac"] = config.intrinsic.scale_fac

        module = reward_cls(
            **kwargs,
        )
        return cls(module, name=name)

    def reset(self, obs_batch: Any) -> None:
        self.module.reset(obs_batch)

    def step(
        self,
        current_obs_tensor: Any,
        next_obs_tensor: Any,
        next_obs_batch: Any,
        done: np.ndarray,
        actions: np.ndarray | None = None,
    ) -> IntrinsicStepResult:
        result = self.module.step(current_obs_tensor, next_obs_tensor, next_obs_batch, done, actions=actions)
        self.last_step_stats = {
            "mean_intrinsic_reward": float(np.mean(result.intrinsic_rewards)),
            f"mean_{self.name}_raw_reward": float(np.mean(result.raw_rewards)),
            f"mean_{self.name}_prediction_error": float(np.mean(result.prediction_errors)),
            f"{self.name}_first_visit_rate": float(np.mean(result.first_visit)),
        }
        return IntrinsicStepResult(
            intrinsic_rewards=result.intrinsic_rewards,
            episode_metrics=result.episode_metrics,
            stats=self.last_step_stats,
        )

    def update(self) -> dict[str, float]:
        return self.module.update()

    def checkpoint_state(self) -> dict[str, Any]:
        return self.module.checkpoint_state()
