"""Common interface for intrinsic reward baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IntrinsicRewardModule(ABC):
    @abstractmethod
    def compute(self, obs: Any, action: int, next_obs: Any, info: dict[str, Any]) -> float:
        """Return an intrinsic reward for a transition."""

    def update(self, *_: Any, **__: Any) -> dict[str, float]:
        return {}


class NoIntrinsicReward(IntrinsicRewardModule):
    def compute(self, obs: Any, action: int, next_obs: Any, info: dict[str, Any]) -> float:
        return 0.0
