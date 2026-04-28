"""CARS potential-based reward shaping."""

from __future__ import annotations

from dataclasses import dataclass

from carsrl.cars.schema import Appraisal


@dataclass(frozen=True)
class CARSRewardShaper:
    beta: float = 0.1
    gamma: float = 0.99
    clip_min: float = -0.05
    clip_max: float = 0.05
    use_confidence: bool = True
    direct_reward: bool = False

    def shape(self, previous: Appraisal, current: Appraisal) -> float:
        if self.direct_reward:
            base = current.phi
        else:
            base = self.gamma * current.phi - previous.phi
        confidence = current.confidence if self.use_confidence else 1.0
        reward = self.beta * confidence * base
        return max(self.clip_min, min(self.clip_max, reward))
