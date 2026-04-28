"""Scheduling logic for SLM calls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AppraisalSchedule:
    mode: str = "every_n"
    interval: int = 8

    def should_call(self, step_in_episode: int, event_triggered: bool = False) -> bool:
        if self.mode == "every_step":
            return True
        if self.mode == "every_n":
            return step_in_episode == 0 or step_in_episode % max(1, self.interval) == 0
        if self.mode == "event":
            return step_in_episode == 0 or event_triggered
        if self.mode == "event_or_every_n":
            return event_triggered or step_in_episode == 0 or step_in_episode % max(1, self.interval) == 0
        raise ValueError(f"Unknown CARS schedule mode: {self.mode}")
