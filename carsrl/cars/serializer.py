"""MiniGrid state serialization for frozen SLM appraisal."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


FALLBACK_OBJECT_BY_IDX = {
    0: "unseen",
    1: "empty",
    2: "wall",
    3: "floor",
    4: "door",
    5: "key",
    6: "ball",
    7: "box",
    8: "goal",
    9: "lava",
    10: "agent",
}

FALLBACK_COLOR_BY_IDX = {
    0: "red",
    1: "green",
    2: "blue",
    3: "purple",
    4: "yellow",
    5: "grey",
}

FALLBACK_STATE_BY_IDX = {
    0: "open",
    1: "closed",
    2: "locked",
}

ACTION_BY_IDX = {
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}

DIRECTION_BY_IDX = {
    0: "right/east",
    1: "down/south",
    2: "left/west",
    3: "up/north",
}


def _load_minigrid_maps() -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
    try:
        from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
    except ImportError:
        return FALLBACK_OBJECT_BY_IDX, FALLBACK_COLOR_BY_IDX, FALLBACK_STATE_BY_IDX
    object_by_idx = {idx: name for name, idx in OBJECT_TO_IDX.items()}
    color_by_idx = {idx: name for name, idx in COLOR_TO_IDX.items()}
    state_by_idx = {idx: name for name, idx in STATE_TO_IDX.items()}
    return object_by_idx, color_by_idx, state_by_idx


def _action_name(action: int | None) -> str:
    if action is None:
        return "none"
    return ACTION_BY_IDX.get(int(action), str(action))


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default) if obj is not None else default


def _format_carrying(carrying: Any) -> str:
    if carrying is None:
        return "nothing"
    color = _safe_getattr(carrying, "color", None)
    kind = _safe_getattr(carrying, "type", None)
    if color and kind:
        return f"{color} {kind}"
    return str(carrying)


def _format_position(agent_pos: Any) -> str:
    if agent_pos is None:
        return "unknown"
    try:
        values = [int(value) for value in agent_pos]
    except TypeError:
        return str(agent_pos)
    return f"({values[0]}, {values[1]})" if len(values) >= 2 else str(tuple(values))


def _iter_image_objects(image: np.ndarray) -> Iterable[tuple[int, int, int, int, int]]:
    if image.ndim != 3 or image.shape[-1] < 3:
        return []
    objects: list[tuple[int, int, int, int, int]] = []
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            obj_idx = int(image[row, col, 0])
            color_idx = int(image[row, col, 1])
            state_idx = int(image[row, col, 2])
            objects.append((row, col, obj_idx, color_idx, state_idx))
    return objects


def _visible_object_summary(obs: Any, max_items: int = 24) -> list[str]:
    if not isinstance(obs, dict) or "image" not in obs:
        return []
    object_by_idx, color_by_idx, state_by_idx = _load_minigrid_maps()
    image = np.asarray(obs["image"])
    items: list[str] = []
    for row, col, obj_idx, color_idx, state_idx in _iter_image_objects(image):
        obj = object_by_idx.get(obj_idx, f"object_{obj_idx}")
        if obj in {"unseen", "empty", "floor"}:
            continue
        color = color_by_idx.get(color_idx, f"color_{color_idx}")
        state = state_by_idx.get(state_idx, f"state_{state_idx}")
        if obj == "door":
            label = f"{state} {color} door at view({row},{col})"
        elif obj in {"key", "ball", "box"}:
            label = f"{color} {obj} at view({row},{col})"
        else:
            label = f"{obj} at view({row},{col})"
        items.append(label)
        if len(items) >= max_items:
            break
    return items


@dataclass
class EpisodeTrace:
    history_length: int = 8
    action_history: list[int] = field(default_factory=list)
    recent_events: list[str] = field(default_factory=list)
    last_action: int | None = None

    def reset(self) -> None:
        self.action_history.clear()
        self.recent_events.clear()
        self.last_action = None

    def observe_transition(
        self,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        self.last_action = action
        self.action_history.append(action)
        del self.action_history[:-self.history_length]

        events = []
        if reward > 0:
            events.append(f"received positive extrinsic reward {reward:.3f}")
        if terminated:
            events.append("episode terminated")
        if truncated:
            events.append("episode truncated")
        if info:
            if info.get("success") is True:
                events.append("mission success")
            if info.get("picked_up") is not None:
                events.append(f"picked up {info['picked_up']}")
        self.recent_events.extend(events)
        del self.recent_events[:-self.history_length]


@dataclass
class MiniGridStateSerializer:
    include_history: bool = False
    history_length: int = 8
    max_visible_objects: int = 24

    def serialize(
        self,
        env: Any,
        obs: Any,
        trace: EpisodeTrace | None = None,
        last_action: int | None = None,
        recent_events: list[str] | None = None,
        action_history: list[int] | None = None,
    ) -> str:
        unwrapped = _safe_getattr(env, "unwrapped", env)
        mission = self._mission(unwrapped, obs)
        carrying = _format_carrying(_safe_getattr(unwrapped, "carrying", None))
        agent_pos = _safe_getattr(unwrapped, "agent_pos", None)
        agent_dir = _safe_getattr(unwrapped, "agent_dir", None)
        direction = DIRECTION_BY_IDX.get(int(agent_dir), str(agent_dir)) if agent_dir is not None else "unknown"

        if trace is not None:
            last_action = trace.last_action if last_action is None else last_action
            recent_events = trace.recent_events if recent_events is None else recent_events
            action_history = trace.action_history if action_history is None else action_history

        visible = _visible_object_summary(obs, max_items=self.max_visible_objects)
        visible_text = "; ".join(visible) if visible else "none listed"

        lines = [
            f"Mission: {mission}",
            f"Agent position: {_format_position(agent_pos)}",
            f"Agent direction: {direction}",
            f"Agent inventory: carrying {carrying}",
            f"Visible objects: {visible_text}",
        ]

        if self.include_history:
            events = recent_events or []
            actions = action_history or []
            lines.append(f"Last action: {_action_name(last_action)}")
            action_names = [_action_name(action) for action in actions[-self.history_length :]]
            lines.append(f"Recent actions: {', '.join(action_names) if action_names else 'none'}")
            lines.append(f"Recent events: {'; '.join(events[-self.history_length:]) if events else 'none'}")
        return "\n".join(lines)

    @staticmethod
    def _mission(unwrapped: Any, obs: Any) -> str:
        if isinstance(obs, dict) and obs.get("mission"):
            return str(obs["mission"])
        mission = _safe_getattr(unwrapped, "mission", None)
        return str(mission) if mission else "unknown"
