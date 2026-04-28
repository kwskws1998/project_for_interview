"""MiniGrid diagnostics for subgoal ordering and reward alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

DISTRACTOR_TYPES = {"ball", "box", "lava"}


def _safe_int_tuple(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        values = [int(item) for item in value]
    except TypeError:
        return None
    if len(values) < 2:
        return None
    return values[0], values[1]


def _object_positions(env: Any) -> dict[str, list[dict[str, Any]]]:
    unwrapped = getattr(env, "unwrapped", env)
    grid = getattr(unwrapped, "grid", None)
    width = int(getattr(unwrapped, "width", 0) or 0)
    height = int(getattr(unwrapped, "height", 0) or 0)
    objects: dict[str, list[dict[str, Any]]] = {}
    if grid is None:
        return objects
    for y in range(height):
        for x in range(width):
            obj = grid.get(x, y)
            if obj is None:
                continue
            obj_type = getattr(obj, "type", "unknown")
            record = {
                "x": x,
                "y": y,
                "color": getattr(obj, "color", ""),
                "is_locked": bool(getattr(obj, "is_locked", False)),
                "is_open": bool(getattr(obj, "is_open", False)),
            }
            objects.setdefault(obj_type, []).append(record)
    return objects


def _min_distance(agent_pos: tuple[int, int] | None, objects: list[dict[str, Any]]) -> float:
    if agent_pos is None or not objects:
        return None
    ax, ay = agent_pos
    return float(min(abs(ax - item["x"]) + abs(ay - item["y"]) for item in objects))


def _visible_counts(obs: Any) -> dict[str, int]:
    if not isinstance(obs, dict) or "image" not in obs:
        return {}
    try:
        from minigrid.core.constants import OBJECT_TO_IDX
    except ImportError:
        return {}
    object_by_idx = {idx: name for name, idx in OBJECT_TO_IDX.items()}
    image = np.asarray(obs["image"])
    counts: dict[str, int] = {}
    if image.ndim != 3 or image.shape[-1] < 1:
        return counts
    for obj_idx in image[..., 0].reshape(-1):
        obj = object_by_idx.get(int(obj_idx), "unknown")
        if obj in {"empty", "floor", "unseen"}:
            continue
        counts[obj] = counts.get(obj, 0) + 1
    return counts


def _carrying_type(env: Any) -> str:
    carrying = getattr(getattr(env, "unwrapped", env), "carrying", None)
    return str(getattr(carrying, "type", "")) if carrying is not None else ""


def extract_minigrid_diagnostics(env: Any, obs: Any, action: int, reward: float, done: bool) -> dict[str, Any]:
    unwrapped = getattr(env, "unwrapped", env)
    agent_pos = _safe_int_tuple(getattr(unwrapped, "agent_pos", None))
    objects = _object_positions(unwrapped)
    doors = objects.get("door", [])
    keys = objects.get("key", [])
    goals = objects.get("goal", [])
    distractors = [item for kind in DISTRACTOR_TYPES for item in objects.get(kind, [])]
    carrying = _carrying_type(unwrapped)
    visible = _visible_counts(obs)

    any_door_open = any(item["is_open"] for item in doors)
    any_door_locked = any(item["is_locked"] for item in doors)
    key_present = bool(keys)
    carrying_key = carrying == "key"

    progress_stage = 0
    if key_present:
        progress_stage = 1
    if carrying_key:
        progress_stage = 2
    if any_door_open:
        progress_stage = 3
    if reward > 0 or done and reward > 0:
        progress_stage = 4

    return {
        "agent_x": agent_pos[0] if agent_pos is not None else None,
        "agent_y": agent_pos[1] if agent_pos is not None else None,
        "carrying": carrying,
        "key_present": key_present,
        "carrying_key": carrying_key,
        "door_open": any_door_open,
        "door_locked": any_door_locked,
        "goal_present": bool(goals),
        "dist_to_key": _min_distance(agent_pos, keys),
        "dist_to_door": _min_distance(agent_pos, doors),
        "dist_to_goal": _min_distance(agent_pos, goals),
        "dist_to_distractor": _min_distance(agent_pos, distractors),
        "visible_key_count": visible.get("key", 0),
        "visible_door_count": visible.get("door", 0),
        "visible_goal_count": visible.get("goal", 0),
        "visible_distractor_count": sum(visible.get(kind, 0) for kind in DISTRACTOR_TYPES),
        "distractor_action": int(action in {3, 4, 5} and sum(visible.get(kind, 0) for kind in DISTRACTOR_TYPES) > 0),
        "progress_stage": progress_stage,
    }


@dataclass
class DiagnosticsState:
    key_first_seen_step: int | None = None
    key_pickup_step: int | None = None
    door_first_seen_step: int | None = None
    door_open_step: int | None = None
    goal_first_seen_step: int | None = None
    success_step: int | None = None
    distractor_actions: int = 0
    stage_sum: float = 0.0
    phi_sum: float = 0.0
    phi_count: int = 0

    def update(self, step_in_episode: int, diag: dict[str, Any], phi: float | None) -> None:
        if diag["visible_key_count"] > 0 and self.key_first_seen_step is None:
            self.key_first_seen_step = step_in_episode
        if diag["carrying_key"] and self.key_pickup_step is None:
            self.key_pickup_step = step_in_episode
        if diag["visible_door_count"] > 0 and self.door_first_seen_step is None:
            self.door_first_seen_step = step_in_episode
        if diag["door_open"] and self.door_open_step is None:
            self.door_open_step = step_in_episode
        if diag["visible_goal_count"] > 0 and self.goal_first_seen_step is None:
            self.goal_first_seen_step = step_in_episode
        if diag["progress_stage"] >= 4 and self.success_step is None:
            self.success_step = step_in_episode
        self.distractor_actions += int(diag["distractor_action"])
        self.stage_sum += float(diag["progress_stage"])
        if phi is not None:
            self.phi_sum += float(phi)
            self.phi_count += 1

    def episode_metrics(self, episode_length: int) -> dict[str, Any]:
        return {
            "key_first_seen_step": self.key_first_seen_step,
            "key_pickup_step": self.key_pickup_step,
            "door_first_seen_step": self.door_first_seen_step,
            "door_open_step": self.door_open_step,
            "goal_first_seen_step": self.goal_first_seen_step,
            "success_step": self.success_step,
            "distractor_actions": self.distractor_actions,
            "mean_progress_stage": self.stage_sum / max(1, episode_length),
            "mean_diag_phi": self.phi_sum / max(1, self.phi_count),
            "phi_count": self.phi_count,
        }


class DiagnosticsTracker:
    def __init__(self, num_envs: int):
        self.states = [DiagnosticsState() for _ in range(num_envs)]
        self.steps = [0 for _ in range(num_envs)]

    def step(
        self,
        envs: Any,
        obs_batch: Any,
        actions: np.ndarray,
        rewards: np.ndarray,
        done: np.ndarray,
        global_step: int,
        wall_time: float,
        phis: np.ndarray | None = None,
    ) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
        records: list[dict[str, Any]] = []
        episode_metrics: dict[int, dict[str, Any]] = {}
        for env_index in range(len(self.states)):
            obs_item = _obs_item(obs_batch, env_index)
            action = int(actions[env_index])
            reward = float(rewards[env_index])
            is_done = bool(done[env_index])
            phi = None if phis is None else float(phis[env_index])
            diag = extract_minigrid_diagnostics(envs.envs[env_index], obs_item, action, reward, is_done)
            self.steps[env_index] += 1
            self.states[env_index].update(self.steps[env_index], diag, phi)
            record = {
                "global_step": global_step,
                "env_index": env_index,
                "episode_step": self.steps[env_index],
                "wall_time": wall_time,
                "action": action,
                "extrinsic_reward": reward,
                "done": int(is_done),
                "phi": phi,
                **diag,
            }
            records.append(record)
            if is_done:
                episode_metrics[env_index] = self.states[env_index].episode_metrics(self.steps[env_index])
                self.states[env_index] = DiagnosticsState()
                self.steps[env_index] = 0
        return records, episode_metrics


def _obs_item(obs_batch: Any, index: int) -> Any:
    if isinstance(obs_batch, dict):
        return {key: value[index] for key, value in obs_batch.items()}
    return obs_batch[index]
