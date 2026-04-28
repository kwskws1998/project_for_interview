"""Actor-critic networks for MiniGrid PPO."""

from __future__ import annotations

import hashlib
import re
from typing import Any


MISSION_FEATURE_DIM = 32
DIRECTION_FEATURE_DIM = 4


def layer_init(layer: Any, std: float = 1.0, bias_const: float = 0.0) -> Any:
    import torch

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def build_minigrid_actor_critic(obs_shape: tuple[int, int, int], action_dim: int) -> Any:
    import torch
    from torch import nn
    from torch.distributions.categorical import Categorical

    class _MiniGridActorCritic(nn.Module):
        def __init__(self, obs_shape_: tuple[int, int, int], action_dim_: int):
            super().__init__()
            channels, height, width = obs_shape_
            self.encoder = nn.Sequential(
                layer_init(nn.Conv2d(channels, 32, kernel_size=3, padding=1)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * height * width, 256)),
                nn.ReLU(),
            )
            self.actor = layer_init(nn.Linear(256, action_dim_), std=0.01)
            self.critic = layer_init(nn.Linear(256, 1), std=1.0)

        def forward(self, obs: Any) -> tuple[Any, Any]:
            hidden = self.encoder(obs)
            return self.actor(hidden), self.critic(hidden).squeeze(-1)

        def get_value(self, obs: Any) -> Any:
            _, value = self.forward(obs)
            return value

        def get_action_and_value(self, obs: Any, action: Any | None = None) -> tuple[Any, Any, Any, Any]:
            logits, value = self.forward(obs)
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            return action, dist.log_prob(action), dist.entropy(), value

    return _MiniGridActorCritic(obs_shape, action_dim)


def _stable_bucket(token: str, size: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False) % size


def _mission_values(raw_mission: Any, batch_size: int) -> list[str]:
    import numpy as np

    if raw_mission is None:
        return [""] * batch_size
    if isinstance(raw_mission, str):
        return [raw_mission] * batch_size
    array = np.asarray(raw_mission, dtype=object)
    if array.ndim == 0:
        return [str(array.item())] * batch_size
    values = [str(item) for item in array.reshape(-1)]
    if len(values) < batch_size:
        values.extend([""] * (batch_size - len(values)))
    return values[:batch_size]


def _mission_features(raw_mission: Any, batch_size: int) -> Any:
    import numpy as np

    features = np.zeros((batch_size, MISSION_FEATURE_DIM), dtype=np.float32)
    for row, mission in enumerate(_mission_values(raw_mission, batch_size)):
        tokens = re.findall(r"[a-z0-9]+", mission.lower())
        if not tokens:
            continue
        for token in tokens:
            features[row, _stable_bucket(token, MISSION_FEATURE_DIM)] = 1.0
    return features


def _direction_features(raw_direction: Any, batch_size: int) -> Any:
    import numpy as np

    features = np.zeros((batch_size, DIRECTION_FEATURE_DIM), dtype=np.float32)
    if raw_direction is None:
        return features
    array = np.asarray(raw_direction)
    if array.ndim == 0:
        values = [array.item()] * batch_size
    else:
        values = list(array.reshape(-1))
    if len(values) < batch_size:
        values.extend([-1] * (batch_size - len(values)))
    for row, value in enumerate(values[:batch_size]):
        try:
            direction = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= direction < DIRECTION_FEATURE_DIM:
            features[row, direction] = 1.0
    return features


def obs_to_tensor(obs: Any, device: str | Any) -> Any:
    """Convert Gymnasium MiniGrid observations to normalized BCHW tensors.

    MiniGrid dict observations contain an image, a mission string, and an agent direction.
    The policy and intrinsic reward models use all three by appending direction and hashed
    mission feature planes to the image channels.
    """
    import numpy as np
    import torch

    image = obs["image"] if isinstance(obs, dict) else obs
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4:
        raise ValueError(f"Expected MiniGrid image obs with 3 or 4 dims, got shape {array.shape}")
    batch_size, height, width, _ = array.shape
    image_chw = np.ascontiguousarray(array.transpose(0, 3, 1, 2)).astype(np.float32) / 10.0

    if isinstance(obs, dict):
        direction_features = _direction_features(obs.get("direction"), batch_size)
        mission_features = _mission_features(obs.get("mission"), batch_size)
    else:
        direction_features = np.zeros((batch_size, DIRECTION_FEATURE_DIM), dtype=np.float32)
        mission_features = np.zeros((batch_size, MISSION_FEATURE_DIM), dtype=np.float32)

    symbolic_features = np.concatenate([direction_features, mission_features], axis=1)
    symbolic_planes = np.broadcast_to(
        symbolic_features[:, :, None, None],
        (batch_size, symbolic_features.shape[1], height, width),
    ).copy()
    combined = np.concatenate([image_chw, symbolic_planes], axis=1)
    return torch.as_tensor(combined, dtype=torch.float32, device=device)
