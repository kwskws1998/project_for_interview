"""MiniGrid environment factories for Gymnasium/Farama APIs."""

from __future__ import annotations

from typing import Any


def make_minigrid_env(env_id: str, seed: int | None = None, render_mode: str | None = None) -> Any:
    import gymnasium as gym
    import minigrid  # noqa: F401

    env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)
    return env


def make_minigrid_env_thunk(env_id: str, seed: int, rank: int = 0) -> Any:
    def thunk() -> Any:
        import gymnasium as gym
        import minigrid  # noqa: F401

        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed + rank)
        return env

    return thunk


def make_vector_minigrid_env(env_id: str, num_envs: int, seed: int) -> Any:
    import gymnasium as gym
    from gymnasium.vector import AutoresetMode

    return gym.vector.SyncVectorEnv(
        [make_minigrid_env_thunk(env_id, seed, rank) for rank in range(num_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
