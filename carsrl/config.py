"""Configuration helpers for CARS experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 2.5e-4
    rollout_steps: int = 128
    num_envs: int = 8
    update_epochs: int = 4
    minibatch_size: int = 256
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass(frozen=True)
class CARSConfig:
    enabled: bool = True
    appraiser: str = "qwen"
    beta: float = 0.1
    gamma: float = 0.99
    reward_clip_min: float = -0.05
    reward_clip_max: float = 0.05
    use_confidence: bool = True
    direct_reward: bool = False
    schedule: str = "every_n"
    interval: int = 8
    neutral_on_skip: bool = True
    include_history: bool = False
    history_length: int = 8
    cache_path: str = "runs/cache/cars_appraisals.jsonl"


@dataclass(frozen=True)
class SLMConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    device: str = "auto"
    dtype: str = "auto"
    load_in_4bit: bool = False
    local_files_only: bool = False
    max_new_tokens: int = 128
    temperature: float = 0.0


@dataclass(frozen=True)
class IntrinsicConfig:
    coef: float = 0.05
    learning_rate: float = 1.0e-4
    embedding_dim: int = 128
    scale_fac: float = 0.5
    reward_clip_min: float = 0.0
    reward_clip_max: float = 1.0
    update_epochs: int = 1
    train_batch_size: int = 256


@dataclass(frozen=True)
class LoggingConfig:
    run_dir: str = "runs"
    csv_name: str = "metrics.csv"
    jsonl_name: str = "events.jsonl"
    diagnostics_name: str = "diagnostics.jsonl"
    diagnostics_interval: int = 1


@dataclass(frozen=True)
class ExperimentConfig:
    algo: str = "ppo_cars"
    env_id: str = "MiniGrid-DoorKey-8x8-v0"
    seed: int = 0
    total_steps: int = 1_000_000
    eval_interval: int = 25_000
    save_interval: int = 100_000
    ppo: PPOConfig = field(default_factory=PPOConfig)
    cars: CARSConfig = field(default_factory=CARSConfig)
    slm: SLMConfig = field(default_factory=SLMConfig)
    intrinsic: IntrinsicConfig = field(default_factory=IntrinsicConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load YAML when PyYAML is installed."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML config files.") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def experiment_config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    defaults = ExperimentConfig()
    base = {
        "algo": defaults.algo,
        "env_id": defaults.env_id,
        "seed": defaults.seed,
        "total_steps": defaults.total_steps,
        "eval_interval": defaults.eval_interval,
        "save_interval": defaults.save_interval,
        "ppo": PPOConfig().__dict__,
        "cars": CARSConfig().__dict__,
        "slm": SLMConfig().__dict__,
        "intrinsic": IntrinsicConfig().__dict__,
        "logging": LoggingConfig().__dict__,
    }
    merged = _merge_dict(base, data)
    return ExperimentConfig(
        algo=merged["algo"],
        env_id=merged["env_id"],
        seed=int(merged["seed"]),
        total_steps=int(merged["total_steps"]),
        eval_interval=int(merged["eval_interval"]),
        save_interval=int(merged["save_interval"]),
        ppo=PPOConfig(**merged["ppo"]),
        cars=CARSConfig(**merged["cars"]),
        slm=SLMConfig(**merged["slm"]),
        intrinsic=IntrinsicConfig(**merged["intrinsic"]),
        logging=LoggingConfig(**merged["logging"]),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return experiment_config_from_dict(load_yaml_config(path))
