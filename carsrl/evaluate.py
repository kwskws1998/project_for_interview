"""Evaluate trained PPO-family checkpoints on MiniGrid."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from carsrl.config import experiment_config_from_dict
from carsrl.envs.minigrid import make_minigrid_env
from carsrl.ppo.model import build_minigrid_actor_critic, obs_to_tensor
from carsrl.utils.seeding import set_global_seeds


def _resolve_device(device: str) -> Any:
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_config_from_checkpoint(checkpoint: dict[str, Any]):
    config_dict = checkpoint.get("config")
    if not isinstance(config_dict, dict):
        raise ValueError("Checkpoint does not contain a config dictionary.")
    return experiment_config_from_dict(config_dict)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    episodes: int,
    seed: int | None,
    device: str,
    output_dir: str | Path | None,
) -> dict[str, Any]:
    import torch

    checkpoint_path = Path(checkpoint_path)
    resolved_device = _resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    config = _load_config_from_checkpoint(checkpoint)
    eval_seed = config.seed if seed is None else seed
    set_global_seeds(eval_seed)

    env = make_minigrid_env(config.env_id, seed=eval_seed)
    obs, _ = env.reset(seed=eval_seed)
    obs_tensor = obs_to_tensor(obs, resolved_device)
    obs_shape = tuple(int(dim) for dim in obs_tensor.shape[1:])
    action_dim = env.action_space.n
    model = build_minigrid_actor_critic(obs_shape, action_dim).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if output_dir is None:
        output_path = checkpoint_path.parent / "evaluation"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    start_time = time.time()
    for episode_idx in range(episodes):
        obs, _ = env.reset(seed=eval_seed + episode_idx)
        done = False
        episode_return = 0.0
        length = 0
        while not done:
            obs_tensor = obs_to_tensor(obs, resolved_device)
            with torch.inference_mode():
                logits, _ = model.forward(obs_tensor)
                action = int(torch.argmax(logits, dim=1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            length += 1
            done = bool(terminated or truncated)
        rows.append(
            {
                "episode": episode_idx,
                "return": episode_return,
                "length": length,
                "success": float(episode_return > 0.0),
            }
        )
    env.close()

    metrics_path = output_path / "eval_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", "return", "length", "success"])
        writer.writeheader()
        writer.writerows(rows)

    returns = np.asarray([row["return"] for row in rows], dtype=np.float32)
    lengths = np.asarray([row["length"] for row in rows], dtype=np.float32)
    successes = np.asarray([row["success"] for row in rows], dtype=np.float32)
    summary = {
        "checkpoint": str(checkpoint_path),
        "env_id": config.env_id,
        "algo": config.algo,
        "seed": eval_seed,
        "episodes": episodes,
        "mean_return": float(np.mean(returns)) if len(returns) else 0.0,
        "std_return": float(np.std(returns)) if len(returns) else 0.0,
        "success_rate": float(np.mean(successes)) if len(successes) else 0.0,
        "mean_length": float(np.mean(lengths)) if len(lengths) else 0.0,
        "wall_time": time.time() - start_time,
        "metrics_path": str(metrics_path),
    }
    with (output_path / "eval_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained CARS/PPO checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
