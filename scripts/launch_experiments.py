"""Build and optionally execute reproducible experiment sweeps."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return data


def _select_envs(envs_config: dict[str, Any], suite: str, explicit_envs: list[str] | None) -> list[str]:
    if explicit_envs:
        return explicit_envs
    envs = envs_config.get(suite)
    if not isinstance(envs, list):
        raise ValueError(f"Unknown environment suite: {suite}")
    return [str(env) for env in envs]


def _select_seeds(envs_config: dict[str, Any], explicit_seeds: list[int] | None) -> list[int]:
    if explicit_seeds:
        return explicit_seeds
    seeds = envs_config.get("seeds", [0, 1, 2, 3])
    return [int(seed) for seed in seeds]


def _command_for(
    algo: str,
    env_id: str,
    seed: int,
    args: argparse.Namespace,
    beta: float | None = None,
) -> list[str]:
    cars_like = {
        "ppo_cars",
        "ppo_rnd_cars",
        "ppo_icm_cars",
        "ppo_ride_cars",
        "ppo_noveld_cars",
        "ppo_random_phi",
        "ppo_shuffled_phi",
        "ppo_heuristic_phi",
        "ppo_cars_no_confidence",
        "ppo_cars_direct",
    }
    intrinsic_like = {
        "ppo_rnd",
        "ppo_rnd_cars",
        "ppo_icm",
        "ppo_icm_cars",
        "ppo_ride",
        "ppo_ride_cars",
        "ppo_noveld",
        "ppo_noveld_cars",
    }
    command = [
        sys.executable,
        "-m",
        "carsrl.train",
        "--config",
        args.config,
        "--algo",
        algo,
        "--env",
        env_id,
        "--seed",
        str(seed),
        "--total_steps",
        str(args.total_steps),
        "--num_envs",
        str(args.num_envs),
        "--rollout_steps",
        str(args.rollout_steps),
        "--minibatch_size",
        str(args.minibatch_size),
        "--update_epochs",
        str(args.update_epochs),
        "--run_dir",
        args.run_dir,
    ]
    if args.device:
        command.extend(["--device", args.device])
    if args.wandb:
        command.append("--wandb")
        command.extend(["--wandb_project", args.wandb_project])
        if args.wandb_entity:
            command.extend(["--wandb_entity", args.wandb_entity])
        group = args.wandb_group or f"{args.suite}_{algo}"
        command.extend(["--wandb_group", group])
        beta_suffix = f"_beta{beta}" if beta is not None else ""
        command.extend(["--wandb_name", f"{algo}_{env_id}_seed{seed}{beta_suffix}"])
        command.extend(["--wandb_mode", args.wandb_mode])
    if algo in cars_like:
        appraiser = args.cars_appraiser
        if algo == "ppo_random_phi":
            appraiser = "random"
        elif algo == "ppo_shuffled_phi" and appraiser == "qwen":
            appraiser = "shuffled_qwen"
        elif algo == "ppo_heuristic_phi":
            appraiser = "heuristic"
        command.extend(["--cars_appraiser", appraiser])
        command.extend(["--cars_schedule", args.cars_schedule])
        command.extend(["--slm_model", args.slm_model])
        command.extend(["--slm_device", args.slm_device])
        command.extend(["--slm_interval", str(args.slm_interval)])
        command.extend(["--slm_dtype", args.slm_dtype])
        command.extend(["--slm_max_new_tokens", str(args.slm_max_new_tokens)])
        command.extend(["--slm_temperature", str(args.slm_temperature)])
        if args.slm_load_in_4bit:
            command.append("--slm_load_in_4bit")
        if args.slm_local_files_only:
            command.append("--slm_local_files_only")
        if beta is not None:
            command.extend(["--beta", str(beta)])
        command.extend(["--cars_cache_path", f"cars_cache_seed{seed}.jsonl"])
        if algo == "ppo_cars_no_confidence":
            command.append("--no_cars_confidence")
        if algo == "ppo_cars_direct":
            command.append("--cars_direct_reward")
        if args.cars_history:
            command.append("--cars_history")
        if args.no_cars_history:
            command.append("--no_cars_history")
        if args.cars_shape_on_skip:
            command.append("--cars_shape_on_skip")
    if algo in intrinsic_like:
        command.extend(["--intrinsic_coef", str(args.intrinsic_coef)])
        command.extend(["--noveld_scale_fac", str(args.noveld_scale_fac)])
    return command


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    envs_config = _load_yaml(Path(args.envs_config))
    envs = _select_envs(envs_config, args.suite, args.envs)
    seeds = _select_seeds(envs_config, args.seeds)
    beta_values = [float(item) for item in envs_config.get("beta_ablation", [args.beta])] if args.beta_ablation else [args.beta]

    commands: list[list[str]] = []
    cars_like = {
        "ppo_cars",
        "ppo_rnd_cars",
        "ppo_icm_cars",
        "ppo_ride_cars",
        "ppo_noveld_cars",
        "ppo_random_phi",
        "ppo_shuffled_phi",
        "ppo_heuristic_phi",
        "ppo_cars_no_confidence",
        "ppo_cars_direct",
    }
    for env_id in envs:
        for algo in args.algos:
            for seed in seeds:
                if algo in cars_like:
                    for beta in beta_values:
                        commands.append(_command_for(algo, env_id, seed, args, beta=beta))
                else:
                    commands.append(_command_for(algo, env_id, seed, args))
    return commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch CARS/PPO/NovelD experiment sweeps.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--envs_config", default="configs/envs.yaml")
    parser.add_argument("--suite", choices=["sanity", "main"], default="sanity")
    parser.add_argument("--envs", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--algos", nargs="+", default=["ppo", "ppo_rnd", "ppo_icm", "ppo_ride", "ppo_noveld", "ppo_cars"])
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--rollout_steps", type=int, default=128)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--run_dir", default="runs/experiments")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--cars_appraiser",
        choices=["qwen", "mock", "random", "heuristic", "shuffled_qwen", "shuffled_mock", "shuffled_heuristic"],
        default="qwen",
    )
    parser.add_argument("--cars_schedule", choices=["every_step", "every_n", "event", "event_or_every_n"], default="every_n")
    parser.add_argument("--slm_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--slm_device", default="auto")
    parser.add_argument("--slm_interval", type=int, default=8)
    parser.add_argument("--slm_dtype", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"], default="auto")
    parser.add_argument("--slm_max_new_tokens", type=int, default=128)
    parser.add_argument("--slm_temperature", type=float, default=0.0)
    parser.add_argument("--slm_load_in_4bit", action="store_true")
    parser.add_argument("--slm_local_files_only", action="store_true")
    parser.add_argument("--cars_history", action="store_true")
    parser.add_argument("--no_cars_history", action="store_true")
    parser.add_argument("--cars_shape_on_skip", action="store_true")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--beta_ablation", action="store_true")
    parser.add_argument("--intrinsic_coef", type=float, default=0.05)
    parser.add_argument("--noveld_scale_fac", type=float, default=0.5)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="carsrl")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Default is dry-run.")
    parser.add_argument("--stop_on_failure", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    commands = build_commands(args)
    for index, command in enumerate(commands, start=1):
        printable = " ".join(command)
        print(f"[{index}/{len(commands)}] {printable}")
        if not args.execute:
            continue
        result = subprocess.run(command, check=False)
        if result.returncode != 0 and args.stop_on_failure:
            raise SystemExit(result.returncode)
    if not args.execute:
        print("Dry-run only. Add --execute to run these commands.")


if __name__ == "__main__":
    main()
