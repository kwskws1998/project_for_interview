"""Training entrypoint for CARS experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace

from carsrl.config import load_experiment_config


ALGO_ALIASES = {
    "ppo_random_phi": {"cars_appraiser": "random"},
    "ppo_shuffled_phi": {"cars_appraiser": "shuffled_qwen"},
    "ppo_heuristic_phi": {"cars_appraiser": "heuristic"},
    "ppo_cars_no_confidence": {"cars_use_confidence": False},
    "ppo_cars_direct": {"cars_direct_reward": True},
}

SUPPORTED_ALGOS = {
    "ppo",
    "ppo_cars",
    "ppo_rnd",
    "ppo_rnd_cars",
    "ppo_icm",
    "ppo_icm_cars",
    "ppo_ride",
    "ppo_ride_cars",
    "ppo_noveld",
    "ppo_noveld_cars",
    *ALGO_ALIASES.keys(),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO/CARS agents on MiniGrid.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--algo", default=None)
    parser.add_argument("--env", dest="env_id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument(
        "--cars_appraiser",
        choices=["qwen", "mock", "random", "heuristic", "shuffled_qwen", "shuffled_mock", "shuffled_heuristic"],
        default=None,
    )
    parser.add_argument("--cars_schedule", choices=["every_step", "every_n", "event", "event_or_every_n"], default=None)
    parser.add_argument("--cars_cache_path", default=None)
    parser.add_argument("--no_cars_confidence", action="store_true")
    parser.add_argument("--cars_direct_reward", action="store_true")
    parser.add_argument("--cars_history", action="store_true", help="Include recent actions/events in CARS state text.")
    parser.add_argument("--no_cars_history", action="store_true")
    parser.add_argument(
        "--cars_shape_on_skip",
        action="store_true",
        help="Apply shaping on scheduler-skipped steps using the stale Phi estimate.",
    )
    parser.add_argument("--intrinsic_coef", type=float, default=None)
    parser.add_argument("--noveld_scale_fac", type=float, default=None)
    parser.add_argument("--intrinsic_learning_rate", type=float, default=None)
    parser.add_argument("--slm_model", default=None)
    parser.add_argument("--slm_device", default=None)
    parser.add_argument("--slm_interval", type=int, default=None)
    parser.add_argument("--slm_dtype", default=None, choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--slm_max_new_tokens", type=int, default=None)
    parser.add_argument("--slm_temperature", type=float, default=None)
    parser.add_argument("--slm_load_in_4bit", action="store_true")
    parser.add_argument("--slm_local_files_only", action="store_true")
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--rollout_steps", type=int, default=None)
    parser.add_argument("--minibatch_size", type=int, default=None)
    parser.add_argument("--update_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--wandb", action="store_true", help="Log training metrics to Weights & Biases.")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true", help="Run the lightweight CARS core smoke check.")
    return parser


def apply_overrides(config_path: str, args: argparse.Namespace):
    cfg = load_experiment_config(config_path)
    if args.algo is not None:
        cfg = replace(cfg, algo=args.algo)
    alias = ALGO_ALIASES.get(cfg.algo, {})
    if args.env_id is not None:
        cfg = replace(cfg, env_id=args.env_id)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if args.total_steps is not None:
        cfg = replace(cfg, total_steps=args.total_steps)
    if args.beta is not None:
        cfg = replace(cfg, cars=replace(cfg.cars, beta=args.beta))
    cars_appraiser = args.cars_appraiser if args.cars_appraiser is not None else alias.get("cars_appraiser")
    if cars_appraiser is not None:
        cfg = replace(cfg, cars=replace(cfg.cars, appraiser=cars_appraiser))
    if args.cars_schedule is not None:
        cfg = replace(cfg, cars=replace(cfg.cars, schedule=args.cars_schedule))
    if args.cars_cache_path is not None:
        cfg = replace(cfg, cars=replace(cfg.cars, cache_path=args.cars_cache_path))
    cars_use_confidence = alias.get("cars_use_confidence")
    if args.no_cars_confidence:
        cars_use_confidence = False
    if cars_use_confidence is not None:
        cfg = replace(cfg, cars=replace(cfg.cars, use_confidence=bool(cars_use_confidence)))
    cars_direct_reward = bool(alias.get("cars_direct_reward", False) or args.cars_direct_reward)
    if cars_direct_reward:
        cfg = replace(cfg, cars=replace(cfg.cars, direct_reward=True))
    if args.cars_history:
        cfg = replace(cfg, cars=replace(cfg.cars, include_history=True))
    if args.no_cars_history:
        cfg = replace(cfg, cars=replace(cfg.cars, include_history=False))
    if args.cars_shape_on_skip:
        cfg = replace(cfg, cars=replace(cfg.cars, neutral_on_skip=False))
    if args.intrinsic_coef is not None:
        cfg = replace(cfg, intrinsic=replace(cfg.intrinsic, coef=args.intrinsic_coef))
    if args.noveld_scale_fac is not None:
        cfg = replace(cfg, intrinsic=replace(cfg.intrinsic, scale_fac=args.noveld_scale_fac))
    if args.intrinsic_learning_rate is not None:
        cfg = replace(cfg, intrinsic=replace(cfg.intrinsic, learning_rate=args.intrinsic_learning_rate))
    if args.slm_model is not None:
        cfg = replace(cfg, slm=replace(cfg.slm, model_name=args.slm_model))
    if args.slm_device is not None:
        cfg = replace(cfg, slm=replace(cfg.slm, device=args.slm_device))
    if args.slm_dtype is not None:
        cfg = replace(cfg, slm=replace(cfg.slm, dtype=args.slm_dtype))
    if args.slm_max_new_tokens is not None:
        cfg = replace(cfg, slm=replace(cfg.slm, max_new_tokens=args.slm_max_new_tokens))
    if args.slm_temperature is not None:
        cfg = replace(cfg, slm=replace(cfg.slm, temperature=args.slm_temperature))
    if args.slm_load_in_4bit:
        cfg = replace(cfg, slm=replace(cfg.slm, load_in_4bit=True))
    if args.slm_interval is not None:
        cfg = replace(cfg, cars=replace(cfg.cars, interval=args.slm_interval))
    if args.slm_local_files_only:
        cfg = replace(cfg, slm=replace(cfg.slm, local_files_only=True))
    if args.num_envs is not None:
        cfg = replace(cfg, ppo=replace(cfg.ppo, num_envs=args.num_envs))
    if args.rollout_steps is not None:
        cfg = replace(cfg, ppo=replace(cfg.ppo, rollout_steps=args.rollout_steps))
    if args.minibatch_size is not None:
        cfg = replace(cfg, ppo=replace(cfg.ppo, minibatch_size=args.minibatch_size))
    if args.update_epochs is not None:
        cfg = replace(cfg, ppo=replace(cfg.ppo, update_epochs=args.update_epochs))
    if args.learning_rate is not None:
        cfg = replace(cfg, ppo=replace(cfg.ppo, learning_rate=args.learning_rate))
    if args.run_dir is not None:
        cfg = replace(cfg, logging=replace(cfg.logging, run_dir=args.run_dir))
    if args.wandb:
        cfg = replace(cfg, logging=replace(cfg.logging, wandb_enabled=True))
    if args.wandb_project is not None:
        cfg = replace(cfg, logging=replace(cfg.logging, wandb_project=args.wandb_project))
    if args.wandb_entity is not None:
        cfg = replace(cfg, logging=replace(cfg.logging, wandb_entity=args.wandb_entity))
    if args.wandb_group is not None:
        cfg = replace(cfg, logging=replace(cfg.logging, wandb_group=args.wandb_group))
    if args.wandb_name is not None:
        cfg = replace(cfg, logging=replace(cfg.logging, wandb_name=args.wandb_name))
    if args.wandb_mode is not None:
        cfg = replace(cfg, logging=replace(cfg.logging, wandb_mode=args.wandb_mode))
    return cfg


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.smoke:
        from scripts.smoke_cars_core import main as smoke_main

        smoke_main()
        return

    cfg = apply_overrides(args.config, args)
    if cfg.algo in SUPPORTED_ALGOS:
        from carsrl.ppo.trainer import PPOTrainer

        result = PPOTrainer(cfg, device=args.device).train()
        print(f"Training complete: run_dir={result.run_dir}")
        print(f"Final checkpoint: {result.final_checkpoint}")
        print(f"Steps={result.global_step}, episodes={result.episodes}")
        return

    raise SystemExit(
        f"Algorithm {cfg.algo!r} is not wired yet. "
        f"Use one of: {', '.join(sorted(SUPPORTED_ALGOS))}."
    )


if __name__ == "__main__":
    main()
