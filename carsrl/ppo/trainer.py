"""PPO trainer for Gymnasium MiniGrid."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
import time
from typing import Any

import numpy as np

from carsrl.cars.coordinator import CARSRolloutCoordinator
from carsrl.config import ExperimentConfig
from carsrl.envs.diagnostics import DiagnosticsTracker
from carsrl.envs.minigrid import make_vector_minigrid_env
from carsrl.ppo.model import build_minigrid_actor_critic, obs_to_tensor
from carsrl.ppo.storage import RolloutBuffer
from carsrl.rewards.coordinator import IntrinsicRewardCoordinator
from carsrl.utils.jsonl import JsonlWriter
from carsrl.utils.seeding import set_global_seeds


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
    "ppo_random_phi",
    "ppo_shuffled_phi",
    "ppo_heuristic_phi",
    "ppo_cars_no_confidence",
    "ppo_cars_direct",
}

CARS_ALGOS = {
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

INTRINSIC_ALGOS = {
    "ppo_rnd",
    "ppo_rnd_cars",
    "ppo_icm",
    "ppo_icm_cars",
    "ppo_ride",
    "ppo_ride_cars",
    "ppo_noveld",
    "ppo_noveld_cars",
}


@dataclass(frozen=True)
class TrainResult:
    run_dir: Path
    global_step: int
    episodes: int
    final_checkpoint: Path


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _mean_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = set().union(*(item.keys() for item in items))
    return {
        key: float(np.mean([item[key] for item in items if key in item]))
        for key in keys
    }


def _transition_obs_batch(obs_out: Any, infos: dict[str, Any], done: np.ndarray) -> Any:
    """Use terminal final_obs for reward computation under SAME_STEP autoreset."""
    if not isinstance(obs_out, dict):
        return obs_out
    final_obs = infos.get("final_obs")
    if final_obs is None:
        final_obs = infos.get("final_observation")
    if final_obs is None:
        return obs_out
    final_mask = infos.get("_final_obs")
    if final_mask is None:
        final_mask = infos.get("_final_observation")

    transition = {key: np.array(value, copy=True) for key, value in obs_out.items()}
    for env_index, is_done in enumerate(done):
        if not bool(is_done):
            continue
        if final_mask is not None and not bool(final_mask[env_index]):
            continue
        final_item = final_obs[env_index]
        if not isinstance(final_item, dict):
            continue
        for key, value in final_item.items():
            if key in transition:
                transition[key][env_index] = value
    return transition


def _write_config(path: Path, config: ExperimentConfig) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, sort_keys=True)


def _init_wandb(config: ExperimentConfig, run_dir: Path) -> Any | None:
    if not config.logging.wandb_enabled:
        return None
    if config.logging.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb logging requested, but wandb is not installed. Run `pip install wandb`.") from exc

    name = config.logging.wandb_name or run_dir.name
    init_kwargs: dict[str, Any] = {
        "project": config.logging.wandb_project,
        "name": name,
        "mode": config.logging.wandb_mode,
        "dir": str(run_dir),
        "config": asdict(config),
        "tags": [config.algo, config.env_id],
    }
    if config.logging.wandb_entity:
        init_kwargs["entity"] = config.logging.wandb_entity
    if config.logging.wandb_group:
        init_kwargs["group"] = config.logging.wandb_group
    return wandb.init(**init_kwargs)


def _is_wandb_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_))


def _wandb_payload(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        if _is_wandb_scalar(value):
            payload[f"{prefix}/{key}"] = float(value) if isinstance(value, (np.integer, np.floating, np.bool_)) else value
        elif isinstance(value, str) and len(value) <= 240:
            payload[f"{prefix}/{key}"] = value
    return payload


def _cars_metrics_for_env(cars_episode_metrics: dict[int, dict[str, Any]] | None, env_index: int) -> dict[str, Any]:
    defaults = {
        "shaped_return": 0.0,
        "slm_calls": 0,
        "cache_hit_rate": 0.0,
        "mean_phi": 0.0,
        "mean_confidence": 0.0,
        "subgoal": "",
    }
    if not cars_episode_metrics:
        return defaults
    return {**defaults, **cars_episode_metrics.get(env_index, {})}


def _intrinsic_metrics_for_env(
    intrinsic_episode_metrics: dict[int, dict[str, Any]] | None,
    env_index: int,
) -> dict[str, Any]:
    defaults = {
        "intrinsic_return": 0.0,
        "intrinsic_unique_states": 0,
        "noveld_unique_states": 0,
    }
    if not intrinsic_episode_metrics:
        return defaults
    return {**defaults, **intrinsic_episode_metrics.get(env_index, {})}


def _episode_record(
    env_index: int,
    extrinsic_return: float,
    episode_length: int,
    global_step: int,
    wall_time: float,
    cars_episode_metrics: dict[int, dict[str, Any]] | None,
    intrinsic_episode_metrics: dict[int, dict[str, Any]] | None,
    diagnostic_episode_metrics: dict[int, dict[str, Any]] | None,
) -> dict[str, Any]:
    cars_metrics = _cars_metrics_for_env(cars_episode_metrics, env_index)
    intrinsic_metrics = _intrinsic_metrics_for_env(intrinsic_episode_metrics, env_index)
    shaped_return = float(cars_metrics["shaped_return"])
    intrinsic_return = float(intrinsic_metrics["intrinsic_return"])
    episode_return = float(extrinsic_return) + shaped_return + intrinsic_return
    diagnostic_metrics = diagnostic_episode_metrics.get(env_index, {}) if diagnostic_episode_metrics else {}
    record = {
        "global_step": global_step,
        "env_index": env_index,
        "episode_return": episode_return,
        "extrinsic_return": float(extrinsic_return),
        "shaped_return": shaped_return,
        "intrinsic_return": intrinsic_return,
        "episode_length": int(episode_length),
        "success": float(extrinsic_return > 0.0),
        "wall_time": wall_time,
        "slm_calls": int(cars_metrics["slm_calls"]),
        "cache_hit_rate": float(cars_metrics["cache_hit_rate"]),
        "mean_phi": float(cars_metrics["mean_phi"]),
        "mean_confidence": float(cars_metrics["mean_confidence"]),
        "subgoal": str(cars_metrics["subgoal"]),
        "intrinsic_unique_states": int(intrinsic_metrics["intrinsic_unique_states"]),
        "noveld_unique_states": int(intrinsic_metrics["noveld_unique_states"]),
    }
    record.update(
        {
            "key_first_seen_step": diagnostic_metrics.get("key_first_seen_step"),
            "key_pickup_step": diagnostic_metrics.get("key_pickup_step"),
            "door_first_seen_step": diagnostic_metrics.get("door_first_seen_step"),
            "door_open_step": diagnostic_metrics.get("door_open_step"),
            "goal_first_seen_step": diagnostic_metrics.get("goal_first_seen_step"),
            "success_step": diagnostic_metrics.get("success_step"),
            "distractor_actions": diagnostic_metrics.get("distractor_actions", 0),
            "mean_progress_stage": diagnostic_metrics.get("mean_progress_stage", 0.0),
            "mean_diag_phi": diagnostic_metrics.get("mean_diag_phi", 0.0),
        }
    )
    return record


def _extract_episode_records(
    infos: dict[str, Any],
    global_step: int,
    wall_time: float,
    cars_episode_metrics: dict[int, dict[str, Any]] | None = None,
    intrinsic_episode_metrics: dict[int, dict[str, Any]] | None = None,
    diagnostic_episode_metrics: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    final_infos = infos.get("final_info") if isinstance(infos, dict) else None
    if final_infos is not None:
        final_mask = infos.get("_final_info")
        if isinstance(final_infos, dict) and "episode" in final_infos:
            episode = final_infos["episode"]
            returns = episode.get("r")
            lengths = episode.get("l")
            if returns is not None and lengths is not None:
                for env_index, episode_return in enumerate(returns):
                    if final_mask is not None and not bool(final_mask[env_index]):
                        continue
                    records.append(
                        _episode_record(
                            env_index=env_index,
                            extrinsic_return=float(episode_return),
                            episode_length=int(lengths[env_index]),
                            global_step=global_step,
                            wall_time=wall_time,
                            cars_episode_metrics=cars_episode_metrics,
                            intrinsic_episode_metrics=intrinsic_episode_metrics,
                            diagnostic_episode_metrics=diagnostic_episode_metrics,
                        )
                    )
            return records

        for env_index, item in enumerate(final_infos):
            if item is None or "episode" not in item:
                continue
            episode = item["episode"]
            records.append(
                _episode_record(
                    env_index=env_index,
                    extrinsic_return=float(episode["r"]),
                    episode_length=int(episode["l"]),
                    global_step=global_step,
                    wall_time=wall_time,
                    cars_episode_metrics=cars_episode_metrics,
                    intrinsic_episode_metrics=intrinsic_episode_metrics,
                    diagnostic_episode_metrics=diagnostic_episode_metrics,
                )
            )
        return records

    episode = infos.get("episode") if isinstance(infos, dict) else None
    if episode is None:
        return records

    returns = episode.get("r")
    lengths = episode.get("l")
    masks = infos.get("_episode")
    if returns is None or lengths is None:
        return records
    for idx, episode_return in enumerate(returns):
        if masks is not None and not masks[idx]:
            continue
        records.append(
            _episode_record(
                env_index=idx,
                extrinsic_return=float(episode_return),
                episode_length=int(lengths[idx]),
                global_step=global_step,
                wall_time=wall_time,
                cars_episode_metrics=cars_episode_metrics,
                intrinsic_episode_metrics=intrinsic_episode_metrics,
                diagnostic_episode_metrics=diagnostic_episode_metrics,
            )
        )
    return records


class PPOTrainer:
    def __init__(self, config: ExperimentConfig, device: str = "auto"):
        self.config = config
        self.device = self._resolve_device(device)

    @staticmethod
    def _resolve_device(device: str) -> Any:
        import torch

        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def train(self) -> TrainResult:
        import numpy as np
        import torch
        from torch import optim

        cfg = self.config
        if cfg.algo not in SUPPORTED_ALGOS:
            raise ValueError(f"Supported algorithms are {sorted(SUPPORTED_ALGOS)}; got {cfg.algo!r}.")

        set_global_seeds(cfg.seed)
        run_dir = Path(cfg.logging.run_dir) / f"{cfg.algo}_{cfg.env_id}_seed{cfg.seed}_{_timestamp()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_config(run_dir / "config.json", cfg)
        wandb_run = _init_wandb(cfg, run_dir)

        envs = make_vector_minigrid_env(cfg.env_id, cfg.ppo.num_envs, cfg.seed)
        action_dim = envs.single_action_space.n
        obs, _ = envs.reset(seed=cfg.seed)
        next_obs = obs_to_tensor(obs, self.device)
        obs_shape = tuple(int(dim) for dim in next_obs.shape[1:])

        model = build_minigrid_actor_critic(obs_shape, action_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.ppo.learning_rate, eps=1e-5)
        buffer = RolloutBuffer(cfg.ppo.rollout_steps, cfg.ppo.num_envs, obs_shape, self.device)

        metrics_path = run_dir / cfg.logging.csv_name
        events_path = run_dir / cfg.logging.jsonl_name
        diagnostics_path = run_dir / cfg.logging.diagnostics_name
        fieldnames = [
            "global_step",
            "env_index",
            "episode_return",
            "extrinsic_return",
            "shaped_return",
            "intrinsic_return",
            "episode_length",
            "success",
            "wall_time",
            "slm_calls",
            "cache_hit_rate",
            "mean_phi",
            "mean_confidence",
            "subgoal",
            "intrinsic_unique_states",
            "noveld_unique_states",
            "key_first_seen_step",
            "key_pickup_step",
            "door_first_seen_step",
            "door_open_step",
            "goal_first_seen_step",
            "success_step",
            "distractor_actions",
            "mean_progress_stage",
            "mean_diag_phi",
        ]

        next_done = np.zeros(cfg.ppo.num_envs, dtype=np.float32)
        cars = CARSRolloutCoordinator.from_config(cfg, run_dir) if cfg.algo in CARS_ALGOS else None
        if cars is not None:
            cars.reset(envs, obs)
        intrinsic = (
            IntrinsicRewardCoordinator.from_config(cfg, obs_shape, action_dim, self.device)
            if cfg.algo in INTRINSIC_ALGOS
            else None
        )
        if intrinsic is not None:
            intrinsic.reset(obs)
        diagnostics = DiagnosticsTracker(cfg.ppo.num_envs)
        global_step = 0
        episodes = 0
        start_time = time.time()
        next_save_step = cfg.save_interval

        with (
            metrics_path.open("w", encoding="utf-8", newline="") as csv_handle,
            JsonlWriter(events_path) as event_writer,
            JsonlWriter(diagnostics_path) as diagnostics_writer,
        ):
            writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
            writer.writeheader()

            while global_step < cfg.total_steps:
                rollout_shaped_rewards: list[float] = []
                rollout_intrinsic_rewards: list[float] = []
                rollout_phis: list[float] = []
                rollout_confidences: list[float] = []
                rollout_intrinsic_stats: list[dict[str, float]] = []
                for step in range(cfg.ppo.rollout_steps):
                    global_step += cfg.ppo.num_envs
                    with torch.no_grad():
                        action, logprob, _, value = model.get_action_and_value(next_obs)

                    actions_np = action.cpu().numpy()
                    obs_out, reward, terminated, truncated, infos = envs.step(actions_np)
                    done = np.logical_or(terminated, truncated).astype(np.float32)
                    next_obs_out_tensor = obs_to_tensor(obs_out, self.device)
                    transition_obs_batch = _transition_obs_batch(obs_out, infos, done)
                    transition_obs_tensor = (
                        next_obs_out_tensor
                        if transition_obs_batch is obs_out
                        else obs_to_tensor(transition_obs_batch, self.device)
                    )
                    shaped_rewards = np.zeros(cfg.ppo.num_envs, dtype=np.float32)
                    intrinsic_rewards = np.zeros(cfg.ppo.num_envs, dtype=np.float32)
                    phis = np.zeros(cfg.ppo.num_envs, dtype=np.float32)
                    confidences = np.zeros(cfg.ppo.num_envs, dtype=np.float32)
                    cars_episode_metrics = None
                    intrinsic_episode_metrics = None
                    diagnostic_episode_metrics = None
                    train_reward = reward
                    if cars is not None:
                        cars_result = cars.step(
                            envs=envs,
                            actions=actions_np,
                            obs_out=obs_out,
                            extrinsic_rewards=reward,
                            terminated=terminated,
                            truncated=truncated,
                            infos=infos,
                        )
                        train_reward = cars_result.total_rewards
                        shaped_rewards = cars_result.shaped_rewards
                        phis = cars_result.phis
                        confidences = cars_result.confidences
                        cars_episode_metrics = cars_result.episode_metrics
                    if intrinsic is not None:
                        intrinsic_result = intrinsic.step(
                            current_obs_tensor=next_obs,
                            next_obs_tensor=transition_obs_tensor,
                            next_obs_batch=transition_obs_batch,
                            done=done,
                            actions=actions_np,
                        )
                        intrinsic_rewards = intrinsic_result.intrinsic_rewards
                        train_reward = np.asarray(train_reward, dtype=np.float32) + intrinsic_rewards
                        intrinsic_episode_metrics = intrinsic_result.episode_metrics
                        rollout_intrinsic_stats.append(intrinsic_result.stats)
                    rollout_shaped_rewards.extend(float(item) for item in shaped_rewards)
                    rollout_intrinsic_rewards.extend(float(item) for item in intrinsic_rewards)
                    rollout_phis.extend(float(item) for item in phis)
                    rollout_confidences.extend(float(item) for item in confidences)

                    buffer.store(step, next_obs, action, logprob, train_reward, next_done, value)

                    wall_time = time.time() - start_time
                    diagnostic_records, diagnostic_episode_metrics = diagnostics.step(
                        envs=envs,
                        obs_batch=transition_obs_batch,
                        actions=actions_np,
                        rewards=reward,
                        done=done,
                        global_step=global_step,
                        wall_time=wall_time,
                        phis=phis if cars is not None else None,
                    )
                    if cfg.logging.diagnostics_interval > 0 and global_step % cfg.logging.diagnostics_interval == 0:
                        for diagnostic_record in diagnostic_records:
                            diagnostics_writer.write(diagnostic_record)

                    for record in _extract_episode_records(
                        infos,
                        global_step,
                        wall_time,
                        cars_episode_metrics=cars_episode_metrics,
                        intrinsic_episode_metrics=intrinsic_episode_metrics,
                        diagnostic_episode_metrics=diagnostic_episode_metrics,
                    ):
                        writer.writerow(record)
                        csv_handle.flush()
                        if wandb_run is not None:
                            wandb_run.log(_wandb_payload("episode", record), step=global_step)
                        episodes += 1

                    next_obs = next_obs_out_tensor
                    next_done = done

                with torch.no_grad():
                    next_value = model.get_value(next_obs)
                buffer.compute_returns_and_advantages(
                    next_value=next_value,
                    next_done=next_done,
                    gamma=cfg.ppo.gamma,
                    gae_lambda=cfg.ppo.gae_lambda,
                )

                batch = buffer.flatten()
                batch_size = cfg.ppo.rollout_steps * cfg.ppo.num_envs
                minibatch_size = min(cfg.ppo.minibatch_size, batch_size)
                indices = np.arange(batch_size)
                update_losses: list[dict[str, float]] = []

                for _ in range(cfg.ppo.update_epochs):
                    np.random.shuffle(indices)
                    for start in range(0, batch_size, minibatch_size):
                        mb_idx = indices[start : start + minibatch_size]
                        _, newlogprob, entropy, newvalue = model.get_action_and_value(batch.obs[mb_idx], batch.actions[mb_idx])
                        logratio = newlogprob - batch.logprobs[mb_idx]
                        ratio = logratio.exp()

                        mb_advantages = batch.advantages[mb_idx]
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.ppo.clip_coef, 1 + cfg.ppo.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        newvalue = newvalue.view(-1)
                        v_loss_unclipped = (newvalue - batch.returns[mb_idx]) ** 2
                        v_clipped = batch.values[mb_idx] + torch.clamp(
                            newvalue - batch.values[mb_idx],
                            -cfg.ppo.clip_coef,
                            cfg.ppo.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - batch.returns[mb_idx]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                        entropy_loss = entropy.mean()
                        loss = pg_loss - cfg.ppo.ent_coef * entropy_loss + cfg.ppo.vf_coef * v_loss

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.ppo.max_grad_norm)
                        optimizer.step()

                        update_losses.append(
                            {
                                "loss": float(loss.detach().cpu()),
                                "pg_loss": float(pg_loss.detach().cpu()),
                                "value_loss": float(v_loss.detach().cpu()),
                                "entropy": float(entropy_loss.detach().cpu()),
                            }
                        )

                intrinsic_update_stats = intrinsic.update() if intrinsic is not None else {}
                if update_losses:
                    mean_losses = {
                        key: float(np.mean([item[key] for item in update_losses]))
                        for key in update_losses[0]
                    }
                    mean_intrinsic_stats = _mean_dicts(rollout_intrinsic_stats)
                    cars_stats = cars.appraiser_stats() if cars is not None else {}
                    update_event = {
                        "type": "ppo_update",
                        "global_step": global_step,
                        "episodes": episodes,
                        "fps": global_step / max(time.time() - start_time, 1e-6),
                        "mean_shaped_reward": float(np.mean(rollout_shaped_rewards)) if rollout_shaped_rewards else 0.0,
                        "mean_intrinsic_reward": float(np.mean(rollout_intrinsic_rewards)) if rollout_intrinsic_rewards else 0.0,
                        "mean_phi": float(np.mean(rollout_phis)) if rollout_phis else 0.0,
                        "mean_confidence": float(np.mean(rollout_confidences)) if rollout_confidences else 0.0,
                        **cars_stats,
                        **mean_intrinsic_stats,
                        **intrinsic_update_stats,
                        **mean_losses,
                    }
                    event_writer.write(update_event)
                    if wandb_run is not None:
                        wandb_run.log(_wandb_payload("update", update_event), step=global_step)

                if cfg.save_interval > 0 and global_step >= next_save_step:
                    self._save_checkpoint(
                        run_dir / f"checkpoint_step{global_step}.pt",
                        model,
                        optimizer,
                        global_step,
                        episodes,
                        extra_state=intrinsic.checkpoint_state() if intrinsic is not None else None,
                    )
                    next_save_step += cfg.save_interval

        final_checkpoint = run_dir / "final_model.pt"
        self._save_checkpoint(
            final_checkpoint,
            model,
            optimizer,
            global_step,
            episodes,
            extra_state=intrinsic.checkpoint_state() if intrinsic is not None else None,
        )
        envs.close()
        if wandb_run is not None:
            wandb_run.summary["final/global_step"] = global_step
            wandb_run.summary["final/episodes"] = episodes
            wandb_run.summary["final/checkpoint"] = str(final_checkpoint)
            wandb_run.finish()
        return TrainResult(
            run_dir=run_dir,
            global_step=global_step,
            episodes=episodes,
            final_checkpoint=final_checkpoint,
        )

    def _save_checkpoint(
        self,
        path: Path,
        model: Any,
        optimizer: Any,
        global_step: int,
        episodes: int,
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "episodes": episodes,
            "config": asdict(self.config),
        }
        if extra_state:
            checkpoint.update(extra_state)
        torch.save(checkpoint, path)
