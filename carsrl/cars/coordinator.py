"""Runtime CARS integration for vectorized PPO rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from carsrl.cars.appraiser import (
    BaseAppraiser,
    HeuristicAppraiser,
    MockAppraiser,
    QwenAppraiser,
    RandomAppraiser,
    ShuffledPhiAppraiser,
)
from carsrl.cars.cache import AppraisalCache
from carsrl.cars.scheduler import AppraisalSchedule
from carsrl.cars.schema import Appraisal
from carsrl.cars.serializer import EpisodeTrace, MiniGridStateSerializer
from carsrl.cars.shaper import CARSRewardShaper
from carsrl.config import ExperimentConfig


@dataclass
class CARSState:
    trace: EpisodeTrace
    previous_appraisal: Appraisal
    step_in_episode: int = 0
    shaped_return: float = 0.0
    slm_calls: int = 0
    phi_sum: float = 0.0
    confidence_sum: float = 0.0
    appraisal_count: int = 0
    last_subgoal: str = ""

    def observe_appraisal(self, appraisal: Appraisal, model_call: bool) -> None:
        self.phi_sum += appraisal.phi
        self.confidence_sum += appraisal.confidence
        self.appraisal_count += 1
        self.last_subgoal = appraisal.subgoal
        if model_call:
            self.slm_calls += 1

    def episode_metrics(self, cache_hit_rate: float) -> dict[str, Any]:
        denom = max(1, self.appraisal_count)
        return {
            "shaped_return": self.shaped_return,
            "slm_calls": self.slm_calls,
            "cache_hit_rate": cache_hit_rate,
            "mean_phi": self.phi_sum / denom,
            "mean_confidence": self.confidence_sum / denom,
            "subgoal": self.last_subgoal,
        }


@dataclass(frozen=True)
class CARSStepResult:
    total_rewards: np.ndarray
    shaped_rewards: np.ndarray
    phis: np.ndarray
    confidences: np.ndarray
    episode_metrics: dict[int, dict[str, Any]]


def _obs_item(obs_batch: Any, index: int) -> Any:
    if isinstance(obs_batch, dict):
        return {key: value[index] for key, value in obs_batch.items()}
    return obs_batch[index]


def _final_obs_item(infos: dict[str, Any], index: int) -> Any | None:
    final_obs = infos.get("final_obs")
    if final_obs is None:
        final_obs = infos.get("final_observation")
    if final_obs is None:
        return None
    mask = infos.get("_final_obs")
    if mask is None:
        mask = infos.get("_final_observation")
    if mask is not None and not bool(mask[index]):
        return None
    return final_obs[index]


class CARSRolloutCoordinator:
    """Applies CARS reward shaping during PPO rollout collection."""

    def __init__(
        self,
        appraiser: BaseAppraiser,
        serializer: MiniGridStateSerializer,
        shaper: CARSRewardShaper,
        schedule: AppraisalSchedule,
        num_envs: int,
        history_length: int,
        cache: AppraisalCache | None = None,
        neutral_on_skip: bool = True,
    ):
        self.appraiser = appraiser
        self.serializer = serializer
        self.shaper = shaper
        self.schedule = schedule
        self.num_envs = num_envs
        self.history_length = history_length
        self.cache = cache
        self.neutral_on_skip = neutral_on_skip
        self.states: list[CARSState] = []

    @classmethod
    def from_config(cls, config: ExperimentConfig, run_dir: Path) -> "CARSRolloutCoordinator":
        cache_path = Path(config.cars.cache_path)
        if not cache_path.is_absolute():
            cache_path = run_dir / cache_path
        cache = AppraisalCache(cache_path)
        appraiser = _build_appraiser(config, cache)
        serializer = MiniGridStateSerializer(
            include_history=config.cars.include_history,
            history_length=config.cars.history_length,
        )
        shaper = CARSRewardShaper(
            beta=config.cars.beta,
            gamma=config.cars.gamma,
            clip_min=config.cars.reward_clip_min,
            clip_max=config.cars.reward_clip_max,
            use_confidence=config.cars.use_confidence,
            direct_reward=config.cars.direct_reward,
        )
        schedule = AppraisalSchedule(mode=config.cars.schedule, interval=config.cars.interval)
        return cls(
            appraiser=appraiser,
            serializer=serializer,
            shaper=shaper,
            schedule=schedule,
            num_envs=config.ppo.num_envs,
            history_length=config.cars.history_length,
            cache=cache,
            neutral_on_skip=config.cars.neutral_on_skip,
        )

    def reset(self, envs: Any, obs_batch: Any) -> None:
        self.states = []
        for index in range(self.num_envs):
            trace = EpisodeTrace(history_length=self.history_length)
            appraisal, model_call = self._appraise(envs.envs[index], _obs_item(obs_batch, index), trace)
            state = CARSState(trace=trace, previous_appraisal=appraisal)
            state.observe_appraisal(appraisal, model_call)
            self.states.append(state)

    def step(
        self,
        envs: Any,
        actions: np.ndarray,
        obs_out: Any,
        extrinsic_rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        infos: dict[str, Any],
    ) -> CARSStepResult:
        if not self.states:
            raise RuntimeError("CARSRolloutCoordinator.reset must be called before step.")

        total_rewards = np.asarray(extrinsic_rewards, dtype=np.float32).copy()
        shaped_rewards = np.zeros(self.num_envs, dtype=np.float32)
        phis = np.zeros(self.num_envs, dtype=np.float32)
        confidences = np.zeros(self.num_envs, dtype=np.float32)
        episode_metrics: dict[int, dict[str, Any]] = {}

        for index in range(self.num_envs):
            done = bool(terminated[index] or truncated[index])
            action = int(actions[index])
            state = self.states[index]
            event_triggered = bool(extrinsic_rewards[index] > 0 or done or action in {3, 5})

            transition_obs = _final_obs_item(infos, index) if done else None
            transition_env = None if transition_obs is not None else envs.envs[index]
            if transition_obs is None:
                transition_obs = _obs_item(obs_out, index)

            state.trace.observe_transition(
                action=action,
                reward=float(extrinsic_rewards[index]),
                terminated=bool(terminated[index]),
                truncated=bool(truncated[index]),
                info=None,
            )

            made_new_estimate = self.schedule.should_call(state.step_in_episode + 1, event_triggered=event_triggered)
            if made_new_estimate:
                current_appraisal, model_call = self._appraise(transition_env, transition_obs, state.trace)
            else:
                current_appraisal = state.previous_appraisal
                model_call = False

            shaped = (
                0.0
                if self.neutral_on_skip and not made_new_estimate
                else self.shaper.shape(state.previous_appraisal, current_appraisal)
            )
            shaped_rewards[index] = shaped
            total_rewards[index] += shaped
            phis[index] = current_appraisal.phi
            confidences[index] = current_appraisal.confidence

            state.shaped_return += float(shaped)
            state.observe_appraisal(current_appraisal, model_call)

            if done:
                episode_metrics[index] = state.episode_metrics(self.cache.stats.hit_rate if self.cache else 0.0)
                self._reset_single_env_state(envs, obs_out, index)
            else:
                state.previous_appraisal = current_appraisal
                state.step_in_episode += 1

        return CARSStepResult(
            total_rewards=total_rewards,
            shaped_rewards=shaped_rewards,
            phis=phis,
            confidences=confidences,
            episode_metrics=episode_metrics,
        )

    def _reset_single_env_state(self, envs: Any, obs_batch: Any, index: int) -> None:
        trace = EpisodeTrace(history_length=self.history_length)
        appraisal, model_call = self._appraise(envs.envs[index], _obs_item(obs_batch, index), trace)
        state = CARSState(trace=trace, previous_appraisal=appraisal)
        state.observe_appraisal(appraisal, model_call)
        self.states[index] = state

    def _appraise(self, env: Any, obs: Any, trace: EpisodeTrace) -> tuple[Appraisal, bool]:
        serialized = self.serializer.serialize(env, obs, trace=trace)
        before_misses = self.cache.stats.misses if self.cache else 0
        appraisal = self.appraiser.appraise(serialized)
        model_call = True if self.cache is None else self.cache.stats.misses > before_misses
        return appraisal, model_call

    def appraiser_stats(self) -> dict[str, Any]:
        stats = getattr(self.appraiser, "stats", None)
        if stats is None:
            return {}
        data = stats.to_dict()
        if self.cache is not None:
            data["appraisal_cache_hit_rate"] = self.cache.stats.hit_rate
            data["appraisal_cache_size"] = len(self.cache)
        return data


def _build_appraiser(config: ExperimentConfig, cache: AppraisalCache | None) -> BaseAppraiser:
    name = config.cars.appraiser.lower()
    if name.startswith("shuffled_"):
        source_name = name.removeprefix("shuffled_")
        base = _build_named_appraiser(source_name, config, cache)
        return ShuffledPhiAppraiser(
            base_appraiser=base,
            cache=cache,
            seed=config.seed,
            cache_namespace=f"shuffled_phi:{source_name}",
        )
    return _build_named_appraiser(name, config, cache)


def _build_named_appraiser(name: str, config: ExperimentConfig, cache: AppraisalCache | None) -> BaseAppraiser:
    if name == "qwen":
        return QwenAppraiser(
            model_name=config.slm.model_name,
            cache=cache,
            device=config.slm.device,
            dtype=config.slm.dtype,
            load_in_4bit=config.slm.load_in_4bit,
            local_files_only=config.slm.local_files_only,
            max_new_tokens=config.slm.max_new_tokens,
            temperature=config.slm.temperature,
        )
    if name == "mock":
        return MockAppraiser(cache=cache, seed=config.seed)
    if name == "random":
        return RandomAppraiser(cache=cache, seed=config.seed)
    if name == "heuristic":
        return HeuristicAppraiser(cache=cache)
    raise ValueError(f"Unknown CARS appraiser: {config.cars.appraiser}")
