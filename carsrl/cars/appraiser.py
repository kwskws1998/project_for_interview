"""Frozen SLM appraisers for CARS."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
import time
from typing import Any

from carsrl.cars.cache import AppraisalCache
from carsrl.cars.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_appraisal_prompt
from carsrl.cars.schema import Appraisal


def _line_value(serialized_state: str, prefix: str) -> str:
    for line in serialized_state.splitlines():
        if line.lower().startswith(prefix.lower()):
            return line.split(":", 1)[1].strip().lower()
    return ""


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _has_carried_key(inventory: str) -> bool:
    return "key" in inventory and "nothing" not in inventory


def _is_legacy_copied_appraisal(appraisal: Appraisal) -> bool:
    return (
        abs(appraisal.phi - 0.25) < 1.0e-6
        and abs(appraisal.confidence - 0.80) < 1.0e-6
        and abs(appraisal.affordance - 0.70) < 1.0e-6
        and abs(appraisal.novelty - 0.30) < 1.0e-6
        and abs(appraisal.risk - 0.10) < 1.0e-6
        and "pick up the visible key" in appraisal.subgoal.lower()
    )


def _appraisal_validation_issue(appraisal: Appraisal, serialized_state: str) -> str | None:
    if appraisal.parse_error:
        return f"parse_error:{appraisal.parse_error}"

    visible = _line_value(serialized_state, "Visible objects")
    inventory = _line_value(serialized_state, "Agent inventory")
    subgoal = appraisal.subgoal.lower()
    carrying_key = _has_carried_key(inventory)
    sees_key = "key" in visible
    sees_locked_door = "locked" in visible and "door" in visible
    sees_open_door = "open" in visible and "door" in visible
    sees_goal = "goal" in visible

    if _is_legacy_copied_appraisal(appraisal):
        return "legacy_example_copy"
    key_pickup_terms = ("pick up key", "pick up a key", "pick up the key", "pick up the visible key", "pickup key")
    if carrying_key and _has_any(subgoal, key_pickup_terms):
        return "subgoal_contradicts_inventory_key"
    if sees_key and not carrying_key and not _has_any(subgoal, key_pickup_terms + ("get key", "get the key", "collect key")):
        return "subgoal_ignores_visible_key"
    if sees_key and not carrying_key and appraisal.phi > 0.50:
        return "phi_too_high_for_visible_key_not_carried"
    if carrying_key and sees_locked_door and appraisal.phi < 0.35:
        return "phi_too_low_for_key_and_locked_door"
    if sees_goal and not sees_locked_door and _has_any(subgoal, ("toggle", "open door", "open the door")):
        return "subgoal_ignores_visible_goal"
    if sees_goal and not sees_locked_door and appraisal.phi < 0.75:
        return "phi_too_low_for_reachable_visible_goal"
    if sees_goal and appraisal.phi < 0.55:
        return "phi_too_low_for_visible_goal"
    if sees_open_door and sees_goal and appraisal.phi < 0.60:
        return "phi_too_low_for_open_path_to_goal"
    if not carrying_key and not sees_key and _has_any(subgoal, key_pickup_terms):
        return "subgoal_mentions_unseen_key"
    if appraisal.confidence > 0.95 and appraisal.phi in {0.0, 1.0}:
        return "overconfident_extreme_score"
    return None


def _correction_instruction(issue: str, serialized_state: str) -> str:
    visible = _line_value(serialized_state, "Visible objects")
    carrying_key = _has_carried_key(_line_value(serialized_state, "Agent inventory"))
    if issue in {"subgoal_ignores_visible_goal", "phi_too_low_for_reachable_visible_goal"}:
        return (
            "The state has a visible goal and no locked door blocks it. "
            "Use subgoal \"reach the goal\" and choose phi in the 0.75 to 1.0 range."
        )
    if issue == "subgoal_contradicts_inventory_key":
        return "The agent already carries a key, so do not ask it to pick up a key."
    if issue in {"subgoal_ignores_visible_key", "phi_too_high_for_visible_key_not_carried"}:
        return (
            "The state has a visible key and the agent is carrying nothing. "
            "Use a key-pickup subgoal and choose phi in the 0.20 to 0.40 range."
        )
    if issue == "phi_too_low_for_key_and_locked_door" or (carrying_key and "locked" in visible and "door" in visible):
        return "The agent carries a key and sees a locked door, so the next subgoal is to open/toggle that door."
    if issue == "subgoal_mentions_unseen_key":
        return "No key is visible or carried, so the next subgoal is to search for the matching key."
    return "Use the state facts to produce a different, state-specific appraisal."


class BaseAppraiser(ABC):
    @abstractmethod
    def appraise(self, serialized_state: str) -> Appraisal:
        """Return an appraisal for a serialized MiniGrid state."""


@dataclass
class AppraiserStats:
    uncached_calls: int = 0
    cache_hits: int = 0
    parse_failures: int = 0
    validation_failures: int = 0
    retry_count: int = 0
    fallback_count: int = 0
    total_latency_sec: float = 0.0
    last_latency_sec: float = 0.0
    last_output_text: str = ""

    @property
    def mean_latency_sec(self) -> float:
        return self.total_latency_sec / self.uncached_calls if self.uncached_calls else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "appraiser_uncached_calls": self.uncached_calls,
            "appraiser_cache_hits": self.cache_hits,
            "appraiser_parse_failures": self.parse_failures,
            "appraiser_validation_failures": self.validation_failures,
            "appraiser_retry_count": self.retry_count,
            "appraiser_fallback_count": self.fallback_count,
            "appraiser_total_latency_sec": self.total_latency_sec,
            "appraiser_mean_latency_sec": self.mean_latency_sec,
            "appraiser_last_latency_sec": self.last_latency_sec,
        }


class CachedAppraiser(BaseAppraiser):
    def __init__(self, cache: AppraisalCache | None = None, cache_namespace: str | None = None):
        self.cache = cache
        self.stats = AppraiserStats()
        self.cache_namespace = cache_namespace or self.__class__.__name__

    def _cache_text(self, serialized_state: str) -> str:
        return f"cache_namespace: {self.cache_namespace}\n{serialized_state}"

    def _appraise_uncached(self, serialized_state: str) -> Appraisal:
        raise NotImplementedError

    def appraise(self, serialized_state: str) -> Appraisal:
        cache_text = self._cache_text(serialized_state)
        if self.cache is not None:
            cached = self.cache.get(cache_text)
            if cached is not None:
                self.stats.cache_hits += 1
                return cached
        start = time.perf_counter()
        appraisal = self._appraise_uncached(serialized_state)
        elapsed = time.perf_counter() - start
        self.stats.uncached_calls += 1
        self.stats.total_latency_sec += elapsed
        self.stats.last_latency_sec = elapsed
        self.stats.last_output_text = appraisal.raw_text
        if appraisal.parse_error:
            self.stats.parse_failures += 1
        if self.cache is not None:
            self.cache.put(cache_text, appraisal)
        return appraisal


class MockAppraiser(CachedAppraiser):
    """Deterministic local appraiser for smoke tests and no-network development."""

    def __init__(self, cache: AppraisalCache | None = None, seed: int = 0):
        super().__init__(cache=cache, cache_namespace="mock_v2")
        self._rng = random.Random(seed)

    def _appraise_uncached(self, serialized_state: str) -> Appraisal:
        text = serialized_state.lower()
        visible = _line_value(serialized_state, "Visible objects")
        inventory = _line_value(serialized_state, "Agent inventory")
        phi = 0.05
        subgoal = "explore"
        if "key" in visible:
            phi += 0.15
            subgoal = "pick up the key"
        if "carrying" in inventory and "key" in inventory:
            phi += 0.35
            subgoal = "open the locked door"
        if "door" in visible:
            phi += 0.20
        if "goal" in visible:
            phi += 0.30
            subgoal = "reach the goal"
        if "locked" in visible and "key" not in inventory and "key" not in visible:
            subgoal = "find the matching key"
        novelty = 0.2 + 0.1 * self._rng.random()
        return Appraisal(
            phi=min(phi, 1.0),
            confidence=0.75,
            subgoal=subgoal,
            affordance=0.6,
            novelty=novelty,
            risk=0.1,
            raw_text='{"source": "mock"}',
        )


class HeuristicAppraiser(CachedAppraiser):
    """Cheap hand-coded control potential for MiniGrid states."""

    def _appraise_uncached(self, serialized_state: str) -> Appraisal:
        visible = _line_value(serialized_state, "Visible objects")
        inventory = _line_value(serialized_state, "Agent inventory")
        has_key = "key" in visible
        carrying_key = "carrying" in inventory and "key" in inventory
        sees_locked_door = "locked" in visible and "door" in visible
        sees_goal = "goal" in visible

        phi = 0.05
        subgoal = "explore"
        if has_key:
            phi = max(phi, 0.25)
            subgoal = "pick up the key"
        if carrying_key:
            phi = max(phi, 0.45)
            subgoal = "open the locked door"
        if sees_locked_door and carrying_key:
            phi = max(phi, 0.60)
        if sees_goal:
            phi = max(phi, 0.80)
            subgoal = "reach the goal"
        return Appraisal(
            phi=phi,
            confidence=0.65,
            subgoal=subgoal,
            affordance=0.5,
            novelty=0.0,
            risk=0.1 if carrying_key else 0.25,
            raw_text='{"source": "heuristic"}',
        )


class RandomAppraiser(CachedAppraiser):
    """Control baseline for random potential scores."""

    def __init__(self, cache: AppraisalCache | None = None, seed: int = 0):
        super().__init__(cache=cache, cache_namespace=f"random_seed_{seed}")
        self._rng = random.Random(seed)

    def _appraise_uncached(self, serialized_state: str) -> Appraisal:
        return Appraisal(
            phi=self._rng.random(),
            confidence=1.0,
            subgoal="random_control",
            affordance=self._rng.random(),
            novelty=self._rng.random(),
            risk=self._rng.random(),
            raw_text='{"source": "random"}',
        )


class ShuffledPhiAppraiser(CachedAppraiser):
    """Control baseline that keeps a base appraiser's Phi distribution but breaks state alignment."""

    def __init__(
        self,
        base_appraiser: BaseAppraiser,
        cache: AppraisalCache | None = None,
        seed: int = 0,
        cache_namespace: str = "shuffled_phi",
        buffer_size: int = 4096,
    ):
        super().__init__(cache=cache, cache_namespace=f"{cache_namespace}:seed{seed}:buffer{buffer_size}")
        self.base_appraiser = base_appraiser
        self._rng = random.Random(seed)
        self._phi_buffer: list[float] = []
        self._buffer_size = buffer_size

    def _appraise_uncached(self, serialized_state: str) -> Appraisal:
        base = self.base_appraiser.appraise(serialized_state)
        if self._phi_buffer:
            shuffled_phi = self._rng.choice(self._phi_buffer)
        else:
            shuffled_phi = base.phi

        self._phi_buffer.append(base.phi)
        if len(self._phi_buffer) > self._buffer_size:
            self._phi_buffer.pop(0)

        return Appraisal(
            phi=shuffled_phi,
            confidence=base.confidence,
            subgoal=f"shuffled_phi:{base.subgoal}"[:160],
            affordance=base.affordance,
            novelty=base.novelty,
            risk=base.risk,
            raw_text=(
                '{"source": "shuffled_phi", '
                f'"base_phi": {base.phi:.6f}, "shuffled_phi": {shuffled_phi:.6f}, '
                f'"base_raw": {base.raw_text!r}' + "}"
            ),
            parse_error=base.parse_error,
        )


class QwenAppraiser(CachedAppraiser):
    """Frozen Qwen appraiser. The model is loaded lazily on first use."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        cache: AppraisalCache | None = None,
        device: str = "auto",
        dtype: str = "auto",
        load_in_4bit: bool = False,
        local_files_only: bool = False,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ):
        namespace = f"qwen:{model_name}:{PROMPT_VERSION}:tokens{max_new_tokens}:temp{temperature}"
        super().__init__(cache=cache, cache_namespace=namespace)
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.local_files_only = local_files_only
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._runtime_device: Any | None = None

    def _resolve_runtime_device(self) -> Any:
        import torch

        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        requested = torch.device(self.device)
        if requested.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Requested SLM device 'mps', but torch.backends.mps.is_available() is False.")
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested SLM device 'cuda', but torch.cuda.is_available() is False.")
        return requested

    def _torch_dtype(self, runtime_device: Any) -> Any:
        import torch

        if self.dtype == "auto":
            if runtime_device.type in {"cuda", "mps"}:
                return torch.float16
            return torch.float32

        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return mapping.get(self.dtype.lower(), "auto")

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        runtime_device = self._resolve_runtime_device()
        self._runtime_device = runtime_device
        model_dtype = self._torch_dtype(runtime_device)
        model_kwargs: dict[str, Any] = {
            "local_files_only": self.local_files_only,
        }
        if self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                dtype=model_dtype,
                **model_kwargs,
            )
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=model_dtype,
                **model_kwargs,
            )
        if not self.load_in_4bit:
            self._model.to(runtime_device)
        self._model.eval()
        if getattr(self._model, "generation_config", None) is not None:
            self._model.generation_config.do_sample = False
            self._model.generation_config.temperature = None
            self._model.generation_config.top_p = None
            self._model.generation_config.top_k = None
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

    def _generate_text(self, user_prompt: str) -> str:
        assert self._model is not None
        assert self._tokenizer is not None

        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer([text], return_tensors="pt")
        runtime_device = self._runtime_device
        if runtime_device is None:
            runtime_device = next(self._model.parameters()).device
        inputs = {k: v.to(runtime_device) for k, v in inputs.items()}
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **generation_kwargs)
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _appraise_uncached(self, serialized_state: str) -> Appraisal:
        self._load()
        assert self._model is not None
        assert self._tokenizer is not None

        first_prompt = build_appraisal_prompt(serialized_state)
        first_text = self._generate_text(first_prompt)
        first_appraisal = Appraisal.from_json_text(first_text)
        first_issue = _appraisal_validation_issue(first_appraisal, serialized_state)
        if first_issue is None:
            return first_appraisal

        self.stats.validation_failures += 1
        self.stats.retry_count += 1
        retry_prompt = (
            f"{first_prompt}\n\n"
            f"Your previous answer was rejected for this reason: {first_issue}.\n"
            f"Correction: {_correction_instruction(first_issue, serialized_state)}\n"
            "Recompute the appraisal from the state. Do not reuse a generic example. "
            "Return strict JSON only."
        )
        retry_text = self._generate_text(retry_prompt)
        retry_appraisal = Appraisal.from_json_text(retry_text)
        retry_issue = _appraisal_validation_issue(retry_appraisal, serialized_state)
        if retry_issue is None:
            return retry_appraisal

        self.stats.validation_failures += 1
        self.stats.fallback_count += 1
        return Appraisal.fallback(
            raw_text=f"{first_text}\n--- retry ---\n{retry_text}",
            parse_error=f"qwen_validation_failed:{first_issue}; retry:{retry_issue}",
        )
