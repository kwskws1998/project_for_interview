"""Preflight checks before running CARS/NovelD experiments.

Default checks are lightweight and do not load Qwen. Add --with-qwen to run
real frozen-SLM semantic appraisal checks before spending GPU time.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""
    elapsed_sec: float = 0.0


class Preflight:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def add(self, name: str, status: str, detail: str = "", elapsed_sec: float = 0.0) -> None:
        self.results.append(CheckResult(name=name, status=status, detail=detail, elapsed_sec=elapsed_sec))

    def check(self, name: str, fn: Any) -> None:
        start = time.perf_counter()
        try:
            detail = fn()
        except AssertionError as exc:
            self.add(name, "fail", str(exc), time.perf_counter() - start)
        except Exception as exc:  # noqa: BLE001 - preflight should report all unexpected failures.
            self.add(name, "fail", f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
        else:
            status = "pass"
            if isinstance(detail, tuple):
                status, detail = detail
            self.add(name, status, str(detail or ""), time.perf_counter() - start)

    def run_command(self, name: str, command: list[str], timeout_sec: int = 120) -> None:
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            self.add(name, "fail", f"timeout after {exc.timeout}s", time.perf_counter() - start)
            return
        output = (completed.stdout + "\n" + completed.stderr).strip()
        tail = output[-1600:] if output else ""
        if completed.returncode == 0:
            self.add(name, "pass", tail, time.perf_counter() - start)
        else:
            self.add(name, "fail", f"exit={completed.returncode}\n{tail}", time.perf_counter() - start)

    def summary(self) -> dict[str, Any]:
        return {
            "passed": sum(result.status == "pass" for result in self.results),
            "warnings": sum(result.status == "warn" for result in self.results),
            "failed": sum(result.status == "fail" for result in self.results),
            "results": [asdict(result) for result in self.results],
        }

    def print_report(self) -> None:
        for result in self.results:
            marker = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[result.status]
            elapsed = f"{result.elapsed_sec:.2f}s"
            print(f"[{marker}] {result.name} ({elapsed})")
            if result.detail:
                print(_indent(result.detail.strip()))
        print("\nJSON summary:")
        print(json.dumps(self.summary(), indent=2, sort_keys=True))


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())


def _check_prompt_contract() -> str:
    from carsrl.cars.prompts import PROMPT_VERSION, build_appraisal_prompt

    match = re.search(r"_v(\d+)$", PROMPT_VERSION)
    assert match is not None, f"Prompt version should end with _vN, got {PROMPT_VERSION!r}"
    assert int(match.group(1)) >= 6, f"Prompt version is too old: {PROMPT_VERSION}"

    prompt = build_appraisal_prompt(
        "Mission: use the key to open the door and reach the goal\n"
        "Agent inventory: carrying yellow key\n"
        "Visible objects: locked yellow door at view(3,4)"
    )
    banned_snippets = ['"phi": 0.25', "pick up the visible key"]
    found = [snippet for snippet in banned_snippets if snippet in prompt]
    assert not found, f"Prompt still contains copyable legacy snippets: {found}"
    return f"PROMPT_VERSION={PROMPT_VERSION}"


def _check_config_defaults() -> str:
    from carsrl.config import CARSConfig

    config = CARSConfig()
    assert config.include_history is False, "Main CARS should default to state-only serializer."
    assert config.neutral_on_skip is True, "Scheduler-skipped steps should be neutral by default."
    assert config.direct_reward is False, "Main CARS must not default to direct reward."
    return "state-only CARS defaults and neutral skipped-step shaping are enabled"


def _check_serializer_history_boundary() -> str:
    from carsrl.cars import MiniGridStateSerializer

    obs = {"image": np.zeros((7, 7, 3), dtype=np.uint8), "mission": "reach the goal", "direction": 0}
    state_only = MiniGridStateSerializer(include_history=False).serialize(
        None,
        obs,
        last_action=2,
        recent_events=["picked up key"],
        action_history=[2],
    )
    with_history = MiniGridStateSerializer(include_history=True).serialize(
        None,
        obs,
        last_action=2,
        recent_events=["picked up key"],
        action_history=[2],
    )
    assert "Last action" not in state_only, "Last action leaked into state-only serializer."
    assert "Recent events" not in state_only, "Recent events leaked into state-only serializer."
    assert "Last action: move forward" in with_history, "History ablation no longer includes last action."
    assert "Recent events: picked up key" in with_history, "History ablation no longer includes recent events."
    return "history fields are excluded by default and included only for --cars_history"


def _check_mission_aware_obs_tensor() -> str:
    from carsrl.ppo.model import obs_to_tensor

    image = np.zeros((7, 7, 3), dtype=np.uint8)
    obs_a = {"image": image, "direction": 0, "mission": "pick up the yellow ball"}
    obs_b = {"image": image, "direction": 1, "mission": "pick up the green key"}
    tensor_a = obs_to_tensor(obs_a, "cpu")
    tensor_b = obs_to_tensor(obs_b, "cpu")
    assert tuple(tensor_a.shape) == (1, 39, 7, 7), f"Expected 39-channel MiniGrid tensor, got {tuple(tensor_a.shape)}"
    assert float((tensor_a - tensor_b).abs().sum()) > 0.0, "Mission/direction changes do not affect policy input."
    return "policy input includes image + 4 direction planes + 32 mission hash planes"


def _check_validation_rules() -> str:
    from carsrl.cars.appraiser import _appraisal_validation_issue
    from carsrl.cars.schema import Appraisal

    cases = [
        (
            "legacy copy",
            Appraisal(0.25, 0.80, "pick up the visible key", 0.70, 0.30, 0.10),
            "Mission: reach goal\nAgent inventory: carrying yellow key\nVisible objects: locked yellow door at view(3,4)",
            True,
        ),
        (
            "visible key ignored",
            Appraisal(0.75, 0.90, "toggle door", 0.90, 0.90, 0.10),
            "Mission: use key\nAgent inventory: carrying nothing\nVisible objects: yellow key at view(2,5)",
            True,
        ),
        (
            "visible goal ignored",
            Appraisal(0.70, 0.90, "toggle", 0.90, 0.90, 0.10),
            "Mission: reach goal\nAgent inventory: carrying nothing\nVisible objects: open yellow door at view(3,2); goal at view(3,4)",
            True,
        ),
        (
            "acceptable key pickup",
            Appraisal(0.35, 0.90, "pick up key", 0.70, 0.50, 0.20),
            "Mission: use key\nAgent inventory: carrying nothing\nVisible objects: yellow key at view(2,5)",
            False,
        ),
        (
            "acceptable goal",
            Appraisal(0.95, 0.95, "reach the goal", 0.95, 0.40, 0.05),
            "Mission: reach goal\nAgent inventory: carrying nothing\nVisible objects: open yellow door at view(3,2); goal at view(3,4)",
            False,
        ),
    ]
    for label, appraisal, state, should_reject in cases:
        issue = _appraisal_validation_issue(appraisal, state)
        if should_reject:
            assert issue is not None, f"Validation failed to reject: {label}"
        else:
            assert issue is None, f"Validation rejected valid case {label}: {issue}"
    return "semantic validation catches legacy copies and key/door/goal contradictions"


def _check_cache_namespace() -> str:
    from carsrl.cars.appraiser import QwenAppraiser
    from carsrl.cars.prompts import PROMPT_VERSION

    appraiser = QwenAppraiser(local_files_only=True)
    namespace = appraiser.cache_namespace
    assert PROMPT_VERSION in namespace, f"Qwen cache namespace does not include {PROMPT_VERSION}: {namespace}"
    return namespace


def _check_minigrid_runtime() -> str:
    from carsrl.envs.minigrid import make_minigrid_env
    from carsrl.ppo.model import obs_to_tensor

    env = make_minigrid_env("MiniGrid-DoorKey-8x8-v0", seed=0)
    obs, _ = env.reset(seed=0)
    tensor = obs_to_tensor(obs, "cpu")
    env.close()
    assert tensor.shape[1] == 39, f"MiniGrid runtime produced unexpected channel count: {tuple(tensor.shape)}"
    return f"MiniGrid reset ok, obs_tensor_shape={tuple(tensor.shape)}"


def _check_qwen_semantics(args: argparse.Namespace) -> str:
    from carsrl.cars.appraiser import QwenAppraiser
    from carsrl.cars.prompts import PROMPT_VERSION

    states = [
        (
            "visible_key",
            "Mission: use the key to open the door and reach the goal\n"
            "Agent position: (3, 4)\n"
            "Agent direction: down/south\n"
            "Agent inventory: carrying nothing\n"
            "Visible objects: yellow key at view(2,5)",
            lambda appraisal: "key" in appraisal.subgoal.lower() and 0.15 <= appraisal.phi <= 0.50,
        ),
        (
            "key_and_locked_door",
            "Mission: use the key to open the door and reach the goal\n"
            "Agent position: (3, 3)\n"
            "Agent direction: right/east\n"
            "Agent inventory: carrying yellow key\n"
            "Visible objects: locked yellow door at view(3,4)",
            lambda appraisal: any(term in appraisal.subgoal.lower() for term in ("door", "toggle", "open"))
            and appraisal.phi >= 0.35,
        ),
        (
            "visible_goal",
            "Mission: use the key to open the door and reach the goal\n"
            "Agent position: (5, 5)\n"
            "Agent direction: right/east\n"
            "Agent inventory: carrying nothing\n"
            "Visible objects: open yellow door at view(3,2); goal at view(3,4)",
            lambda appraisal: "goal" in appraisal.subgoal.lower() and appraisal.phi >= 0.75,
        ),
    ]

    appraiser = QwenAppraiser(
        model_name=args.qwen_model,
        device=args.qwen_device,
        dtype=args.qwen_dtype,
        local_files_only=args.qwen_local_files_only,
        max_new_tokens=args.qwen_max_new_tokens,
    )
    observed: list[dict[str, Any]] = []
    for label, state, predicate in states:
        appraisal = appraiser.appraise(state)
        observed.append({"label": label, **appraisal.to_dict()})
        assert appraisal.parse_error is None, f"{label} failed parse/validation: {appraisal.parse_error}"
        assert predicate(appraisal), f"{label} appraisal is semantically suspicious: {appraisal.to_dict()}"

    stats = appraiser.stats.to_dict()
    assert stats["appraiser_fallback_count"] == 0, f"Qwen fallback used: {stats}"
    status = "pass"
    detail = {
        "prompt_version": PROMPT_VERSION,
        "stats": stats,
        "observed": observed,
    }
    if stats["appraiser_validation_failures"] > 0:
        status = "warn"
    return status, json.dumps(detail, indent=2, ensure_ascii=False, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CARS experiment preflight checks.")
    parser.add_argument("--skip-compileall", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-train-smoke", action="store_true")
    parser.add_argument("--with-qwen", action="store_true")
    parser.add_argument("--qwen_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--qwen_device", default="auto")
    parser.add_argument("--qwen_dtype", default="auto")
    parser.add_argument("--qwen_max_new_tokens", type=int, default=128)
    parser.add_argument("--qwen_local_files_only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    preflight = Preflight()

    preflight.check("prompt contract", _check_prompt_contract)
    preflight.check("CARS config defaults", _check_config_defaults)
    preflight.check("serializer history boundary", _check_serializer_history_boundary)
    preflight.check("mission-aware observation tensor", _check_mission_aware_obs_tensor)
    preflight.check("appraisal validation rules", _check_validation_rules)
    preflight.check("Qwen cache namespace", _check_cache_namespace)
    preflight.check("MiniGrid runtime", _check_minigrid_runtime)

    if not args.skip_compileall:
        preflight.run_command(
            "compileall",
            [sys.executable, "-m", "compileall", "carsrl", "scripts", "tests"],
            timeout_sec=120,
        )
    if not args.skip_pytest:
        preflight.run_command("pytest", [sys.executable, "-m", "pytest", "tests"], timeout_sec=180)
    if not args.skip_train_smoke:
        preflight.run_command(
            "mock PPO+CARS train smoke",
            [
                sys.executable,
                "-m",
                "carsrl.train",
                "--algo",
                "ppo_cars",
                "--env",
                "MiniGrid-DoorKey-8x8-v0",
                "--seed",
                "101",
                "--total_steps",
                "32",
                "--num_envs",
                "2",
                "--rollout_steps",
                "16",
                "--minibatch_size",
                "32",
                "--update_epochs",
                "1",
                "--cars_appraiser",
                "mock",
                "--slm_interval",
                "8",
                "--run_dir",
                "runs/preflight",
                "--device",
                "cpu",
            ],
            timeout_sec=180,
        )
    if args.with_qwen:
        preflight.check("real Qwen semantic appraisal", lambda: _check_qwen_semantics(args))

    preflight.print_report()
    summary = preflight.summary()
    raise SystemExit(1 if summary["failed"] else 0)


if __name__ == "__main__":
    main()
