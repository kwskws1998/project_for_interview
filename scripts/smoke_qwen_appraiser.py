"""Smoke test the real Qwen appraiser on one MiniGrid state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carsrl.cars import AppraisalCache, MiniGridStateSerializer, QwenAppraiser
from carsrl.cars.serializer import EpisodeTrace
from carsrl.envs.minigrid import make_minigrid_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test Qwen CARS appraisal on one MiniGrid state.")
    parser.add_argument("--env", default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--cache", default="runs/qwen_smoke/appraisals.jsonl")
    parser.add_argument("--local_files_only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    env = make_minigrid_env(args.env, seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    serializer = MiniGridStateSerializer(include_history=False)
    trace = EpisodeTrace()
    serialized = serializer.serialize(env, obs, trace=trace)
    cache = AppraisalCache(args.cache)
    appraiser = QwenAppraiser(
        model_name=args.model,
        cache=cache,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        local_files_only=args.local_files_only,
    )
    appraisal = appraiser.appraise(serialized)
    env.close()

    print("Serialized state:")
    print(serialized)
    print("\nAppraisal:")
    print(json.dumps(appraisal.to_dict(), indent=2, ensure_ascii=False))
    print("\nStats:")
    stats = appraiser.stats.to_dict()
    stats["cache_hit_rate"] = cache.stats.hit_rate
    stats["cache_size"] = len(cache)
    print(json.dumps(stats, indent=2, sort_keys=True))
    if appraisal.parse_error:
        raise SystemExit(f"Qwen appraisal parse failed: {appraisal.parse_error}")


if __name__ == "__main__":
    main()
