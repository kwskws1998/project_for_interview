"""Aggregate and plot CARS experiment results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

_CACHE_ROOT = Path("/tmp/carsrl_cache")
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def _run_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for config_path in root.rglob("config.json"):
        run_dir = config_path.parent
        metrics_path = run_dir / "metrics.csv"
        events_path = run_dir / "events.jsonl"
        diagnostics_path = run_dir / "diagnostics.jsonl"
        if not metrics_path.exists() and not events_path.exists():
            continue
        config = _read_json(config_path)
        records.append(
            {
                "run_dir": run_dir,
                "config": config,
                "metrics_path": metrics_path,
                "events_path": events_path,
                "diagnostics_path": diagnostics_path,
            }
        )
    return records


def _steps_to_threshold(metrics: pd.DataFrame, threshold: float, window: int) -> float:
    if metrics.empty or "success" not in metrics:
        return np.nan
    data = metrics.sort_values("global_step").copy()
    rolling = data["success"].rolling(window=max(1, min(window, len(data))), min_periods=1).mean()
    passed = data.loc[rolling >= threshold, "global_step"]
    if passed.empty:
        return np.nan
    return float(passed.iloc[0])


def _auc_success(metrics: pd.DataFrame) -> float:
    if metrics.empty or "success" not in metrics or "global_step" not in metrics:
        return np.nan
    data = metrics.sort_values("global_step").copy()
    if len(data) < 2:
        return np.nan
    curve = data["success"].expanding().mean().to_numpy(dtype=np.float32)
    steps = data["global_step"].to_numpy(dtype=np.float32)
    span = float(max(steps[-1] - steps[0], 1.0))
    return float(np.trapezoid(curve, steps) / span)


def _summarize_run(record: dict[str, Any], window: int) -> dict[str, Any]:
    config = record["config"]
    metrics = pd.read_csv(record["metrics_path"]) if record["metrics_path"].exists() else pd.DataFrame()
    events = _read_jsonl(record["events_path"])
    diagnostics = _read_jsonl(record["diagnostics_path"])
    algo = config.get("algo", "unknown")
    env_id = config.get("env_id", "unknown")
    seed = int(config.get("seed", -1))
    cars = config.get("cars", {}) if isinstance(config.get("cars"), dict) else {}
    intrinsic = config.get("intrinsic", {}) if isinstance(config.get("intrinsic"), dict) else {}

    last_metrics = metrics.tail(window) if not metrics.empty else pd.DataFrame()
    last_events = events.tail(window) if not events.empty else pd.DataFrame()

    def metric_mean(name: str) -> float:
        if last_metrics.empty or name not in last_metrics:
            return np.nan
        return float(last_metrics[name].mean())

    def event_mean(name: str) -> float:
        if last_events.empty or name not in last_events:
            return np.nan
        return float(last_events[name].mean())

    def diagnostic_mean(name: str) -> float:
        if diagnostics.empty or name not in diagnostics:
            return np.nan
        return float(diagnostics[name].mean())

    phi_progress_corr = np.nan
    if not diagnostics.empty and {"phi", "progress_stage"}.issubset(diagnostics.columns):
        aligned = diagnostics[["phi", "progress_stage"]].dropna()
        if len(aligned) > 1 and aligned["phi"].std() > 0 and aligned["progress_stage"].std() > 0:
            phi_progress_corr = float(aligned["phi"].corr(aligned["progress_stage"]))

    final_step = np.nan
    if not metrics.empty and "global_step" in metrics:
        final_step = float(metrics["global_step"].max())
    elif not events.empty and "global_step" in events:
        final_step = float(events["global_step"].max())

    return {
        "run_dir": str(record["run_dir"]),
        "algo": algo,
        "env_id": env_id,
        "seed": seed,
        "beta": cars.get("beta", np.nan),
        "cars_appraiser": cars.get("appraiser", ""),
        "intrinsic_coef": intrinsic.get("coef", np.nan),
        "episodes": int(len(metrics)),
        "final_global_step": final_step,
        "success_rate_last_window": metric_mean("success"),
        "extrinsic_return_last_window": metric_mean("extrinsic_return"),
        "episode_return_last_window": metric_mean("episode_return"),
        "shaped_return_last_window": metric_mean("shaped_return"),
        "intrinsic_return_last_window": metric_mean("intrinsic_return"),
        "auc_success": _auc_success(metrics),
        "steps_to_50_success": _steps_to_threshold(metrics, 0.5, window),
        "steps_to_80_success": _steps_to_threshold(metrics, 0.8, window),
        "mean_fps_last_window": event_mean("fps"),
        "mean_shaped_reward_last_window": event_mean("mean_shaped_reward"),
        "mean_intrinsic_reward_last_window": event_mean("mean_intrinsic_reward"),
        "mean_phi_last_window": event_mean("mean_phi"),
        "mean_confidence_last_window": event_mean("mean_confidence"),
        "rnd_loss_last_window": event_mean("rnd_loss"),
        "icm_loss_last_window": event_mean("icm_loss"),
        "ride_loss_last_window": event_mean("ride_loss"),
        "noveld_loss_last_window": event_mean("noveld_loss"),
        "appraiser_uncached_calls": event_mean("appraiser_uncached_calls"),
        "appraiser_parse_failures": event_mean("appraiser_parse_failures"),
        "appraiser_mean_latency_sec": event_mean("appraiser_mean_latency_sec"),
        "appraisal_cache_hit_rate": event_mean("appraisal_cache_hit_rate"),
        "mean_progress_stage": diagnostic_mean("progress_stage"),
        "mean_distractor_actions": metric_mean("distractor_actions"),
        "mean_visible_distractor_count": diagnostic_mean("visible_distractor_count"),
        "phi_progress_corr": phi_progress_corr,
        "mean_diag_phi": diagnostic_mean("phi"),
    }


def _plot_metric(curves: pd.DataFrame, output_dir: Path, metric: str, ylabel: str, window: int) -> None:
    if curves.empty or metric not in curves:
        return
    for env_id, env_df in curves.groupby("env_id"):
        plt.figure(figsize=(8, 5))
        for algo, algo_df in env_df.groupby("algo"):
            per_seed = []
            for _, seed_df in algo_df.groupby("seed"):
                seed_df = seed_df.sort_values("global_step")
                if seed_df.empty:
                    continue
                smoothed = seed_df[metric].rolling(window=max(1, min(window, len(seed_df))), min_periods=1).mean()
                per_seed.append(pd.DataFrame({"global_step": seed_df["global_step"], metric: smoothed}))
            if not per_seed:
                continue
            merged = pd.concat(per_seed, ignore_index=True)
            grouped = merged.groupby("global_step")[metric].mean().reset_index()
            plt.plot(grouped["global_step"], grouped[metric], label=algo)
        plt.title(f"{env_id}: {ylabel}")
        plt.xlabel("Environment steps")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        safe_env = env_id.replace("/", "_")
        plt.savefig(output_dir / f"{safe_env}_{metric}.png", dpi=160)
        plt.close()


def _plot_metric_by_x(
    curves: pd.DataFrame,
    output_dir: Path,
    x_metric: str,
    y_metric: str,
    ylabel: str,
    filename_suffix: str,
    window: int,
) -> None:
    if curves.empty or x_metric not in curves or y_metric not in curves:
        return
    for env_id, env_df in curves.groupby("env_id"):
        plt.figure(figsize=(8, 5))
        for algo, algo_df in env_df.groupby("algo"):
            per_seed = []
            for _, seed_df in algo_df.groupby("seed"):
                seed_df = seed_df.sort_values(x_metric)
                if seed_df.empty:
                    continue
                smoothed = seed_df[y_metric].rolling(window=max(1, min(window, len(seed_df))), min_periods=1).mean()
                per_seed.append(pd.DataFrame({x_metric: seed_df[x_metric], y_metric: smoothed}))
            if not per_seed:
                continue
            merged = pd.concat(per_seed, ignore_index=True)
            grouped = merged.groupby(x_metric)[y_metric].mean().reset_index()
            plt.plot(grouped[x_metric], grouped[y_metric], label=algo)
        plt.title(f"{env_id}: {ylabel}")
        plt.xlabel(x_metric)
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        safe_env = env_id.replace("/", "_")
        plt.savefig(output_dir / f"{safe_env}_{filename_suffix}.png", dpi=160)
        plt.close()


def _plot_diagnostics(diagnostics: pd.DataFrame, output_dir: Path, window: int) -> None:
    if diagnostics.empty:
        return
    if {"phi", "progress_stage"}.issubset(diagnostics.columns):
        for env_id, env_df in diagnostics.groupby("env_id"):
            plt.figure(figsize=(7, 5))
            for algo, algo_df in env_df.groupby("algo"):
                aligned = algo_df[["phi", "progress_stage"]].dropna()
                if aligned.empty:
                    continue
                grouped = aligned.groupby("progress_stage")["phi"].mean().reset_index()
                plt.plot(grouped["progress_stage"], grouped["phi"], marker="o", label=algo)
            plt.title(f"{env_id}: Phi vs progress stage")
            plt.xlabel("Progress stage")
            plt.ylabel("Mean Phi")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"{env_id.replace('/', '_')}_phi_progress_alignment.png", dpi=160)
            plt.close()

    for metric, ylabel in [
        ("visible_distractor_count", "Visible distractors"),
        ("distractor_action", "Distractor actions"),
        ("progress_stage", "Progress stage"),
        ("phi", "Phi"),
    ]:
        if metric not in diagnostics:
            continue
        _plot_metric(diagnostics, output_dir, metric, ylabel, window)


def _plot_episode_timing(summary_df: pd.DataFrame, output_dir: Path) -> None:
    timing_columns = [
        "key_first_seen_step",
        "key_pickup_step",
        "door_first_seen_step",
        "door_open_step",
        "goal_first_seen_step",
        "success_step",
    ]
    existing = [col for col in timing_columns if col in summary_df.columns]
    if not existing:
        return
    data = summary_df.dropna(subset=existing, how="all")
    if data.empty:
        return
    for env_id, env_df in data.groupby("env_id"):
        grouped = env_df.groupby("algo")[existing].mean()
        if grouped.empty:
            continue
        ax = grouped.plot(kind="bar", figsize=(10, 5))
        ax.set_title(f"{env_id}: subgoal timing")
        ax.set_ylabel("Episode step")
        plt.tight_layout()
        plt.savefig(output_dir / f"{env_id.replace('/', '_')}_subgoal_timing.png", dpi=160)
        plt.close()


def aggregate(root: str | Path, output_dir: str | Path, window: int) -> pd.DataFrame:
    root = Path(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _run_records(root)
    summaries = [_summarize_run(record, window=window) for record in records]
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    curve_frames = []
    diagnostic_frames = []
    for record in records:
        metrics_path = record["metrics_path"]
        config = record["config"]
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            if not metrics.empty:
                metrics["algo"] = config.get("algo", "unknown")
                metrics["env_id"] = config.get("env_id", "unknown")
                metrics["seed"] = int(config.get("seed", -1))
                metrics["run_dir"] = str(record["run_dir"])
                curve_frames.append(metrics)
        diagnostics = _read_jsonl(record["diagnostics_path"])
        if not diagnostics.empty:
            diagnostics["algo"] = config.get("algo", "unknown")
            diagnostics["env_id"] = config.get("env_id", "unknown")
            diagnostics["seed"] = int(config.get("seed", -1))
            diagnostics["run_dir"] = str(record["run_dir"])
            diagnostic_frames.append(diagnostics)
    curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    diagnostic_frames = [frame for frame in diagnostic_frames if not frame.empty and not frame.dropna(how="all").empty]
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True) if diagnostic_frames else pd.DataFrame()
    if not curves.empty:
        curves.to_csv(output_dir / "episode_curves.csv", index=False)
        _plot_metric(curves, output_dir, "success", "Success rate", window)
        _plot_metric(curves, output_dir, "extrinsic_return", "Extrinsic return", window)
        _plot_metric(curves, output_dir, "episode_return", "Training return", window)
        _plot_metric(curves, output_dir, "episode_length", "Episode length", window)
        _plot_metric(curves, output_dir, "distractor_actions", "Distractor actions per episode", window)
        _plot_metric(curves, output_dir, "mean_progress_stage", "Mean progress stage per episode", window)
        _plot_metric_by_x(curves, output_dir, "wall_time", "success", "Success rate", "success_by_wall_time", window)
        _plot_episode_timing(curves, output_dir)
    if not diagnostics.empty:
        diagnostics.to_csv(output_dir / "diagnostics_curves.csv", index=False)
        _plot_diagnostics(diagnostics, output_dir, window)
    return summary_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate and plot CARS experiment runs.")
    parser.add_argument("--root", default="runs")
    parser.add_argument("--output_dir", default="runs/plots")
    parser.add_argument("--window", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = aggregate(args.root, args.output_dir, args.window)
    print(f"Wrote {len(summary)} run summaries to {Path(args.output_dir) / 'summary.csv'}")


if __name__ == "__main__":
    main()
