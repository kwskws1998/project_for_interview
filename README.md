# CARS: Cognitive Appraisal Reward Shaping

Research prototype for **CARS: Cognitive Appraisal Reward Shaping with Frozen Small Language Models for Sparse-Reward Reinforcement Learning**.

The goal is to compare novelty-only intrinsic rewards against a frozen small language model used as a cognitive goal-progress potential estimator in sparse-reward MiniGrid tasks.

## Research Setup

CARS uses PPO as the base RL algorithm. A frozen SLM, by default `Qwen/Qwen2.5-1.5B-Instruct`, receives a compact text serialization of the MiniGrid state and returns a JSON appraisal:

```json
{
  "phi": 0.62,
  "confidence": 0.84,
  "subgoal": "open the locked door",
  "affordance": 0.78,
  "novelty": 0.20,
  "risk": 0.05
}
```

The main method does not use this as direct reward. It uses potential-based shaping:

```text
r_shape = clip(beta * confidence * (gamma * Phi(s_next) - Phi(s)), -0.05, 0.05)
r_total = r_env + r_shape
```

Important implementation guardrails:

- PPO and intrinsic baselines now observe MiniGrid image, agent direction, and a hashed mission feature vector, so KeyCorridor target missions are not hidden from the policy.
- The main CARS serializer excludes recent action/event history by default to keep Phi closer to a state potential. Use `--cars_history` only for the history ablation.
- Scheduler-skipped CARS steps are neutral by default. Use `--cars_shape_on_skip` only to reproduce the older stale-Phi shaping behavior.
- The Qwen prompt no longer contains a copyable numeric JSON example. Qwen outputs are validated for state consistency, retried once on semantic contradictions, and logged as fallback failures if still invalid.

## Current Implementation Status

This repository is being built incrementally.

- Portion 1: project scaffold, configs, MiniGrid serializer, frozen appraiser interface, JSONL cache, reward shaping utility.
- Portion 2: PPO training loop for Gymnasium/MiniGrid.
- Portion 3: CARS integration into PPO rollout collection and smoke test.
- Portion 4: NovelD/RND/ICM/RIDE-style baseline modules inside the same PPO pipeline.
- Portion 5: experiment launcher, plotting, and reproducibility report.
- Portion 9: shuffled-Phi control baseline for testing whether CARS gains come from semantic state alignment rather than the marginal Phi distribution alone.
- Portion 10: RND intrinsic reward baseline in the same PPO/MiniGrid pipeline.
- Portion 11: ICM and RIDE curiosity baselines in the same PPO/MiniGrid pipeline.
- Portion 12: review hardening for Qwen appraisal validation, mission-aware policy inputs, neutral skipped-step shaping, and state-only CARS defaults.

## Target Environments

Sanity checks:

```text
MiniGrid-DoorKey-8x8-v0
MiniGrid-DoorKey-16x16-v0
```

Main benchmarks:

```text
MiniGrid-KeyCorridorS4R3-v0
MiniGrid-KeyCorridorS5R3-v0
MiniGrid-KeyCorridorS6R3-v0
MiniGrid-ObstructedMaze-2Dlh-v0
MiniGrid-ObstructedMaze-2Dlhb-v0
MiniGrid-ObstructedMaze-Full-v0
```

## Planned Commands

Current PPO-only smoke command:

```bash
python -m carsrl.train --algo ppo --env MiniGrid-DoorKey-8x8-v0 --seed 0 --total_steps 256 --num_envs 1 --rollout_steps 64 --minibatch_size 64 --update_epochs 1 --run_dir runs/smoke
```

Current PPO+CARS mock smoke command:

```bash
python -m carsrl.train --algo ppo_cars --cars_appraiser mock --env MiniGrid-DoorKey-8x8-v0 --seed 0 --total_steps 64 --num_envs 1 --rollout_steps 16 --minibatch_size 16 --update_epochs 1 --run_dir runs/smoke
```

Real Qwen appraiser smoke command:

```bash
python scripts/smoke_qwen_appraiser.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --env MiniGrid-DoorKey-8x8-v0 \
  --seed 0 \
  --device auto \
  --cache runs/qwen_smoke/appraisals.jsonl
```

Real PPO+CARS Qwen smoke command:

```bash
python -m carsrl.train \
  --algo ppo_cars \
  --cars_appraiser qwen \
  --slm_model Qwen/Qwen2.5-1.5B-Instruct \
  --env MiniGrid-DoorKey-8x8-v0 \
  --seed 0 \
  --total_steps 32 \
  --num_envs 1 \
  --rollout_steps 16 \
  --minibatch_size 16 \
  --update_epochs 1 \
  --slm_interval 16 \
  --run_dir runs/qwen_smoke
```

Current PPO+RND smoke command:

```bash
python -m carsrl.train --algo ppo_rnd --env MiniGrid-DoorKey-8x8-v0 --seed 0 --total_steps 64 --num_envs 1 --rollout_steps 16 --minibatch_size 16 --update_epochs 1 --intrinsic_coef 0.05 --run_dir runs/smoke
```

Current PPO+ICM/RIDE smoke commands:

```bash
python -m carsrl.train --algo ppo_icm --env MiniGrid-DoorKey-8x8-v0 --seed 0 --total_steps 64 --num_envs 1 --rollout_steps 16 --minibatch_size 16 --update_epochs 1 --intrinsic_coef 0.05 --run_dir runs/smoke
python -m carsrl.train --algo ppo_ride --env MiniGrid-DoorKey-8x8-v0 --seed 0 --total_steps 64 --num_envs 1 --rollout_steps 16 --minibatch_size 16 --update_epochs 1 --intrinsic_coef 0.05 --run_dir runs/smoke
```

Current PPO+NovelD smoke command:

```bash
python -m carsrl.train --algo ppo_noveld --env MiniGrid-DoorKey-8x8-v0 --seed 0 --total_steps 64 --num_envs 1 --rollout_steps 16 --minibatch_size 16 --update_epochs 1 --intrinsic_coef 0.05 --run_dir runs/smoke
```

CARS control baseline smoke commands:

```bash
python -m carsrl.train --algo ppo_random_phi --env MiniGrid-DoorKey-8x8-v0 --seed 0 --cars_appraiser random --total_steps 64 --num_envs 1 --rollout_steps 16 --minibatch_size 16 --update_epochs 1 --run_dir runs/smoke
python -m carsrl.train --algo ppo_shuffled_phi --env MiniGrid-DoorKey-8x8-v0 --seed 0 --cars_appraiser shuffled_mock --total_steps 64 --num_envs 1 --rollout_steps 16 --minibatch_size 16 --update_epochs 1 --run_dir runs/smoke
python -m carsrl.train --algo ppo_heuristic_phi --env MiniGrid-DoorKey-8x8-v0 --seed 0 --cars_appraiser heuristic --total_steps 64 --num_envs 1 --rollout_steps 16 --minibatch_size 16 --update_epochs 1 --run_dir runs/smoke
python -m carsrl.train --algo ppo_cars_no_confidence --env MiniGrid-DoorKey-8x8-v0 --seed 0 --cars_appraiser mock --no_cars_confidence --total_steps 64 --num_envs 1 --rollout_steps 16 --minibatch_size 16 --update_epochs 1 --run_dir runs/smoke
python -m carsrl.train --algo ppo_cars_direct --env MiniGrid-DoorKey-8x8-v0 --seed 0 --cars_appraiser mock --cars_direct_reward --total_steps 64 --num_envs 1 --rollout_steps 16 --minibatch_size 16 --update_epochs 1 --run_dir runs/smoke
python -m carsrl.train --algo ppo_rnd_cars --env MiniGrid-DoorKey-8x8-v0 --seed 0 --cars_appraiser mock --intrinsic_coef 0.05 --total_steps 64 --num_envs 1 --rollout_steps 16 --minibatch_size 16 --update_epochs 1 --run_dir runs/smoke
python -m carsrl.train --algo ppo_icm_cars --env MiniGrid-DoorKey-8x8-v0 --seed 0 --cars_appraiser mock --intrinsic_coef 0.05 --total_steps 64 --num_envs 1 --rollout_steps 16 --minibatch_size 16 --update_epochs 1 --run_dir runs/smoke
python -m carsrl.train --algo ppo_ride_cars --env MiniGrid-DoorKey-8x8-v0 --seed 0 --cars_appraiser mock --intrinsic_coef 0.05 --total_steps 64 --num_envs 1 --rollout_steps 16 --minibatch_size 16 --update_epochs 1 --run_dir runs/smoke
python -m carsrl.train --algo ppo_noveld_cars --env MiniGrid-DoorKey-8x8-v0 --seed 0 --cars_appraiser mock --intrinsic_coef 0.05 --total_steps 64 --num_envs 1 --rollout_steps 16 --minibatch_size 16 --update_epochs 1 --run_dir runs/smoke
```

Experiment launcher dry-run:

```bash
python scripts/launch_experiments.py --suite sanity --algos ppo ppo_rnd ppo_icm ppo_ride ppo_noveld ppo_cars --total_steps 1000000
```

Ablation launcher dry-run:

```bash
python scripts/launch_experiments.py --suite sanity --algos ppo ppo_rnd ppo_icm ppo_ride ppo_noveld ppo_cars ppo_random_phi ppo_shuffled_phi ppo_heuristic_phi ppo_cars_no_confidence ppo_cars_direct ppo_rnd_cars ppo_icm_cars ppo_ride_cars ppo_noveld_cars --beta_ablation --total_steps 1000000
```

Actually execute the generated sweep:

```bash
python scripts/launch_experiments.py --suite sanity --algos ppo ppo_rnd ppo_icm ppo_ride ppo_noveld ppo_cars --total_steps 1000000 --execute --stop_on_failure
```

Enable Weights & Biases logging for a single run:

```bash
python -m carsrl.train \
  --algo ppo_cars \
  --env MiniGrid-DoorKey-8x8-v0 \
  --seed 0 \
  --cars_appraiser mock \
  --total_steps 64 \
  --wandb \
  --wandb_project carsrl \
  --wandb_group smoke \
  --wandb_mode offline
```

Evaluate a trained checkpoint:

```bash
python -m carsrl.evaluate --checkpoint runs/smoke/example/final_model.pt --episodes 20
```

Aggregate and plot runs:

```bash
python scripts/plot_results.py --root runs/experiments --output_dir runs/plots
```

Diagnostics outputs include:

```text
diagnostics.jsonl
runs/plots/diagnostics_curves.csv
runs/plots/*_phi_progress_alignment.png
runs/plots/*_progress_stage.png
runs/plots/*_visible_distractor_count.png
runs/plots/*_distractor_action.png
runs/plots/*_success_by_wall_time.png
runs/plots/*_subgoal_timing.png
```

Planned main CARS command shape:

```bash
python -m carsrl.train --algo ppo_cars --env MiniGrid-DoorKey-8x8-v0 --seed 0 --beta 0.1 --slm_model Qwen/Qwen2.5-1.5B-Instruct --slm_interval 8
```

## RTX 3090 Main Training

Use the MacBook for code edits and smoke tests. Use the RTX 3090 machine for the main sweeps, especially Qwen-backed CARS runs.

Recommended CUDA environment setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install the PyTorch CUDA wheel that matches the server driver/CUDA runtime.
# Example only; if the server has a different CUDA stack, use the PyTorch selector.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

python -m pip install -r requirements.txt
```

Check GPU visibility:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Download and smoke-test Qwen on the 3090:

```bash
python scripts/smoke_qwen_appraiser.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --env MiniGrid-DoorKey-8x8-v0 \
  --seed 0 \
  --device cuda \
  --dtype float16 \
  --max_new_tokens 96 \
  --cache runs/qwen_smoke/appraisals_3090.jsonl
```

After the model is downloaded once, add `--local_files_only` for reproducible offline runs.

Main CARS command on RTX 3090:

```bash
python -m carsrl.train \
  --algo ppo_cars \
  --cars_appraiser qwen \
  --env MiniGrid-DoorKey-8x8-v0 \
  --seed 0 \
  --beta 0.1 \
  --device cuda \
  --slm_model Qwen/Qwen2.5-1.5B-Instruct \
  --slm_device cuda \
  --slm_dtype float16 \
  --slm_max_new_tokens 96 \
  --slm_interval 8 \
  --cars_cache_path cars_cache_seed0.jsonl \
  --total_steps 1000000 \
  --num_envs 8 \
  --rollout_steps 128 \
  --minibatch_size 256 \
  --run_dir runs/experiments_3090
```

Shuffled-Phi control on RTX 3090:

```bash
python -m carsrl.train \
  --algo ppo_shuffled_phi \
  --cars_appraiser shuffled_qwen \
  --env MiniGrid-DoorKey-8x8-v0 \
  --seed 0 \
  --beta 0.1 \
  --device cuda \
  --slm_model Qwen/Qwen2.5-1.5B-Instruct \
  --slm_device cuda \
  --slm_dtype float16 \
  --slm_max_new_tokens 96 \
  --slm_interval 8 \
  --cars_cache_path cars_cache_shuffled_seed0.jsonl \
  --total_steps 1000000 \
  --num_envs 8 \
  --rollout_steps 128 \
  --minibatch_size 256 \
  --run_dir runs/experiments_3090
```

Launcher dry-run for 3090:

```bash
python scripts/launch_experiments.py \
  --suite sanity \
  --algos ppo ppo_rnd ppo_icm ppo_ride ppo_noveld ppo_cars \
  --device cuda \
  --slm_device cuda \
  --slm_dtype float16 \
  --slm_max_new_tokens 96 \
  --slm_interval 8 \
  --total_steps 1000000
```

Actually execute:

```bash
python scripts/launch_experiments.py \
  --suite sanity \
  --algos ppo ppo_rnd ppo_icm ppo_ride ppo_noveld ppo_cars \
  --device cuda \
  --slm_device cuda \
  --slm_dtype float16 \
  --slm_max_new_tokens 96 \
  --slm_interval 8 \
  --total_steps 1000000 \
  --wandb \
  --wandb_project carsrl \
  --execute \
  --stop_on_failure
```

3090 notes:

- A 24 GB RTX 3090 should comfortably hold Qwen2.5-1.5B in float16 plus the MiniGrid PPO policy. Start without 4-bit quantization.
- Avoid running many Qwen-backed CARS jobs concurrently on one GPU; each process loads its own Qwen copy. Run CARS sweeps sequentially, or parallelize PPO/NovelD baselines separately.
- Keep `--slm_interval 8` or `--cars_schedule event_or_every_n` for main experiments. `every_step` is useful for ablation but expensive.
- Use per-run caches such as `cars_cache_seed0.jsonl`; repeated states then skip Qwen generation.
- If VRAM becomes tight, try `--slm_load_in_4bit`, but report that separately because quantization changes the appraiser.
- Watch `appraiser_validation_failures`, `appraiser_retry_count`, and `appraiser_fallback_count` in `events.jsonl`; these are the diagnostics that show when the frozen SLM is or is not behaving like a semantic potential estimator.

Smoke test command for the current scaffold:

```bash
python scripts/smoke_cars_core.py
```

Preflight check before any expensive run:

```bash
python scripts/preflight_checks.py
```

Optional real-Qwen semantic preflight:

```bash
python scripts/preflight_checks.py --with-qwen --qwen_device cuda --qwen_dtype float16 --qwen_local_files_only
```

## Fair Comparison Note

NovelD predates the 2023 MiniGrid & MiniWorld benchmark paper and used the earlier `gym-minigrid` environment family. For a fair CARS comparison, NovelD should be reimplemented or adapted as an intrinsic reward module inside the same current Gymnasium/Farama MiniGrid PPO pipeline, with identical seeds, observation preprocessing, PPO hyperparameters, and training budgets.

The potential-based shaping equation should be described as a PBRS-inspired formulation, not as a strict Ng et al. policy-invariance guarantee. Confidence scaling, clipping, partial observability, scheduled stale estimates, and optional history all weaken the exact theorem conditions. The default implementation therefore keeps history off and logs alignment diagnostics explicitly.
