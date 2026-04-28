# Implementation Notes

## Source Assets

The workspace includes:

- `NovelD-master.zip`: official NovelD code. It uses an actor-learner/V-trace style implementation derived from RIDE, not a simple PPO loop.
- `PPO-PyTorch-master.zip`: minimal PPO reference implementation, originally built for classic Gym tasks.
- `NovelD- A Simple yet Effective Exploration Criterion.pdf`
- `Proximal Policy Optimization Algorithms.pdf`
- `QWEN2.5-MATH TECHNICAL REPORT- TOWARD MATHEMATICAL EXPERT MODEL VIA SELFIMPROVEMENT.pdf`

## Design Decision

For fair comparison, this project should not compare CARS running on a new PPO stack against NovelD numbers from the 2021 paper. Instead, implement PPO once and plug reward modules into that same pipeline:

- no intrinsic reward
- RND intrinsic reward
- ICM intrinsic reward
- RIDE intrinsic reward
- NovelD-style intrinsic reward
- CARS potential shaping
- random/shuffled/heuristic potential controls

This keeps MiniGrid version, observation preprocessing, random seeds, PPO update logic, and logging protocol aligned.

## Intrinsic Reward Baselines

`ppo_rnd` implements a standard random network distillation intrinsic reward inside the shared PPO trainer:

```text
r_intrinsic = coef * prediction_error(s_next)
```

The reward-time prediction error is the L2 distance between a frozen random target embedding and a trainable predictor embedding; the predictor itself is trained with MSE.

`ppo_icm` implements an intrinsic curiosity module with a learned encoder, inverse dynamics model, and forward dynamics model:

```text
r_intrinsic = coef * || forward(phi(s), a) - phi(s_next) ||_2
```

The update minimizes forward prediction MSE plus a small inverse-action cross-entropy term.

`ppo_ride` implements a RIDE-style impact reward using the same learned ICM embedding:

```text
r_intrinsic = coef * || phi(s_next) - phi(s) ||_2 / sqrt(N_episode(s_next))
```

## NovelD Baseline

`ppo_noveld` reimplements the key NovelD reward idea inside the shared PPO trainer:

```text
r_intrinsic = coef * first_visit(s_next) * max(error(s_next) - scale_fac * error(s), 0)
```

where `error` is the L2 distance between a frozen random target embedding and a trainable predictor embedding. This is not a verbatim import of the official actor-learner code; it is a same-pipeline baseline so PPO, MiniGrid version, observation preprocessing, seeds, logging, and training budget are controlled.

## Control Baselines And Ablations

The trainer supports these CARS-related algorithm aliases:

- `ppo_random_phi`: CARS potential shaping with random Phi scores.
- `ppo_shuffled_phi`: CARS potential shaping where Phi values come from the selected base appraiser's historical Phi distribution but are assigned to different states.
- `ppo_heuristic_phi`: CARS potential shaping with a hand-coded heuristic Phi.
- `ppo_cars_no_confidence`: CARS potential shaping without multiplying by confidence.
- `ppo_cars_direct`: direct Phi reward ablation, intentionally not the main method.
- `ppo_rnd_cars`: combined RND intrinsic reward plus CARS potential shaping.
- `ppo_icm_cars`: combined ICM intrinsic reward plus CARS potential shaping.
- `ppo_ride_cars`: combined RIDE intrinsic reward plus CARS potential shaping.
- `ppo_noveld_cars`: combined NovelD intrinsic reward plus CARS potential shaping.

These aliases keep their own `algo` names in logs and run directories while reusing the same PPO rollout code.

## MiniGrid Version Note

NovelD 2021 used the earlier MiniGrid ecosystem. The 2023 MiniGrid & MiniWorld NeurIPS benchmark paper formalized the environment family later. Experiments here should report the exact `minigrid` and `gymnasium` package versions.

## Hardware Plan

Use the MacBook as a development and smoke-test machine. Treat the RTX 3090 24 GB machine as the main experiment target.

For 3090 runs, use `--device cuda`, `--slm_device cuda`, and `--slm_dtype float16`. Qwen2.5-1.5B should fit comfortably in fp16 together with the PPO policy and MiniGrid rollout buffers. The launcher runs commands sequentially by default, which is preferable for CARS because every active CARS process loads its own Qwen model. PPO-only and NovelD-only runs can be parallelized more aggressively if desired.

The 4-bit option is exposed as `--slm_load_in_4bit`, but it should be treated as a separate systems ablation rather than the default main result.
