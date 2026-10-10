# 2026-10-10 — First RunPod session: local baseline (B0) + GRPO smoke run

Raw files: [`results/2026-10-10-grpo-smoke/`](../results/2026-10-10-grpo-smoke/)
Adapter (not in git): `checkpoints/2026-10-10-grpo-smoke-adapter/` (local) and `/workspace/runs/grpo_smoke/` (pod volume)

## Setup
- Pod: RunPod **Secure Cloud L40S 48 GB**, $1.09/h, pod id `0iruizy6rvu97e`, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, 60 GB volume at `/workspace`.
- Two Community Cloud pods ($0.79/h) were deleted first: the machine (`kldbzozpc4vi`) had a broken GPU (`CUDA unknown error` even with the image's own PyTorch). Lesson: check `torch.cuda.is_available()` right after a pod boots.
- Code: branch `feat/local-policy-grpo`. Libraries: torch 2.14.1+cu130, transformers 4.57.6, peft 0.21.2.
- Model: `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- Data: CMU HotpotQA server unreachable, so the official copy on Hugging Face (`hotpotqa/hotpot_qa`, `distractor`) was converted to the original JSON format. Counts match the originals: train 90,447, dev 7,405.
- Splits (`configs/evaluation/splits.json`, `holdout_v1_ids.json`, bootstrapped once, now frozen): train 85,573 / validation 4,452 / holdout 300; skipped malformed rows: train 422, dev 36.
  - Manifest hashes: holdout `8d545138…d24d`, splits `a0c78cb1…5211`.
  - The holdout is not the same 300 questions as the earlier OpenRouter run.

## Preflight
Passed. Peak GPU memory 15.1 GB / 44.4 GB, old/new log-prob drift 0.002 (bf16), LoRA trainable params 40.4M of 7.66B.

## Training config (`configs/training/grpo_smoke.json`)
64 train questions × 4 rollouts (G=4), 1 epoch, LoRA r=16 α=32 dropout 0 on all projections, lr 1e-5, ε=0.2, no KL, temperature 1.0, max 128 new tokens, top_k=3, max_searches=3, default `RewardConfig`, **`train_on_invalid: true`** (switched on after B0 showed ~43% invalid episodes, mostly invented citation titles).

Training took ~13 min (~12 s/step), peak GPU memory 17 GB.

| | Steps 1–32 | Steps 33–64 |
|---|---|---|
| Mean group reward | 0.95 | 1.15 |
| Invalid rollouts | 46% | 51% |
| EM of valid rollouts | 44% | 64% |
| Skipped steps (zero reward variance) | 13% | 22% |
| Searches per rollout | 2.07 | 1.90 |

## Holdout results (300 questions, greedy, no retries)

| Metric | B0 (un-tuned) | GRPO smoke | Δ |
|---|---|---|---|
| Completed episodes | 170 (56.7%) | 192 (64.0%) | +22 |
| EM, failures counted as 0 | 22.3% | 25.3% | +3.0 pts |
| EM (completed only) | 39.4% | 39.6% | ≈ |
| F1 (completed) | 0.501 | 0.508 | ≈ |
| Evidence coverage (completed) | 0.472 | 0.426 | −0.05 |
| Citation precision (completed) | 0.633 | 0.591 | −0.04 |
| Retrieval recall (completed) | 0.829 | 0.845 | ≈ |
| Searches per episode (completed) | 1.92 | 1.93 | ≈ |

Failure kinds:

| | B0 | GRPO smoke |
|---|---|---|
| evidence (invented/unknown citation) | 85 | 79 |
| protocol (action format) | 37 | 23 |
| budget (no answer in time) | 8 | 6 |

## Interpretation
- Main effect: **fewer invalid episodes** (130 → 108), mainly fewer format errors (37 → 23). That raises all-rows EM by 3 points.
- Answer quality on completed episodes is unchanged. Evidence coverage and citation precision dropped slightly, probably because some of the newly completed episodes are weaker ones that used to fail.
- With 300 questions, a 3-point EM difference is within noise (SE ≈ 2.4 pts). This is a pipeline smoke test, not a result to claim.
- Invented citation titles are still the biggest failure (79). 64 updates at lr 1e-5 is very little training.

## Next steps (candidates)
1. Longer run: more questions (e.g. 256–512) and/or lr 2e-5; `group_size: 8` to cut zero-variance steps.
2. Prompt fix for citations (list the exact retrieved titles in the prompt) to reduce evidence failures. Would need a new B0, since the prompt changes.
3. Paired significance test (bootstrap/McNemar) once a real run is done.

## Cost
Balance $15.00 → $13.71 at 12:08 UTC (pod still running at that point, uptime 69 min).
