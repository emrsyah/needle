# Needle — Next-Session Handoff

## Current position

Needle has a deterministic HotpotQA research substrate and an OpenRouter Qwen baseline. The project has **not** run local Qwen inference or GRPO training yet.

The detailed implementation roadmap is [docs/plans/2026-09-17-needle-grpo-roadmap.md](../plans/2026-09-17-needle-grpo-roadmap.md). Do not duplicate or replace that plan; use this document for current context and the next handoff.

## Completed

- Immutable HotpotQA models and tiny fixtures.
- Deterministic BM25 retrieval.
- Resettable, bounded `SearchEnvironment` with `SEARCH[...]` and `ANSWER[...]` actions.
- Citation validation and immutable `EpisodeTrajectory`.
- Framework-independent deterministic reward evaluator.
- Strict action parser and OpenRouter Qwen baseline runner.
- Reproducible split preparation, immutable 300-example holdout convention, JSONL evaluation analysis, and evaluation configs.
- Immutable interactive rollout records and a policy-independent collector with typed valid/invalid/exhausted/transport outcomes.

Key current files:

- `src/needle/rewards/`
- `src/needle/training/rollouts.py`
- `src/needle/training/collector.py`
- `scripts/evaluate_subset.py`
- `scripts/analyze_eval.py`

## Baseline context

The recorded 300-example OpenRouter Qwen run completed 257/300 episodes with retries. Exact match was 29% over all rows and 33.9% over completed rows. This is an inference/protocol baseline, not a training result.

## Immediate next work

1. Implement the local Qwen2.5-7B-Instruct policy adapter with Transformers/PEFT.
2. Add the CUDA/offline preflight: model/tokenizer loading, one forward/backward pass, one interactive episode, token/action log-prob capture, and VRAM reporting.
3. Add the smallest provider launcher for one remote GPU job.
4. Run a first short GRPO experiment: 50–100 training examples, `G=4`, one checkpoint, then evaluate on the untouched 300-example holdout.

The training path must use local model weights and offline loading. OpenRouter remains an ancillary baseline only; its API key must not be used by the training CLI.

## Compute decision

Modal is not currently usable for this account's GPU workflow. Use **RunPod L40S** as the default provider for the first remote run:

- 48GB VRAM, 16 vCPU, about 62GB RAM.
- User screenshot showed `$1.09/hour`, billed per millisecond.
- Add roughly 50GB persistent storage; the 30GB container disk is likely tight for model cache and checkpoints.
- With a total budget of `$30`, target roughly 20–25 active hours and stop/delete the Pod whenever it is idle.

A second provider screenshot showed an L40S around `$0.97/hour`, but the UI presented a reservation/pool and a roughly `$697/month` estimate. Do not reserve it until its billing is confirmed as pay-as-you-go with no monthly commitment. Beam's published on-demand pricing can be cheaper, but availability and billing mode must be verified before using it.

## Git state at handoff

- Branch: `codex/needle-grpo-roadmap`
- Latest implementation commit before this document: `14dd044 feat: add interactive rollout boundary`
- `main`/`origin/main` still points to the earlier baseline commit; this branch is intended to be merged through a pull request.
- Commit author must remain Emirsyah only; do not add an AI/Codex co-author.

## Verification note

The repository was clean before adding this handoff. A fresh local test/lint run was not completed in the current Windows shell because `uv` could not access its global cache and the globally installed pytest hit a `pyreadline`/Python 3.13 compatibility error. Re-run the project commands in a clean `uv` environment before merging:

```text
uv run ruff check .
uv run ruff format --check .
uv run pytest
```
