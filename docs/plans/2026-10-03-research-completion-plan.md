# Needle — End-to-End Research Completion Plan

Date: 2026-10-03
Companion docs:
- Implementation detail for Phases 1–3: [2026-09-17-needle-grpo-roadmap.md](2026-09-17-needle-grpo-roadmap.md) (source of truth; not duplicated here)
- Literature and directions: [../research/2026-10-03-related-work-and-directions.md](../research/2026-10-03-related-work-and-directions.md)
- Current state: [../handoffs/2026-10-03-needle-grpo.md](../handoffs/2026-10-03-needle-grpo.md)

## Working thesis

> Training a search agent with GRPO using **turn-level evidence credit** and **correctness-gated efficiency penalties** yields more grounded (better-cited, fewer shortcut answers) and cheaper (fewer searches) multi-hop QA than outcome-only or flat trajectory-reward GRPO, at equal or better answer accuracy.

Differentiation target: CaRR/C‑GRPO (citation rubrics, trajectory-level) and STAMP (provenance credit, web search). Needle's lever is exact per-search attribution from deterministic BM25 + HotpotQA gold supporting facts.

## Budget and guardrails

- Compute: ~$30 total on RunPod L40S (~$1.09/h) → ~20–25 GPU-hours. Stop/delete pods when idle.
- The 300-example `holdout_v1` is never used for training, prompt tuning, or hyperparameter selection. Tune on `validation` only.
- Every run records config, git SHA, dataset manifest hash, seed, and library versions.
- Commits authored by Emirsyah only.

If the budget runs out before Phase 5, the minimum publishable result is Phase 4 (main comparison) on HotpotQA with one seed plus a clear limitations section.

---

## Phase 0 — Close out current state (local, ~0.5 day)

- [ ] Re-run `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest` in a clean `uv` env; fix anything broken.
- [ ] Commit research notes and this plan.
- [ ] Read CaRR and STAMP in full; write a 1-paragraph "how Needle differs" note into the research doc. Adjust the thesis if novelty collapses.

**Exit:** green CI, thesis confirmed or revised.

## Phase 1 — Local policy + GPU preflight (roadmap Chunk 3)

- [ ] Optional `training` dependency group (torch, transformers, peft, bitsandbytes), pinned.
- [ ] `src/needle/training/local_policy.py`: Qwen2.5-7B-Instruct, offline loading, LoRA (QLoRA fallback), captures prompt/completion IDs, action masks, per-token log-probs.
- [ ] `scripts/preflight_training.py`: CUDA, VRAM, one forward/backward, one interactive episode in <120s, peak memory <90%.
- [ ] Smoke-test locally with a tiny Qwen (e.g. 0.5B) on CPU/any GPU — infra only, not a result.
- [ ] Minimal RunPod launcher/runbook: pod spec, 50GB volume, HF cache pre-download, `HF_HUB_OFFLINE=1`.
- [ ] Run preflight on L40S. (~1 GPU-hour)

**Exit:** preflight passes on L40S with 7B.

## Phase 2 — Local baseline (the real comparison point)

- [ ] `scripts/evaluate_checkpoint.py` for local models, `max_retries=0`.
- [ ] Evaluate un-tuned local Qwen2.5-7B on `holdout_v1` (300). Store compact summary under `results/`. (~1–2 GPU-hours)
- [ ] Rollout parity check vs. OpenRouter on a small sample (schema/scoring parity, not identical text).
- [ ] Add the new metrics now so every later run has them:
  - **Shortcut rate**: correct answer with citation precision or evidence coverage below threshold.
  - Zero-variance group fraction (training-time).
  - First-query retrieval recall.

**Exit:** `B0` = local un-tuned baseline numbers on holdout.

## Phase 3 — GRPO smoke run (roadmap Chunk 4)

- [ ] `src/needle/training/grpo.py` + `tests/training/test_grpo_math.py` (hand-computed loss, masking, clipping).
- [ ] `scripts/train_grpo.py`, `configs/training/grpo_smoke.json` (G=4, 50–100 train examples, ε=0.2, no KL).
- [ ] Run on L40S (~2–3 GPU-hours). Watch: reward trend, zero-variance fraction, invalid-action rate, searches per episode.
- [ ] Evaluate checkpoint on holdout.
- [ ] Decision: if zero-variance >~50%, switch to G=8 or difficulty-filter the train set (score train questions with `B0` rollouts, keep those with mixed success).

**Exit:** a training pipeline that moves reward on train without collapsing; first holdout number.

## Phase 4 — Main experiments (the paper's core)

Four conditions, identical data/budget/seed/steps, only the reward/advantage differs:

| ID | Condition | Change |
|----|-----------|--------|
| R0 | Outcome-only | reward = EM (or F1); Search-R1-style baseline |
| R1 | Flat trajectory reward | current `RewardConfig` total |
| R2 | R1 + gated efficiency | search/duplicate penalties only applied when answer correct |
| R3 | R2 + turn-level credit | per-search advantage from evidence attribution (`information_gain` / gold-paragraph hits) |

Implementation:
- [ ] Reward gating flag in `RewardConfig` (small change).
- [ ] Per-component group normalization option (cheap; include in R1–R3 if it helps on validation).
- [ ] Turn-level advantage: per action span, `A_turn = A_traj + λ · normalized(evidence_gain_turn)`; tune λ on validation only.
- [ ] Unit test for turn-advantage assignment to the correct token spans.

Runs:
- [ ] Scale training to ~500–1000 examples if budget allows (estimate cost from Phase 3 throughput first).
- [ ] 4 conditions × 1 seed (~3 GPU-hours each ≈ 12h). Add a 2nd seed for R1 and R3 if budget remains.
- [ ] Evaluate all on holdout.

**Exit:** table of EM, F1, evidence coverage, citation precision, retrieval recall, searches/episode, shortcut rate, invalid rate for B0, R0–R3.

## Phase 5 — Analysis and robustness

- [ ] Bootstrap 95% CIs / paired significance on holdout (300 is small — report CIs, not just means).
- [ ] Reward ablations: leave-one-component-out on R1 (validation set, short runs) if budget allows.
- [ ] Efficiency analysis: accuracy vs. searches/episode; does R2/R3 avoid the "answer without searching" collapse?
- [ ] Shortcut analysis: shortcut rate per condition; 20–30 qualitative traces (good, shortcut, over-search, failure).
- [ ] Generalization (eval-only, no training): 2WikiMultiHopQA and/or MuSiQue subset with the same BM25 setup.
- [ ] Training dynamics plots: reward, zero-variance fraction, search count, invalid rate over steps.

## Phase 6 — Write-up

- [ ] Outline: Intro → Related work (Search-R1, CaRR, STAMP, EVO-RAG, credit assignment, efficiency) → Method (environment, reward, gating, turn credit) → Setup → Results → Analysis → Limitations → Conclusion.
- [ ] Limitations to state honestly: single dataset for training, BM25 over distractor paragraphs (not open-web), small holdout, few seeds, 7B LoRA only.
- [ ] Figures: system diagram, main results table, accuracy–efficiency plot, credit-assignment illustration on one trace.
- [ ] Reproducibility: release code, configs, split manifests, compact result JSONL, run commands.
- [ ] Target venue/workshop and deadline: _TBD_.

## Rough compute allocation (~22 GPU-hours)

| Phase | GPU-hours |
|-------|-----------|
| 1 Preflight | 1 |
| 2 Local baseline | 2 |
| 3 Smoke run + eval | 3 |
| 4 Main runs + evals | 12–14 |
| 5 Generalization evals | 2 |
| Slack for failures | 2 |

## Open decisions

- Outcome reward for R0: EM or F1?
- Turn-credit signal: `information_gain` as-is, or binary "retrieved a new gold paragraph"?
- Train set size after Phase 3 throughput measurement.
- Venue and deadline.
