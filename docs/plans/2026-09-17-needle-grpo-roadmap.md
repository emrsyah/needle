# Needle GRPO Research Roadmap Implementation Plan

> **For agentic workers:** REQUIRED: Use subagent-driven-development or executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Needle from the deterministic evaluator and OpenRouter baseline to a first defensible interactive GRPO result without over-engineering the research prototype.

**Architecture:** Keep `data`, `retrieval`, `environment`, and `rewards` framework-independent and deterministic. Add a thin rollout/training boundary that runs fresh environments per rollout, preserves raw action traces, and converts invalid episodes into explicit protocol outcomes. Use OpenRouter only for baseline inference; local model weights are required for gradient-based GRPO.

**Tech Stack:** Python 3.11, `uv`, PyTorch, Transformers, Qwen2.5-7B-Instruct, TRL/GRPO, the existing BM25 environment, and JSONL experiment logs.

---

## Current state and constraints

- Deterministic data models, BM25 retrieval, search environment, trajectory, reward breakdown, strict action parser, OpenRouter runner, retry/canonicalization, and baseline harness exist.
- A 300-example dev subset has been evaluated. It is a holdout reference, not training data.
- The latest 300-example Qwen run with two inference retries completed 257/300 episodes; exact match was 29% over all rows and 33.9% over completed rows.
- The latest prompt budget rules are currently uncommitted in the working tree. Isolate and commit them before starting a new training branch; do not overwrite unrelated user changes.
- Do not add more broad correctness tests. Keep only smoke tests and invariants around new training boundaries.

## Research decisions to lock

- Training rollouts use `max_retries=0`; retries are baseline/debug behavior only.
- A failed protocol/environment rollout receives a transparent fixed protocol reward (initially `-1.0`) and is logged separately from valid trajectory rewards.
- `information_gain` remains diagnostic and is not part of the first GRPO scalar.
- The 300-example holdout is never used for training or prompt selection.
- The first GRPO experiment uses the same Qwen2.5-7B-Instruct checkpoint as the OpenRouter baseline, if GPU memory allows.
- The first experiment optimizes signal, not final benchmark performance: 50–100 training examples, 4 rollouts per prompt, one short run, then holdout evaluation.

---

## Chunk 1: Freeze the data and baseline reference

### Task 1: Isolate the current baseline state

**Files:**
- Modify: `src/needle/inference/runner.py`
- Modify: `tests/inference/test_inference.py`

- [ ] Run `git status` and `git diff` first; create a feature branch from the current commit, commit only the in-scope prompt budget rules, and leave unrelated dirty paths untouched.
- [ ] Record the complete `git status --short` output, including untracked paths, before editing; never use `git add .`, `git clean`, stash, reset, or checkout to hide unrelated work.
- [ ] Before staging, inspect `git diff -- src/needle/inference/runner.py tests/inference/test_inference.py`; after staging, inspect `git diff --cached --name-only` and `git diff --cached --` and verify that only intended hunks are present. If an unrelated hunk or untracked path appears, stop and preserve it rather than committing it.
- [ ] Before staging, inspect `git diff -- src/needle/inference/runner.py tests/inference/test_inference.py`; after staging, inspect `git diff --cached --name-only` and `git diff --cached --` and verify that only intended hunks are present. If an unrelated hunk appears, stop and preserve it rather than committing it.
- [ ] Run the focused inference smoke test and the full quality gate with a workspace-local pytest temp directory.
- [ ] Commit only these two intended files with author `Emirsyah`; update documentation/configuration in separate owned commits.

### Task 2: Make the split reproducible

**Files:**
- Create: `scripts/prepare_splits.py`
- Create: `configs/evaluation/splits.json`
- Create: `configs/evaluation/holdout_v1_ids.json`
- Test: one smoke invocation, not a large unit-test suite

- [ ] Load the official train/dev source through the existing validated model boundary, skipping malformed rows only with a counted and logged reason.
- [ ] Record the exact source URL, dataset filename/version, download date, byte count, and SHA-256 for every source file used.
- [ ] Define source membership explicitly: `holdout_v1` is the persisted 300-ID dev sample; `train` and `validation` come only from the train source. Deduplicate by first occurrence of the original non-empty `_id` string, sort IDs by their UTF-8 byte sequence, and never trim/case-fold IDs.
- [ ] Assign train/validation by `digest = sha256(f"{seed}:{question_id}".encode("utf-8"))`, `bucket = int.from_bytes(digest[:8], "big") / 2**64`, placing `bucket < 0.95` in train and the rest in validation; record algorithm version and seed.
- [ ] Produce deterministic `train`, `validation`, and `holdout` ID files using that fixed seed and algorithm, plus the source position for each ID.
- [ ] Assert zero ID overlap between splits and print counts plus skipped-row counts.
- [ ] Persist the 300 holdout IDs and a manifest hash as `holdout_v1`; do not silently replace it with a new random sample.
- [ ] Record the corpus construction and BM25 configuration (`top_k`, tokenizer revision, and `max_searches`) alongside the split manifest.
- [ ] Commit the split manifest and the script, not the downloaded dataset.

### Task 3: Add lightweight result analysis

**Files:**
- Create: `scripts/analyze_eval.py`
- Create: `configs/evaluation/shared_environment.json`
- Create: `configs/evaluation/openrouter_baseline.json`
- Create: `configs/evaluation/local_baseline.json`
- Create: `configs/evaluation/grpo_eval.json`
- Modify: `scripts/evaluate_subset.py` only if a missing field blocks analysis

- [ ] Read `episodes.jsonl` and group failures into protocol, budget, transport, retrieval, answer, and evidence categories.
- [ ] Print reward quantiles, completion rate, exact match, F1, evidence coverage, citation precision, retrieval recall, search count, and token totals.
- [ ] Verify that the report can be regenerated from JSONL without network access.
- [ ] Create `configs/evaluation/shared_environment.json` containing dataset manifest hash, prompt-builder revision, seed policy, `top_k`, and `max_searches`.
- [ ] Create separate `configs/evaluation/openrouter_baseline.json`, `configs/evaluation/local_baseline.json`, and `configs/evaluation/grpo_eval.json` containing model/tokenizer revisions, generation settings, and `max_retries=0`.
- [ ] Define the local unfine-tuned checkpoint with the exact model revision, tokenizer revision, prompt serialization, and generation config used by GRPO as the primary baseline. Keep OpenRouter as an ancillary provider/transport reference only; never claim an OpenRouter-vs-local change is a training improvement.
- [ ] Store only compact summaries under `results/`; keep large raw runs outside Git unless explicitly needed.
- [ ] Run and store a separate 300-example Qwen baseline with `max_retries=0`; keep the earlier retry-enabled run as a protocol-repair reference, not as the GRPO comparison baseline.

**Exit gate:** the same command can regenerate baseline metrics and the 300-example holdout is immutable by convention.

**Script ownership:** `evaluate_subset.py` owns baseline collection and row serialization; `analyze_eval.py` is read-only analysis; `train_grpo.py` owns training orchestration and checkpoint paths. Do not move training logic into the baseline harness.

---

## Chunk 2: Build the interactive rollout boundary

### Task 4: Represent valid and invalid rollouts

**Files:**
- Create: `src/needle/training/__init__.py`
- Create: `src/needle/training/rollouts.py`
- Create: `tests/training/test_rollouts.py`

- [ ] Add immutable records for action steps, observation text, generated token IDs, action-token masks, old log-probabilities, model metadata, terminal trajectory, failure kind, and scalar training reward.
- [ ] Define canonical per-turn serialization: store exact prompt text and prompt IDs separately from completion IDs; use right padding; action-token spans are offsets within completion IDs; with shifted causal logits `logits[:, :-1] -> input_ids[:, 1:]`, completion token `j` is scored at full target index `prompt_length + j` and logit index `prompt_length + j - 1`.
- [ ] Store detached `old_logp` as one value per generated action token; valid token-bearing rollouts must have a non-empty action mask, while invalid/transport rollouts have no gradient-bearing sample.
- [ ] Keep `EpisodeTrajectory` as the source of truth for valid episodes; do not put training state into `SearchEnvironment`.
- [ ] Define the protocol reward mapping explicitly: valid episode uses `RewardBreakdown.total`; invalid action and exhausted search terminate immediately with fixed `-1.0`; transport failure is logged separately and contributes no gradient sample unless a later policy-error policy is explicitly chosen.
- [ ] Preserve every raw model action and validator error for later error analysis.
- [ ] Serialize one rollout as JSON-safe metadata plus tensor sidecars or a documented tensor serialization; keep the framework-independent record separate from trainer tensors.
- [ ] Add smoke coverage for one valid episode and one invalid episode; assert invalid/exhausted reward is exactly `-1.0` and transport errors create no gradient-bearing sample.

### Task 5: Implement a policy-independent interactive collector

**Files:**
- Create: `src/needle/training/collector.py`
- Create: `tests/training/test_collector.py`

- [ ] Define a small policy interface that receives the current prompt/observation and returns text plus model metadata.
- [ ] Make the collector own environment transitions and rollout status; make the local policy own tokenization/generation/log-probabilities; make `grpo.py` own grouping, advantages, loss, and optimizer steps.
- [ ] Run a fresh `SearchEnvironment` for every rollout; never reuse history across group members.
- [ ] Use the existing strict parser and canonical title handling, but set retries to zero.
- [ ] Stop on a valid answer, protocol failure, transport failure, or search-budget exhaustion.
- [ ] Return an immutable rollout record that can be scored offline.
- [ ] Verify the collector with a deterministic fake policy before connecting Transformers.

**Exit gate:** a local fake policy can produce `question -> search -> answer -> trajectory -> reward` and every failure has a typed/logged outcome.

---

## Chunk 3: Connect local Qwen inference

### Task 6: Add a local policy adapter

**Files:**
- Modify: `pyproject.toml` with an optional training dependency group
- Create: `src/needle/training/local_policy.py`
- Create: `scripts/run_local_episode.py`
- Create: `scripts/preflight_training.py`

- [ ] Load the same Qwen2.5-7B-Instruct checkpoint locally with an explicit dtype/device configuration; use LoRA/PEFT as the default training mode rather than full fine-tuning.
- [ ] Add a preflight command that reports GPU name, VRAM, dtype, trainable parameter count, and estimated model memory before loading the trainer.
- [ ] Define the intended 7B gate as an NVIDIA GPU with at least 24 GiB VRAM using `torch_dtype=torch.bfloat16`, `device_map={"": "cuda:0"}`, BF16 LoRA, or at least 16 GiB using 4-bit NF4 QLoRA with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)` and pinned quantization dependencies; use batch size 1, right padding, max sequence length 2048, and one 128-token action generation.
- [ ] Make `python scripts/preflight_training.py --model-path <local-path> --mode lora|qlora` pass only when CUDA is available, tokenizer/model/adapter load with `local_files_only=True`, one forward/backward plus optimizer allocation and one interactive environment path (question prompt -> one search action -> one 128-token-cap action generation -> terminal answer) complete within 120 seconds, and `torch.cuda.max_memory_allocated()` across the whole path is below 90% of total VRAM. Report peak allocated and reserved memory separately; CUDA OOM, missing quantization support, network access, or timeout is a failed gate.
- [ ] Define the fallback: if neither 7B gate passes, use a smaller Qwen checkpoint only for infrastructure smoke testing and do not treat that result as the 7B research result.
- [ ] Reuse the exact prompt builder and action grammar used by the OpenRouter baseline.
- [ ] Generate one action per environment turn, preserving input IDs, output IDs, action text, and per-token log probabilities needed by policy optimization.
- [ ] Run one local episode and compare its parsed action/trajectory fields with the collector contract.
- [ ] Record exact Python, PyTorch, Transformers, TRL, PEFT, tokenizer, CUDA, and model revision metadata in every local run.
- [ ] Pin the training dependency versions in the lock/environment manifest and explicitly allow a local Hugging Face cache path while forbidding network/provider fallback during training and local checkpoint evaluation.
- [ ] Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for training/local evaluation, and use `local_files_only=True` in every local tokenizer/model load path. OpenRouter baseline evaluation uses its separate online entrypoint/config and is never imported by the training path.
- [ ] Add no trainer dependency until local single-episode inference works.

### Task 7: Check rollout parity

**Files:**
- Create: `scripts/compare_policies.py`
- Modify: `scripts/analyze_eval.py` only if needed

- [ ] Run the local policy and OpenRouter policy on a tiny fixed sample with deterministic generation settings.
- [ ] Compare protocol validity, number of searches, answer text, citations, and reward components. Do not require matching provider token counts or exact text.
- [ ] Treat provider/model differences as expected; the requirement is schema and scoring parity, not identical text.

**Ownership:** `evaluate_subset.py` remains the OpenRouter/scripted/BM25 baseline harness; `evaluate_checkpoint.py` owns local-checkpoint evaluation; `analyze_eval.py` only reads completed JSONL; `train_grpo.py` owns trainer orchestration; `grpo.py` owns optimizer/math; `grpo_smoke.json` is written once in the training task. Commit sequence is: prompt baseline, split manifest, analysis/config, rollout boundary, local policy, then trainer.

**Exit gate:** local Qwen can generate an interactive rollout with enough metadata to calculate a policy loss.

---

## Chunk 4: First interactive GRPO smoke test

### Task 8: Implement the group rollout and advantage path

**Files:**
- Create: `src/needle/training/grpo.py`
- Create: `configs/training/grpo_smoke.json`
- Create: `scripts/train_grpo.py`

- [ ] For each prompt, collect `G=4` independent rollouts with the same environment configuration and no retries.
- [ ] Score each rollout with the immutable evaluator or explicit protocol reward and preserve `valid`, `invalid`, `exhausted`, and `transport_error` statuses.
- [ ] Normalize rewards within each prompt group and log raw rewards, mean, standard deviation, and advantages. Skip optimizer updates for groups with no valid token-bearing samples or zero reward variance, while still logging them.
- [ ] Use only generated action-token spans for optimization; do not train on search-observation tokens as if they were sampled actions.
- [ ] Use one optimizer update immediately after each collected group. Treat the collection log-probability as `old_logp`; compute the sequence mean log-probability over the action-token mask and use group-normalized advantage `A`.
- [ ] Implement the documented clipped objective `-mean(min(exp(logp-old_logp)*A, clamp(exp(logp-old_logp), 1-epsilon, 1+epsilon)*A))` with `epsilon=0.2`; log clipping fraction and gradient norm. Reference-model KL is disabled for the first smoke run but its hook and metric are reserved.
- [ ] Keep the first run short: 50–100 training examples, one epoch or a fixed small number of optimizer steps, and one checkpoint.
- [ ] Use TRL/Transformers components where they fit, but do not force the stock single-completion interface onto the multi-turn environment. A thin custom trainer/loop is the default.
- [ ] Save JSONL metrics and a resumable checkpoint containing model, adapter, optimizer, scheduler, RNG, global step, config, and dataset manifest hash.
- [ ] Make the training CLI require a local model path and reject OpenRouter URLs/API-key configuration; OpenRouter metadata cannot be used for policy optimization.

**Numerical verification:**

- [ ] Add `tests/training/test_grpo_math.py` with a synthetic two-turn rollout whose prompt/observation tokens and action-token spans are known.
- [ ] Use batch size 2, right padding, CPU float64, and explicit synthetic masks: row 0 action positions `[3,4]`, row 1 action positions `[2,3]`; assert every action token is included once and all prompt/observation/padding positions are excluded.
- [ ] Assert the exact sequence-mean convention `mean(action_token_logp)` per rollout, `ratio=exp(new_mean-old_mean)`, clipping at `epsilon=0.2`, group-normalized advantage, and final loss against hand-calculated decimal values.
- [ ] Assert that recomputing old log-probabilities before an optimizer step reproduces collection values within `1e-6`; run this math test with deterministic CPU float64 and `torch.use_deterministic_algorithms(True)`.
- [ ] Assert empty action masks produce no loss/gradient sample and transport failures produce no optimizer sample.
- [ ] Add configuration-boundary and network-blocked tests proving the training CLI rejects OpenRouter URLs, API-key settings, and provider fallback, accepts only a local model path/cache, sets offline loading, and makes no network call.

### Task 9: Evaluate the first checkpoint

**Files:**
- Keep: `scripts/evaluate_subset.py` as the OpenRouter/scripted/BM25 baseline harness; add local checkpoint support only in the new evaluator below
- Create: `scripts/evaluate_checkpoint.py`

- [ ] Evaluate the checkpoint on the untouched 300-example holdout with `max_retries=0`.
- [ ] Compare against the unfine-tuned local Qwen baseline from the exact same checkpoint/tokenizer/config, plus scripted and BM25+oracle baselines. Report OpenRouter only as an ancillary reference.
- [ ] Report both all-row metrics and valid-completion-only metrics; never hide protocol failures.
- [ ] Inspect reward histograms and per-episode traces before claiming improvement.
- [ ] Reload the saved checkpoint in a fresh process and verify that evaluation metrics and model outputs are reproducible under the same seed/config.
- [ ] Seed Python, NumPy, PyTorch, and CUDA; use temperature `0.0`, fixed tokenizer/model revisions, deterministic algorithms where supported, exact discrete-action comparison, and metric tolerance `1e-6` on the same device.

**Smoke-test gate:** training runs end-to-end on exactly 50 prompts; at least 40/50 prompts produce two or more token-bearing rollouts; at least 13/50 groups have reward standard deviation greater than `1e-8`; loss and gradients are finite with gradient norm in `(0, 1e3)`; define `absolute_delta = sqrt(sum((p_after-p_before)^2))` and `relative_delta = absolute_delta / max(sqrt(sum(p_before^2)), 1e-12)` over trainable parameters, require `absolute_delta > 1e-10`, `relative_delta > 1e-8`, and a changed SHA-256 checksum of flattened FP32 trainable parameters; checkpoint save/load reproduces deterministic discrete actions exactly and metrics within `1e-6`; and the checkpoint can be evaluated offline without network access.

---

## Chunk 5: Minimal ablations and scale-up

### Task 10: Run only the ablations that answer research questions

**Files:**
- Create: `configs/training/ablation_*.json`
- Modify: `scripts/train_grpo.py`
- Modify: `scripts/analyze_eval.py`

- [ ] Compare reward with and without search-cost penalty.
- [ ] Compare answer-only reward against answer-plus-evidence reward.
- [ ] Keep protocol handling, dataset split, seed policy, and rollout count fixed across ablations.
- [ ] Use one seed first; add more seeds only after an effect is visible.
- [ ] Stop an ablation if it clearly collapses to endless search, no-citation answers, or a constant reward.

### Task 11: Scale the promising configuration

**Files:**
- Modify: `configs/training/grpo_smoke.json` into a versioned experiment config
- Modify: `scripts/train_grpo.py`
- Modify: `scripts/evaluate_checkpoint.py`

- [ ] Move from 50–100 training examples to 500, then 1,000 only if the smoke run shows signal.
- [ ] Increase from 4 to 8 rollouts only when group variance is too noisy, not by default.
- [ ] Evaluate every checkpoint on the fixed 300 holdout and a second random holdout.
- [ ] Track answer EM/F1, evidence metrics, retrieval recall, search cost, invalid rate, average searches, and token/episode.
- [ ] Run on Kaggle/Modal only after the local smoke path is reproducible.

**Exit gate:** there is a repeatable improvement over the raw Qwen baseline on held-out questions without protocol validity collapsing.

For the first claimed improvement, use a pre-registered comparison: at least +5 percentage points all-row EM or +0.10 mean total reward over the raw baseline, with protocol failure rate not worsening by more than 5 percentage points. Confirm the direction on a second seed before scaling further. Report the exact holdout metrics, not only completed-row metrics.

---

## Chunk 6: Research packaging after the first result

### Task 12: Make experiment comparison easy

**Files:**
- Create: `docs/experiments/README.md`
- Create: `results/README.md`
- Modify: `README.md`

- [ ] Document each run by config, code commit, dataset split, model checkpoint, seed, provider/device, and output path.
- [ ] Keep JSONL as the source of truth; add W&B/Langfuse only if local JSONL becomes insufficient.
- [ ] Record negative results and failed runs, especially protocol and reward-collapse failures.
- [ ] Update the project status from “prototype” to the exact current research milestone.

## Things explicitly out of scope until the first GRPO result

- Dense retrieval or a new corpus index.
- Fuzzy semantic citation aliases.
- Large-scale test-suite expansion.
- Prompt perfection beyond the current action/budget rules.
- Multi-model sweeps.
- Complex distributed training infrastructure.
- Replacing the deterministic evaluator with a learned judge.
