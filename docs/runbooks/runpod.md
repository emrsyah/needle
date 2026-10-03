# RunPod Runbook — First Local Baseline and GRPO Smoke Run

Target: one RunPod **L40S (48 GB)** pod, ~$1.09/h. Budget ~$30 total. **Stop the pod whenever you are not actively running something.**

Approximate GPU time for this runbook: preflight ~0.5 h, baseline eval ~1–2 h, smoke training ~2–4 h, checkpoint eval ~1–2 h.

## 0. Create the pod

- GPU: L40S × 1, on-demand (not a reservation/monthly pool).
- Template: any recent **RunPod PyTorch** image (CUDA 12.x).
- Volume: **network/persistent volume ≥ 50 GB mounted at `/workspace`**. Everything (repo, model, data, outputs) lives there so it survives stopping the pod.
- Open the web terminal or SSH in.

## 1. Set up the repo (online, once)

```bash
cd /workspace
git clone https://github.com/emrsyah/needle.git   # or your fork URL
cd needle
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv sync --group training
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True` and the GPU name.

## 2. Download the model and data (online, once)

```bash
export HF_HOME=/workspace/hf
uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-7B-Instruct'))"
```

Copy the printed path (`/workspace/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/<commit>`). Its `<commit>` is recorded automatically as the model revision.

```bash
export MODEL=/workspace/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/<commit>
mkdir -p /workspace/data && cd /workspace/data
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
cd /workspace/needle
```

## 3. Create the split manifests (once)

```bash
uv run python scripts/prepare_splits.py \
  --train-source /workspace/data/hotpot_train_v1.1.json \
  --dev-source /workspace/data/hotpot_dev_distractor_v1.json \
  --bootstrap-holdout
```

This writes `configs/evaluation/splits.json` and `configs/evaluation/holdout_v1_ids.json`. **Commit these two files** (push from the pod, or download them and commit locally). Run with `--bootstrap-holdout` only this once; afterwards the holdout is frozen.

Note: this 300-question holdout is a new sample, so it is not the same 300 questions as the earlier OpenRouter run. That run stays an ancillary reference only.

## 4. Preflight (~5 min)

From here on everything runs offline. Do **not** set any `OPENROUTER_*` variable on the pod; the scripts refuse to run if one is present.

```bash
unset OPENROUTER_API_KEY
uv run python scripts/preflight_training.py --model-path "$MODEL" --output /workspace/preflight.json
```

It must end with `"passed": true`. Check `peak_allocated_gib` (should be well under 43 GiB) and `episode_and_update_seconds`.

## 5. Local un-tuned baseline on the holdout (B0)

```bash
uv run python scripts/evaluate_checkpoint.py \
  --model-path "$MODEL" \
  --source /workspace/data/hotpot_dev_distractor_v1.json \
  --output-dir /workspace/results/b0_local_base --name b0_local_base
```

Tip: try `--limit 20` first to check speed, then run the full 300. Use `tmux` so a closed browser tab does not kill the run.

## 6. GRPO smoke training

```bash
uv run python scripts/train_grpo.py \
  --model-path "$MODEL" \
  --train-source /workspace/data/hotpot_train_v1.1.json \
  --output-dir /workspace/runs/grpo_smoke
```

Config: `configs/training/grpo_smoke.json` (64 questions × 4 rollouts, LoRA r=16, lr 1e-5). While it runs, watch `metrics.jsonl`:

- `skipped_reason` = `zero_reward_variance` on most steps → the group gives no learning signal; raise `group_size` to 8.
- Most rollouts `invalid` and `train_on_invalid` is false → the model never learns the format; set `"train_on_invalid": true`.
- `searches` dropping to 0 with falling `evidence_coverage` → shortcut answering; lower the search cost in `"reward": {"search_cost_weight": 0.0}`.

Resume after a stop: add `--resume /workspace/runs/grpo_smoke/checkpoint-latest`.

## 7. Evaluate the checkpoint

```bash
uv run python scripts/evaluate_checkpoint.py \
  --model-path "$MODEL" \
  --adapter-path /workspace/runs/grpo_smoke/checkpoint-final/adapter \
  --source /workspace/data/hotpot_dev_distractor_v1.json \
  --output-dir /workspace/results/grpo_smoke --name grpo_smoke
uv run python scripts/analyze_eval.py --episodes /workspace/results/grpo_smoke/episodes.jsonl
```

## 8. Bring results home, then stop the pod

Download `results/*/summary.json`, `runs/grpo_smoke/metrics.jsonl`, `run_metadata.json`, and the `checkpoint-final/adapter` folder (small, ~80 MB). Then **stop** (or terminate) the pod.
