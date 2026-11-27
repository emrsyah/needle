#!/usr/bin/env bash
# Prompt-only (few-shot) baseline B0-FS on the frozen 300-question holdout (issue #9).
#
# Run on a RunPod L40S pod that has followed docs/runbooks/runpod.md steps 0-3
# (repo at /workspace/needle, model snapshot downloaded, HotpotQA dev file present):
#
#   cd /workspace/needle && git pull && bash scripts/run_fewshot_b0.sh
#
# Expected cost: ~0.5-1.0 GPU-hour (~$0.55-1.10 at $1.10/h). Hard cap: the run is
# killed after MAX_HOURS (default 2.0 h = ~$2.20) so it cannot exceed the $2.50 budget.
# STOP THE POD afterwards.
set -euo pipefail

MODEL="${MODEL:-$(ls -d /workspace/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/ | head -n 1)}"
DEV="${DEV:-/workspace/data/hotpot_dev_distractor_v1.json}"
OUT="${OUT:-results/$(date +%Y-%m-%d)-b0-fewshot}"
MAX_HOURS="${MAX_HOURS:-2.0}"

unset OPENROUTER_API_KEY OPENROUTER_BASE_URL || true
export HF_HOME="${HF_HOME:-/workspace/hf}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

test -d "$MODEL" || { echo "model snapshot not found: $MODEL (runbook step 2)"; exit 1; }
test -f "$DEV" || { echo "HotpotQA dev file not found: $DEV (runbook step 2)"; exit 1; }
mkdir -p "$OUT"

# 20-question speed check first; abort if the full run would not fit the time cap.
start=$(date +%s)
timeout 30m uv run python scripts/evaluate_checkpoint.py --model-path "$MODEL" --source "$DEV" \
  --output-dir "$OUT/speedcheck" --name b0_fewshot --few-shot --limit 20
per_q=$(( ( $(date +%s) - start ) / 20 ))
estimate_h=$(python3 -c "print(round(${per_q} * 300 / 3600, 2))")
echo "speed check: ~${per_q}s/question, full run ~${estimate_h} h"
python3 -c "import sys; sys.exit(0 if ${estimate_h} < ${MAX_HOURS} else 1)" \
  || { echo "estimated run exceeds MAX_HOURS=${MAX_HOURS}; stopping"; exit 2; }

timeout "$(python3 -c "print(int(${MAX_HOURS} * 3600))")s" \
  uv run python scripts/evaluate_checkpoint.py --model-path "$MODEL" --source "$DEV" \
  --output-dir "$OUT" --name b0_fewshot --few-shot
uv run python scripts/analyze_eval.py --episodes "$OUT/episodes.jsonl" | tee "$OUT/analysis.txt"
echo "done: $OUT/summary.json -- download $OUT/ and STOP THE POD"
