"""Preflight a local model before paying for a training run.

Loads the model offline with LoRA, plays one interactive episode with the real
collector, runs one GRPO forward/backward and optimizer step on it, and reports
time and peak GPU memory. Exits non-zero if any gate fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from needle.training.runtime import (
    ConfigurationError,
    enforce_offline,
    run_metadata,
    validate_local_model_path,
)

enforce_offline()

import torch  # noqa: E402

from needle.data.hotpotqa import load_hotpotqa  # noqa: E402
from needle.training import RolloutCollector  # noqa: E402
from needle.training.grpo import rollout_mean_log_prob, rollout_traces  # noqa: E402
from needle.training.local_policy import (  # noqa: E402
    GenerationConfig,
    LocalPolicy,
    completion_log_probs,
)

DEFAULT_FIXTURE = Path("tests/fixtures/tiny_hotpotqa.json")


def preflight(args: argparse.Namespace) -> dict:
    from needle.training.runtime import load_model_and_tokenizer

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_path = validate_local_model_path(args.model_path)
    cuda = torch.cuda.is_available()
    if not cuda and not args.allow_cpu:
        raise SystemExit("CUDA is not available (use --allow-cpu for a tiny-model smoke test)")
    report: dict[str, object] = {"gates": {}}
    started = time.time()
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    model, tokenizer = load_model_and_tokenizer(
        model_path,
        dtype=config["dtype"] if cuda else "float32",
        lora=config["lora"],
        gradient_checkpointing=config["gradient_checkpointing"],
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    report["trainable_parameters"] = sum(parameter.numel() for parameter in trainable)
    report["total_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    report["load_seconds"] = round(time.time() - started, 1)
    optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"])

    example = load_hotpotqa(args.source)[0]
    generation = GenerationConfig(**config["generation"])
    policy = LocalPolicy(model, tokenizer, generation=generation, model_name=str(model_path))
    policy.reseed(config["seed"])
    collector = RolloutCollector(
        policy,
        top_k=config["environment"]["top_k"],
        max_searches=config["environment"]["max_searches"],
    )
    episode_started = time.time()
    record = collector.collect(example)
    report["episode"] = {
        "status": record.status.value,
        "turns": len(record.steps),
        "actions": [step.raw_action for step in record.steps],
        "failure": record.failure_message,
        "reward": record.training_reward,
        "completion_tokens": [len(trace.completion_token_ids) for trace in rollout_traces(record)],
        "prompt_tokens": [len(trace.prompt_token_ids) for trace in rollout_traces(record)],
    }

    traces = rollout_traces(record)
    report["gates"]["token_trace_captured"] = bool(traces)
    if traces:
        model.train()
        optimizer.zero_grad(set_to_none=True)

        def log_prob_fn(trace):
            return completion_log_probs(model, trace.prompt_token_ids, trace.completion_token_ids)

        new_mean = rollout_mean_log_prob(record, log_prob_fn)
        old_values = [value for trace in traces for value in trace.old_log_probs]
        drift = abs(float(new_mean.detach()) - sum(old_values) / len(old_values))
        report["old_new_logp_drift"] = drift
        report["gates"]["old_logp_reproducible"] = drift < args.drift_tolerance
        (-new_mean).backward()
        optimizer.step()
        report["gates"]["backward_and_step"] = True
    report["episode_and_update_seconds"] = round(time.time() - episode_started, 1)
    report["gates"]["within_time_limit"] = report["episode_and_update_seconds"] <= args.time_limit

    if cuda:
        total = torch.cuda.get_device_properties(0).total_memory
        peak_allocated = torch.cuda.max_memory_allocated()
        report["gpu"] = torch.cuda.get_device_name(0)
        report["vram_total_gib"] = round(total / 2**30, 2)
        report["peak_allocated_gib"] = round(peak_allocated / 2**30, 2)
        report["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
        report["gates"]["peak_memory_below_90pct"] = peak_allocated < 0.9 * total
    report["metadata"] = run_metadata(model_path)
    report["passed"] = all(report["gates"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/training/grpo_smoke.json"))
    parser.add_argument("--source", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--drift-tolerance", type=float, default=1e-2)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    try:
        validate_local_model_path(args.model_path)
    except ConfigurationError as error:
        parser.error(str(error))
    report = preflight(args)
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
