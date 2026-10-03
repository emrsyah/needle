"""Train a LoRA adapter on Needle with interactive multi-turn GRPO.

Local weights only: the model path must be a local directory, Hugging Face runs
offline, and provider (OpenRouter) credentials in the environment are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from needle.training.runtime import (
    ConfigurationError,
    enforce_offline,
    load_examples,
    load_split_ids,
    run_metadata,
    validate_local_model_path,
)

enforce_offline()

import torch  # noqa: E402

from needle.rewards import RewardConfig  # noqa: E402
from needle.training import RolloutCollector, RolloutRecord  # noqa: E402
from needle.training.grpo import (  # noqa: E402
    clipped_surrogate_loss,
    is_trainable,
    rollout_mean_log_prob,
    rollout_old_mean,
    score_group,
)
from needle.training.local_policy import (  # noqa: E402
    GenerationConfig,
    LocalPolicy,
    completion_log_probs,
)


def rollout_seed(seed: int, question_id: str, member: int) -> int:
    digest = hashlib.sha256(f"{seed}:{question_id}:{member}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def select_train_ids(ids: tuple[str, ...], count: int, seed: int) -> list[str]:
    """Deterministic sample of ``count`` training IDs."""
    ordered = sorted(ids, key=lambda question_id: rollout_seed(seed, question_id, -1))
    return ordered[:count]


def _record_summary(record: RolloutRecord, advantage: float) -> dict[str, object]:
    reward = record.reward_breakdown
    return {
        "status": record.status.value,
        "failure_kind": record.failure_kind.value if record.failure_kind else None,
        "reward": record.training_reward,
        "advantage": advantage,
        "turns": len(record.steps),
        "searches": len(record.trajectory.searches) if record.trajectory else None,
        "exact_match": reward.answer_exact_match if reward else None,
        "f1": reward.answer_f1 if reward else None,
        "evidence_coverage": reward.evidence_coverage if reward else None,
        "citation_precision": reward.citation_precision if reward else None,
    }


def save_checkpoint(model, optimizer, directory: Path, step: int, config: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory / "adapter")
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_rng": random.getstate(),
        },
        directory / "trainer_state.pt",
    )
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    from needle.training.runtime import load_model_and_tokenizer

    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_path = validate_local_model_path(args.model_path)
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    train_ids = select_train_ids(
        load_split_ids(args.splits, "train"), config["num_train_examples"], config["seed"]
    )
    examples = load_examples(args.train_source, train_ids)

    model, tokenizer = load_model_and_tokenizer(
        model_path,
        dtype=config["dtype"],
        adapter_path=args.resume / "adapter" if args.resume else None,
        lora=config["lora"],
        gradient_checkpointing=config["gradient_checkpointing"],
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"])
    start_step = 0
    if args.resume:
        state = torch.load(args.resume / "trainer_state.pt", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])

    policy = LocalPolicy(
        model,
        tokenizer,
        generation=GenerationConfig(**config["generation"]),
        model_name=config["base_model"],
    )
    collector = RolloutCollector(
        policy,
        top_k=config["environment"]["top_k"],
        max_searches=config["environment"]["max_searches"],
        reward_config=RewardConfig(**config.get("reward", {})),
    )

    def log_prob_fn(trace):
        return completion_log_probs(model, trace.prompt_token_ids, trace.completion_token_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = run_metadata(
        model_path,
        {
            "config": config,
            "config_path": str(args.config),
            "splits": str(args.splits),
            "splits_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
            "train_ids": train_ids,
        },
    )
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    schedule = [example for _ in range(config["epochs"]) for example in examples]
    group_size = config["group_size"]
    metrics_path = args.output_dir / "metrics.jsonl"
    rollouts_path = args.output_dir / "rollouts.jsonl"
    with (
        metrics_path.open("a", encoding="utf-8") as metrics_file,
        rollouts_path.open("a", encoding="utf-8") as rollouts_file,
    ):
        for step in range(start_step, len(schedule)):
            example = schedule[step]
            started = time.time()
            records = []
            for member in range(group_size):
                policy.reseed(rollout_seed(config["seed"], f"{example.question_id}:{step}", member))
                records.append(collector.collect(example))
            advantages, stats = score_group(records, train_on_invalid=config["train_on_invalid"])

            loss_value = clip_value = grad_norm = None
            ratio_drift = None
            if stats.skipped_reason is None:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                clip_total = 0.0
                drifts = []
                for record, advantage in zip(records, advantages, strict=True):
                    if advantage == 0.0 or not is_trainable(
                        record, train_on_invalid=config["train_on_invalid"]
                    ):
                        continue
                    new_mean = rollout_mean_log_prob(record, log_prob_fn)
                    old_mean = rollout_old_mean(record)
                    drifts.append(abs(float(new_mean.detach()) - old_mean))
                    device = new_mean.device
                    loss, clip_fraction = clipped_surrogate_loss(
                        new_mean.unsqueeze(0),
                        torch.tensor([old_mean], device=device),
                        torch.tensor([advantage], device=device),
                        epsilon=config["epsilon"],
                    )
                    (loss / group_size).backward()
                    total_loss += float(loss.detach()) / group_size
                    clip_total += float(clip_fraction)
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable, config["max_grad_norm"])
                )
                optimizer.step()
                loss_value = total_loss
                clip_value = clip_total / max(stats.trainable, 1)
                ratio_drift = max(drifts) if drifts else None

            row = {
                "step": step + 1,
                "question_id": example.question_id,
                "reward_mean": stats.reward_mean,
                "reward_std": stats.reward_std,
                "trainable": stats.trainable,
                "skipped_reason": stats.skipped_reason,
                "loss": loss_value,
                "clip_fraction": clip_value,
                "grad_norm": grad_norm,
                "max_old_new_logp_drift": ratio_drift,
                "seconds": round(time.time() - started, 2),
                "rollouts": [
                    _record_summary(record, advantage)
                    for record, advantage in zip(records, advantages, strict=True)
                ],
            }
            if torch.cuda.is_available():
                row["cuda_max_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
            for record in records:
                rollouts_file.write(record.to_json() + "\n")
            rollouts_file.flush()
            print(
                f"step {step + 1}/{len(schedule)} reward_mean={stats.reward_mean:.3f} "
                f"std={stats.reward_std:.3f} skipped={stats.skipped_reason} "
                f"loss={loss_value} {row['seconds']}s",
                flush=True,
            )
            if config["save_every"] and (step + 1) % config["save_every"] == 0:
                save_checkpoint(
                    model, optimizer, args.output_dir / "checkpoint-latest", step + 1, config
                )

    save_checkpoint(model, optimizer, args.output_dir / "checkpoint-final", len(schedule), config)
    print(f"Saved final adapter to {args.output_dir / 'checkpoint-final' / 'adapter'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/grpo_smoke.json"))
    parser.add_argument("--model-path", required=True, help="local model directory")
    parser.add_argument("--train-source", type=Path, required=True, help="hotpot_train_v1.1.json")
    parser.add_argument("--splits", type=Path, default=Path("configs/evaluation/splits.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="checkpoint directory to resume from")
    args = parser.parse_args()
    try:
        validate_local_model_path(args.model_path)
    except ConfigurationError as error:
        parser.error(str(error))
    train(args)


if __name__ == "__main__":
    main()
