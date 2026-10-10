"""Evaluate a local model (optionally with a LoRA adapter) on a fixed question set.

Writes ``summary.json`` and ``episodes.jsonl`` in the same row schema as
``scripts/evaluate_subset.py`` so ``scripts/analyze_eval.py`` can read it.
Greedy decoding, no retries, offline loading only.
"""

from __future__ import annotations

import argparse
import json
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

from needle.data import QuestionExample  # noqa: E402
from needle.training import RolloutCollector, RolloutRecord  # noqa: E402
from needle.training.local_policy import GenerationConfig, LocalPolicy  # noqa: E402


def episode_row(baseline: str, example: QuestionExample, record: RolloutRecord) -> dict:
    outputs = [step.raw_action for step in record.steps if step.raw_action is not None]
    total_tokens = sum(
        len(step.tokens.prompt_token_ids) + len(step.tokens.completion_token_ids)
        for step in record.steps
        if step.tokens is not None
    )
    common = {
        "baseline": baseline,
        "question_id": example.question_id,
        "rollout_status": record.status.value,
        "outputs": outputs,
        "providers": ["local"],
        "total_tokens": total_tokens,
    }
    if record.trajectory is None or record.reward_breakdown is None:
        return {
            **common,
            "status": "failed",
            "failure_kind": record.failure_kind.value if record.failure_kind else None,
            "error": record.failure_message,
        }
    reward = record.reward_breakdown
    trajectory = record.trajectory
    return {
        **common,
        "status": "completed",
        "answer": trajectory.answer.answer,
        "citations": [
            f"{citation.document_title}|{citation.sentence_index}"
            for citation in trajectory.answer.citations
        ],
        "searches": [search.query for search in trajectory.searches],
        "metrics": {
            "answer_exact_match": reward.answer_exact_match,
            "answer_f1": reward.answer_f1,
            "evidence_coverage": reward.evidence_coverage,
            "citation_precision": reward.citation_precision,
            "retrieval_recall": reward.retrieval_recall,
            "search_cost": reward.search_cost,
            "duplicate_query_penalty": reward.duplicate_query_penalty,
            "information_gain": list(reward.information_gain),
            "total": reward.total,
        },
    }


def summarize(rows: list[dict]) -> dict:
    completed = [row for row in rows if row["status"] == "completed"]

    def mean(metric: str) -> float | None:
        if not completed:
            return None
        return round(sum(row["metrics"][metric] for row in completed) / len(completed), 4)

    return {
        "n": len(rows),
        "completed": len(completed),
        "failed": len(rows) - len(completed),
        "answer_exact_match_all_rows_fail_as_zero": round(
            sum(row["metrics"]["answer_exact_match"] for row in completed) / len(rows), 4
        )
        if rows
        else None,
        "answer_exact_match_completed_only": mean("answer_exact_match"),
        "answer_f1_completed_only": mean("answer_f1"),
        "evidence_coverage_completed_only": mean("evidence_coverage"),
        "citation_precision_completed_only": mean("citation_precision"),
        "retrieval_recall_completed_only": mean("retrieval_recall"),
        "total_completed_only": mean("total"),
        "mean_searches_completed_only": round(
            sum(len(row["searches"]) for row in completed) / len(completed), 4
        )
        if completed
        else None,
    }


def evaluate(args: argparse.Namespace) -> None:
    from needle.training.runtime import load_model_and_tokenizer

    model_path = validate_local_model_path(args.model_path)
    ids = load_split_ids(args.ids, args.split)
    if args.limit:
        ids = ids[: args.limit]
    examples = load_examples(args.source, ids)
    model, tokenizer = load_model_and_tokenizer(
        model_path, dtype=args.dtype, adapter_path=args.adapter_path
    )
    model.eval()
    policy = LocalPolicy(
        model,
        tokenizer,
        generation=GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False),
        model_name=args.name,
    )
    collector = RolloutCollector(policy, top_k=args.top_k, max_searches=args.max_searches)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with (args.output_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for index, example in enumerate(examples, start=1):
            row = episode_row(args.name, example, collector.collect(example))
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{index}/{len(examples)} {row['status']}", flush=True)

    summary = {
        "metadata": run_metadata(
            model_path,
            {
                "adapter_path": str(args.adapter_path) if args.adapter_path else None,
                "ids_manifest": str(args.ids),
                "source": str(args.source),
                "top_k": args.top_k,
                "max_searches": args.max_searches,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "max_retries": 0,
            },
        ),
        "summary": {args.name: summarize(rows)},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["summary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--source", type=Path, required=True, help="hotpot_dev_distractor_v1.json")
    parser.add_argument("--ids", type=Path, default=Path("configs/evaluation/holdout_v1_ids.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="local")
    parser.add_argument(
        "--split",
        default="holdout_v1",
        help="split name inside --ids; use 'validation' with splits.json and the train source",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    try:
        validate_local_model_path(args.model_path)
    except ConfigurationError as error:
        parser.error(str(error))
    evaluate(args)


if __name__ == "__main__":
    main()
