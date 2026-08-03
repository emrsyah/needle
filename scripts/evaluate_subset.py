"""Run lightweight scripted, BM25, and Qwen baseline evaluations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from needle.data import Document, EvidenceRef, QuestionExample
from needle.environment import EpisodeTrajectory, SearchEnvironment
from needle.inference import EpisodeRunner, EpisodeRunnerError, OpenRouterClient, OpenRouterConfig
from needle.rewards import RewardBreakdown, evaluate


def _load_subset(path: Path, limit: int) -> tuple[tuple[QuestionExample, ...], int]:
    raw_rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list):
        raise ValueError("dataset top level must be a list")

    examples: list[QuestionExample] = []
    skipped = 0
    for raw in raw_rows:
        try:
            documents = tuple(
                Document(title=title, sentences=tuple(sentences))
                for title, sentences in raw["context"]
            )
            supporting_facts = frozenset(
                EvidenceRef(title, index) for title, index in raw["supporting_facts"]
            )
            examples.append(
                QuestionExample(
                    question_id=raw["_id"],
                    question=raw["question"],
                    gold_answer=raw["answer"],
                    documents=documents,
                    supporting_facts=supporting_facts,
                )
            )
        except (KeyError, TypeError, ValueError):
            skipped += 1
        if len(examples) == limit:
            break
    return tuple(examples), skipped


def _new_environment(example: QuestionExample, top_k: int, max_searches: int) -> SearchEnvironment:
    return SearchEnvironment(example, top_k=top_k, max_searches=max_searches)


def _oracle_trajectory(
    example: QuestionExample,
    queries: Iterable[str],
    top_k: int,
    max_searches: int,
) -> EpisodeTrajectory:
    environment = _new_environment(example, top_k, max_searches)
    environment.reset()
    for query in queries:
        if environment.remaining_searches == 0:
            break
        environment.search(query)
    environment.answer(example.gold_answer, tuple(sorted(example.supporting_facts, key=_fact_key)))
    trajectory = environment.trajectory
    if trajectory is None:
        raise AssertionError("oracle answer should create a trajectory")
    return trajectory


def _fact_key(fact: EvidenceRef) -> tuple[str, int]:
    return fact.document_title, fact.sentence_index


def _scripted_trajectory(
    example: QuestionExample, top_k: int, max_searches: int
) -> EpisodeTrajectory:
    document_titles = dict.fromkeys(
        fact.document_title for fact in sorted(example.supporting_facts, key=_fact_key)
    )
    return _oracle_trajectory(example, document_titles, top_k, max_searches)


def _bm25_oracle_trajectory(
    example: QuestionExample, top_k: int, max_searches: int
) -> EpisodeTrajectory:
    return _oracle_trajectory(example, (example.question,), top_k, max_searches)


class _RecordingClient:
    def __init__(self, client: OpenRouterClient) -> None:
        self.client = client
        self.outputs: list[str] = []
        self.providers: list[str] = []
        self.total_tokens = 0

    def complete(self, messages):
        result = self.client.complete(messages)
        self.outputs.append(result.text)
        if result.provider:
            self.providers.append(result.provider)
        if result.usage.total_tokens is not None:
            self.total_tokens += result.usage.total_tokens
        return result


def _reward_metrics(reward: RewardBreakdown) -> dict[str, object]:
    return {
        "answer_exact_match": reward.answer_exact_match,
        "answer_f1": reward.answer_f1,
        "evidence_coverage": reward.evidence_coverage,
        "citation_precision": reward.citation_precision,
        "retrieval_recall": reward.retrieval_recall,
        "search_cost": reward.search_cost,
        "duplicate_query_penalty": reward.duplicate_query_penalty,
        "information_gain": reward.information_gain,
        "total": reward.total,
    }


def _trajectory_row(
    baseline: str,
    example: QuestionExample,
    trajectory: EpisodeTrajectory,
    *,
    outputs: list[str] | None = None,
    providers: list[str] | None = None,
    total_tokens: int = 0,
) -> dict[str, object]:
    reward = evaluate(example, trajectory)
    return {
        "baseline": baseline,
        "question_id": example.question_id,
        "status": "completed",
        "answer": trajectory.answer.answer,
        "citations": [
            f"{citation.document_title}|{citation.sentence_index}"
            for citation in trajectory.answer.citations
        ],
        "searches": [search.query for search in trajectory.searches],
        "metrics": _reward_metrics(reward),
        "outputs": outputs or [],
        "providers": sorted(set(providers or [])),
        "total_tokens": total_tokens,
    }


def _failed_row(
    baseline: str,
    example: QuestionExample,
    error: Exception,
    *,
    outputs: list[str] | None = None,
    providers: list[str] | None = None,
    total_tokens: int = 0,
) -> dict[str, object]:
    return {
        "baseline": baseline,
        "question_id": example.question_id,
        "status": "failed",
        "error": str(error),
        "outputs": outputs or [],
        "providers": sorted(set(providers or [])),
        "total_tokens": total_tokens,
    }


def _mean(rows: list[dict[str, object]], metric: str) -> float | None:
    completed = [row for row in rows if row["status"] == "completed"]
    if not completed:
        return None
    return round(sum(row["metrics"][metric] for row in completed) / len(completed), 4)  # type: ignore[index]


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    baselines = sorted({row["baseline"] for row in rows})
    summary: dict[str, object] = {}
    for baseline in baselines:
        baseline_rows = [row for row in rows if row["baseline"] == baseline]
        summary[baseline] = {
            "n": len(baseline_rows),
            "completed": sum(row["status"] == "completed" for row in baseline_rows),
            "failed": sum(row["status"] == "failed" for row in baseline_rows),
            "answer_exact_match_all_rows_fail_as_zero": round(
                sum(row.get("metrics", {}).get("answer_exact_match", 0.0) for row in baseline_rows)
                / len(baseline_rows),
                4,
            ),
            "answer_exact_match_completed_only": _mean(baseline_rows, "answer_exact_match"),
            "answer_f1_completed_only": _mean(baseline_rows, "answer_f1"),
            "evidence_coverage_completed_only": _mean(baseline_rows, "evidence_coverage"),
            "citation_precision_completed_only": _mean(baseline_rows, "citation_precision"),
            "retrieval_recall_completed_only": _mean(baseline_rows, "retrieval_recall"),
            "total_completed_only": _mean(baseline_rows, "total"),
            "mean_total_tokens": round(
                sum(row["total_tokens"] for row in baseline_rows) / len(baseline_rows), 2
            ),
        }
    return summary


def run(args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]]]:
    examples, skipped = _load_subset(args.dataset, args.subset_size)
    if not examples:
        raise ValueError("dataset contains no valid examples")

    rows: list[dict[str, object]] = []
    for example in examples:
        rows.append(
            _trajectory_row(
                "scripted",
                example,
                _scripted_trajectory(example, args.top_k, args.max_searches),
            )
        )
        rows.append(
            _trajectory_row(
                "bm25_oracle",
                example,
                _bm25_oracle_trajectory(example, args.top_k, args.max_searches),
            )
        )

    qwen_config = OpenRouterConfig.from_env(
        model=args.model,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
        allow_fallbacks=True,
        title="Needle baseline evaluation",
    )
    qwen_client = OpenRouterClient(qwen_config)
    for example in examples:
        recorder = _RecordingClient(qwen_client)
        try:
            environment = _new_environment(example, args.top_k, args.max_searches)
            trajectory = EpisodeRunner(
                recorder,
                max_turns=args.max_searches + 1,
                max_retries=args.max_retries,
            ).run(environment)
            rows.append(
                _trajectory_row(
                    "qwen",
                    example,
                    trajectory,
                    outputs=recorder.outputs,
                    providers=recorder.providers,
                    total_tokens=recorder.total_tokens,
                )
            )
        except EpisodeRunnerError as error:
            rows.append(
                _failed_row(
                    "qwen",
                    example,
                    error,
                    outputs=recorder.outputs,
                    providers=recorder.providers,
                    total_tokens=recorder.total_tokens,
                )
            )

    metadata = {
        "dataset": str(args.dataset),
        "dataset_valid_examples_used": len(examples),
        "dataset_rows_skipped_before_subset": skipped,
        "subset_size_requested": args.subset_size,
        "top_k": args.top_k,
        "max_searches": args.max_searches,
        "max_retries": args.max_retries,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    return {"metadata": metadata, "summary": _summarize(rows)}, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--model", default="qwen/qwen-2.5-7b-instruct")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.subset_size <= 0 or args.top_k <= 0 or args.max_searches <= 0:
        parser.error("subset-size, top-k, and max-searches must be positive")
    if args.max_retries < 0 or args.max_tokens <= 0 or args.seed < 0:
        parser.error(
            "max-retries must be non-negative; max-tokens must be positive; "
            "seed must be non-negative"
        )

    summary, rows = run(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {len(rows)} episode rows to {args.output_dir}")


if __name__ == "__main__":
    main()
