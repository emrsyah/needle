"""Summarize a train_grpo.py metrics.jsonl in fixed-size step blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(rows: list[dict], block: int) -> list[dict]:
    blocks = []
    for start in range(0, len(rows), block):
        chunk = rows[start : start + block]
        rollouts = [rollout for row in chunk for rollout in row["rollouts"]]

        def values(key: str, rollouts: list[dict] = rollouts) -> list[float]:
            return [rollout[key] for rollout in rollouts if rollout[key] is not None]

        blocks.append(
            {
                "steps": f"{start + 1}-{start + len(chunk)}",
                "reward_mean": _mean([row["reward_mean"] for row in chunk]),
                "invalid_rate": _mean([float(r["status"] != "valid") for r in rollouts]),
                "exact_match_valid": _mean(values("exact_match")),
                "evidence_coverage_valid": _mean(values("evidence_coverage")),
                "citation_precision_valid": _mean(values("citation_precision")),
                "searches_valid": _mean(values("searches")),
                "skipped_steps": sum(row["skipped_reason"] is not None for row in chunk),
            }
        )
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--block", type=int, default=32)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.metrics.read_text(encoding="utf-8").splitlines()]
    for entry in summarize(rows, args.block):
        print(json.dumps(entry))


if __name__ == "__main__":
    main()
