"""Analyze completed Needle episode JSONL without network access."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

METRICS = (
    "answer_exact_match",
    "answer_f1",
    "evidence_coverage",
    "citation_precision",
    "retrieval_recall",
    "search_cost",
    "duplicate_query_penalty",
    "total",
)
FAILURE_CATEGORIES = ("protocol", "budget", "transport", "retrieval", "answer", "evidence")
COMPLETED_STATUSES = {"completed", "complete", "valid", "success"}


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _read_rows(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"could not read episodes JSONL: {path}: {error}") from error

    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"episode row at {path}:{line_number} must be an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"episodes JSONL is empty: {path}")
    return rows


def _completed(row: dict[str, object]) -> bool:
    status = row.get("status")
    return isinstance(status, str) and status.lower() in COMPLETED_STATUSES


def _failure_category(row: dict[str, object]) -> str:
    explicit = row.get("failure_kind") or row.get("error_category")
    if isinstance(explicit, str):
        normalized = explicit.lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "invalid_action": "protocol",
            "parse": "protocol",
            "exhausted": "budget",
            "search_budget": "budget",
            "network": "transport",
            "http": "transport",
            "citation": "evidence",
        }
        category = aliases.get(normalized, normalized)
        if category in FAILURE_CATEGORIES:
            return category

    error = row.get("error", "")
    text = error.lower() if isinstance(error, str) else ""
    patterns = (
        ("transport", ("transport", "openrouter", "http ", "timeout", "connection")),
        ("budget", ("exhaust", "max_turn", "search budget", "too many searches")),
        ("evidence", ("citation", "evidence", "supporting fact")),
        ("retrieval", ("retriev", "search", "query", "document")),
        ("answer", ("answer", "gold")),
        ("protocol", ("parse", "action", "format", "invalid")),
    )
    for category, needles in patterns:
        if any(needle in text for needle in needles):
            return category
    return "protocol"


def _metric(row: dict[str, object], name: str) -> float | None:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return _finite_number(metrics.get(name))


def _search_count(row: dict[str, object]) -> float | None:
    searches = row.get("searches")
    if isinstance(searches, list):
        return float(len(searches))
    for key in ("searches_used", "search_count"):
        value = _finite_number(row.get(key))
        if value is not None:
            return value
    return None


def _token_count(row: dict[str, object]) -> float | None:
    value = _finite_number(row.get("total_tokens"))
    if value is not None:
        return value
    usage = row.get("usage")
    if isinstance(usage, dict):
        return _finite_number(usage.get("total_tokens"))
    return None


def _information_gain(row: dict[str, object]) -> list[float]:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict) or not isinstance(metrics.get("information_gain"), list):
        return []
    return [
        value for item in metrics["information_gain"] if (value := _finite_number(item)) is not None
    ]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _quantile(values: list[float], probability: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _distribution(values: list[float]) -> dict[str, object]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "p0": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p100": None,
        }
    return {
        "n": len(values),
        "mean": _mean(values),
        "p0": round(_quantile(values, 0.0), 6),
        "p25": round(_quantile(values, 0.25), 6),
        "p50": round(_quantile(values, 0.5), 6),
        "p75": round(_quantile(values, 0.75), 6),
        "p100": round(_quantile(values, 1.0), 6),
    }


def _metric_report(
    rows: list[dict[str, object]], completed_rows: list[dict[str, object]]
) -> dict[str, object]:
    report: dict[str, object] = {}
    for metric in METRICS:
        completed_values = [
            value for row in completed_rows if (value := _metric(row, metric)) is not None
        ]
        all_values = [_metric(row, metric) or 0.0 for row in rows]
        report[metric] = {
            "all_rows_failure_as_zero": _mean(all_values),
            "completed_only": _mean(completed_values),
        }
    return report


def _failure_report(
    rows: list[dict[str, object]],
) -> tuple[dict[str, int], dict[str, list[dict[str, object]]]]:
    counts = Counter({category: 0 for category in FAILURE_CATEGORIES})
    examples: dict[str, list[dict[str, object]]] = {category: [] for category in FAILURE_CATEGORIES}
    for row in rows:
        if _completed(row):
            continue
        category = _failure_category(row)
        counts[category] += 1
        if len(examples[category]) < 3:
            examples[category].append(
                {
                    "question_id": row.get("question_id"),
                    "error": row.get("error"),
                }
            )
    return dict(counts), examples


def _analyze_baseline(rows: list[dict[str, object]]) -> dict[str, object]:
    completed_rows = [row for row in rows if _completed(row)]
    failure_counts, failure_examples = _failure_report(rows)
    search_values = [value for row in completed_rows if (value := _search_count(row)) is not None]
    token_values = [value for row in rows if (value := _token_count(row)) is not None]
    information_gain_values = [value for row in completed_rows for value in _information_gain(row)]
    return {
        "n": len(rows),
        "completed": len(completed_rows),
        "failed": len(rows) - len(completed_rows),
        "completion_rate": round(len(completed_rows) / len(rows), 6),
        "failure_categories": failure_counts,
        "failure_examples": failure_examples,
        "metrics": _metric_report(rows, completed_rows),
        "reward_distribution_completed_only": _distribution(
            [value for row in completed_rows if (value := _metric(row, "total")) is not None]
        ),
        "information_gain_per_search_completed_only": _distribution(information_gain_values),
        "searches_completed_only": _distribution(search_values),
        "tokens": {
            "rows_with_tokens": len(token_values),
            "total": round(sum(token_values), 6),
            "mean_per_row": _mean(token_values),
        },
    }


def analyze(path: Path) -> dict[str, object]:
    """Return a JSON-serializable report derived only from one episodes JSONL file."""
    rows = _read_rows(path)
    by_baseline: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        baseline = row.get("baseline", "unknown")
        key = baseline if isinstance(baseline, str) and baseline else "unknown"
        by_baseline.setdefault(key, []).append(row)
    return {
        "source": str(path),
        "rows": len(rows),
        "baselines": {
            baseline: _analyze_baseline(baseline_rows)
            for baseline, baseline_rows in sorted(by_baseline.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.episodes)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
