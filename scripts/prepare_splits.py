"""Prepare reproducible HotpotQA train, validation, and holdout manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from needle.data import QuestionExample
from needle.data.hotpotqa import parse_hotpotqa_example

DEFAULT_TRAIN_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
DEFAULT_DEV_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
SPLIT_ALGORITHM = "sha256_seed_question_id_v1"
TRAIN_FRACTION = 0.95


@dataclass(frozen=True, slots=True)
class ValidatedRow:
    """A validated example plus its original source position."""

    example: QuestionExample
    source_position: int


@dataclass(frozen=True, slots=True)
class SourceRows:
    """Validated unique rows and counted skip reasons for one source file."""

    rows: tuple[ValidatedRow, ...]
    skipped_rows: tuple[dict[str, object], ...]


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_date(path: Path, explicit_date: str | None) -> str:
    if explicit_date is not None:
        try:
            datetime.strptime(explicit_date, "%Y-%m-%d")
        except ValueError as error:
            raise ValueError("download date must use YYYY-MM-DD") from error
        return explicit_date
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).date().isoformat()


def _source_metadata(
    path: Path,
    *,
    url: str,
    dataset_version: str,
    download_date: str | None,
) -> dict[str, object]:
    payload_size = path.stat().st_size
    return {
        "dataset": "HotpotQA",
        "version": dataset_version,
        "filename": path.name,
        "source_url": url,
        "download_date": _download_date(path, download_date),
        "download_date_source": "argument" if download_date else "file_mtime_utc",
        "byte_count": payload_size,
        "sha256": _sha256_file(path),
    }


def _load_source(path: Path) -> SourceRows:
    try:
        raw_rows = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"source file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"source file is not valid JSON: {path}: {error.msg}") from error

    if not isinstance(raw_rows, list):
        raise ValueError(f"source top level must be a list: {path}")

    rows: list[ValidatedRow] = []
    skipped: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for position, raw_row in enumerate(raw_rows):
        raw_id = raw_row.get("_id") if isinstance(raw_row, dict) else None
        try:
            example = parse_hotpotqa_example(raw_row, position)
        except (KeyError, TypeError, ValueError) as error:
            skipped_row = {
                "source_position": position,
                "question_id": raw_id if isinstance(raw_id, str) else None,
                "reason": str(error),
                "category": "malformed",
            }
            skipped.append(skipped_row)
            print(
                f"SKIP source_position={position} question_id={raw_id!r} "
                f"category=malformed reason={error}",
                file=sys.stderr,
            )
            continue

        if example.question_id in seen_ids:
            skipped_row = {
                "source_position": position,
                "question_id": example.question_id,
                "reason": "duplicate _id; first occurrence kept",
                "category": "duplicate_id",
            }
            skipped.append(skipped_row)
            print(
                f"SKIP source_position={position} question_id={example.question_id!r} "
                "category=duplicate_id reason=first occurrence kept",
                file=sys.stderr,
            )
            continue

        seen_ids.add(example.question_id)
        rows.append(ValidatedRow(example=example, source_position=position))

    return SourceRows(rows=tuple(rows), skipped_rows=tuple(skipped))


def _id_sort_key(question_id: str) -> bytes:
    return question_id.encode("utf-8")


def _assign_train_validation(question_ids: list[str], seed: int) -> tuple[list[str], list[str]]:
    train: list[str] = []
    validation: list[str] = []
    for question_id in question_ids:
        digest = hashlib.sha256(f"{seed}:{question_id}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") / 2**64
        (train if bucket < TRAIN_FRACTION else validation).append(question_id)
    return train, validation


def _rows_by_id(rows: SourceRows) -> dict[str, ValidatedRow]:
    return {row.example.question_id: row for row in rows.rows}


def _load_existing_holdout(path: Path) -> tuple[tuple[str, ...], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(
            f"holdout manifest does not exist: {path}; use --bootstrap-holdout once "
            "to persist the intended v1 sample"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(f"holdout manifest is not valid JSON: {path}") from error

    if not isinstance(payload, dict) or payload.get("name") != "holdout_v1":
        raise ValueError(f"holdout manifest must have name='holdout_v1': {path}")
    ids = payload.get("ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(item, str) for item in ids):
        raise ValueError(f"holdout manifest ids must be a non-empty string list: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"holdout manifest contains duplicate IDs: {path}")
    manifest_hash = payload.get("manifest_sha256")
    if not isinstance(manifest_hash, str):
        raise ValueError(f"holdout manifest is missing manifest_sha256: {path}")
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    expected_hash = _sha256_bytes(_canonical_json(unsigned))
    if manifest_hash != expected_hash:
        raise ValueError(f"holdout manifest hash mismatch: {path}")
    return tuple(ids), manifest_hash


def _select_holdout(
    rows: SourceRows, path: Path, *, bootstrap: bool, size: int
) -> tuple[tuple[str, ...], str | None]:
    if path.exists():
        holdout_ids, manifest_hash = _load_existing_holdout(path)
        if len(holdout_ids) != size:
            raise ValueError(
                f"existing holdout_v1 has {len(holdout_ids)} IDs, expected {size}; "
                "refusing to replace it"
            )
        return holdout_ids, manifest_hash
    if not bootstrap:
        _load_existing_holdout(path)
    if len(rows.rows) < size:
        raise ValueError(
            f"cannot bootstrap holdout_v1 with {size} IDs from {len(rows.rows)} valid dev rows"
        )
    selected = tuple(row.example.question_id for row in rows.rows[:size])
    print(
        f"BOOTSTRAP holdout_v1={size} IDs from first valid dev rows; "
        "this is explicit and will be immutable on later runs",
        file=sys.stderr,
    )
    return selected, None


def _holdout_payload(
    holdout_ids: tuple[str, ...],
    dev_rows_by_id: dict[str, ValidatedRow],
    *,
    source_metadata: dict[str, object],
    selection: dict[str, object],
) -> dict[str, object]:
    entries = [
        {"question_id": question_id, "source_position": dev_rows_by_id[question_id].source_position}
        for question_id in holdout_ids
    ]
    return {
        "name": "holdout_v1",
        "source": source_metadata,
        "selection": selection,
        "id_order": "persisted_source_order",
        "entries": entries,
        "ids": list(holdout_ids),
    }


def _write_hashed_json(path: Path, payload: dict[str, object], hash_field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(hash_field, None)
    manifest_hash = _sha256_bytes(_canonical_json(unsigned))
    output = dict(unsigned)
    output[hash_field] = manifest_hash
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_hash


def prepare(args: argparse.Namespace) -> dict[str, object]:
    train_source = _load_source(args.train_source)
    dev_source = _load_source(args.dev_source)
    dev_rows_by_id = _rows_by_id(dev_source)
    holdout_path = args.output_dir / "holdout_v1_ids.json"
    holdout_ids, holdout_hash = _select_holdout(
        dev_source,
        holdout_path,
        bootstrap=args.bootstrap_holdout,
        size=args.holdout_size,
    )
    missing_holdout_ids = sorted(set(holdout_ids) - set(dev_rows_by_id), key=_id_sort_key)
    if missing_holdout_ids:
        raise ValueError(
            "holdout_v1 IDs missing from dev source: "
            + ", ".join(repr(item) for item in missing_holdout_ids[:10])
        )

    train_ids, validation_ids = _assign_train_validation(
        sorted((row.example.question_id for row in train_source.rows), key=_id_sort_key), args.seed
    )
    split_sets = {
        "train": set(train_ids),
        "validation": set(validation_ids),
        "holdout": set(holdout_ids),
    }
    if split_sets["train"] & split_sets["validation"]:
        raise AssertionError("train and validation IDs overlap")
    if (split_sets["train"] | split_sets["validation"]) & split_sets["holdout"]:
        raise AssertionError("train/validation and holdout IDs overlap")

    train_by_id = _rows_by_id(train_source)
    if holdout_hash is None:
        holdout_payload = _holdout_payload(
            holdout_ids,
            dev_rows_by_id,
            source_metadata=_source_metadata(
                args.dev_source,
                url=args.dev_url,
                dataset_version=args.dataset_version,
                download_date=args.dev_download_date or args.download_date,
            ),
            selection={
                "method": "first_valid_rows",
                "size": args.holdout_size,
                "explicit_bootstrap_required": True,
            },
        )
        holdout_hash = _write_hashed_json(holdout_path, holdout_payload, "manifest_sha256")

    split_payload: dict[str, Any] = {
        "name": "needle_evaluation_splits",
        "version": 1,
        "algorithm": {
            "name": SPLIT_ALGORITHM,
            "seed": args.seed,
            "train_fraction": TRAIN_FRACTION,
            "bucket": (
                "int.from_bytes(sha256(f'{seed}:{question_id}'.encode('utf-8'))[:8], 'big') / 2**64"
            ),
            "id_sort": "UTF-8 byte sequence",
            "source_position_indexing": "zero_based",
        },
        "sources": {
            "train": _source_metadata(
                args.train_source,
                url=args.train_url,
                dataset_version=args.dataset_version,
                download_date=args.train_download_date or args.download_date,
            ),
            "dev": _source_metadata(
                args.dev_source,
                url=args.dev_url,
                dataset_version=args.dataset_version,
                download_date=args.dev_download_date or args.download_date,
            ),
        },
        "skipped_rows": {
            "train": list(train_source.skipped_rows),
            "dev": list(dev_source.skipped_rows),
        },
        "splits": {
            "train": [
                {
                    "question_id": question_id,
                    "source_position": train_by_id[question_id].source_position,
                }
                for question_id in train_ids
            ],
            "validation": [
                {
                    "question_id": question_id,
                    "source_position": train_by_id[question_id].source_position,
                }
                for question_id in validation_ids
            ],
            "holdout_v1": {
                "source": "dev",
                "manifest": "holdout_v1_ids.json",
                "manifest_sha256": holdout_hash,
                "count": len(holdout_ids),
            },
        },
        "corpus": {
            "construction": "per-example native HotpotQA context documents",
            "retriever": "rank_bm25.BM25Okapi",
            "tokenizer_revision": "needle.retrieval.bm25.tokenize@working-tree",
            "top_k": args.top_k,
            "max_searches": args.max_searches,
        },
    }
    split_hash = _write_hashed_json(
        args.output_dir / "splits.json", split_payload, "manifest_sha256"
    )

    print(
        json.dumps(
            {
                "train_valid": len(train_source.rows),
                "train": len(train_ids),
                "validation": len(validation_ids),
                "holdout_v1": len(holdout_ids),
                "train_skipped": len(train_source.skipped_rows),
                "dev_skipped": len(dev_source.skipped_rows),
                "holdout_manifest_sha256": holdout_hash,
                "split_manifest_sha256": split_hash,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return split_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--dev-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("configs/evaluation"))
    parser.add_argument("--train-url", default=DEFAULT_TRAIN_URL)
    parser.add_argument("--dev-url", default=DEFAULT_DEV_URL)
    parser.add_argument("--dataset-version", default="v1.1")
    parser.add_argument("--download-date")
    parser.add_argument("--train-download-date")
    parser.add_argument("--dev-download-date")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--holdout-size", type=int, default=300)
    parser.add_argument("--bootstrap-holdout", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-searches", type=int, default=3)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.seed < 0:
        raise SystemExit("seed must be non-negative")
    if args.holdout_size <= 0 or args.top_k <= 0 or args.max_searches <= 0:
        raise SystemExit("holdout-size, top-k, and max-searches must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
