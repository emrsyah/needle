"""Offline model loading, run metadata, and split loading for local training/evaluation.

Call :func:`enforce_offline` before importing ``transformers`` in a script.
"""

from __future__ import annotations

import json
import os
import platform
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from needle.data import QuestionExample
from needle.data.hotpotqa import parse_hotpotqa_example

OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
_URL_PATTERN = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
_FORBIDDEN_ENV_PREFIXES = ("OPENROUTER",)


class ConfigurationError(ValueError):
    """A training/evaluation configuration violates the offline local-model contract."""


def enforce_offline() -> None:
    """Force Hugging Face libraries into offline mode for this process."""
    os.environ.update(OFFLINE_ENV)


def validate_local_model_path(
    model_path: str | Path, environ: Mapping[str, str] | None = None
) -> Path:
    """Reject URLs, missing paths, and provider credentials; return the resolved path."""
    environ = os.environ if environ is None else environ
    forbidden = sorted(
        key for key in environ if key.upper().startswith(_FORBIDDEN_ENV_PREFIXES) and environ[key]
    )
    if forbidden:
        raise ConfigurationError(
            "provider credentials are not allowed in the local training path: "
            + ", ".join(forbidden)
        )
    text = str(model_path)
    if not text or _URL_PATTERN.match(text) or "openrouter" in text.lower():
        raise ConfigurationError(f"model path must be a local directory, got {text!r}")
    path = Path(text).expanduser()
    if not path.is_dir():
        raise ConfigurationError(f"model path does not exist or is not a directory: {path}")
    if not (path / "config.json").is_file():
        raise ConfigurationError(f"model path has no config.json: {path}")
    return path.resolve()


def load_model_and_tokenizer(
    model_path: str | Path,
    *,
    dtype: str = "bfloat16",
    device: str | None = None,
    adapter_path: str | Path | None = None,
    lora: Mapping[str, Any] | None = None,
    gradient_checkpointing: bool = False,
) -> tuple[Any, Any]:
    """Load a local causal LM (optionally with a LoRA adapter) with no network access.

    ``adapter_path`` loads a saved adapter; ``lora`` creates a fresh trainable one.
    """
    enforce_offline()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = validate_local_model_path(model_path)
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch_dtype)
    model.to(device)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, str(adapter_path), is_trainable=lora is not None, local_files_only=True
        )
    elif lora is not None:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **dict(lora)))
    return model, tokenizer


def _model_revision(model_path: Path) -> str | None:
    """Best-effort Hugging Face snapshot commit hash from a cache path."""
    parts = model_path.resolve().parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def run_metadata(model_path: str | Path, extra: Mapping[str, Any] | None = None) -> dict:
    """Library versions, hardware, and model revision recorded with every local run."""
    import peft
    import torch
    import transformers

    path = Path(model_path)
    metadata: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_path": str(path),
        "model_revision": _model_revision(path),
        "offline": {key: os.environ.get(key) for key in OFFLINE_ENV},
    }
    if extra:
        metadata.update(extra)
    return metadata


def load_split_ids(manifest_path: str | Path, split: str) -> tuple[str, ...]:
    """Read question IDs for ``train``/``validation`` from splits.json or a holdout manifest."""
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if payload.get("name") == "holdout_v1":
        return tuple(payload["ids"])
    entries = payload["splits"][split]
    if isinstance(entries, dict):
        raise ConfigurationError(
            f"split {split!r} lives in its own manifest: {entries.get('manifest')}"
        )
    return tuple(entry["question_id"] for entry in entries)


def load_examples(source_path: str | Path, ids: Iterable[str]) -> tuple[QuestionExample, ...]:
    """Load the raw HotpotQA file and return examples for ``ids`` in the given order.

    Only the requested rows are validated, so malformed rows that the split
    preparation already skipped do not abort loading.
    """
    wanted = list(ids)
    wanted_set = set(wanted)
    payload = json.loads(Path(source_path).read_text(encoding="utf-8"))
    by_id: dict[str, QuestionExample] = {}
    for position, raw in enumerate(payload):
        if isinstance(raw, dict) and raw.get("_id") in wanted_set and raw["_id"] not in by_id:
            by_id[raw["_id"]] = parse_hotpotqa_example(raw, position)
    missing = [question_id for question_id in wanted if question_id not in by_id]
    if missing:
        raise ConfigurationError(f"{len(missing)} IDs missing from {source_path}: {missing[:3]}")
    return tuple(by_id[question_id] for question_id in wanted)


__all__ = [
    "ConfigurationError",
    "OFFLINE_ENV",
    "enforce_offline",
    "load_examples",
    "load_model_and_tokenizer",
    "load_split_ids",
    "run_metadata",
    "validate_local_model_path",
]
