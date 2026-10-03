import json
from pathlib import Path

import pytest

from needle.training.runtime import (
    ConfigurationError,
    load_examples,
    load_split_ids,
    validate_local_model_path,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "tiny_hotpotqa.json"


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    return tmp_path


def test_accepts_local_model_directory(model_dir):
    assert validate_local_model_path(model_dir, environ={}) == model_dir.resolve()


@pytest.mark.parametrize(
    "path",
    ["https://openrouter.ai/api/v1", "hf://Qwen/Qwen2.5-7B-Instruct", "qwen/openrouter-model", ""],
)
def test_rejects_remote_or_provider_model_paths(path):
    with pytest.raises(ConfigurationError):
        validate_local_model_path(path, environ={})


def test_rejects_provider_credentials(model_dir):
    with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY"):
        validate_local_model_path(model_dir, environ={"OPENROUTER_API_KEY": "secret"})


def test_rejects_directory_without_model_config(tmp_path):
    with pytest.raises(ConfigurationError, match="config.json"):
        validate_local_model_path(tmp_path, environ={})


def test_loads_requested_examples_in_order_and_ignores_other_rows(tmp_path):
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))
    malformed = {"_id": "broken", "question": ""}
    source = tmp_path / "source.json"
    source.write_text(json.dumps([malformed, *rows]), encoding="utf-8")
    ids = [row["_id"] for row in rows][::-1]
    assert [example.question_id for example in load_examples(source, ids)] == ids
    with pytest.raises(ConfigurationError, match="missing"):
        load_examples(source, ["absent"])


def test_reads_split_and_holdout_manifests(tmp_path):
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps({"splits": {"train": [{"question_id": "a"}], "holdout_v1": {"manifest": "h"}}}),
        encoding="utf-8",
    )
    holdout = tmp_path / "holdout.json"
    holdout.write_text(json.dumps({"name": "holdout_v1", "ids": ["x", "y"]}), encoding="utf-8")
    assert load_split_ids(splits, "train") == ("a",)
    assert load_split_ids(holdout, "holdout_v1") == ("x", "y")
    with pytest.raises(ConfigurationError):
        load_split_ids(splits, "holdout_v1")
