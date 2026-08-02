import json
from pathlib import Path

import pytest

from needle.data.models import EvidenceRef, HotpotQAValidationError

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "tiny_hotpotqa.json"


def _payload() -> list[dict[str, object]]:
    return [
        {
            "_id": "example-001",
            "question": "Where was Ada born?",
            "answer": "London",
            "context": [
                ["Ada Lovelace", ["Ada was a mathematician.", "Ada was born in London."]],
                ["London", ["London is in England."]],
            ],
            "supporting_facts": [["Ada Lovelace", 1]],
        }
    ]


def _write_payload(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "hotpotqa.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _load(path: Path):
    from needle.data.hotpotqa import load_hotpotqa

    return load_hotpotqa(path)


def test_loads_fixture_into_ordered_domain_examples() -> None:
    examples = _load(FIXTURE_PATH)

    assert [example.question_id for example in examples] == [
        "needle-001",
        "needle-002",
        "needle-003",
    ]
    assert [example.gold_answer for example in examples] == [
        "United States",
        "Pacific Ocean",
        "Quito",
    ]
    first = examples[0]
    assert [document.title for document in first.documents] == [
        "The Left Hand of Darkness",
        "Ursula K. Le Guin",
        "Berkeley",
    ]
    assert first.documents[0].sentences == (
        "The Left Hand of Darkness is a 1969 science fiction novel.",
        "It was written by Ursula K. Le Guin.",
    )
    assert first.supporting_facts == frozenset(
        {
            EvidenceRef("The Left Hand of Darkness", 1),
            EvidenceRef("Ursula K. Le Guin", 1),
        }
    )


@pytest.mark.parametrize("field", ["_id", "question", "answer", "context", "supporting_facts"])
def test_rejects_missing_core_field(tmp_path: Path, field: str) -> None:
    payload = _payload()
    del payload[0][field]

    with pytest.raises(HotpotQAValidationError, match=rf"example 0.*{field}"):
        _load(_write_payload(tmp_path, payload))


@pytest.mark.parametrize(
    "field, value",
    [("_id", ""), ("_id", 1), ("question", ""), ("question", 1), ("answer", ""), ("answer", 1)],
)
def test_rejects_empty_or_nonstring_core_text(tmp_path: Path, field: str, value: object) -> None:
    payload = _payload()
    payload[0][field] = value

    with pytest.raises(HotpotQAValidationError, match=rf"example 0.*{field}"):
        _load(_write_payload(tmp_path, payload))


def test_rejects_non_list_top_level(tmp_path: Path) -> None:
    with pytest.raises(HotpotQAValidationError, match=r"top-level.*list"):
        _load(_write_payload(tmp_path, {"example": _payload()[0]}))


def test_rejects_non_object_example(tmp_path: Path) -> None:
    with pytest.raises(HotpotQAValidationError, match=r"example 0.*object"):
        _load(_write_payload(tmp_path, ["not an object"]))


@pytest.mark.parametrize(
    "context", [[], "not a list", [["Ada Lovelace"]], [["Ada Lovelace", [], "extra"]]]
)
def test_rejects_empty_or_malformed_context(tmp_path: Path, context: object) -> None:
    payload = _payload()
    payload[0]["context"] = context

    with pytest.raises(HotpotQAValidationError, match=r"example 0.*context"):
        _load(_write_payload(tmp_path, payload))


@pytest.mark.parametrize("title", ["", 1])
def test_rejects_empty_or_nonstring_context_title(tmp_path: Path, title: object) -> None:
    payload = _payload()
    payload[0]["context"] = [[title, ["Ada was born in London."]]]

    with pytest.raises(HotpotQAValidationError, match=r"example 0.*context\[0\].*title"):
        _load(_write_payload(tmp_path, payload))


def test_rejects_duplicate_context_title(tmp_path: Path) -> None:
    payload = _payload()
    payload[0]["context"] = [["Ada", ["One."]], ["Ada", ["Two."]]]

    with pytest.raises(HotpotQAValidationError, match=r"example 0.*context.*duplicate.*Ada"):
        _load(_write_payload(tmp_path, payload))


@pytest.mark.parametrize(
    "sentences", [[], "not a list", ["Ada was born in London.", ""], ["Ada was born in London.", 1]]
)
def test_rejects_empty_or_nonstring_context_sentences(tmp_path: Path, sentences: object) -> None:
    payload = _payload()
    payload[0]["context"] = [["Ada Lovelace", sentences]]

    with pytest.raises(HotpotQAValidationError, match=r"example 0.*context\[0\].*sentences"):
        _load(_write_payload(tmp_path, payload))


@pytest.mark.parametrize(
    "facts", [[], "not a list", [["Ada Lovelace"]], [["Ada Lovelace", 0, "extra"]]]
)
def test_rejects_empty_or_malformed_supporting_facts(tmp_path: Path, facts: object) -> None:
    payload = _payload()
    payload[0]["supporting_facts"] = facts

    with pytest.raises(HotpotQAValidationError, match=r"example 0.*supporting_facts"):
        _load(_write_payload(tmp_path, payload))


@pytest.mark.parametrize("title", ["", 1, "Unknown"])
def test_rejects_invalid_or_unknown_supporting_fact_title(tmp_path: Path, title: object) -> None:
    payload = _payload()
    payload[0]["supporting_facts"] = [[title, 0]]

    with pytest.raises(HotpotQAValidationError, match=r"example 0.*supporting_facts\[0\].*title"):
        _load(_write_payload(tmp_path, payload))


@pytest.mark.parametrize("index", [True, -1, 2])
def test_rejects_invalid_supporting_fact_index(tmp_path: Path, index: object) -> None:
    payload = _payload()
    payload[0]["supporting_facts"] = [["Ada Lovelace", index]]

    with pytest.raises(HotpotQAValidationError, match=r"example 0.*supporting_facts\[0\].*index"):
        _load(_write_payload(tmp_path, payload))


def test_rejects_duplicate_supporting_fact(tmp_path: Path) -> None:
    payload = _payload()
    payload[0]["supporting_facts"] = [["Ada Lovelace", 1], ["Ada Lovelace", 1]]

    with pytest.raises(
        HotpotQAValidationError, match=r"example 0.*supporting_facts\[1\].*duplicate"
    ):
        _load(_write_payload(tmp_path, payload))


def test_chains_invalid_json_as_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{not JSON", encoding="utf-8")

    with pytest.raises(HotpotQAValidationError, match=r"invalid JSON") as error:
        _load(path)

    assert isinstance(error.value.__cause__, json.JSONDecodeError)


def test_ignores_extra_official_fields_at_all_levels(tmp_path: Path) -> None:
    payload = _payload()
    payload[0]["level"] = "easy"
    payload[0]["type"] = "bridge"
    payload[0]["extra"] = {"ignored": True}

    examples = _load(_write_payload(tmp_path, payload))

    assert examples[0].question_id == "example-001"
