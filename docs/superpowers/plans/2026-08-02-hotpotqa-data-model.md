# HotpotQA Data Model Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add immutable HotpotQA domain objects and a validated loader for a deterministic three-example controlled corpus.

**Architecture:** Standard-library frozen dataclasses form the only representation exposed downstream. A boundary loader validates native HotpotQA core JSON, converts mutable JSON collections to immutable domain objects, and reports source-aware validation failures. Retrieval, rewards, model inference, and dataset downloading remain outside this milestone.

**Tech Stack:** Python 3.11, dataclasses, json, pathlib, pytest, Ruff, uv

---

## File Map

- Create: `src/needle/data/models.py` — immutable domain objects and the shared validation exception.
- Create: `src/needle/data/hotpotqa.py` — native-core HotpotQA JSON loader and boundary validation.
- Create: `src/needle/data/__init__.py` — supported public data API.
- Create: `tests/data/test_models.py` — model invariants, immutability, and exact document lookup.
- Create: `tests/data/test_hotpotqa.py` — fixture loading and malformed-input behavior.
- Create: `tests/fixtures/tiny_hotpotqa.json` — three deterministic multi-hop examples.

## Chunk 1: Immutable Domain Objects

### Task 1: Implement the model contract using TDD

**Files:**

- Create: `src/needle/data/models.py`
- Create: `tests/data/test_models.py`

- [ ] **Step 1: Write the failing model tests**

Create `tests/data/test_models.py`:

```python
from dataclasses import FrozenInstanceError

import pytest

from needle.data.models import (
    Document,
    EvidenceRef,
    HotpotQAValidationError,
    QuestionExample,
)


def documents() -> tuple[Document, ...]:
    return (
        Document("Maya", ("Maya moved to Bandung.",)),
        Document("Bandung", ("Bandung is in West Java.",)),
    )


def example(
    *,
    question_id: str = "needle-test",
    corpus: tuple[Document, ...] | None = None,
    facts: frozenset[EvidenceRef] | None = None,
) -> QuestionExample:
    return QuestionExample(
        question_id=question_id,
        question="Which province contains the city Maya moved to?",
        gold_answer="West Java",
        documents=documents() if corpus is None else corpus,
        supporting_facts=(
            frozenset({EvidenceRef("Maya", 0), EvidenceRef("Bandung", 0)})
            if facts is None
            else facts
        ),
    )


def test_domain_objects_are_immutable_and_evidence_is_hashable() -> None:
    evidence = EvidenceRef("Maya", 0)
    assert {evidence, EvidenceRef("Maya", 0)} == {evidence}

    with pytest.raises(FrozenInstanceError):
        evidence.sentence_index = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        documents()[0].title = "Changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        example().question = "Changed?"  # type: ignore[misc]


def test_document_order_and_exact_title_lookup_are_preserved() -> None:
    item = example()

    assert tuple(document.title for document in item.documents) == ("Maya", "Bandung")
    assert item.document_for_title("Maya") is item.documents[0]
    with pytest.raises(HotpotQAValidationError, match=r"needle-test.*documents.*maya"):
        item.document_for_title("maya")


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (lambda: EvidenceRef("", 0), "document_title"),
        (lambda: EvidenceRef("Maya", -1), "sentence_index"),
        (lambda: EvidenceRef("Maya", True), "sentence_index"),
        (lambda: Document("", ("Sentence.",)), "title"),
        (lambda: Document("Maya", ()), "sentences"),
        (lambda: Document("Maya", ("",)), "sentences"),
    ],
)
def test_local_fields_are_validated(factory, field: str) -> None:
    with pytest.raises(HotpotQAValidationError, match=field):
        factory()


def test_mutable_collections_are_rejected() -> None:
    with pytest.raises(HotpotQAValidationError, match="sentences"):
        Document("Maya", ["Sentence."])  # type: ignore[arg-type]

    with pytest.raises(HotpotQAValidationError, match=r"needle-test.*documents"):
        QuestionExample(
            question_id="needle-test",
            question="Question?",
            gold_answer="Answer",
            documents=list(documents()),  # type: ignore[arg-type]
            supporting_facts=frozenset({EvidenceRef("Maya", 0)}),
        )

    with pytest.raises(HotpotQAValidationError, match=r"needle-test.*supporting_facts"):
        QuestionExample(
            question_id="needle-test",
            question="Question?",
            gold_answer="Answer",
            documents=documents(),
            supporting_facts={EvidenceRef("Maya", 0)},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("corpus", "facts", "field"),
    [
        ((), frozenset({EvidenceRef("Maya", 0)}), "documents"),
        (documents(), frozenset(), "supporting_facts"),
        (
            (Document("Maya", ("One.",)), Document("Maya", ("Two.",))),
            frozenset({EvidenceRef("Maya", 0)}),
            "documents",
        ),
        (documents(), frozenset({EvidenceRef("Missing", 0)}), "supporting_facts"),
        (documents(), frozenset({EvidenceRef("maya", 0)}), "supporting_facts"),
        (documents(), frozenset({EvidenceRef("Maya", 1)}), "supporting_facts"),
    ],
)
def test_example_rejects_invalid_corpus_or_facts(
    corpus: tuple[Document, ...],
    facts: frozenset[EvidenceRef],
    field: str,
) -> None:
    with pytest.raises(HotpotQAValidationError, match=rf"needle-test.*{field}"):
        example(corpus=corpus, facts=facts)


@pytest.mark.parametrize(
    ("question_id", "question", "gold_answer", "field"),
    [
        ("", "Question?", "Answer", "question_id"),
        ("needle-test", "", "Answer", "question"),
        ("needle-test", "Question?", "", "gold_answer"),
    ],
)
def test_example_rejects_empty_required_text(
    question_id: str,
    question: str,
    gold_answer: str,
    field: str,
) -> None:
    with pytest.raises(HotpotQAValidationError, match=field):
        QuestionExample(
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
            documents=documents(),
            supporting_facts=frozenset({EvidenceRef("Maya", 0)}),
        )
```

- [ ] **Step 2: Run the tests to verify the red state**

Run: `uv run pytest tests/data/test_models.py -v`

Expected: collection fails because `needle.data.models` does not exist.

- [ ] **Step 3: Implement the immutable models**

Create `src/needle/data/models.py`:

```python
"""Immutable domain models for controlled HotpotQA examples."""

from dataclasses import dataclass


class HotpotQAValidationError(ValueError):
    """Raised when HotpotQA data violates Needle's data contract."""


def _non_empty_string(value: object, field: str, context: str = "") -> None:
    if not isinstance(value, str) or not value:
        raise HotpotQAValidationError(f"{context}{field} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """An exact reference to one sentence in a titled document."""

    document_title: str
    sentence_index: int

    def __post_init__(self) -> None:
        _non_empty_string(self.document_title, "document_title")
        if (
            isinstance(self.sentence_index, bool)
            or not isinstance(self.sentence_index, int)
            or self.sentence_index < 0
        ):
            raise HotpotQAValidationError(
                "sentence_index must be a non-negative integer, not a boolean"
            )


@dataclass(frozen=True, slots=True)
class Document:
    """A titled document with evidence-addressable ordered sentences."""

    title: str
    sentences: tuple[str, ...]

    def __post_init__(self) -> None:
        _non_empty_string(self.title, "title")
        if not isinstance(self.sentences, tuple) or not self.sentences:
            raise HotpotQAValidationError("sentences must be a non-empty tuple")
        for sentence in self.sentences:
            _non_empty_string(sentence, "sentences")


@dataclass(frozen=True, slots=True)
class QuestionExample:
    """One question, its controlled corpus, and gold supporting facts."""

    question_id: str
    question: str
    gold_answer: str
    documents: tuple[Document, ...]
    supporting_facts: frozenset[EvidenceRef]

    def __post_init__(self) -> None:
        _non_empty_string(self.question_id, "question_id")
        context = f"example {self.question_id!r}: "
        _non_empty_string(self.question, "question", context)
        _non_empty_string(self.gold_answer, "gold_answer", context)

        if not isinstance(self.documents, tuple) or not self.documents:
            raise HotpotQAValidationError(f"{context}documents must be a non-empty tuple")
        if not all(isinstance(document, Document) for document in self.documents):
            raise HotpotQAValidationError(f"{context}documents must contain Document objects")
        titles = tuple(document.title for document in self.documents)
        if len(set(titles)) != len(titles):
            raise HotpotQAValidationError(f"{context}documents contain duplicate titles")

        if not isinstance(self.supporting_facts, frozenset) or not self.supporting_facts:
            raise HotpotQAValidationError(
                f"{context}supporting_facts must be a non-empty frozenset"
            )
        if not all(isinstance(fact, EvidenceRef) for fact in self.supporting_facts):
            raise HotpotQAValidationError(
                f"{context}supporting_facts must contain EvidenceRef objects"
            )

        documents_by_title = {document.title: document for document in self.documents}
        for fact in self.supporting_facts:
            document = documents_by_title.get(fact.document_title)
            if document is None:
                raise HotpotQAValidationError(
                    f"{context}supporting_facts references unknown document {fact.document_title!r}"
                )
            if fact.sentence_index >= len(document.sentences):
                raise HotpotQAValidationError(
                    f"{context}supporting_facts sentence_index {fact.sentence_index} is out "
                    f"of range for document {fact.document_title!r}"
                )

    def document_for_title(self, title: str) -> Document:
        """Return a document using exact case-sensitive title matching."""
        for document in self.documents:
            if document.title == title:
                return document
        raise HotpotQAValidationError(
            f"example {self.question_id!r}: documents contain no title {title!r}"
        )
```

- [ ] **Step 4: Run focused and repository tests**

Run: `uv run pytest tests/data/test_models.py -v`

Expected: all focused model tests pass.

Run: `uv run pytest`

Expected: the complete suite passes.

- [ ] **Step 5: Commit the model slice**

```text
git add src/needle/data/models.py tests/data/test_models.py
git commit -m "feat: add immutable HotpotQA models"
```

## Chunk 2: Native JSON Boundary and Controlled Fixture

### Task 2: Add the deterministic fixture

**Files:**

- Create: `tests/fixtures/tiny_hotpotqa.json`
- Create: `tests/data/test_hotpotqa.py`

- [ ] **Step 1: Write a fixture contract test**

Create `tests/data/test_hotpotqa.py`:

```python
import json
from pathlib import Path


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "tiny_hotpotqa.json"


def test_fixture_has_three_multi_hop_examples_with_distractors() -> None:
    raw = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert [item["_id"] for item in raw] == ["needle-001", "needle-002", "needle-003"]
    assert all(
        {"_id", "question", "answer", "context", "supporting_facts"} <= item.keys() for item in raw
    )
    assert all(len(item["context"]) == 3 for item in raw)
    assert all(len(item["supporting_facts"]) == 2 for item in raw)
    assert [item["answer"] for item in raw] == ["United States", "Pacific Ocean", "Quito"]
    for item in raw:
        context_titles = {document[0] for document in item["context"]}
        supporting_titles = {fact[0] for fact in item["supporting_facts"]}
        assert len(supporting_titles) == 2
        assert supporting_titles < context_titles
        assert len(context_titles - supporting_titles) == 1
```

- [ ] **Step 2: Run the fixture test to verify it fails because the file is absent**

Run: `uv run pytest tests/data/test_hotpotqa.py -v`

Expected: the test fails with `FileNotFoundError` for `tiny_hotpotqa.json`.

- [ ] **Step 3: Create the exact fixture**

Create `tests/fixtures/tiny_hotpotqa.json`:

```json
[
  {
    "_id": "needle-001",
    "question": "Which country is the birthplace of the author of The Left Hand of Darkness?",
    "answer": "United States",
    "context": [
      ["The Left Hand of Darkness", ["The Left Hand of Darkness is a science fiction novel.", "It was written by Ursula K. Le Guin."]],
      ["Ursula K. Le Guin", ["Ursula K. Le Guin was an American author.", "She was born in Berkeley, California, United States."]],
      ["Berkeley", ["Berkeley is a city in California.", "It is known for the University of California, Berkeley."]]
    ],
    "supporting_facts": [["The Left Hand of Darkness", 1], ["Ursula K. Le Guin", 1]]
  },
  {
    "_id": "needle-002",
    "question": "What ocean borders the country where Machu Picchu is located?",
    "answer": "Pacific Ocean",
    "context": [
      ["Machu Picchu", ["Machu Picchu is a fifteenth-century Inca citadel.", "It is located in Peru."]],
      ["Peru", ["Peru is a country on the western side of South America.", "Peru has a coastline on the Pacific Ocean."]],
      ["Atlantic Ocean", ["The Atlantic Ocean lies between the Americas and Europe.", "It does not border Peru."]]
    ],
    "supporting_facts": [["Machu Picchu", 1], ["Peru", 1]]
  },
  {
    "_id": "needle-003",
    "question": "What is the capital of the country where the Galapagos Islands are located?",
    "answer": "Quito",
    "context": [
      ["Galapagos Islands", ["The Galapagos Islands are an archipelago in the Pacific Ocean.", "They are part of Ecuador."]],
      ["Ecuador", ["Ecuador is a country in northwestern South America.", "Its capital city is Quito."]],
      ["Guayaquil", ["Guayaquil is Ecuador's largest city.", "It is not Ecuador's capital."]]
    ],
    "supporting_facts": [["Galapagos Islands", 1], ["Ecuador", 1]]
  }
]
```

- [ ] **Step 4: Run the fixture test**

Run: `uv run pytest tests/data/test_hotpotqa.py -v`

Expected: the fixture test passes.

- [ ] **Step 5: Commit the fixture**

```text
git add tests/fixtures/tiny_hotpotqa.json tests/data/test_hotpotqa.py
git commit -m "test: add tiny HotpotQA corpus"
```

### Task 3: Implement successful loading and invalid-input behavior

**Files:**

- Create: `src/needle/data/hotpotqa.py`
- Modify: `tests/data/test_hotpotqa.py`

- [ ] **Step 1: Extend the tests with positive loading and invalid cases**

Add to `tests/data/test_hotpotqa.py`:

```python
import pytest

from needle.data.hotpotqa import load_hotpotqa
from needle.data.models import EvidenceRef, HotpotQAValidationError


VALID_EXAMPLE = {
    "_id": "valid-001",
    "question": "What is the answer?",
    "answer": "Answer",
    "context": [
        ["Supporting A", ["Sentence A."]],
        ["Supporting B", ["Sentence B."]],
        ["Distractor", ["Distractor sentence."]],
    ],
    "supporting_facts": [["Supporting A", 0], ["Supporting B", 0]],
}


def write_payload(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "examples.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_hotpotqa_preserves_order_answer_and_supporting_facts() -> None:
    examples = load_hotpotqa(FIXTURE_PATH)

    assert tuple(item.question_id for item in examples) == (
        "needle-001",
        "needle-002",
        "needle-003",
    )
    first = examples[0]
    assert first.gold_answer == "United States"
    assert tuple(document.title for document in first.documents) == (
        "The Left Hand of Darkness",
        "Ursula K. Le Guin",
        "Berkeley",
    )
    assert first.documents[0].sentences[1] == "It was written by Ursula K. Le Guin."
    assert first.supporting_facts == frozenset(
        {
            EvidenceRef("The Left Hand of Darkness", 1),
            EvidenceRef("Ursula K. Le Guin", 1),
        }
    )
    assert examples[1].gold_answer == "Pacific Ocean"
    assert examples[2].gold_answer == "Quito"


@pytest.mark.parametrize("field", ["_id", "question", "answer", "context", "supporting_facts"])
def test_load_hotpotqa_rejects_missing_required_field(tmp_path: Path, field: str) -> None:
    payload = {key: value for key, value in VALID_EXAMPLE.items() if key != field}

    with pytest.raises(HotpotQAValidationError, match=rf"{field}.*required"):
        load_hotpotqa(write_payload(tmp_path, [payload]))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("_id", ""),
        ("question", ""),
        ("question", 42),
        ("answer", ""),
        ("answer", None),
    ],
)
def test_load_hotpotqa_rejects_invalid_required_text(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = {**VALID_EXAMPLE, field: value}

    with pytest.raises(HotpotQAValidationError, match=rf"{field}.*non-empty"):
        load_hotpotqa(write_payload(tmp_path, [payload]))


def test_load_hotpotqa_ignores_additional_official_fields(tmp_path: Path) -> None:
    payload = {**VALID_EXAMPLE, "type": "bridge", "level": "easy"}

    assert load_hotpotqa(write_payload(tmp_path, [payload]))[0].question_id == "valid-001"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, r"top level.*array"),
        (["not an object"], r"example 0.*object"),
        ([{**VALID_EXAMPLE, "context": []}], r"context.*non-empty"),
        ([{**VALID_EXAMPLE, "context": [["Bad"]]}], r"context\[0\].*title.*sentences"),
        ([{**VALID_EXAMPLE, "context": [["Title", []]]}], r"context\[0\].*sentences.*non-empty"),
        ([{**VALID_EXAMPLE, "context": [["", ["Sentence."]]]}], r"context\[0\].*title.*non-empty"),
        ([{**VALID_EXAMPLE, "context": [[42, ["Sentence."]]]}], r"context\[0\].*title.*non-empty"),
        (
            [{**VALID_EXAMPLE, "context": [["Repeated", ["One."]], ["Repeated", ["Two."]]]}],
            r"duplicate.*title",
        ),
        ([{**VALID_EXAMPLE, "context": [["Title", [""]]]}], r"context\[0\].*sentence.*non-empty"),
        ([{**VALID_EXAMPLE, "supporting_facts": []}], r"supporting_facts.*non-empty"),
        (
            [{**VALID_EXAMPLE, "supporting_facts": [["Bad"]]}],
            r"supporting_facts\[0\].*title.*index",
        ),
        ([{**VALID_EXAMPLE, "supporting_facts": [["Missing", 0]]}], r"unknown document"),
        (
            [{**VALID_EXAMPLE, "supporting_facts": [["Supporting A", True]]}],
            r"sentence_index.*integer",
        ),
        ([{**VALID_EXAMPLE, "supporting_facts": [["Supporting A", -1]]}], r"sentence_index.*range"),
        ([{**VALID_EXAMPLE, "supporting_facts": [["Supporting A", 1]]}], r"sentence_index.*range"),
        (
            [{**VALID_EXAMPLE, "supporting_facts": [["Supporting A", 0], ["Supporting A", 0]]}],
            r"duplicate.*supporting fact",
        ),
    ],
)
def test_load_hotpotqa_rejects_invalid_payload(
    tmp_path: Path, payload: object, message: str
) -> None:
    with pytest.raises(HotpotQAValidationError, match=message):
        load_hotpotqa(write_payload(tmp_path, payload))


def test_load_hotpotqa_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(HotpotQAValidationError, match=r"top level JSON"):
        load_hotpotqa(path)
```

- [ ] **Step 2: Run the loader tests to verify the red state**

Run: `uv run pytest tests/data/test_hotpotqa.py -v`

Expected: collection fails because `needle.data.hotpotqa` does not exist.

- [ ] **Step 3: Implement the boundary loader**

Create `src/needle/data/hotpotqa.py`:

```python
"""Load and validate native-core HotpotQA JSON."""

import json
from pathlib import Path

from .models import Document, EvidenceRef, HotpotQAValidationError, QuestionExample


def _error(position: int, example_id: object, field: str, detail: str) -> HotpotQAValidationError:
    identity = repr(example_id) if isinstance(example_id, str) and example_id else "unknown id"
    return HotpotQAValidationError(f"example {position} ({identity}) field {field!r}: {detail}")


def _required(raw: dict[str, object], position: int, field: str) -> object:
    if field not in raw:
        raise _error(position, raw.get("_id"), field, "is required")
    return raw[field]


def _text(value: object, position: int, example_id: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise _error(position, example_id, field, "must be a non-empty string")
    return value


def _load_example(raw: object, position: int) -> QuestionExample:
    if not isinstance(raw, dict):
        raise _error(position, None, "example", "must be an object")

    question_id = _text(_required(raw, position, "_id"), position, raw.get("_id"), "_id")
    question = _text(_required(raw, position, "question"), position, question_id, "question")
    answer = _text(_required(raw, position, "answer"), position, question_id, "answer")

    raw_context = _required(raw, position, "context")
    if not isinstance(raw_context, list) or not raw_context:
        raise _error(position, question_id, "context", "must be a non-empty list")

    documents: list[Document] = []
    sentences_by_title: dict[str, tuple[str, ...]] = {}
    for document_position, raw_document in enumerate(raw_context):
        field = f"context[{document_position}]"
        if not isinstance(raw_document, list) or len(raw_document) != 2:
            raise _error(position, question_id, field, "must contain title and sentences")
        title = _text(raw_document[0], position, question_id, f"{field}.title")
        raw_sentences = raw_document[1]
        if not isinstance(raw_sentences, list) or not raw_sentences:
            raise _error(position, question_id, f"{field}.sentences", "must be a non-empty list")
        if title in sentences_by_title:
            raise _error(position, question_id, field, f"duplicate document title {title!r}")
        sentences = tuple(
            _text(sentence, position, question_id, f"{field}.sentence[{sentence_position}]")
            for sentence_position, sentence in enumerate(raw_sentences)
        )
        documents.append(Document(title, sentences))
        sentences_by_title[title] = sentences

    raw_facts = _required(raw, position, "supporting_facts")
    if not isinstance(raw_facts, list) or not raw_facts:
        raise _error(position, question_id, "supporting_facts", "must be a non-empty list")

    facts: set[EvidenceRef] = set()
    for fact_position, raw_fact in enumerate(raw_facts):
        field = f"supporting_facts[{fact_position}]"
        if not isinstance(raw_fact, list) or len(raw_fact) != 2:
            raise _error(position, question_id, field, "must contain title and sentence index")
        title = _text(raw_fact[0], position, question_id, f"{field}.document_title")
        if title not in sentences_by_title:
            raise _error(position, question_id, field, f"references unknown document {title!r}")
        index = raw_fact[1]
        if isinstance(index, bool) or not isinstance(index, int):
            raise _error(position, question_id, f"{field}.sentence_index", "must be an integer")
        if index < 0 or index >= len(sentences_by_title[title]):
            raise _error(position, question_id, f"{field}.sentence_index", "is out of range")
        fact = EvidenceRef(title, index)
        if fact in facts:
            raise _error(position, question_id, field, "duplicate supporting fact")
        facts.add(fact)

    return QuestionExample(
        question_id=question_id,
        question=question,
        gold_answer=answer,
        documents=tuple(documents),
        supporting_facts=frozenset(facts),
    )


def load_hotpotqa(path: str | Path) -> tuple[QuestionExample, ...]:
    """Load a local native-core HotpotQA JSON array."""
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise HotpotQAValidationError(f"top level JSON: {error.msg}") from error
    if not isinstance(raw, list):
        raise HotpotQAValidationError("top level must be an array of examples")
    return tuple(_load_example(item, position) for position, item in enumerate(raw))
```

- [ ] **Step 4: Run the focused loader tests**

Run: `uv run pytest tests/data/test_hotpotqa.py -v`

Expected: all fixture, positive-loader, malformed-input, and invalid-JSON tests pass.

- [ ] **Step 5: Commit the loader slice**

```text
git add src/needle/data/hotpotqa.py tests/data/test_hotpotqa.py
git commit -m "feat: load validated HotpotQA data"
```

### Task 4: Expose the supported data API and run final gates

**Files:**

- Create: `src/needle/data/__init__.py`
- Modify: `tests/data/test_hotpotqa.py`

- [ ] **Step 1: Add a failing public API test**

Add to `tests/data/test_hotpotqa.py`:

```python
def test_public_data_api_exports_models_error_and_loader() -> None:
    from needle.data import (
        Document,
        EvidenceRef,
        HotpotQAValidationError,
        QuestionExample,
        load_hotpotqa,
    )

    assert Document.__module__ == "needle.data.models"
    assert EvidenceRef.__module__ == "needle.data.models"
    assert HotpotQAValidationError.__module__ == "needle.data.models"
    assert QuestionExample.__module__ == "needle.data.models"
    assert callable(load_hotpotqa)
```

- [ ] **Step 2: Run the public API test to verify the red state**

Run: `uv run pytest tests/data/test_hotpotqa.py::test_public_data_api_exports_models_error_and_loader -v`

Expected: import fails because `needle.data` does not yet export the API.

- [ ] **Step 3: Create the public exports**

Create `src/needle/data/__init__.py`:

```python
"""Public data models and loaders for Needle."""

from .hotpotqa import load_hotpotqa
from .models import Document, EvidenceRef, HotpotQAValidationError, QuestionExample

__all__ = [
    "Document",
    "EvidenceRef",
    "HotpotQAValidationError",
    "QuestionExample",
    "load_hotpotqa",
]
```

- [ ] **Step 4: Run all quality gates**

Run:

```text
uv run pytest tests/data -v
uv run ruff check .
uv run ruff format --check .
uv run pytest
git diff --check
```

Expected: every command exits zero, all tests pass, and no whitespace errors are reported.

- [ ] **Step 5: Commit the public API**

```text
git add src/needle/data/__init__.py tests/data/test_hotpotqa.py
git commit -m "feat: expose HotpotQA data API"
```

## Final Acceptance

- [ ] `load_hotpotqa("tests/fixtures/tiny_hotpotqa.json")` returns three immutable examples in fixture order.
- [ ] Every example contains two gold supporting facts and one distractor document.
- [ ] Invalid native-core shapes fail with `HotpotQAValidationError` containing example position or ID and the failing field.
- [ ] No runtime dependency was added.
- [ ] No BM25, reward, model, RL, download, normalization, or citation code was added.
