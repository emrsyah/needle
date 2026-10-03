"""Load validated HotpotQA JSON data into immutable domain models."""

import json
from pathlib import Path

from needle.data.models import Document, EvidenceRef, HotpotQAValidationError, QuestionExample


def _error(example_position: int, field: str, detail: str, question_id: object = None) -> None:
    """Raise a validation error annotated with an example position or identifier."""
    identity = (
        f"example {example_position} ({question_id!r})"
        if isinstance(question_id, str) and question_id
        else f"example {example_position}"
    )
    raise HotpotQAValidationError(f"{identity}: {field} {detail}")


def _non_empty_text(
    value: object, example_position: int, field: str, question_id: object = None
) -> str:
    if not isinstance(value, str) or not value:
        _error(example_position, field, "must be a non-empty string", question_id)
    return value


def _load_documents(
    example: dict[object, object], position: int, question_id: str
) -> tuple[Document, ...]:
    context = example["context"]
    if not isinstance(context, list) or not context:
        _error(position, "context", "must be a non-empty list", question_id)

    documents: list[Document] = []
    seen_titles: set[str] = set()
    for context_index, item in enumerate(context):
        item_field = f"context[{context_index}]"
        if not isinstance(item, list) or len(item) != 2:
            _error(position, item_field, "must be exactly [title, sentences]", question_id)

        title = _non_empty_text(item[0], position, f"{item_field}.title", question_id)
        if title in seen_titles:
            _error(position, item_field, f"contains duplicate title {title!r}", question_id)
        seen_titles.add(title)

        sentences = item[1]
        if not isinstance(sentences, list) or not sentences:
            _error(position, f"{item_field}.sentences", "must be a non-empty list", question_id)
        validated_sentences = tuple(
            _non_empty_text(
                sentence, position, f"{item_field}.sentences[{sentence_index}]", question_id
            )
            for sentence_index, sentence in enumerate(sentences)
        )
        documents.append(Document(title=title, sentences=validated_sentences))

    return tuple(documents)


def _load_supporting_facts(
    example: dict[object, object],
    position: int,
    question_id: str,
    documents: tuple[Document, ...],
) -> frozenset[EvidenceRef]:
    supporting_facts = example["supporting_facts"]
    if not isinstance(supporting_facts, list) or not supporting_facts:
        _error(position, "supporting_facts", "must be a non-empty list", question_id)

    documents_by_title = {document.title: document for document in documents}
    facts: list[EvidenceRef] = []
    seen_facts: set[tuple[str, int]] = set()
    for fact_index, item in enumerate(supporting_facts):
        item_field = f"supporting_facts[{fact_index}]"
        if not isinstance(item, list) or len(item) != 2:
            _error(position, item_field, "must be exactly [title, index]", question_id)

        title = _non_empty_text(item[0], position, f"{item_field}.title", question_id)
        document = documents_by_title.get(title)
        if document is None:
            _error(
                position,
                f"{item_field}.title",
                f"references unknown document {title!r}",
                question_id,
            )

        sentence_index = item[1]
        if isinstance(sentence_index, bool) or not isinstance(sentence_index, int):
            _error(
                position,
                f"{item_field}.index",
                "must be an integer (booleans are not allowed)",
                question_id,
            )
        if sentence_index < 0 or sentence_index >= len(document.sentences):
            _error(position, f"{item_field}.index", "is out of range", question_id)

        key = (title, sentence_index)
        if key in seen_facts:
            _error(position, item_field, f"is a duplicate fact {key!r}", question_id)
        seen_facts.add(key)
        facts.append(EvidenceRef(document_title=title, sentence_index=sentence_index))

    return frozenset(facts)


def load_hotpotqa(path: str | Path) -> tuple[QuestionExample, ...]:
    """Read a HotpotQA JSON file, validating its native core fields at the boundary."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise HotpotQAValidationError("invalid JSON in HotpotQA data") from error

    if not isinstance(payload, list):
        raise HotpotQAValidationError("top-level HotpotQA data must be a list")

    examples: list[QuestionExample] = []
    seen_question_ids: dict[str, int] = {}
    for position, raw_example in enumerate(payload):
        parsed = parse_hotpotqa_example(raw_example, position)
        question_id = parsed.question_id
        if question_id in seen_question_ids:
            _error(
                position,
                "_id",
                f"is a duplicate of earlier identifier at example {seen_question_ids[question_id]}",
                question_id,
            )
        seen_question_ids[question_id] = position
        examples.append(parsed)

    return tuple(examples)


def parse_hotpotqa_example(raw_example: object, position: int = 0) -> QuestionExample:
    """Validate and convert one native HotpotQA row.

    Keeping row parsing separate lets dataset-preparation tools report malformed rows
    individually while preserving this module as the single validation boundary.
    """
    if not isinstance(raw_example, dict):
        _error(position, "example", "must be an object")

    for field in ("_id", "question", "answer", "context", "supporting_facts"):
        if field not in raw_example:
            _error(position, field, "is required", raw_example.get("_id"))

    question_id = _non_empty_text(raw_example["_id"], position, "_id")
    question = _non_empty_text(raw_example["question"], position, "question", question_id)
    answer = _non_empty_text(raw_example["answer"], position, "answer", question_id)
    documents = _load_documents(raw_example, position, question_id)
    supporting_facts = _load_supporting_facts(raw_example, position, question_id, documents)
    return QuestionExample(
        question_id=question_id,
        question=question,
        gold_answer=answer,
        documents=documents,
        supporting_facts=supporting_facts,
    )
