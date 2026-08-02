from dataclasses import FrozenInstanceError

import pytest

from needle.data.models import (
    Document,
    EvidenceRef,
    HotpotQAValidationError,
    QuestionExample,
)


def make_document(
    title: str = "Ada Lovelace", sentences: tuple[str, ...] = ("She wrote notes.",)
) -> Document:
    return Document(title=title, sentences=sentences)


def make_evidence(title: str = "Ada Lovelace", index: int = 0) -> EvidenceRef:
    return EvidenceRef(document_title=title, sentence_index=index)


def make_example(**overrides: object) -> QuestionExample:
    values: dict[str, object] = {
        "question_id": "needle-test",
        "question": "Who wrote the notes?",
        "gold_answer": "Ada Lovelace",
        "documents": (make_document(),),
        "supporting_facts": frozenset({make_evidence()}),
    }
    values.update(overrides)
    return QuestionExample(**values)  # type: ignore[arg-type]


def test_models_are_frozen_and_evidence_is_hashable() -> None:
    evidence = make_evidence()
    document = make_document()
    example = make_example()

    assert {evidence} == {make_evidence()}
    with pytest.raises(FrozenInstanceError):
        evidence.sentence_index = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        document.title = "Changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        example.question = "Changed"  # type: ignore[misc]


def test_documents_and_sentences_preserve_supplied_order() -> None:
    first = make_document("First", ("First zero.", "First one."))
    second = make_document("Second", ("Second zero.",))
    example = make_example(
        documents=(first, second),
        supporting_facts=frozenset({make_evidence("Second")}),
    )

    assert example.documents == (first, second)
    assert first.sentences == ("First zero.", "First one.")


def test_document_for_title_uses_exact_case_sensitive_lookup() -> None:
    document = make_document("Ada Lovelace")
    example = make_example(documents=(document,), supporting_facts=frozenset({make_evidence()}))

    assert example.document_for_title("Ada Lovelace") is document
    with pytest.raises(
        HotpotQAValidationError, match=r"example 'needle-test': documents.*ada lovelace"
    ):
        example.document_for_title("ada lovelace")


@pytest.mark.parametrize("title", ["", 1, None])
def test_evidence_ref_requires_non_empty_string_title(title: object) -> None:
    with pytest.raises(HotpotQAValidationError, match="document_title"):
        EvidenceRef(document_title=title, sentence_index=0)  # type: ignore[arg-type]


@pytest.mark.parametrize("index", [True, False, -1, 1.5, "0"])
def test_evidence_ref_rejects_boolean_negative_and_non_integer_indexes(index: object) -> None:
    with pytest.raises(
        HotpotQAValidationError, match=r"sentence_index must be a non-negative integer"
    ):
        EvidenceRef(document_title="Ada Lovelace", sentence_index=index)  # type: ignore[arg-type]


@pytest.mark.parametrize("title", ["", 1, None])
def test_document_requires_non_empty_string_title(title: object) -> None:
    with pytest.raises(HotpotQAValidationError, match="title"):
        Document(title=title, sentences=("Sentence.",))  # type: ignore[arg-type]


@pytest.mark.parametrize("sentences", [(), [], ("",), (1,), ("Valid.", "")])
def test_document_requires_non_empty_tuple_of_non_empty_string_sentences(sentences: object) -> None:
    with pytest.raises(HotpotQAValidationError, match="sentences"):
        Document(title="Ada Lovelace", sentences=sentences)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["question_id", "question", "gold_answer"])
@pytest.mark.parametrize("value", ["", 1, None])
def test_question_example_requires_non_empty_string_text_fields(field: str, value: object) -> None:
    values: dict[str, object] = {field: value}

    pattern = field if field == "question_id" else rf"example 'needle-test': {field}"
    with pytest.raises(HotpotQAValidationError, match=pattern):
        make_example(**values)


@pytest.mark.parametrize("documents", [(), [], ("not a document",)])
def test_question_example_requires_non_empty_tuple_of_documents(documents: object) -> None:
    with pytest.raises(HotpotQAValidationError, match=r"example 'needle-test': documents"):
        make_example(documents=documents)


def test_question_example_rejects_duplicate_document_titles() -> None:
    duplicate = make_document("Ada Lovelace", ("Another sentence.",))

    with pytest.raises(
        HotpotQAValidationError, match=r"example 'needle-test': documents.*duplicate"
    ):
        make_example(documents=(make_document(), duplicate))


@pytest.mark.parametrize(
    "facts", [frozenset(), set(), {make_evidence()}, (make_evidence(),), frozenset({object()})]
)
def test_question_example_requires_non_empty_frozenset_of_evidence_refs(facts: object) -> None:
    with pytest.raises(HotpotQAValidationError, match=r"example 'needle-test': supporting_facts"):
        make_example(supporting_facts=facts)


def test_question_example_rejects_unknown_and_lowercase_evidence_titles() -> None:
    with pytest.raises(
        HotpotQAValidationError, match=r"example 'needle-test': supporting_facts.*unknown"
    ):
        make_example(supporting_facts=frozenset({make_evidence("Unknown")}))
    with pytest.raises(
        HotpotQAValidationError, match=r"example 'needle-test': supporting_facts.*unknown"
    ):
        make_example(supporting_facts=frozenset({make_evidence("ada lovelace")}))


def test_question_example_rejects_out_of_range_evidence_index() -> None:
    with pytest.raises(
        HotpotQAValidationError, match=r"example 'needle-test': supporting_facts.*range"
    ):
        make_example(supporting_facts=frozenset({make_evidence(index=1)}))
