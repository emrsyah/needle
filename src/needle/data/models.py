"""Immutable HotpotQA domain models."""

from dataclasses import dataclass


class HotpotQAValidationError(ValueError):
    """Raised when HotpotQA domain data violates its required shape."""


def _require_non_empty_text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise HotpotQAValidationError(f"{field} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """A reference to one sentence in a document."""

    document_title: str
    sentence_index: int

    def __post_init__(self) -> None:
        _require_non_empty_text(self.document_title, "document_title")
        if isinstance(self.sentence_index, bool) or not isinstance(self.sentence_index, int):
            raise HotpotQAValidationError("sentence_index must be an integer")
        if self.sentence_index < 0:
            raise HotpotQAValidationError("sentence_index must be non-negative")


@dataclass(frozen=True, slots=True)
class Document:
    """An ordered, titled collection of source sentences."""

    title: str
    sentences: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_non_empty_text(self.title, "title")
        if not isinstance(self.sentences, tuple) or not self.sentences:
            raise HotpotQAValidationError("sentences must be a non-empty tuple")
        for index, sentence in enumerate(self.sentences):
            if not isinstance(sentence, str) or not sentence:
                raise HotpotQAValidationError(f"sentences[{index}] must be a non-empty string")


@dataclass(frozen=True, slots=True)
class QuestionExample:
    """A question with its controlled corpus and supporting evidence."""

    question_id: str
    question: str
    gold_answer: str
    documents: tuple[Document, ...]
    supporting_facts: frozenset[EvidenceRef]

    def __post_init__(self) -> None:
        _require_non_empty_text(self.question_id, "question_id")
        _require_non_empty_text(self.question, "question")
        _require_non_empty_text(self.gold_answer, "gold_answer")

        if not isinstance(self.documents, tuple) or not self.documents:
            raise HotpotQAValidationError("documents must be a non-empty tuple")

        titles: set[str] = set()
        for document in self.documents:
            if not isinstance(document, Document):
                raise HotpotQAValidationError("documents must contain Document instances")
            if document.title in titles:
                raise HotpotQAValidationError(f"duplicate document title: {document.title!r}")
            titles.add(document.title)

        if not isinstance(self.supporting_facts, frozenset) or not self.supporting_facts:
            raise HotpotQAValidationError("supporting_facts must be a non-empty frozenset")

        for fact in self.supporting_facts:
            if not isinstance(fact, EvidenceRef):
                raise HotpotQAValidationError("supporting_facts must contain EvidenceRef instances")
            document = self.document_for_title(fact.document_title)
            if fact.sentence_index >= len(document.sentences):
                raise HotpotQAValidationError(
                    f"question {self.question_id!r} has out-of-range supporting fact "
                    f"{fact.document_title!r}[{fact.sentence_index}]"
                )

    def document_for_title(self, title: str) -> Document:
        """Return the document whose title exactly matches *title*."""
        for document in self.documents:
            if document.title == title:
                return document
        raise HotpotQAValidationError(
            f"question {self.question_id!r} documents do not contain title {title!r}"
        )
