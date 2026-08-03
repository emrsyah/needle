"""Bounded, resettable BM25 search sessions."""

from collections.abc import Sequence
from dataclasses import dataclass

from needle.data import EvidenceRef, QuestionExample
from needle.retrieval import BM25Retriever, SearchResult


class SearchEnvironmentError(RuntimeError):
    """Raised when a search is attempted outside an active session."""


@dataclass(frozen=True, slots=True)
class QuestionObservation:
    """The question and search budget supplied at session start."""

    question_id: str
    question: str
    max_searches: int


@dataclass(frozen=True, slots=True)
class SearchObservation:
    """One successful ranked search within a session."""

    turn: int
    query: str
    results: tuple[SearchResult, ...]
    remaining_searches: int


@dataclass(frozen=True, slots=True)
class AnswerObservation:
    """A terminal answer with citations from the controlled corpus."""

    answer: str
    citations: tuple[EvidenceRef, ...]
    searches_used: int


@dataclass(frozen=True, slots=True)
class EpisodeTrajectory:
    """The immutable record of a completed search-and-answer episode."""

    question: QuestionObservation
    searches: tuple[SearchObservation, ...]
    answer: AnswerObservation


class SearchEnvironment:
    """Run a bounded search session over one question's document corpus."""

    def __init__(self, example: QuestionExample, top_k: int = 3, max_searches: int = 5) -> None:
        if not isinstance(example, QuestionExample):
            raise TypeError("example must be a QuestionExample")
        self._validate_positive_int(top_k, "top_k")
        self._validate_positive_int(max_searches, "max_searches")

        self._example = example
        self._top_k = top_k
        self._max_searches = max_searches
        self._retriever = BM25Retriever(example.documents)
        self._history: list[SearchObservation] = []
        self._started = False
        self._terminated = False
        self._question_observation: QuestionObservation | None = None
        self._answer_observation: AnswerObservation | None = None
        self._trajectory: EpisodeTrajectory | None = None

    @staticmethod
    def _validate_positive_int(value: object, field: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field} must be a positive integer, not a boolean")
        if value <= 0:
            raise ValueError(f"{field} must be a positive integer")

    @property
    def example(self) -> QuestionExample:
        """Return the controlled question and document corpus."""
        return self._example

    @property
    def history(self) -> tuple[SearchObservation, ...]:
        """Return successful searches in this session."""
        return tuple(self._history)

    @property
    def remaining_searches(self) -> int:
        """Return unused searches in the current session."""
        return self._max_searches - len(self._history)

    @property
    def started(self) -> bool:
        """Whether reset has started the current session."""
        return self._started

    @property
    def terminated(self) -> bool:
        """Whether the current session has a terminal answer."""
        return self._terminated

    @property
    def answer_observation(self) -> AnswerObservation | None:
        """Return the terminal answer, when the session has one."""
        return self._answer_observation

    @property
    def trajectory(self) -> EpisodeTrajectory | None:
        """Return the completed trajectory, when the session has one."""
        return self._trajectory

    def reset(self) -> QuestionObservation:
        """Start a fresh session and return its question observation."""
        self._history.clear()
        self._started = True
        self._terminated = False
        self._answer_observation = None
        self._trajectory = None
        self._question_observation = QuestionObservation(
            question_id=self._example.question_id,
            question=self._example.question,
            max_searches=self._max_searches,
        )
        return self._question_observation

    def search(self, query: object) -> SearchObservation:
        """Rank *query* and record its successful search observation."""
        if not self._started:
            raise SearchEnvironmentError("reset must be called before searching")
        if self._terminated:
            raise SearchEnvironmentError("session is terminated")
        if self.remaining_searches == 0:
            raise SearchEnvironmentError("search budget is exhausted")

        if not isinstance(query, str):
            self._retriever.search(query, top_k=self._top_k)  # type: ignore[arg-type]
            raise AssertionError("BM25Retriever must reject non-string queries")

        normalized_query = query.strip()
        results = self._retriever.search(normalized_query, top_k=self._top_k)
        observation = SearchObservation(
            turn=len(self._history) + 1,
            query=normalized_query,
            results=results,
            remaining_searches=self.remaining_searches - 1,
        )
        self._history.append(observation)
        return observation

    def answer(self, answer: str, citations: Sequence[EvidenceRef] = ()) -> AnswerObservation:
        """Terminate the active session with a validated answer and citations."""
        if not self._started:
            raise SearchEnvironmentError("reset must be called before answering")
        if self._terminated:
            raise SearchEnvironmentError("session is terminated")

        if not isinstance(answer, str) or not (normalized_answer := answer.strip()):
            raise SearchEnvironmentError("answer must be a non-empty string")
        if isinstance(citations, str):
            raise SearchEnvironmentError("citations must be a sequence of EvidenceRef instances")
        try:
            normalized_citations = tuple(citations)
        except TypeError as error:
            raise SearchEnvironmentError(
                "citations must be a sequence of EvidenceRef instances"
            ) from error

        documents_by_title = {document.title: document for document in self._example.documents}
        for citation in normalized_citations:
            if not isinstance(citation, EvidenceRef):
                raise SearchEnvironmentError("citations must contain EvidenceRef instances")
            document = documents_by_title.get(citation.document_title)
            if document is None:
                raise SearchEnvironmentError(
                    f"citation references unknown document {citation.document_title!r}"
                )
            if citation.sentence_index >= len(document.sentences):
                raise SearchEnvironmentError(
                    f"citation sentence index is out of range for {citation.document_title!r}"
                )
        if len(set(normalized_citations)) != len(normalized_citations):
            raise SearchEnvironmentError("citations must not contain duplicates")

        question = self._question_observation
        if question is None:
            raise AssertionError("started sessions must have a question observation")
        observation = AnswerObservation(
            answer=normalized_answer,
            citations=normalized_citations,
            searches_used=len(self._history),
        )
        self._answer_observation = observation
        self._trajectory = EpisodeTrajectory(question, tuple(self._history), observation)
        self._terminated = True
        return observation
