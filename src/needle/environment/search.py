"""Bounded, resettable BM25 search sessions."""

from dataclasses import dataclass

from needle.data import QuestionExample
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

    def reset(self) -> QuestionObservation:
        """Start a fresh session and return its question observation."""
        self._history.clear()
        self._started = True
        return QuestionObservation(
            question_id=self._example.question_id,
            question=self._example.question,
            max_searches=self._max_searches,
        )

    def search(self, query: object) -> SearchObservation:
        """Rank *query* and record its successful search observation."""
        if not self._started:
            raise SearchEnvironmentError("reset must be called before searching")
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
