"""Deterministic lexical retrieval over Needle documents."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from needle.data import Document

_WORD_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    """Return case-folded Unicode word tokens from *text*."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return tuple(_WORD_TOKEN_PATTERN.findall(text.casefold()))


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One immutable, ranked document result."""

    rank: int
    score: float
    document: Document


class BM25Retriever:
    """Rank an ordered corpus of documents using BM25Okapi."""

    def __init__(self, documents: Sequence[Document]) -> None:
        if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
            raise TypeError("documents must be a non-empty sequence of Document instances")

        self.documents = tuple(documents)
        if not self.documents:
            raise ValueError("documents must be a non-empty sequence of Document instances")

        seen_titles: set[str] = set()
        for document in self.documents:
            if not isinstance(document, Document):
                raise TypeError("documents must contain only Document instances")
            if document.title in seen_titles:
                raise ValueError(f"documents contain duplicate title: {document.title!r}")
            seen_titles.add(document.title)

        corpus: list[tuple[str, ...]] = []
        for document in self.documents:
            tokens = tokenize(" ".join((document.title, *document.sentences)))
            if not tokens:
                raise ValueError(
                    f"document {document.title!r} must contain at least one searchable token"
                )
            corpus.append(tokens)
        self._index = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 5) -> tuple[SearchResult, ...]:
        """Return up to *top_k* documents in deterministic descending-score order."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        query_tokens = tokenize(query)
        if not query_tokens:
            raise ValueError("query must contain at least one token")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be a positive integer, not a boolean")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        scores = self._index.get_scores(query_tokens)
        limit = min(top_k, len(self.documents))
        ranked_indices = sorted(
            range(len(self.documents)), key=lambda index: (-float(scores[index]), index)
        )
        return tuple(
            SearchResult(rank=rank, score=float(scores[index]), document=self.documents[index])
            for rank, index in enumerate(ranked_indices[:limit], start=1)
        )
