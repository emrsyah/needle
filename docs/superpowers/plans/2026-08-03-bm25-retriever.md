# BM25 Retriever Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and verify a deterministic `rank-bm25` retriever over Needle `Document` objects.

**Architecture:** Needle performs one shared tokenization step for corpus and query, delegates scoring to `BM25Okapi`, and applies its own stable top-k ordering. The public retrieval API exposes immutable result records without leaking NumPy arrays or library internals.

**Tech Stack:** Python 3.11, rank-bm25, pytest, Ruff, uv

---

## Chunk 1: Complete Retriever Slice

### Task 1: Add dependency, retriever, tests, and public API

**Files:**

- Modify: `pyproject.toml`
- Generate: `uv.lock`
- Create: `src/needle/retrieval/bm25.py`
- Create: `src/needle/retrieval/__init__.py`
- Create: `tests/retrieval/test_bm25.py`

- [ ] Add runtime dependency `rank-bm25>=0.2.2,<0.3` with uv and update the lockfile.
- [ ] Implement `tokenize(text: str) -> tuple[str, ...]` using `casefold()` and Unicode letter/number word tokens, with punctuation and underscores as separators.
- [ ] Implement frozen, slotted `SearchResult(rank: int, score: float, document: Document)`.
- [ ] Implement `BM25Retriever(documents)` with a non-empty immutable corpus, exact unique titles, nonempty searchable title-plus-sentences tokens per document, and `BM25Okapi` scoring.
- [ ] Implement `search(query, top_k=5)` with query/top-k validation, descending scores, input-position tie breaks, and corpus-size capping.
- [ ] Re-export `BM25Retriever`, `SearchResult`, and `tokenize` from `needle.retrieval` through exact `__all__`.
- [ ] Test tokenizer case/punctuation behavior, corpus validation, query/top-k validation including bool, original object identity, score/rank types, expected HotpotQA fixture retrieval, top-k capping, and deterministic zero-score ties.
- [ ] Run `uv run pytest tests/retrieval -v`, the full test suite, `uv run ruff check .`, `uv run ruff format --check .`, and `git diff --check`.
- [ ] Commit implementation with no coauthor trailer and leave the feature branch clean.

## Acceptance

- `BM25Retriever(example.documents).search(query, top_k)` returns immutable ranked results.
- Queries targeting each hop of the tiny corpus retrieve the intended supporting document first.
- No dependency or code unrelated to lexical retrieval is introduced.
