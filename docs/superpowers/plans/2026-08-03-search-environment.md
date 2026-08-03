# Search Environment Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a bounded, resettable BM25 search session over one `QuestionExample`.

**Architecture:** `SearchEnvironment` owns mutable episode state while exposing only frozen observations and tuple history. It delegates lexical ranking and query validation to the existing `BM25Retriever` and adds reset/budget behavior without introducing an RL framework.

**Tech Stack:** Python 3.11, existing Needle data/retrieval packages, pytest, Ruff

---

## Chunk 1: Complete Search Session

### Task 1: Implement environment, public API, and tests

**Files:**

- Create: `src/needle/environment/search.py`
- Create: `src/needle/environment/__init__.py`
- Create: `tests/environment/test_search.py`

- [ ] Implement frozen, slotted `QuestionObservation` and `SearchObservation` records.
- [ ] Implement `SearchEnvironmentError(RuntimeError)`.
- [ ] Implement `SearchEnvironment(example, top_k=3, max_searches=5)` with validated non-boolean positive configuration and an internal `BM25Retriever`.
- [ ] Implement read-only `example`, `history`, `remaining_searches`, and `started` state.
- [ ] Implement `reset()` to clear history, mark started, and return the question observation.
- [ ] Implement `search(query)` to reject pre-reset/exhausted states, delegate ranking, avoid state mutation on invalid queries, and append one immutable observation on success.
- [ ] Re-export the four supported environment symbols through exact `__all__`.
- [ ] Test lifecycle, reset, budget consumption/exhaustion, invalid-query non-mutation, repeated queries, result equivalence with BM25, object immutability, configuration validation, and public API.
- [ ] Run focused/full pytest, repository Ruff lint/format, and `git diff --check`.
- [ ] Commit without a coauthor trailer and leave the feature branch clean.

## Acceptance

- A tiny HotpotQA example can be reset and searched across multiple recorded turns.
- Every successful observation contains deterministic ranked documents and correct remaining budget.
- No answer, citation, reward, or RL framework code is added.
