# BM25 Retriever Design

## Goal

Add a deterministic lexical search baseline over a `QuestionExample` controlled corpus. The retriever returns ranked `Document` objects and scores that later agent-environment code can expose as search observations.

## Decision

Use `rank-bm25` and its `BM25Okapi` implementation. The library expects a pre-tokenized corpus and query, so Needle owns a small explicit tokenizer. This is lighter than BM25S for the current per-example corpus and safer than maintaining a custom BM25 formula.

## API

```python
retriever = BM25Retriever(example.documents)
results = retriever.search("Pacific Ocean coastline", top_k=2)
```

`SearchResult` is an immutable record containing a one-based rank, a float score, and the original `Document` object. `BM25Retriever` stores documents in input order and exposes a read-only tuple.

## Text and Ranking

- Indexed text is the document title followed by all sentences.
- Tokenization uses Unicode word tokens after `casefold()`; punctuation separates terms.
- Corpus and query use the same tokenizer.
- Ranking sorts by descending BM25 score, then original corpus position for deterministic ties.
- A query with no matching terms still returns deterministic zero-scored top-k documents, matching ordinary top-k retrieval behavior.
- `top_k` is capped by corpus size.

## Validation

- Corpus must be a non-empty sequence of `Document` objects with unique exact titles.
- Query must be a string producing at least one token.
- `top_k` must be a positive integer and must reject booleans.

## Files

```text
src/needle/retrieval/__init__.py
src/needle/retrieval/bm25.py
tests/retrieval/test_bm25.py
pyproject.toml
uv.lock
```

## Scope

This milestone does not add dense retrieval, persistence, global Wikipedia indexing, agent actions, rewards, model inference, or RL training.
