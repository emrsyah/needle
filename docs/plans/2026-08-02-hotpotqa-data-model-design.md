# HotpotQA Data Model Design

## Goal

Give every later Needle subsystem one deterministic representation of a question, its controlled document corpus, its gold answer, and its gold supporting evidence. The milestone ends with a small HotpotQA-shaped JSON fixture that can be loaded and validated without network access.

## Approaches Considered

1. Raw dictionaries are the fastest to write, but key names, sentence indexes, and evidence references can silently drift between retrieval, environment, and reward code.
2. Frozen standard-library dataclasses provide typed, immutable domain objects without adding a runtime dependency. This is the chosen approach.
3. Pydantic would provide rich schema errors and serialization, but that dependency and abstraction are unnecessary for the small initial boundary.

## Domain Model

`EvidenceRef` identifies one sentence by the exact pair `(document_title, sentence_index)`. It is immutable and hashable so coverage and utilization code can later use sets safely.

`Document` contains an exact title and an ordered tuple of sentences. Sentence order is part of evidence identity and must never be normalized or reordered by the loader.

`QuestionExample` contains:

- a stable question ID;
- the question text;
- the gold answer;
- an ordered tuple of documents forming the controlled corpus;
- a frozen set of gold supporting evidence references.

Document titles remain case-sensitive because HotpotQA supporting facts refer back to titles exactly. Text normalization for answer metrics belongs in the future reward layer, not the data model.

## Input Format

The fixture follows the relevant fields of native HotpotQA JSON:

```json
{
  "_id": "needle-001",
  "question": "...",
  "answer": "...",
  "context": [
    ["Document title", ["Sentence zero.", "Sentence one."]]
  ],
  "supporting_facts": [
    ["Document title", 1]
  ]
}
```

The top level is an array of examples. Additional official-dataset fields such as `type` or `level` may be present and are ignored. Required fields and their core shapes are validated explicitly.

## Validation Rules

- Question ID, question, gold answer, document titles, and sentences must be non-empty strings.
- Each example must contain at least one document and one supporting fact.
- Document titles must be unique within an example.
- Every supporting-fact title must resolve to a document using exact matching.
- Every supporting-fact sentence index must be a non-negative integer within that document.
- Duplicate supporting facts are rejected rather than silently collapsed.
- Boolean values are not accepted as sentence indexes even though Python treats booleans as integers.

Malformed data raises `HotpotQAValidationError`, a `ValueError` subclass. Messages include the example position or ID and the failing field so fixture and dataset failures are diagnosable.

## Components

```text
src/needle/data/
  __init__.py       public data API
  models.py         immutable domain objects
  hotpotqa.py       JSON parsing and validation

tests/data/
  test_models.py
  test_hotpotqa.py

tests/fixtures/
  tiny_hotpotqa.json
```

The package does not expose file-format dictionaries beyond the loader. Downstream code receives only domain objects.

## Tiny Controlled Corpus

The fixture contains three synthetic, HotpotQA-shaped multi-hop questions. Each example has two supporting documents and at least one distractor. Synthetic content keeps the test deterministic and small while exercising the same evidence-linking structure as the public dataset.

## Testing

- Model tests verify immutability, hashable evidence references, and document lookup behavior.
- Loader tests verify the fixture's IDs, answers, document order, sentences, and gold evidence.
- Focused invalid-input tests cover duplicate titles, missing supporting documents, duplicate evidence, boolean or out-of-range indexes, empty fields, and malformed top-level JSON shape.
- The complete repository continues to pass Ruff and pytest.

## Non-goals

- Downloading the full HotpotQA dataset.
- BM25 indexing or search.
- Citation parsing.
- Reward computation.
- Model inference or reinforcement learning.
