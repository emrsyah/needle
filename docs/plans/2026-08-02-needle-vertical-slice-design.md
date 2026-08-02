# Needle Vertical Slice Design

## Goal

Build the smallest deterministic end-to-end slice of Needle before adding reinforcement learning. One HotpotQA question must be able to enter an environment, invoke BM25 search over a controlled corpus, produce an answer with evidence references, and receive an inspectable reward breakdown.

## Chosen Approach

Use an environment-first Python package with framework-independent domain types and reward functions.

Two alternatives were considered:

- A notebook-first prototype would be faster for an initial demonstration but harder to test, reuse, and run consistently on Modal or Kaggle.
- A trainer-first prototype built directly around TRL would reach GRPO sooner but would couple environment bugs to training behavior and make reward validation difficult.

The environment-first approach adds a small amount of setup while keeping the retrieval, trajectory, and reward logic independently testable. TRL, PEFT, and model inference are intentionally deferred until the deterministic slice is correct.

## Architecture

The initial package is divided into four boundaries:

1. `data`: normalized HotpotQA examples and controlled document corpora.
2. `retrieval`: a BM25 search interface returning ranked document passages.
3. `environment`: action parsing and trajectory state for `search` and `answer` actions.
4. `rewards`: pure functions that score answer correctness, supporting-evidence coverage, evidence utilization, search cost, and duplicate queries.

Trainer integration will consume these boundaries later rather than being embedded inside them.

## Data Flow

```text
HotpotQA example
    -> controlled corpus
    -> Needle environment
    -> search(query)
    -> ranked passages
    -> answer(text, citations)
    -> trajectory
    -> reward component breakdown
```

Every episode retains the question, gold answer, gold supporting facts, issued queries, retrieved passages, final answer, citations, termination reason, and individual reward components.

## First Scaffold

The first implementation step creates a Python package using a `src` layout, a minimal test suite, configuration for formatting and static checks, and a smoke test. It does not download datasets, models, or indexes.

Planned top-level structure:

```text
needle/
  pyproject.toml
  README.md
  src/needle/
  tests/
  configs/
  scripts/
  docs/plans/
```

## Error Handling

- Invalid actions must produce explicit validation errors rather than silently changing episode state.
- Search results must use stable identifiers so citations and gold evidence can be compared deterministically.
- Episodes must terminate on a final answer or a configurable maximum number of search steps.
- Reward components must remain individually visible; only the final aggregator produces a scalar reward.

## Testing Strategy

- Unit tests cover action parsing, normalized answers, evidence identifiers, every reward component, and reward aggregation.
- Retrieval tests use a tiny in-memory corpus with known rankings.
- An integration test runs one complete scripted episode without a language model.
- Model and GRPO smoke tests are added only after the deterministic integration test passes.

## Non-goals for This Slice

- Full HotpotQA ingestion.
- GPU or Modal execution.
- Qwen inference.
- TRL/GRPO training.
- Dense retrieval.
- Large-scale experiment tracking.

These are later milestones and must not complicate the first deterministic slice.
