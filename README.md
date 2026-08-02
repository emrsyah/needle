# Needle

Needle is a research project for training an evidence-seeking agentic RAG system. It studies how reinforcement learning can help an agent search for relevant evidence, use that evidence in its answer, and avoid unnecessary retrieval steps.

## Status

Early research prototype. The current milestone is a deterministic HotpotQA vertical slice; model inference and reinforcement learning are not implemented yet.

## Development

Requirements:

- [uv](https://docs.astral.sh/uv/)

Set up the environment:

```bash
uv sync --dev
```

Run the quality gates:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```
