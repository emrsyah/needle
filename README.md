# Needle

Needle is a research project for training an evidence-seeking agentic RAG system. It studies how reinforcement learning can help an agent search for relevant evidence, use that evidence in its answer, and avoid unnecessary retrieval steps.

## Status

Research prototype. The deterministic HotpotQA environment, an OpenRouter baseline, a local Qwen policy, and an interactive GRPO trainer are implemented. GPU runs follow [docs/runbooks/runpod.md](docs/runbooks/runpod.md).

## Development

Requirements:

- [uv](https://docs.astral.sh/uv/)

Set up the environment:

```bash
uv sync --dev
```

For local model training/evaluation (PyTorch, Transformers, PEFT), add the training group. A plain `uv sync --dev` removes it again:

```bash
uv sync --dev --group training
```

Run the quality gates:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```
