"""Local Transformers policy that records the token data needed for GRPO.

This module imports ``torch`` lazily through its callers; install the optional
``training`` dependency group (``uv sync --group training``) before using it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .rollouts import ModelMetadata, PolicyResponse, TokenTrace

PromptFormatter = Callable[[str], list[int]]


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Sampling settings for one action generation."""

    max_new_tokens: int = 128
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.do_sample and self.temperature <= 0:
            raise ValueError("temperature must be positive when sampling")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


def chat_prompt_formatter(tokenizer: Any) -> PromptFormatter:
    """Format the raw collector prompt as one user chat turn, like the OpenRouter baseline."""

    def format_prompt(prompt_text: str) -> list[int]:
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=True,
        )
        if isinstance(token_ids, dict) or hasattr(token_ids, "keys"):
            token_ids = token_ids["input_ids"]
        return [int(token_id) for token_id in token_ids]

    return format_prompt


def completion_log_probs(
    model: Any, prompt_ids: Sequence[int], completion_ids: Sequence[int]
) -> torch.Tensor:
    """Return per-token log-probabilities of ``completion_ids`` given ``prompt_ids``.

    Uses the canonical shifted causal alignment: completion token ``j`` is scored by
    the logit at index ``len(prompt_ids) + j - 1``. Gradients flow if enabled.
    """
    if not prompt_ids or not completion_ids:
        raise ValueError("prompt_ids and completion_ids must be non-empty")
    device = next(model.parameters()).device
    input_ids = torch.tensor([list(prompt_ids) + list(completion_ids)], device=device)
    logits = model(input_ids=input_ids).logits[0]
    start = len(prompt_ids) - 1
    scored = logits[start : start + len(completion_ids)].float()
    targets = input_ids[0, len(prompt_ids) :]
    return torch.log_softmax(scored, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


class LocalPolicy:
    """Generate one action per turn with a local causal LM and record its token trace."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        generation: GenerationConfig | None = None,
        model_name: str = "local",
        prompt_formatter: PromptFormatter | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.generation = generation or GenerationConfig()
        self.model_name = model_name
        self.prompt_formatter = prompt_formatter or chat_prompt_formatter(tokenizer)
        self.seed = seed

    def reseed(self, seed: int | None) -> None:
        """Set the sampling seed used by the next ``act`` calls."""
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

    def _eos_ids(self) -> set[int]:
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        if eos is None:
            eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is None:
            return set()
        return {int(eos)} if isinstance(eos, int) else {int(item) for item in eos}

    @torch.no_grad()
    def act(self, prompt_text: str, observation_text: str) -> PolicyResponse:
        del observation_text  # the prompt already contains the full search history
        prompt_ids = self.prompt_formatter(prompt_text)
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([prompt_ids], device=device)
        was_training = self.model.training
        self.model.eval()
        try:
            kwargs: dict[str, object] = {
                "max_new_tokens": self.generation.max_new_tokens,
                "do_sample": self.generation.do_sample,
                "attention_mask": torch.ones_like(input_ids),
                # Override the checkpoint's generation_config (Qwen ships top_k=20 and
                # repetition_penalty>1) so samples come from the same softmax that
                # old_log_probs scores, and greedy decoding is a pure argmax.
                "repetition_penalty": 1.0,
            }
            if self.generation.do_sample:
                kwargs["temperature"] = self.generation.temperature
                kwargs["top_p"] = self.generation.top_p
                kwargs["top_k"] = 0
            else:
                kwargs["temperature"] = None
                kwargs["top_p"] = None
                kwargs["top_k"] = None
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = next(iter(self._eos_ids()), 0)
            kwargs["pad_token_id"] = pad_id
            output = self.model.generate(input_ids, **kwargs)
            completion_ids = [int(token) for token in output[0, len(prompt_ids) :].tolist()]
            eos_ids = self._eos_ids()
            for index, token in enumerate(completion_ids):
                if token in eos_ids:
                    completion_ids = completion_ids[: index + 1]
                    break
            if not completion_ids:
                raise ValueError("model generated no tokens")
            # Teacher-forced recomputation, same code path as the trainer.
            log_probs = completion_log_probs(self.model, prompt_ids, completion_ids)
        finally:
            if was_training:
                self.model.train()

        action_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        tokens = TokenTrace(
            prompt_text=prompt_text,
            prompt_token_ids=tuple(prompt_ids),
            completion_token_ids=tuple(completion_ids),
            action_token_spans=((0, len(completion_ids)),),
            action_token_mask=(True,) * len(completion_ids),
            old_log_probs=tuple(float(value) for value in log_probs.tolist()),
        )
        metadata = ModelMetadata(
            model=self.model_name,
            provider="local",
            extras=(
                ("seed", self.seed),
                ("do_sample", self.generation.do_sample),
                ("temperature", self.generation.temperature if self.generation.do_sample else 0.0),
                ("top_p", self.generation.top_p if self.generation.do_sample else 1.0),
                ("top_k", 0),
                ("repetition_penalty", 1.0),
            ),
        )
        return PolicyResponse(action_text=action_text, model_metadata=metadata, tokens=tokens)


__all__ = ["GenerationConfig", "LocalPolicy", "chat_prompt_formatter", "completion_log_probs"]
