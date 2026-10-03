from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from needle.data.hotpotqa import load_hotpotqa  # noqa: E402
from needle.training import RolloutCollector  # noqa: E402
from needle.training.grpo import rollout_mean_log_prob, rollout_old_mean  # noqa: E402
from needle.training.local_policy import (  # noqa: E402
    GenerationConfig,
    LocalPolicy,
    completion_log_probs,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "tiny_hotpotqa.json"
VOCAB = 64
EOS = 1


class FakeTokenizer:
    eos_token_id = EOS
    pad_token_id = 0

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(97 + token % 26) for token in token_ids if token != EOS)


def _formatter(prompt_text):
    return [2 + (ord(char) % (VOCAB - 2)) for char in prompt_text[-48:]]


def _tiny_lora_model():
    torch.manual_seed(0)
    config = transformers.Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        eos_token_id=EOS,
    )
    model = transformers.Qwen2ForCausalLM(config)
    lora = peft.LoraConfig(
        task_type="CAUSAL_LM", r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj"]
    )
    return peft.get_peft_model(model, lora)


def _policy(model):
    return LocalPolicy(
        model,
        FakeTokenizer(),
        generation=GenerationConfig(max_new_tokens=8, do_sample=True),
        prompt_formatter=_formatter,
        seed=3,
    )


def test_act_records_reproducible_old_log_probs():
    model = _tiny_lora_model()
    policy = _policy(model)
    policy.reseed(3)
    response = policy.act("Question: who?", "(none)")
    trace = response.tokens
    assert trace is not None
    assert trace.action_token_spans == ((0, len(trace.completion_token_ids)),)
    assert all(trace.action_token_mask)
    assert len(trace.old_log_probs) == len(trace.completion_token_ids)

    model.train()
    recomputed = completion_log_probs(model, trace.prompt_token_ids, trace.completion_token_ids)
    assert recomputed.tolist() == pytest.approx(list(trace.old_log_probs), abs=1e-6)


def test_collector_episode_supports_a_grpo_update():
    model = _tiny_lora_model()
    policy = _policy(model)
    example = load_hotpotqa(FIXTURE)[0]
    record = RolloutCollector(policy, top_k=3, max_searches=3).collect(example)
    # Random weights emit gibberish, so the episode is a protocol failure with tokens.
    assert record.steps[0].tokens is not None
    assert record.steps[0].tokens.prompt_text == record.steps[0].prompt_text

    model.train()

    def log_prob_fn(trace):
        return completion_log_probs(model, trace.prompt_token_ids, trace.completion_token_ids)

    new_mean = rollout_mean_log_prob(record, log_prob_fn)
    assert float(new_mean.detach()) == pytest.approx(rollout_old_mean(record), abs=1e-5)
    (-new_mean).backward()
    lora_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "lora_" in name and parameter.requires_grad
    ]
    assert lora_grads and any(grad is not None and grad.abs().sum() > 0 for grad in lora_grads)
