"""Offline tests for the OpenRouter baseline transport and runner."""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from needle.data import EvidenceRef, load_hotpotqa
from needle.environment import SearchEnvironment
from needle.inference import (
    CompletionResult,
    CompletionUsage,
    EpisodeRunner,
    EpisodeRunnerError,
    OpenRouterClient,
    OpenRouterConfig,
    OpenRouterHTTPError,
    OpenRouterResponseError,
)

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "tiny_hotpotqa.json"


def config(**kwargs: object) -> OpenRouterConfig:
    return OpenRouterConfig(api_key="secret", model="qwen/qwen3-8b", **kwargs)


def test_client_builds_reproducible_openrouter_request() -> None:
    requests = []

    def transport(request):
        requests.append(request)
        return 200, json.dumps(
            {
                "id": "resp-1",
                "model": "qwen/qwen3-8b",
                "provider": "mock",
                "choices": [{"message": {"content": "SEARCH[Ursula]"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }
        ).encode()

    client = OpenRouterClient(
        config(
            seed=7,
            provider_order=("openai", "anthropic"),
            allow_fallbacks=True,
            http_referer="https://needle.test",
            title="Needle",
        ),
        transport,
    )
    result = client.complete(({"role": "user", "content": "hello"},), stop=("END",))
    request = requests[0]
    payload = json.loads(request.body)

    assert request.url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer secret"
    assert request.headers["HTTP-Referer"] == "https://needle.test"
    assert request.headers["X-OpenRouter-Title"] == "Needle"
    assert "secret" not in repr(request)
    assert payload == {
        "model": "qwen/qwen3-8b",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
        "stream": False,
        "seed": 7,
        "stop": ["END"],
        "provider": {
            "require_parameters": True,
            "order": ["openai", "anthropic"],
            "allow_fallbacks": True,
        },
    }
    assert result == CompletionResult(
        "SEARCH[Ursula]",
        "resp-1",
        "qwen/qwen3-8b",
        "mock",
        CompletionUsage(10, 2, 12),
        request_body(result),
    )


def request_body(result: CompletionResult) -> str:
    return result.raw_json


def test_client_parses_missing_usage_and_rejects_errors() -> None:
    def response(status, payload):
        return lambda request: (status, json.dumps(payload).encode())

    payload = {"id": "x", "choices": [{"message": {"content": "ANSWER[ok]"}}]}
    result = OpenRouterClient(config(), response(200, payload)).complete(
        ({"role": "user", "content": "q"},)
    )
    assert result.usage == CompletionUsage(None, None, None)
    with pytest.raises(OpenRouterHTTPError, match="HTTP 401"):
        OpenRouterClient(config(), response(401, {"error": "bad"})).complete(
            ({"role": "user", "content": "q"},)
        )
    with pytest.raises(OpenRouterResponseError):
        OpenRouterClient(config(), response(200, {"choices": []})).complete(
            ({"role": "user", "content": "q"},)
        )
    with pytest.raises(OpenRouterResponseError, match="contains an error"):
        OpenRouterClient(
            config(), response(200, {"error": {"message": "provider failed"}})
        ).complete(({"role": "user", "content": "q"},))
    with pytest.raises(OpenRouterResponseError, match="complete response"):
        OpenRouterClient(
            config(),
            response(
                200,
                {
                    "id": "x",
                    "choices": [
                        {
                            "message": {"content": "ANSWER[partial"},
                            "finish_reason": "length",
                        }
                    ],
                },
            ),
        ).complete(({"role": "user", "content": "q"},))


def test_config_is_immutable_and_rejects_conflicting_providers() -> None:
    value = config(provider_order=["mock"])
    assert value.provider_order == ("mock",)
    with pytest.raises(FrozenInstanceError):
        value.model = "other"  # type: ignore[misc]
    with pytest.raises(ValueError, match="cannot both"):
        config(provider_order=("a",), provider_only=("b",))


def test_prompt_is_deterministic_and_runner_completes_fake_episode() -> None:
    example = load_hotpotqa(FIXTURE_PATH)[0]
    environment = SearchEnvironment(example, top_k=1, max_searches=2)
    prompts = []
    responses = iter(("SEARCH[Ursula]", "ANSWER[United States] CITATIONS[Ursula K. Le Guin|1]"))

    class FakeClient:
        def complete(self, messages):
            prompt = messages[0]["content"]
            assert messages[0]["role"] == "user"
            if prompts:
                assert "Search 1: Ursula" in prompt
                assert "[Ursula K. Le Guin]" in prompt
                assert "1: She was born in Berkeley" in prompt
            else:
                assert "Search results so far:\n(none)" in prompt
                assert "Ursula K. Le Guin|1; The Left Hand of Darkness|1" in prompt
                assert "after the final closing bracket" in prompt
                assert "Copy citation titles exactly as shown" in prompt
            prompts.append(prompt)
            return CompletionResult(
                next(responses), "id", "mock", None, CompletionUsage(1, 1, 2), "{}"
            )

    trajectory = EpisodeRunner(FakeClient()).run(environment)  # type: ignore[arg-type]
    assert trajectory.answer.citations == (EvidenceRef("Ursula K. Le Guin", 1),)
    assert len(prompts) == 2


def test_runner_retries_parser_error_with_validator_feedback() -> None:
    example = load_hotpotqa(FIXTURE_PATH)[0]
    prompts = []
    responses = iter(
        (
            "ANSWER[United States] CITATIONS[Ursula K. Le Guin|1] trailing",
            "ANSWER[United States] CITATIONS[Ursula K. Le Guin|1]",
        )
    )

    class RetryingClient:
        def complete(self, messages):
            prompts.append(messages[0]["content"])
            return CompletionResult(
                next(responses), "id", "mock", None, CompletionUsage(1, 1, 2), "{}"
            )

    trajectory = EpisodeRunner(RetryingClient(), max_retries=1).run(  # type: ignore[arg-type]
        SearchEnvironment(example, top_k=1, max_searches=1)
    )

    assert trajectory.answer.answer == "United States"
    assert len(prompts) == 2
    assert "Validator feedback: the previous action was rejected." in prompts[1]
    assert "action does not match the Needle action grammar" in prompts[1]


def test_runner_retries_invalid_citation_with_validator_feedback() -> None:
    example = load_hotpotqa(FIXTURE_PATH)[0]
    prompts = []
    responses = iter(
        (
            "SEARCH[Ursula]",
            "ANSWER[United States] CITATIONS[Ursula K. Le Guin|99]",
            "ANSWER[United States] CITATIONS[Ursula K. Le Guin|1]",
        )
    )

    class RetryingClient:
        def complete(self, messages):
            prompts.append(messages[0]["content"])
            return CompletionResult(
                next(responses), "id", "mock", None, CompletionUsage(1, 1, 2), "{}"
            )

    trajectory = EpisodeRunner(RetryingClient(), max_retries=1).run(  # type: ignore[arg-type]
        SearchEnvironment(example, top_k=1, max_searches=1)
    )

    assert trajectory.answer.citations == (EvidenceRef("Ursula K. Le Guin", 1),)
    assert "Valid citation titles from retrieved results" in prompts[2]
    assert "- Ursula K. Le Guin" in prompts[2]
    assert "Use CITATIONS[] if no valid title" in prompts[2]


def test_runner_canonicalizes_harmless_citation_title_variation() -> None:
    example = load_hotpotqa(FIXTURE_PATH)[0]
    environment = SearchEnvironment(example, top_k=1, max_searches=1)
    responses = iter(("SEARCH[Ursula]", "ANSWER[United States] CITATIONS[ ursula k. le guin | 1 ]"))

    class FakeClient:
        def complete(self, messages):
            return CompletionResult(
                next(responses), "id", "mock", None, CompletionUsage(1, 1, 2), "{}"
            )

    trajectory = EpisodeRunner(FakeClient(), max_turns=2).run(environment)  # type: ignore[arg-type]

    assert trajectory.answer.citations == (EvidenceRef("Ursula K. Le Guin", 1),)


def test_runner_reports_malformed_output_and_exhaustion() -> None:
    example = load_hotpotqa(FIXTURE_PATH)[0]

    class BadClient:
        def complete(self, messages):
            return CompletionResult(
                "not an action", "id", "mock", None, CompletionUsage(1, 1, 2), "{}"
            )

    with pytest.raises(EpisodeRunnerError, match="model action failed"):
        EpisodeRunner(BadClient()).run(SearchEnvironment(example))  # type: ignore[arg-type]

    class SearchOnlyClient:
        def complete(self, messages):
            return CompletionResult(
                "SEARCH[Ursula]", "id", "mock", None, CompletionUsage(1, 1, 2), "{}"
            )

    with pytest.raises(EpisodeRunnerError, match="exhausted"):
        EpisodeRunner(SearchOnlyClient(), max_turns=1).run(SearchEnvironment(example))  # type: ignore[arg-type]
