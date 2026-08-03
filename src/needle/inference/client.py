"""Dependency-free OpenRouter chat-completions client."""

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


class OpenRouterError(RuntimeError):
    """Base error for OpenRouter configuration, transport, or decoding."""


class OpenRouterHTTPError(OpenRouterError):
    """The endpoint returned a non-success HTTP status."""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"OpenRouter returned HTTP {status}: {body[:300]}")


class OpenRouterResponseError(OpenRouterError):
    """The endpoint response did not have the expected chat-completion shape."""


@dataclass(frozen=True, slots=True)
class OpenRouterConfig:
    """Immutable request settings; the API key is intentionally hidden in repr."""

    api_key: str = field(repr=False)
    model: str
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    seed: int | None = None
    timeout: float = 60.0
    http_referer: str | None = None
    title: str | None = None
    provider_order: tuple[str, ...] | None = None
    provider_only: tuple[str, ...] | None = None
    allow_fallbacks: bool = False
    require_parameters: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ValueError("api_key must be a non-empty string")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP(S) URL")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be greater than 0 and at most 1")
        if isinstance(self.max_tokens, bool) or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(self.max_tokens, int):
            raise TypeError("max_tokens must be an integer")
        if self.seed is not None and (isinstance(self.seed, bool) or self.seed < 0):
            raise ValueError("seed must be a non-negative integer")
        if self.seed is not None and not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if not isinstance(self.allow_fallbacks, bool) or not isinstance(
            self.require_parameters, bool
        ):
            raise TypeError("allow_fallbacks and require_parameters must be booleans")
        if self.provider_order is not None and self.provider_only is not None:
            raise ValueError("provider_order and provider_only cannot both be configured")
        for name in ("provider_order", "provider_only"):
            providers = getattr(self, name)
            if providers is not None:
                if isinstance(providers, str) or not providers:
                    raise ValueError(f"{name} must contain non-empty provider names")
                normalized = tuple(providers)
                if any(not isinstance(p, str) or not p.strip() for p in normalized):
                    raise ValueError(f"{name} must contain non-empty provider names")
                object.__setattr__(self, name, normalized)

    @classmethod
    def from_env(cls, model: str, **kwargs: Any) -> "OpenRouterConfig":
        """Build config from ``OPENROUTER_API_KEY`` without logging the secret."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        return cls(api_key=api_key, model=model, **kwargs)


@dataclass(frozen=True, slots=True)
class HTTPTransportRequest:
    """Fully materialized request passed to the injectable HTTP transport."""

    url: str
    headers: Mapping[str, str] = field(repr=False)
    body: bytes
    timeout: float


HTTPTransport = Callable[[HTTPTransportRequest], tuple[int, bytes]]


@dataclass(frozen=True, slots=True)
class CompletionUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True, slots=True)
class CompletionResult:
    text: str
    response_id: str
    model: str
    provider: str | None
    usage: CompletionUsage
    raw_json: str


def _urllib_transport(request: HTTPTransportRequest) -> tuple[int, bytes]:
    http_request = urllib.request.Request(
        request.url, data=request.body, headers=dict(request.headers), method="POST"
    )
    try:
        with urllib.request.urlopen(http_request, timeout=request.timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
    except urllib.error.URLError as error:
        raise OpenRouterError(f"OpenRouter transport failed: {error.reason}") from error


class OpenRouterClient:
    """Call OpenRouter's OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(self, config: OpenRouterConfig, transport: HTTPTransport | None = None) -> None:
        self.config = config
        self._transport = transport or _urllib_transport

    def complete(
        self, messages: Sequence[Mapping[str, str]], *, stop: Sequence[str] | None = None
    ) -> CompletionResult:
        if not messages:
            raise ValueError("messages must not be empty")
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [dict(message) for message in messages],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        if self.config.seed is not None:
            body["seed"] = self.config.seed
        if stop is not None:
            body["stop"] = list(stop)
        provider: dict[str, Any] = {"require_parameters": self.config.require_parameters}
        if self.config.provider_order is not None:
            provider["order"] = list(self.config.provider_order)
        if self.config.provider_only is not None:
            provider["only"] = list(self.config.provider_only)
        provider["allow_fallbacks"] = self.config.allow_fallbacks
        body["provider"] = provider
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if self.config.http_referer:
            headers["HTTP-Referer"] = self.config.http_referer
        if self.config.title:
            headers["X-OpenRouter-Title"] = self.config.title
        request = HTTPTransportRequest(
            url=self.config.base_url.rstrip("/") + "/chat/completions",
            headers=headers,
            body=json.dumps(body, separators=(",", ":")).encode(),
            timeout=self.config.timeout,
        )
        status, payload = self._transport(request)
        raw_json = payload.decode("utf-8", errors="replace")
        if not 200 <= status < 300:
            raise OpenRouterHTTPError(status, raw_json)
        try:
            data = json.loads(raw_json)
            if data.get("error") is not None:
                raise OpenRouterResponseError("OpenRouter response contains an error")
            choice = data["choices"][0]
            if choice.get("error") is not None:
                raise OpenRouterResponseError("OpenRouter choice contains an error")
            finish_reason = choice.get("finish_reason")
            if finish_reason not in (None, "stop"):
                raise OpenRouterResponseError(
                    f"completion ended before a complete response: {finish_reason}"
                )
            text = choice["message"]["content"]
            response_id = data["id"]
            model = data.get("model", self.config.model)
        except OpenRouterResponseError:
            raise
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise OpenRouterResponseError("malformed OpenRouter completion response") from error
        if not all(isinstance(value, str) for value in (text, response_id, model)):
            raise OpenRouterResponseError("completion response contains invalid text metadata")
        usage_data = data.get("usage") or {}
        usage = CompletionUsage(
            _token_value(usage_data, "prompt_tokens"),
            _token_value(usage_data, "completion_tokens"),
            _token_value(usage_data, "total_tokens"),
        )
        provider_name = data.get("provider") or data.get("provider_name") or choice.get("provider")
        if provider_name is not None and not isinstance(provider_name, str):
            raise OpenRouterResponseError("provider metadata must be a string")
        return CompletionResult(text, response_id, model, provider_name, usage, raw_json)


def _token_value(data: Mapping[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise OpenRouterResponseError(f"usage.{key} must be an integer")
    return value
