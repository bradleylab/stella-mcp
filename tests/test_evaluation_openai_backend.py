"""Tests for the OpenAI-compatible agent backend without network access."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from evaluation.agent_runner import AgentToolCall, AgentTurn
from evaluation.openai_chat_backend import (
    OPENAI_API_BASE,
    OpenAIChatBackend,
    _validated_endpoint,
    build_openai_chat_backend,
)


class FakeCompletions:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.request: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> Any:
        self.request = kwargs
        return self.response


def _client(response: Any) -> tuple[Any, FakeCompletions]:
    completions = FakeCompletions(response)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_openai_chat_backend_converts_tools_messages_and_usage() -> None:
    response = SimpleNamespace(
        model="resolved-model",
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            function=SimpleNamespace(
                                name="inspect_model", arguments='{"model_id":"m"}'
                            ),
                        )
                    ],
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=SimpleNamespace(cached_tokens=40),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
        ),
    )
    client, completions = _client(response)
    backend = OpenAIChatBackend(
        client,
        provider="openai",
        model="requested-model",
        endpoint="https://api.openai.com/v1",
        sampling_mode="temperature",
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "previous", "name": "list_models", "arguments": "{}"}],
        },
        {"role": "tool", "tool_call_id": "previous", "content": "{}"},
    ]
    tools = [
        {
            "name": "inspect_model",
            "description": "Inspect a model",
            "inputSchema": {"type": "object", "properties": {}},
        }
    ]

    turn = asyncio.run(
        backend.complete(
            messages,
            tools,
            {"temperature": 0, "seed": 20260713, "max_completion_tokens": 4096},
        )
    )

    assert turn == AgentTurn(
        content=None,
        tool_calls=(AgentToolCall("call-1", "inspect_model", '{"model_id":"m"}'),),
        stop_reason="tool_calls",
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cached_tokens": 40,
            "reasoning_tokens": 5,
        },
    )
    assert completions.request["temperature"] == 0
    assert "seed" not in completions.request
    assert completions.request["max_completion_tokens"] == 4096
    assert completions.request["parallel_tool_calls"] is False
    assert completions.request["tools"][0]["function"]["strict"] is False
    assert completions.request["messages"][2]["tool_calls"][0]["function"] == {
        "name": "list_models",
        "arguments": "{}",
    }
    assert completions.request["messages"][3]["tool_call_id"] == "previous"
    assert backend.metadata()["effective_model_request"] == {
        "temperature": 0,
        "max_completion_tokens": 4096,
    }
    assert backend.metadata()["resolved_model"] == "resolved-model"


def test_openai_chat_backend_rejects_unknown_sampling_mode() -> None:
    with pytest.raises(ValueError, match="sampling mode"):
        OpenAIChatBackend(
            object(),
            provider="openai",
            model="model",
            endpoint="https://api.openai.com/v1",
            sampling_mode="unsupported",
        )


def test_personal_backend_ignores_ambient_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_client(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("OPENAI_API_KEY", "test-personal-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://institution.invalid/v1")
    monkeypatch.setattr(
        "evaluation.openai_chat_backend._async_openai_class",
        lambda: fake_client,
    )

    backend = build_openai_chat_backend(
        provider="openai",
        model="test-model",
        sampling_mode="none",
    )

    assert captured["base_url"] == OPENAI_API_BASE
    assert captured["api_key"] == "test-personal-key"
    assert backend.metadata()["provider"] == "openai"


def test_washu_backend_uses_selected_institutional_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_client(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("OPENAI_BASE_URL", "https://institution.example/models/v1")
    monkeypatch.setattr(
        "evaluation.openai_chat_backend._async_openai_class",
        lambda: fake_client,
    )
    monkeypatch.setattr(
        "evaluation.openai_chat_backend._washu_access_token",
        lambda: "test-institution-token",
    )

    backend = build_openai_chat_backend(
        provider="washu",
        model="test-model",
        sampling_mode="both",
    )

    assert captured["base_url"] == "https://institution.example/models/v1"
    assert captured["api_key"] == "test-institution-token"
    assert backend.metadata()["provider"] == "washu"


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://institution.example/v1",
        "https://name:secret@institution.example/v1",
        "https://institution.example/v1?token=secret",
        "https://institution.example/v1#fragment",
    ],
)
def test_endpoint_validation_rejects_unsafe_urls(endpoint: str) -> None:
    with pytest.raises(ValueError, match="endpoint"):
        _validated_endpoint(endpoint)
