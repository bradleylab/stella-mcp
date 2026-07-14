"""OpenAI-compatible Chat Completions backend for agent evaluation."""

from __future__ import annotations

import importlib.metadata
import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .agent_runner import AgentToolCall, AgentTurn

OPENAI_API_BASE = "https://api.openai.com/v1"
REQUEST_TIMEOUT_SECONDS = 120.0
SAMPLING_MODES = {
    "both": {"temperature", "seed"},
    "temperature": {"temperature"},
    "seed": {"seed"},
    "none": set(),
}


class OpenAIChatBackend:
    """Adapt an OpenAI-compatible async client to ``AgentBackend``."""

    def __init__(
        self,
        client: Any,
        *,
        provider: str,
        model: str,
        endpoint: str,
        sampling_mode: str,
    ) -> None:
        if sampling_mode not in SAMPLING_MODES:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")
        self._client = client
        self._provider = provider
        self._model = model
        self._endpoint = endpoint.rstrip("/")
        self._sampling_controls = SAMPLING_MODES[sampling_mode]
        self._effective_model_request: dict[str, Any] | None = None
        self._resolved_model: str | None = None

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model_request: dict[str, Any],
    ) -> AgentTurn:
        request_controls = {
            name: model_request[name] for name in self._sampling_controls if name in model_request
        }
        request_controls["max_completion_tokens"] = model_request["max_completion_tokens"]
        chat_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description") or "",
                    "parameters": tool["inputSchema"],
                    "strict": False,
                },
            }
            for tool in tools
        ]
        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=_chat_messages(messages),
            tools=chat_tools,
            tool_choice="auto",
            parallel_tool_calls=False,
            **request_controls,
        )
        if not completion.choices:
            raise RuntimeError("Chat Completions response contained no choices")

        self._effective_model_request = request_controls
        self._resolved_model = getattr(completion, "model", None)
        choice = completion.choices[0]
        message = choice.message
        calls = tuple(
            AgentToolCall(
                call_id=call.id,
                name=call.function.name,
                arguments_json=call.function.arguments,
            )
            for call in (message.tool_calls or [])
        )
        return AgentTurn(
            content=message.content,
            tool_calls=calls,
            stop_reason=choice.finish_reason,
            usage=_usage_dict(getattr(completion, "usage", None)),
        )

    def metadata(self) -> dict[str, Any]:
        try:
            sdk_version = importlib.metadata.version("openai")
        except importlib.metadata.PackageNotFoundError:
            sdk_version = None
        return {
            "provider": self._provider,
            "api": "chat.completions",
            "model": self._model,
            "resolved_model": self._resolved_model,
            "endpoint": self._endpoint,
            "openai_sdk": sdk_version,
            "parallel_tool_calls": False,
            "tool_schema_strict": False,
            "effective_model_request": self._effective_model_request,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "automatic_retries": 0,
        }


def _chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for message in messages:
        converted_message = {
            "role": message["role"],
            "content": message.get("content"),
        }
        if message["role"] == "assistant" and message.get("tool_calls"):
            converted_message["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
                for call in message["tool_calls"]
            ]
        if message["role"] == "tool":
            converted_message["tool_call_id"] = message["tool_call_id"]
        converted.append(converted_message)
    return converted


def _usage_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    values = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            values[name] = value
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    for name, details in (
        ("cached_tokens", prompt_details),
        ("reasoning_tokens", completion_details),
    ):
        value = getattr(details, name, None) if details is not None else None
        if isinstance(value, int) and not isinstance(value, bool):
            values[name] = value
    return values


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable is not set: {name}")
    return value


def _validated_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("LLM endpoint must be an absolute HTTPS URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("LLM endpoint must not contain credentials, a query, or a fragment")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _washu_access_token() -> str:
    import httpx

    tenant_id = _required_environment("WUSTL_TENANT_ID")
    response = httpx.post(
        f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
        data={
            "client_id": _required_environment("WUSTL_CLIENT_ID"),
            "client_secret": _required_environment("WUSTL_CLIENT_SECRET"),
            "scope": _required_environment("WUSTL_API_SCOPE"),
            "grant_type": "client_credentials",
        },
        timeout=30.0,
    )
    response.raise_for_status()
    token = response.json().get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("WashU OAuth response did not contain an access token")
    return token


def _async_openai_class() -> Any:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Agent evaluation requires the optional agent-eval dependencies"
        ) from exc
    return AsyncOpenAI


def build_openai_chat_backend(
    *, provider: str, model: str, sampling_mode: str
) -> OpenAIChatBackend:
    """Build a credential-safe backend from the selected environment route."""
    async_openai = _async_openai_class()

    if provider == "openai":
        endpoint = _validated_endpoint(OPENAI_API_BASE)
        api_key = _required_environment("OPENAI_API_KEY")
    elif provider == "washu":
        endpoint = _validated_endpoint(_required_environment("OPENAI_BASE_URL"))
        api_key = _washu_access_token()
    else:
        raise ValueError(f"Unknown provider: {provider}")

    client = async_openai(
        api_key=api_key,
        base_url=endpoint,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )
    return OpenAIChatBackend(
        client,
        provider=provider,
        model=model,
        endpoint=endpoint,
        sampling_mode=sampling_mode,
    )
