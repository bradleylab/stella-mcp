"""Run free-form model backends against the live Stella MCP stdio server."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .runner import (
    REPO_ROOT,
    _artifact_evidence,
    _capabilities,
    _content_text,
    _replace_tokens,
    evaluate_expectation,
    sanitize_text,
)

DEFAULT_AGENT_SCENARIOS = Path(__file__).with_name("agent_scenarios.json")
DEFAULT_SCENARIO_TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class AgentToolCall:
    """One function-tool call requested by a model backend."""

    call_id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class AgentTurn:
    """Provider-neutral assistant turn returned by a model backend."""

    content: str | None
    tool_calls: tuple[AgentToolCall, ...] = ()
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class AgentBackend(Protocol):
    """Minimal interface required by the free-form evaluation loop."""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model_request: dict[str, Any],
    ) -> AgentTurn: ...

    def metadata(self) -> dict[str, Any]: ...


def load_agent_scenarios(path: Path = DEFAULT_AGENT_SCENARIOS) -> dict[str, Any]:
    """Load and validate the fixed free-form evaluation protocol."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError(f"Unsupported agent scenario document: {path}")
    if not isinstance(document.get("system_prompt"), str) or not document["system_prompt"]:
        raise ValueError("Agent protocol requires a non-empty system_prompt")
    model_request = document.get("model_request")
    if not isinstance(model_request, dict):
        raise ValueError("Agent protocol requires a model_request object")
    temperature = model_request.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("model_request.temperature must be numeric")
    seed = model_request.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("model_request.seed must be an integer")
    max_completion_tokens = model_request.get("max_completion_tokens")
    if (
        isinstance(max_completion_tokens, bool)
        or not isinstance(max_completion_tokens, int)
        or max_completion_tokens < 1
    ):
        raise ValueError("model_request.max_completion_tokens must be a positive integer")
    max_rounds = document.get("max_tool_rounds")
    if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
        raise ValueError("max_tool_rounds must be a positive integer")
    scenarios = document.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Agent protocol requires a non-empty scenarios array")

    ids = [scenario.get("id") for scenario in scenarios if isinstance(scenario, dict)]
    if len(ids) != len(scenarios) or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("Every agent scenario requires a non-empty string id")
    if len(ids) != len(set(ids)):
        raise ValueError("Agent scenario ids must be unique")

    artifact_owners: dict[str, str] = {}
    for scenario in scenarios:
        scenario_id = scenario["id"]
        if not isinstance(scenario.get("description"), str) or not scenario["description"]:
            raise ValueError(f"Agent scenario {scenario_id} requires a description")
        if not isinstance(scenario.get("prompt"), str) or not scenario["prompt"]:
            raise ValueError(f"Agent scenario {scenario_id} requires a prompt")
        requires = scenario.get("requires", [])
        if not isinstance(requires, list) or any(
            not isinstance(name, str) or not name for name in requires
        ):
            raise ValueError(f"Agent scenario {scenario_id} has invalid requires")
        required_order = scenario.get("required_tool_order")
        if (
            not isinstance(required_order, list)
            or not required_order
            or any(not isinstance(name, str) or not name for name in required_order)
        ):
            raise ValueError(f"Agent scenario {scenario_id} has invalid required_tool_order")
        checks = scenario.get("checks")
        if not isinstance(checks, list) or not checks:
            raise ValueError(f"Agent scenario {scenario_id} requires post-run checks")
        for check in checks:
            if not isinstance(check, dict):
                raise ValueError(f"Agent scenario {scenario_id} has an invalid check")
            if not isinstance(check.get("tool"), str) or not check["tool"]:
                raise ValueError(f"Agent scenario {scenario_id} has a check without a tool")
            if not isinstance(check.get("arguments", {}), dict):
                raise ValueError(f"Agent scenario {scenario_id} has invalid check arguments")
            if not isinstance(check.get("expect"), dict):
                raise ValueError(f"Agent scenario {scenario_id} has invalid check expectations")
        artifacts = scenario.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError(f"Agent scenario {scenario_id} requires expected artifacts")
        for artifact in artifacts:
            if not isinstance(artifact, str) or not artifact:
                raise ValueError(f"Agent scenario {scenario_id} has an invalid artifact path")
            artifact_path = Path(artifact)
            if artifact_path.is_absolute() or ".." in artifact_path.parts:
                raise ValueError(f"Agent scenario {scenario_id} has unsafe artifact path")
            owner = artifact_owners.setdefault(artifact, scenario_id)
            if owner != scenario_id:
                raise ValueError(
                    f"Agent artifact {artifact!r} is shared by scenarios {owner!r} and "
                    f"{scenario_id!r}"
                )
    return document


def _select_scenarios(
    document: dict[str, Any], selected_ids: set[str] | None
) -> list[dict[str, Any]]:
    scenarios = document["scenarios"]
    if not selected_ids:
        return scenarios
    known = {scenario["id"] for scenario in scenarios}
    unknown = selected_ids - known
    if unknown:
        raise ValueError(f"Unknown agent scenario ids: {', '.join(sorted(unknown))}")
    return [scenario for scenario in scenarios if scenario["id"] in selected_ids]


def preflight_agent_artifacts(
    scenario_path: Path,
    artifact_dir: Path,
    selected_ids: set[str] | None = None,
) -> None:
    """Reject expected output files that could satisfy checks from a prior run."""
    document = load_agent_scenarios(scenario_path)
    scenarios = _select_scenarios(document, selected_ids)
    available = _capabilities()
    runnable_scenarios = [
        scenario for scenario in scenarios if not set(scenario.get("requires", [])) - available
    ]
    expected_paths = [
        artifact_dir / name
        for scenario in runnable_scenarios
        for name in scenario.get("artifacts", [])
    ]
    preexisting = sorted(str(path) for path in expected_paths if path.exists())
    if preexisting:
        raise FileExistsError(
            "Expected agent-evaluation artifacts already exist: " + ", ".join(preexisting)
        )


def evaluate_tool_order(actual: list[str], required: list[str]) -> list[str]:
    """Return failures when ``required`` is not an ordered subsequence of ``actual``."""
    next_index = 0
    for name in actual:
        if next_index < len(required) and name == required[next_index]:
            next_index += 1
    if next_index == len(required):
        return []
    return [
        "required successful tool order not observed; "
        f"missing from {required[next_index:]!r} after actual sequence {actual!r}"
    ]


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _sum_usage(total: dict[str, int], usage: dict[str, int]) -> None:
    for name, value in usage.items():
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        total[name] = total.get(name, 0) + value


def _catalog_tools(tools: Any) -> list[dict[str, Any]]:
    return [tool.model_dump(mode="json") for tool in tools.tools]


def _tool_result_payload(result: Any) -> dict[str, Any]:
    return {
        "is_error": bool(result.isError),
        "content": [item.model_dump(mode="json") for item in result.content],
        "structured_content": result.structuredContent,
    }


def _assistant_message(turn: AgentTurn) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": turn.content}
    if turn.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.call_id,
                "name": call.name,
                "arguments": call.arguments_json,
            }
            for call in turn.tool_calls
        ]
    return message


async def _execute_tool_call(
    session: ClientSession,
    call: AgentToolCall,
    round_number: int,
    call_number: int,
    redactions: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    try:
        arguments = json.loads(call.arguments_json)
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must decode to an object")
    except (json.JSONDecodeError, ValueError) as exc:
        message = str(exc)
        payload = {
            "is_error": True,
            "content": [],
            "structured_content": {
                "error": {
                    "code": "invalid_tool_arguments",
                    "category": "model_output",
                    "message": message,
                }
            },
        }
        event = {
            "round": round_number,
            "index": call_number,
            "call_id": call.call_id,
            "tool": call.name,
            "arguments_json": sanitize_text(call.arguments_json, redactions),
            "called_mcp": False,
            "is_error": True,
            "error_code": "invalid_tool_arguments",
            "structured_keys": ["error"],
            "text": message,
        }
        return event, payload, False

    result = await session.call_tool(call.name, arguments)
    structured = result.structuredContent or {}
    event = {
        "round": round_number,
        "index": call_number,
        "call_id": call.call_id,
        "tool": call.name,
        "arguments": _replace_tokens(arguments, redactions),
        "called_mcp": True,
        "is_error": bool(result.isError),
        "error_code": (structured.get("error") or {}).get("code"),
        "structured_keys": sorted(structured),
        "text": sanitize_text(_content_text(result), redactions),
    }
    return event, _tool_result_payload(result), not result.isError


async def _run_scenario(
    session: ClientSession,
    backend: AgentBackend,
    scenario: dict[str, Any],
    document: dict[str, Any],
    tools: list[dict[str, Any]],
    artifact_dir: Path,
    replacements: dict[str, str],
    redactions: dict[str, str],
) -> dict[str, Any]:
    started = time.perf_counter()
    prompt = _replace_tokens(scenario["prompt"], replacements)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": document["system_prompt"]},
        {"role": "user", "content": prompt},
    ]
    events: list[dict[str, Any]] = []
    successful_tools: list[str] = []
    usage: dict[str, int] = {}
    final_response: str | None = None
    stop_reason = "tool_round_cap"
    backend_failure: str | None = None

    for round_number in range(1, document["max_tool_rounds"] + 1):
        try:
            turn = await backend.complete(messages, tools, document["model_request"])
        except Exception as exc:
            backend_failure = sanitize_text(f"{type(exc).__name__}: {exc}", redactions)
            stop_reason = "backend_error"
            break

        _sum_usage(usage, turn.usage)
        messages.append(_assistant_message(turn))
        if not turn.tool_calls:
            final_response = turn.content or ""
            stop_reason = turn.stop_reason or "final_response"
            break

        for call_number, call in enumerate(turn.tool_calls):
            event, payload, successful = await _execute_tool_call(
                session,
                call,
                round_number,
                call_number,
                redactions,
            )
            events.append(event)
            if successful:
                successful_tools.append(call.name)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": json.dumps(payload, separators=(",", ":")),
                }
            )

    check_results = []
    check_failures: list[str] = []
    for index, check in enumerate(scenario.get("checks", [])):
        arguments = _replace_tokens(check.get("arguments", {}), replacements)
        result = await session.call_tool(check["tool"], arguments)
        failures = evaluate_expectation(result, check.get("expect", {}))
        structured = result.structuredContent or {}
        check_results.append(
            {
                "index": index,
                "tool": check["tool"],
                "status": "passed" if not failures else "failed",
                "is_error": bool(result.isError),
                "error_code": (structured.get("error") or {}).get("code"),
                "structured_keys": sorted(structured),
                "text": sanitize_text(_content_text(result), redactions),
                "failures": failures,
            }
        )
        check_failures.extend(f"check {index}: {failure}" for failure in failures)

    artifacts = _artifact_evidence(artifact_dir, scenario.get("artifacts", []))
    missing_artifacts = [item["path"] for item in artifacts if not item["exists"]]
    order_failures = evaluate_tool_order(successful_tools, scenario.get("required_tool_order", []))
    failures = [*order_failures, *check_failures]
    if final_response is None:
        failures.append(f"no final response: {stop_reason}")
    if backend_failure:
        failures.append(backend_failure)
    failures.extend(f"missing artifact: {path}" for path in missing_artifacts)

    return {
        "id": scenario["id"],
        "description": scenario["description"],
        "status": "passed" if not failures else "failed",
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "prompt": sanitize_text(prompt, redactions),
        "stop_reason": stop_reason,
        "final_response": (
            sanitize_text(final_response, redactions) if final_response is not None else None
        ),
        "usage": usage,
        "required_tool_order": scenario.get("required_tool_order", []),
        "successful_tool_order": successful_tools,
        "tool_calls": events,
        "checks": check_results,
        "artifacts": artifacts,
        "missing_artifacts": missing_artifacts,
        "failures": failures,
    }


async def run_agent_evaluation(
    backend: AgentBackend,
    scenario_path: Path,
    artifact_dir: Path,
    selected_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Run selected free-form scenarios through fresh real MCP sessions."""
    document = load_agent_scenarios(scenario_path)
    scenarios = _select_scenarios(document, selected_ids)

    preflight_agent_artifacts(scenario_path, artifact_dir, selected_ids)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    replacements = {
        "${ARTIFACT_DIR}": str(artifact_dir.resolve()),
        "${REPO_ROOT}": str(REPO_ROOT),
    }
    redactions = {value: token for token, value in replacements.items()}
    available = _capabilities()
    results: list[dict[str, Any]] = []
    catalog_hashes: set[str] = set()
    tool_count: int | None = None
    server_name: str | None = None
    started = time.perf_counter()

    for scenario in scenarios:
        missing = sorted(set(scenario.get("requires", [])) - available)
        if missing:
            results.append(
                {
                    "id": scenario["id"],
                    "description": scenario["description"],
                    "status": "skipped",
                    "missing_capabilities": missing,
                    "tool_calls": [],
                    "checks": [],
                    "artifacts": [],
                    "failures": [],
                }
            )
            continue

        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "stella_mcp.server"],
            cwd=REPO_ROOT,
        )
        scenario_started = time.perf_counter()
        try:
            with anyio.fail_after(DEFAULT_SCENARIO_TIMEOUT_SECONDS):
                async with stdio_client(server) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        initialized = await session.initialize()
                        listed_tools = await session.list_tools()
                        catalog = _catalog_tools(listed_tools)
                        catalog_json = json.dumps(
                            catalog,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        catalog_hashes.add(hashlib.sha256(catalog_json).hexdigest())
                        tool_count = len(catalog)
                        server_name = initialized.serverInfo.name
                        results.append(
                            await _run_scenario(
                                session,
                                backend,
                                scenario,
                                document,
                                catalog,
                                artifact_dir,
                                replacements,
                                redactions,
                            )
                        )
        except TimeoutError:
            artifacts = _artifact_evidence(artifact_dir, scenario.get("artifacts", []))
            missing_artifacts = [item["path"] for item in artifacts if not item["exists"]]
            results.append(
                {
                    "id": scenario["id"],
                    "description": scenario["description"],
                    "status": "failed",
                    "duration_ms": round((time.perf_counter() - scenario_started) * 1000, 3),
                    "prompt": scenario["prompt"],
                    "stop_reason": "scenario_timeout",
                    "final_response": None,
                    "usage": {},
                    "required_tool_order": scenario.get("required_tool_order", []),
                    "successful_tool_order": [],
                    "tool_calls": [],
                    "checks": [],
                    "artifacts": artifacts,
                    "missing_artifacts": missing_artifacts,
                    "failures": [f"scenario exceeded {DEFAULT_SCENARIO_TIMEOUT_SECONDS} seconds"],
                }
            )

    counts = {
        status: sum(item["status"] == status for item in results)
        for status in ("passed", "failed", "skipped")
    }
    total_usage: dict[str, int] = {}
    for result in results:
        _sum_usage(total_usage, result.get("usage", {}))
    scenario_bytes = scenario_path.read_bytes()
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "stella_mcp": importlib.metadata.version("stella-mcp"),
            "mcp": importlib.metadata.version("mcp"),
            "capabilities": sorted(available),
        },
        "backend": backend.metadata(),
        "protocol": {
            "scenario_path": _portable_path(scenario_path),
            "scenario_sha256": hashlib.sha256(scenario_bytes).hexdigest(),
            "requested_model_request": document["model_request"],
            "max_tool_rounds": document["max_tool_rounds"],
            "scenario_timeout_seconds": DEFAULT_SCENARIO_TIMEOUT_SECONDS,
            "server_name": server_name,
            "tool_count": tool_count,
            "tool_catalog_sha256": (
                next(iter(catalog_hashes)) if len(catalog_hashes) == 1 else sorted(catalog_hashes)
            ),
        },
        "summary": {
            "scenarios": len(results),
            **counts,
            "tool_calls": sum(len(item.get("tool_calls", [])) for item in results),
            "tool_errors": sum(
                call["is_error"] for item in results for call in item.get("tool_calls", [])
            ),
            "usage": total_usage,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        },
        "scenarios": results,
    }
