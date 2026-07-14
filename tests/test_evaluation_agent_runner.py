"""Tests for the provider-neutral free-form agent evaluation loop."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from evaluation.agent_reporting import render_agent_markdown
from evaluation.agent_runner import (
    AgentToolCall,
    AgentTurn,
    evaluate_tool_order,
    load_agent_scenarios,
    run_agent_evaluation,
)

ROOT = Path(__file__).resolve().parents[1]


class ScriptedBackend:
    def __init__(self, turns: list[AgentTurn]) -> None:
        self._turns = iter(turns)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model_request: dict[str, Any],
    ) -> AgentTurn:
        assert messages
        assert any(tool["name"] == "build_model" for tool in tools)
        assert model_request == {
            "temperature": 0,
            "seed": 20260713,
            "max_completion_tokens": 4096,
        }
        return next(self._turns)

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": "scripted",
            "model": "test",
            "effective_sampling": {"temperature": 0, "seed": 20260713},
        }


class FixedProtocolBackend:
    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model_request: dict[str, Any],
    ) -> AgentTurn:
        assert tools
        assert model_request == {
            "temperature": 0,
            "seed": 20260713,
            "max_completion_tokens": 4096,
        }
        if any(message["role"] == "assistant" for message in messages):
            return AgentTurn(content="Completed the requested Stella workflow.", stop_reason="stop")

        prompt = messages[1]["content"]
        if "Agent Evaluation Growth" in prompt:
            calls = [
                _call(
                    "build",
                    "build_model",
                    {
                        "name": "Agent Evaluation Growth",
                        "model_id": "agent_growth",
                        "sim_specs": {
                            "start": 0,
                            "stop": 5,
                            "dt": 1,
                            "method": "Euler",
                            "time_units": "Years",
                        },
                        "stocks": [
                            {"name": "Population", "initial_value": "100", "units": "people"}
                        ],
                        "flows": [
                            {
                                "name": "growth",
                                "equation": "Population * growth_rate",
                                "to_stock": "Population",
                                "units": "people/Year",
                            }
                        ],
                        "auxs": [{"name": "growth_rate", "equation": "0.1", "units": "1/Year"}],
                    },
                ),
                _call("validate", "validate_model", {"model_id": "agent_growth"}),
                _call("simulate", "simulate", {"model_id": "agent_growth"}),
                _call("render", "render_diagram", {"model_id": "agent_growth"}),
                _call(
                    "save",
                    "save_model",
                    {
                        "model_id": "agent_growth",
                        "filepath": str(self.artifact_dir / "agent_growth.stmx"),
                        "compat_mode": "strict",
                    },
                ),
            ]
        elif "built-in SIR template" in prompt:
            calls = [
                _call("load", "load_template", {"template_name": "sir", "model_id": "agent_sir"}),
                _call(
                    "update",
                    "update_aux",
                    {"model_id": "agent_sir", "name": "beta", "equation": "0.2"},
                ),
                _call("validate", "validate_model", {"model_id": "agent_sir"}),
                _call("simulate", "simulate", {"model_id": "agent_sir"}),
                _call("render", "render_diagram", {"model_id": "agent_sir"}),
                _call(
                    "save",
                    "save_model",
                    {
                        "model_id": "agent_sir",
                        "filepath": str(self.artifact_dir / "agent_sir.stmx"),
                        "compat_mode": "strict",
                    },
                ),
            ]
        else:
            calls = [
                _call(
                    "read",
                    "read_model",
                    {
                        "filepath": str(
                            ROOT / "tests/fixtures/compat_corpus/stella_4_1_1_accumulator.stmx"
                        ),
                        "model_id": "agent_analysis",
                        "compat_mode": "strict",
                    },
                ),
                _call("inspect", "inspect_model", {"model_id": "agent_analysis"}),
                _call("simulate", "simulate", {"model_id": "agent_analysis"}),
                _call(
                    "compare",
                    "compare_scenarios",
                    {
                        "model_id": "agent_analysis",
                        "scenarios": [{"name": "double rate", "overrides": {"rate": 2}}],
                        "save_comparison_csv": str(
                            self.artifact_dir / "agent_accumulator_scenarios.csv"
                        ),
                    },
                ),
                _call(
                    "sensitivity",
                    "sensitivity_analysis",
                    {
                        "model_id": "agent_analysis",
                        "parameters": [{"name": "rate", "start": 1, "stop": 3, "steps": 3}],
                        "output": {"variable": "Accumulator", "metric": "final"},
                        "save_sweep_csv": str(
                            self.artifact_dir / "agent_accumulator_sensitivity.csv"
                        ),
                    },
                ),
                _call(
                    "save",
                    "save_model",
                    {
                        "model_id": "agent_analysis",
                        "filepath": str(self.artifact_dir / "agent_accumulator.stmx"),
                        "compat_mode": "strict",
                    },
                ),
            ]
        return AgentTurn(content=None, tool_calls=tuple(calls), stop_reason="tool_calls")

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": "scripted",
            "api": "test",
            "model": "fixed-protocol-reference",
            "resolved_model": "fixed-protocol-reference",
            "endpoint": "offline",
            "effective_model_request": {
                "temperature": 0,
                "seed": 20260713,
                "max_completion_tokens": 4096,
            },
        }


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> AgentToolCall:
    return AgentToolCall(call_id, name, json.dumps(arguments))


def _write_protocol(path: Path, *, max_tool_rounds: int = 4) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "system_prompt": "Complete the task with Stella MCP tools.",
                "model_request": {
                    "temperature": 0,
                    "seed": 20260713,
                    "max_completion_tokens": 4096,
                },
                "max_tool_rounds": max_tool_rounds,
                "scenarios": [
                    {
                        "id": "scripted_build",
                        "description": "Build and save a stock model.",
                        "requires": [],
                        "prompt": "Build scripted_model and save ${ARTIFACT_DIR}/model.stmx.",
                        "required_tool_order": ["build_model", "save_model"],
                        "checks": [
                            {
                                "tool": "inspect_model",
                                "arguments": {
                                    "model_id": "scripted_model",
                                    "include_validation": True,
                                },
                                "expect": {
                                    "is_error": False,
                                    "fields": {
                                        "model.name": "Scripted",
                                        "model.counts.stocks": 1,
                                        "validation.passed": True,
                                    },
                                },
                            }
                        ],
                        "artifacts": ["model.stmx"],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_agent_scenarios_and_order_contract() -> None:
    document = load_agent_scenarios()

    assert [scenario["id"] for scenario in document["scenarios"]] == [
        "construct_growth",
        "modify_sir_template",
        "analyze_accumulator",
    ]
    assert (
        evaluate_tool_order(
            ["build_model", "inspect_model", "validate_model", "save_model"],
            ["build_model", "validate_model", "save_model"],
        )
        == []
    )
    assert evaluate_tool_order(["save_model", "build_model"], ["build_model", "save_model"])


def test_fixed_agent_protocol_is_executable_with_scripted_backend(tmp_path: Path) -> None:
    pytest.importorskip("pysd")
    artifacts = tmp_path / "artifacts"

    result = asyncio.run(
        run_agent_evaluation(
            FixedProtocolBackend(artifacts),
            ROOT / "evaluation/agent_scenarios.json",
            artifacts,
        )
    )

    assert result["summary"]["scenarios"] == 3
    assert result["summary"]["passed"] == 3
    assert result["summary"]["failed"] == 0
    assert result["summary"]["skipped"] == 0
    assert result["summary"]["tool_calls"] == 17
    assert all(scenario["checks"][0]["status"] == "passed" for scenario in result["scenarios"])
    assert str(tmp_path) not in json.dumps(result)
    markdown = render_agent_markdown(result)
    assert "# Stella MCP Free-Form Agent Evaluation" in markdown
    assert "| `construct_growth` | passed | stop | 5 | 0 |" in markdown
    assert "`agent_growth.stmx`" in markdown
    assert str(tmp_path) not in markdown


def test_agent_runner_recovers_from_malformed_call_and_scores_outcome(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    artifacts = tmp_path / "artifacts"
    _write_protocol(protocol)
    backend = ScriptedBackend(
        [
            AgentTurn(
                content=None,
                tool_calls=(AgentToolCall("bad", "build_model", "{"),),
                stop_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            ),
            AgentTurn(
                content=None,
                tool_calls=(
                    AgentToolCall(
                        "build",
                        "build_model",
                        json.dumps(
                            {
                                "name": "Scripted",
                                "model_id": "scripted_model",
                                "stocks": [{"name": "Stock", "initial_value": "1"}],
                            }
                        ),
                    ),
                    AgentToolCall(
                        "save",
                        "save_model",
                        json.dumps(
                            {
                                "model_id": "scripted_model",
                                "filepath": str(artifacts / "model.stmx"),
                                "compat_mode": "strict",
                            }
                        ),
                    ),
                ),
                stop_reason="tool_calls",
                usage={"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
            ),
            AgentTurn(
                content="Built and saved the model.",
                stop_reason="stop",
                usage={"prompt_tokens": 30, "completion_tokens": 6, "total_tokens": 36},
            ),
        ]
    )

    result = asyncio.run(run_agent_evaluation(backend, protocol, artifacts))

    assert result["summary"]["passed"] == 1
    assert result["summary"]["tool_calls"] == 3
    assert result["summary"]["tool_errors"] == 1
    assert result["summary"]["usage"] == {
        "prompt_tokens": 60,
        "completion_tokens": 12,
        "total_tokens": 72,
    }
    scenario = result["scenarios"][0]
    assert scenario["status"] == "passed"
    assert scenario["tool_calls"][0]["error_code"] == "invalid_tool_arguments"
    assert scenario["successful_tool_order"] == ["build_model", "save_model"]
    assert scenario["checks"][0]["status"] == "passed"
    assert scenario["artifacts"][0]["exists"] is True
    assert "${ARTIFACT_DIR}" in scenario["prompt"]


def test_agent_runner_reports_tool_round_cap(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    artifacts = tmp_path / "artifacts"
    _write_protocol(protocol, max_tool_rounds=1)
    backend = ScriptedBackend(
        [
            AgentTurn(
                content=None,
                tool_calls=(AgentToolCall("list", "list_models", "{}"),),
                stop_reason="tool_calls",
            )
        ]
    )

    result = asyncio.run(run_agent_evaluation(backend, protocol, artifacts))

    scenario = result["scenarios"][0]
    assert scenario["status"] == "failed"
    assert scenario["stop_reason"] == "tool_round_cap"
    assert "no final response: tool_round_cap" in scenario["failures"]


def test_agent_runner_rejects_stale_expected_artifact(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "model.stmx").write_text("stale", encoding="utf-8")
    _write_protocol(protocol)

    with pytest.raises(FileExistsError, match="model.stmx"):
        asyncio.run(run_agent_evaluation(ScriptedBackend([]), protocol, artifacts))


def test_agent_scenario_loader_rejects_unsafe_artifact_path(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    _write_protocol(protocol)
    document = json.loads(protocol.read_text(encoding="utf-8"))
    document["scenarios"][0]["artifacts"] = ["../outside.stmx"]
    protocol.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe artifact path"):
        load_agent_scenarios(protocol)
