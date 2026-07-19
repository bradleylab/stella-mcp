"""Markdown rendering for free-form agent evaluation reports."""

from __future__ import annotations

from typing import Any


def render_agent_markdown(report: dict[str, Any]) -> str:
    """Render the machine-readable agent report as concise Markdown."""
    summary = report["summary"]
    backend = report["backend"]
    lines = [
        "# Stella MCP Free-Form Agent Evaluation",
        "",
        "## Run",
        "",
        f"- Provider: `{backend.get('provider')}`",
        f"- API: `{backend.get('api')}`",
        f"- Requested model: `{backend.get('model')}`",
        f"- Resolved model: `{backend.get('resolved_model')}`",
        f"- Endpoint: `{backend.get('endpoint')}`",
        f"- Effective model request: `{backend.get('effective_model_request')}`",
        "",
        "## Summary",
        "",
        f"- Protocol scenarios: {summary['protocol_scenarios']}",
        f"- Repetitions per scenario: {report['protocol']['runs_per_scenario']}",
        f"- Scenario runs: {summary['scenario_runs']}",
        f"- Passed runs: {summary['passed']}",
        f"- Failed runs: {summary['failed']}",
        f"- Skipped runs: {summary['skipped']}",
        f"- MCP tool calls: {summary['tool_calls']}",
        f"- Tool errors: {summary['tool_errors']}",
        f"- Usage: `{summary['usage']}`",
        "",
        "These are raw repeated outcomes, not an estimated general success rate.",
        "",
        "## Dimension Counts",
        "",
        "| Dimension | Passed | Recovered | Failed | Skipped |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, counts in summary["dimensions"].items():
        lines.append(
            f"| {name} | {counts['passed']} | {counts['recovered']} | "
            f"{counts['failed']} | {counts['skipped']} |"
        )
    lines.extend(
        [
            "",
            "## Scenario Runs",
            "",
            "| Scenario | Run | Overall | Workflow | Semantic | Artifacts | Completion | "
            "Tool health | Calls | Errors |",
            "|---|---:|---|---|---|---|---|---|---:|---:|",
        ]
    )
    for scenario in report["scenarios"]:
        calls = scenario.get("tool_calls", [])
        errors = sum(call.get("is_error", False) for call in calls)
        dimensions = scenario.get("dimensions", {})
        lines.append(
            f"| `{scenario['id']}` | {scenario.get('run_index', '')} | {scenario['status']} | "
            f"{dimensions.get('workflow', {}).get('status', '')} | "
            f"{dimensions.get('semantic', {}).get('status', '')} | "
            f"{dimensions.get('artifacts', {}).get('status', '')} | "
            f"{dimensions.get('completion', {}).get('status', '')} | "
            f"{dimensions.get('tool_health', {}).get('status', '')} | "
            f"{len(calls)} | {errors} |"
        )

    for scenario in report["scenarios"]:
        lines.extend(
            ["", f"### {scenario['id']} / run {scenario.get('run_index', '')}", ""]
        )
        lines.append(
            "Successful tool order: "
            + (
                " -> ".join(f"`{name}`" for name in scenario.get("successful_tool_order", []))
                or "none"
            )
        )
        if scenario.get("failures"):
            lines.extend(["", "Failures:"])
            lines.extend(f"- {failure}" for failure in scenario["failures"])
        if scenario.get("artifacts"):
            lines.extend(["", "Artifacts:"])
            prefix = scenario.get("artifact_subdirectory", "")
            for artifact in scenario["artifacts"]:
                artifact_path = "/".join(
                    part for part in (prefix, artifact["path"]) if part
                )
                if artifact["exists"]:
                    lines.append(
                        f"- `{artifact_path}`: {artifact['bytes']} bytes, "
                        f"SHA-256 `{artifact['sha256']}`"
                    )
                else:
                    lines.append(f"- `{artifact_path}`: missing")
        if scenario.get("final_response") is not None:
            lines.extend(["", "Final response:", ""])
            lines.extend(
                f"> {line}" if line else ">" for line in scenario["final_response"].splitlines()
            )
    return "\n".join(lines)
