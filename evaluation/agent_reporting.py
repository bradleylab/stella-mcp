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
        f"- Provider: `{backend['provider']}`",
        f"- API: `{backend['api']}`",
        f"- Requested model: `{backend['model']}`",
        f"- Resolved model: `{backend.get('resolved_model')}`",
        f"- Endpoint: `{backend['endpoint']}`",
        f"- Effective model request: `{backend.get('effective_model_request')}`",
        "",
        "## Summary",
        "",
        f"- Scenarios: {summary['scenarios']}",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        f"- Skipped: {summary['skipped']}",
        f"- MCP tool calls: {summary['tool_calls']}",
        f"- Tool errors: {summary['tool_errors']}",
        f"- Usage: `{summary['usage']}`",
        "",
        "## Scenarios",
        "",
        "| Scenario | Status | Stop reason | Calls | Errors |",
        "|---|---|---|---:|---:|",
    ]
    for scenario in report["scenarios"]:
        calls = scenario.get("tool_calls", [])
        errors = sum(call.get("is_error", False) for call in calls)
        lines.append(
            f"| `{scenario['id']}` | {scenario['status']} | "
            f"{scenario.get('stop_reason', '')} | {len(calls)} | {errors} |"
        )

    for scenario in report["scenarios"]:
        lines.extend(["", f"### {scenario['id']}", ""])
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
            for artifact in scenario["artifacts"]:
                if artifact["exists"]:
                    lines.append(
                        f"- `{artifact['path']}`: {artifact['bytes']} bytes, "
                        f"SHA-256 `{artifact['sha256']}`"
                    )
                else:
                    lines.append(f"- `{artifact['path']}`: missing")
        if scenario.get("final_response") is not None:
            lines.extend(["", "Final response:", ""])
            lines.extend(f"> {line}" for line in scenario["final_response"].splitlines())
    return "\n".join(lines)
