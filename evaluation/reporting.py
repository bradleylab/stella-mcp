"""Human-readable reporting for Stella MCP evaluation results."""

from __future__ import annotations

from typing import Any


def render_markdown(result: dict[str, Any]) -> str:
    """Render a concise Markdown report from evaluation JSON."""
    summary = result["summary"]
    protocol = result["protocol"]
    environment = result["environment"]
    lines = [
        "# Stella MCP Evaluation Report",
        "",
        f"Generated: {result['generated_at']}",
        "",
        "## Environment",
        "",
        f"- stella-mcp: {environment['stella_mcp']}",
        f"- mcp: {environment['mcp']}",
        f"- Python: {environment['python']}",
        f"- Platform: {environment['platform']}",
        f"- Capabilities: {', '.join(environment['capabilities']) or 'core only'}",
        "",
        "## Protocol",
        "",
        f"- Server: {protocol['server_name']}",
        f"- Tools: {protocol['tool_count']}",
        f"- Tool catalog SHA-256: `{protocol['tool_catalog_sha256']}`",
        f"- Initial resources: {protocol['resource_count_at_start']}",
        f"- Prompts: {protocol['prompt_count']}",
        "",
        "## Summary",
        "",
        f"- Scenarios: {summary['scenarios']}",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        f"- Skipped: {summary['skipped']}",
        f"- Tool calls: {summary['tool_calls']}",
        f"- Duration: {summary['duration_ms']} ms",
        "",
        "## Scenarios",
        "",
        "| Scenario | Status | Steps | Duration (ms) |",
        "|---|---:|---:|---:|",
    ]
    for scenario in result["scenarios"]:
        duration = scenario.get("duration_ms", "-")
        lines.append(
            f"| `{scenario['id']}` | {scenario['status']} | "
            f"{len(scenario['steps'])} | {duration} |"
        )
    lines.append("")

    failures = [scenario for scenario in result["scenarios"] if scenario["status"] == "failed"]
    if failures:
        lines.extend(["## Failures", ""])
        for scenario in failures:
            lines.append(f"### {scenario['id']}")
            for step in scenario["steps"]:
                for failure in step["failures"]:
                    lines.append(f"- `{step['tool']}`: {failure}")
            for path in scenario.get("missing_artifacts", []):
                lines.append(f"- Missing artifact: `{path}`")
            lines.append("")
    return "\n".join(lines)
