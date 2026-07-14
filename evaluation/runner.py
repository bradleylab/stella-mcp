"""Run deterministic MCP workflows and write machine-readable evidence."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .reporting import render_markdown

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIOS = Path(__file__).with_name("scenarios.json")


def load_scenarios(path: Path = DEFAULT_SCENARIOS) -> dict[str, Any]:
    """Load and minimally validate the scenario document."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("scenarios"), list):
        raise ValueError(f"Unsupported evaluation scenario document: {path}")
    ids = [scenario.get("id") for scenario in data["scenarios"]]
    if any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("Every evaluation scenario requires a non-empty string id")
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation scenario ids must be unique")
    return data


def _replace_tokens(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for token, replacement in replacements.items():
            value = value.replace(token, replacement)
        return value
    if isinstance(value, list):
        return [_replace_tokens(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_tokens(item, replacements) for key, item in value.items()}
    return value


def _lookup(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(path)
    return current


def evaluate_expectation(result: Any, expectation: dict[str, Any]) -> list[str]:
    """Return assertion failures for one MCP result."""
    failures: list[str] = []
    actual_error = bool(result.isError)
    expected_error = bool(expectation.get("is_error", False))
    if actual_error != expected_error:
        failures.append(f"is_error expected {expected_error}, got {actual_error}")

    structured = result.structuredContent or {}
    for path, expected in expectation.get("fields", {}).items():
        try:
            actual = _lookup(structured, path)
        except (KeyError, IndexError, ValueError):
            failures.append(f"missing structured field {path}")
            continue
        if actual != expected:
            failures.append(f"{path} expected {expected!r}, got {actual!r}")

    for path in expectation.get("nonempty", []):
        try:
            actual = _lookup(structured, path)
        except (KeyError, IndexError, ValueError):
            failures.append(f"missing structured field {path}")
            continue
        if not actual:
            failures.append(f"{path} is empty")

    for path in expectation.get("finite", []):
        try:
            actual = _lookup(structured, path)
        except (KeyError, IndexError, ValueError):
            failures.append(f"missing structured field {path}")
            continue
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            failures.append(f"{path} is not numeric")
        elif not math.isfinite(actual):
            failures.append(f"{path} is not finite")
    return failures


def _content_text(result: Any) -> str:
    return "\n".join(
        item.text for item in result.content if getattr(item, "type", None) == "text"
    )


def sanitize_text(text: str, redactions: dict[str, str]) -> str:
    """Replace machine-specific paths with stable report tokens."""
    for path, token in sorted(redactions.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(path, token)
    return text


def _artifact_evidence(artifact_dir: Path, names: list[str]) -> list[dict[str, Any]]:
    evidence = []
    for name in names:
        path = artifact_dir / name
        if not path.is_file():
            evidence.append({"path": name, "exists": False})
            continue
        content = path.read_bytes()
        evidence.append(
            {
                "path": name,
                "exists": True,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return evidence


def _capabilities() -> set[str]:
    capabilities = set()
    if importlib.util.find_spec("pysd") is not None:
        capabilities.add("sim")
    return capabilities


async def run_evaluation(
    scenario_path: Path,
    artifact_dir: Path,
    selected_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Run selected scenarios through a real stdio MCP client session."""
    document = load_scenarios(scenario_path)
    scenarios = document["scenarios"]
    if selected_ids:
        known = {scenario["id"] for scenario in scenarios}
        unknown = selected_ids - known
        if unknown:
            raise ValueError(f"Unknown scenario ids: {', '.join(sorted(unknown))}")
        scenarios = [scenario for scenario in scenarios if scenario["id"] in selected_ids]

    artifact_dir.mkdir(parents=True, exist_ok=True)
    available = _capabilities()
    replacements = {
        "${ARTIFACT_DIR}": str(artifact_dir.resolve()),
        "${REPO_ROOT}": str(REPO_ROOT),
    }
    redactions = {value: token for token, value in replacements.items()}
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "stella_mcp.server"],
        cwd=REPO_ROOT,
    )
    scenario_results: list[dict[str, Any]] = []
    started = time.perf_counter()

    with anyio.fail_after(180):
        async with stdio_client(server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                initialized = await session.initialize()
                tools = await session.list_tools()
                resources = await session.list_resources()
                prompts = await session.list_prompts()
                catalog_json = json.dumps(
                    [tool.model_dump(mode="json") for tool in tools.tools],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")

                for scenario in scenarios:
                    missing = sorted(set(scenario.get("requires", [])) - available)
                    if missing:
                        scenario_results.append(
                            {
                                "id": scenario["id"],
                                "status": "skipped",
                                "missing_capabilities": missing,
                                "steps": [],
                                "artifacts": [],
                            }
                        )
                        continue

                    scenario_started = time.perf_counter()
                    step_results = []
                    for index, step in enumerate(scenario["steps"]):
                        arguments = _replace_tokens(step.get("arguments", {}), replacements)
                        step_started = time.perf_counter()
                        result = await session.call_tool(step["tool"], arguments)
                        failures = evaluate_expectation(result, step.get("expect", {}))
                        structured = result.structuredContent or {}
                        step_results.append(
                            {
                                "index": index,
                                "tool": step["tool"],
                                "status": "passed" if not failures else "failed",
                                "duration_ms": round((time.perf_counter() - step_started) * 1000, 3),
                                "is_error": bool(result.isError),
                                "error_code": (structured.get("error") or {}).get("code"),
                                "structured_keys": sorted(structured),
                                "text": sanitize_text(_content_text(result), redactions),
                                "failures": failures,
                            }
                        )

                    artifacts = _artifact_evidence(artifact_dir, scenario.get("artifacts", []))
                    artifact_failures = [item["path"] for item in artifacts if not item["exists"]]
                    status = "passed"
                    if any(step["status"] == "failed" for step in step_results) or artifact_failures:
                        status = "failed"
                    scenario_results.append(
                        {
                            "id": scenario["id"],
                            "description": scenario["description"],
                            "status": status,
                            "duration_ms": round(
                                (time.perf_counter() - scenario_started) * 1000,
                                3,
                            ),
                            "steps": step_results,
                            "artifacts": artifacts,
                            "missing_artifacts": artifact_failures,
                        }
                    )

    counts = {
        status: sum(item["status"] == status for item in scenario_results)
        for status in ("passed", "failed", "skipped")
    }
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
        "protocol": {
            "server_name": initialized.serverInfo.name,
            "tool_count": len(tools.tools),
            "tool_catalog_sha256": hashlib.sha256(catalog_json).hexdigest(),
            "resource_count_at_start": len(resources.resources),
            "prompt_count": len(prompts.prompts),
        },
        "summary": {
            "scenarios": len(scenario_results),
            **counts,
            "tool_calls": sum(len(item["steps"]) for item in scenario_results),
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        },
        "scenarios": scenario_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--artifact-dir", type=Path, default=Path("results/evaluation/artifacts"))
    parser.add_argument("--output-json", type=Path, default=Path("results/evaluation/latest.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("results/evaluation/latest.md"))
    parser.add_argument("--require", action="append", default=[])
    args = parser.parse_args()

    available = _capabilities()
    missing_required = sorted(set(args.require) - available)
    if missing_required:
        parser.error(f"missing required capabilities: {', '.join(missing_required)}")

    result = asyncio.run(
        run_evaluation(
            args.scenarios,
            args.artifact_dir,
            selected_ids=set(args.scenario) or None,
        )
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(result) + "\n", encoding="utf-8")
    return 1 if result["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
