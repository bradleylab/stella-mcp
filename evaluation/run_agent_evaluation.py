"""Command-line entry point for the free-form agent evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .agent_reporting import render_agent_markdown
from .agent_runner import (
    DEFAULT_AGENT_SCENARIOS,
    preflight_agent_artifacts,
    run_agent_evaluation,
)
from .openai_chat_backend import SAMPLING_MODES, build_openai_chat_backend


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openai", "washu"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sampling-mode", choices=sorted(SAMPLING_MODES), required=True)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_AGENT_SCENARIOS)
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("results/evaluation/agent-artifacts"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/evaluation/agent-evaluation.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("results/evaluation/agent-evaluation.md"),
    )
    args = parser.parse_args()

    selected_ids = set(args.scenario) or None
    output_paths = [args.output_json.resolve(), args.output_markdown.resolve()]
    if len(set(output_paths)) != len(output_paths):
        parser.error("JSON and Markdown outputs must be different files")
    existing_outputs = [str(path) for path in output_paths if path.exists()]
    if existing_outputs:
        parser.error("Result files already exist: " + ", ".join(existing_outputs))
    preflight_agent_artifacts(args.scenarios, args.artifact_dir, selected_ids)

    backend = build_openai_chat_backend(
        provider=args.provider,
        model=args.model,
        sampling_mode=args.sampling_mode,
    )
    report = asyncio.run(
        run_agent_evaluation(
            backend,
            args.scenarios,
            args.artifact_dir,
            selected_ids=selected_ids,
        )
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_agent_markdown(report) + "\n", encoding="utf-8")
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
