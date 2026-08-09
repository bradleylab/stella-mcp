"""Tests for the deterministic evaluation harness."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from evaluation.runner import (
    DEFAULT_SCENARIOS,
    evaluate_expectation,
    run_evaluation,
    sanitize_text,
)


def test_expectation_checks_nested_fields_and_finite_values() -> None:
    result = SimpleNamespace(
        is_error=False,
        structured_content={"validation": {"passed": True}, "value": 1.5, "items": [1]},
    )

    failures = evaluate_expectation(
        result,
        {
            "is_error": False,
            "fields": {"validation.passed": True},
            "nonempty": ["items"],
            "finite": ["value"],
        },
    )

    assert failures == []


def test_expectation_reports_missing_and_mismatched_fields() -> None:
    result = SimpleNamespace(is_error=True, structured_content={"value": float("nan")})

    failures = evaluate_expectation(
        result,
        {
            "is_error": False,
            "fields": {"missing.path": True},
            "finite": ["value"],
        },
    )

    assert failures == [
        "is_error expected False, got True",
        "missing structured field missing.path",
        "value is not finite",
    ]


def test_report_text_redacts_longest_machine_path_first() -> None:
    text = "/repo/results/model.stmx written from /repo"

    sanitized = sanitize_text(
        text,
        {"/repo/results": "${ARTIFACT_DIR}", "/repo": "${REPO_ROOT}"},
    )

    assert sanitized == "${ARTIFACT_DIR}/model.stmx written from ${REPO_ROOT}"


def test_core_evaluation_scenario_runs_over_stdio(tmp_path: Path) -> None:
    result = asyncio.run(
        run_evaluation(DEFAULT_SCENARIOS, tmp_path, selected_ids={"build_growth"})
    )

    assert result["protocol"]["server_name"] == "stella-mcp"
    assert result["protocol"]["tool_count"] == 44
    assert result["summary"] == {
        "scenarios": 1,
        "passed": 1,
        "failed": 0,
        "skipped": 0,
        "tool_calls": 4,
        "duration_ms": result["summary"]["duration_ms"],
    }
    assert result["scenarios"][0]["artifacts"][0]["exists"] is True
