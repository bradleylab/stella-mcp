"""Integrity checks for the recorded 0.12.0 free-form agent evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "evaluation"
ARTIFACTS = RESULTS / "0.12.0-agent-artifacts"
PROTOCOL = ROOT / "tests" / "fixtures" / "evaluation" / "agent_scenarios_v1.json"


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_recorded_agent_evidence_matches_protocol_and_artifacts() -> None:
    report = _load("0.12.0-agent-evaluation.json")

    assert report["protocol"]["scenario_sha256"] == _sha256(PROTOCOL)
    assert report["backend"]["provider"] == "openai"
    assert report["backend"]["model"] == "gpt-5.6-sol"
    assert report["backend"]["resolved_model"] == "gpt-5.6-sol"
    assert report["backend"]["effective_model_request"] == {
        "max_completion_tokens": 4096,
        "reasoning_effort": "none",
    }
    assert report["summary"]["passed"] == 2
    assert report["summary"]["failed"] == 1
    assert report["summary"]["skipped"] == 0
    assert report["summary"]["tool_calls"] == 17
    assert report["summary"]["tool_errors"] == 0

    scenarios = {scenario["id"]: scenario for scenario in report["scenarios"]}
    assert scenarios["construct_growth"]["failures"] == [
        "check 0: model.sim_specs.time_units expected 'Years', got 'Year'"
    ]
    assert scenarios["modify_sir_template"]["status"] == "passed"
    assert scenarios["analyze_accumulator"]["status"] == "passed"

    artifact_records = [
        artifact for scenario in report["scenarios"] for artifact in scenario["artifacts"]
    ]
    assert len(artifact_records) == 5
    for artifact in artifact_records:
        path = ARTIFACTS / artifact["path"]
        assert artifact["exists"] is True
        assert path.stat().st_size == artifact["bytes"]
        assert _sha256(path) == artifact["sha256"]

    serialized = json.dumps(report)
    assert "/Users/" not in serialized
    assert "sk-" not in serialized


def test_preflight_failure_is_preserved_separately() -> None:
    preflight = _load("0.12.0-agent-preflight-failure.json")
    report = _load("0.12.0-agent-evaluation.json")

    assert preflight["protocol"]["scenario_sha256"] == report["protocol"]["scenario_sha256"]
    assert preflight["summary"]["scenarios"] == 3
    assert preflight["summary"]["passed"] == 0
    assert preflight["summary"]["failed"] == 3
    assert preflight["summary"]["skipped"] == 0
    assert preflight["summary"]["tool_calls"] == 0
    assert preflight["summary"]["tool_errors"] == 0
    assert preflight["summary"]["usage"] == {}
    assert all(scenario["stop_reason"] == "backend_error" for scenario in preflight["scenarios"])
    assert all(
        any("reasoning_effort" in failure for failure in scenario["failures"])
        for scenario in preflight["scenarios"]
    )
