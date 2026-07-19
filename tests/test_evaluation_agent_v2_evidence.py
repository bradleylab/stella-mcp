"""Integrity checks for the recorded 0.13.0 agent protocol-v2 evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "evaluation"
ARTIFACTS = RESULTS / "0.13.0-agent-artifacts"
PROTOCOL = ROOT / "evaluation" / "agent_scenarios.json"
REPORT = RESULTS / "0.13.0-agent-evaluation.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_recorded_agent_v2_evidence_matches_protocol_and_artifacts() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["schema_version"] == 2
    assert report["protocol"]["scenario_sha256"] == _sha256(PROTOCOL)
    assert report["protocol"]["runs_per_scenario"] == 3
    assert report["backend"]["provider"] == "openai"
    assert report["backend"]["model"] == "gpt-5.6-sol"
    assert report["backend"]["resolved_model"] == "gpt-5.6-sol"
    assert report["backend"]["effective_model_request"] == {
        "max_completion_tokens": 4096,
        "reasoning_effort": "none",
    }
    assert report["environment"]["stella_mcp"] == "0.13.0"

    summary = report["summary"]
    assert summary["protocol_scenarios"] == 3
    assert summary["scenario_runs"] == 9
    assert summary["passed"] == 9
    assert summary["failed"] == 0
    assert summary["skipped"] == 0
    assert summary["tool_calls"] == 51
    assert summary["tool_errors"] == 0
    assert all(
        counts == {"passed": 9, "recovered": 0, "failed": 0, "skipped": 0}
        for counts in summary["dimensions"].values()
    )

    assert Counter(scenario["id"] for scenario in report["scenarios"]) == {
        "construct_growth": 3,
        "modify_sir_template": 3,
        "analyze_accumulator": 3,
    }
    assert all(scenario["status"] == "passed" for scenario in report["scenarios"])
    assert all(scenario["failures"] == [] for scenario in report["scenarios"])
    assert all(
        all(call["is_error"] is False for call in scenario["tool_calls"])
        for scenario in report["scenarios"]
    )

    artifact_records = [
        (scenario["artifact_subdirectory"], artifact)
        for scenario in report["scenarios"]
        for artifact in scenario["artifacts"]
    ]
    assert len(artifact_records) == 15
    for subdirectory, artifact in artifact_records:
        path = ARTIFACTS / subdirectory / artifact["path"]
        assert artifact["exists"] is True
        assert path.stat().st_size == artifact["bytes"]
        assert _sha256(path) == artifact["sha256"]

    serialized = json.dumps(report)
    assert "/Users/" not in serialized
    assert "sk-" not in serialized
