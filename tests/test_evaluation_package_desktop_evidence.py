"""Tests for retained Stella package desktop evidence generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.package_desktop_evidence import (
    generate_package_desktop_evidence,
    render_desktop_evidence_markdown,
)


def test_package_desktop_evidence_validates_all_cases(tmp_path: Path) -> None:
    pytest.importorskip("pysd")

    report = generate_package_desktop_evidence(output_dir=tmp_path)

    assert report["application"] == "Stella Professional"
    assert report["application_version"] == "4.1.1"
    assert len(report["cases"]) == 6
    assert all(case["opened_without_repair"] for case in report["cases"])
    assert all(case["run_completed"] for case in report["cases"])
    assert all(case["saved_reimport_strict"] for case in report["cases"])
    assert all(case["identifier_renames"] == [] for case in report["cases"])
    assert all(case["computational_changes"] == [] for case in report["cases"])
    by_id = {case["id"]: case for case in report["cases"]}
    assert all(
        case["serialization_changes"] == []
        for case_id, case in by_id.items()
        if case_id != "package_scalar_graphical_function"
    )
    assert [
        change["path"]
        for change in by_id["package_scalar_graphical_function"]["serialization_changes"]
    ] == [
        "/auxiliaries/seasonal_multiplier/graphical_function/gf_type",
        "/flows/response_input/equation",
    ]
    assert all((tmp_path / Path(case["parity_report"]).name).is_file() for case in report["cases"])
    markdown = render_desktop_evidence_markdown(report)
    assert "no canvas-scale loops" in markdown
    assert "Identifier renames" in markdown
    assert "Serialization rewrites" in markdown
    assert "graphical_function/gf_type" in markdown
