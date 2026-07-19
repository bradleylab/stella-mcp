"""Tests for generated, explicitly reviewed numeric-fidelity evidence."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.numeric_fidelity_report import (
    generate_numeric_fidelity_report,
    render_numeric_fidelity_markdown,
    write_numeric_fidelity_report,
)


def test_numeric_fidelity_report_retains_raw_reviewed_discrepancies(
    tmp_path: Path,
) -> None:
    report = generate_numeric_fidelity_report()
    cases = {case["id"]: case for case in report["cases"]}

    assert report["scientific_pass_threshold"] is None
    assert len(cases) == 9
    assert all(case["alignment"]["pass_threshold"] is None for case in cases.values())
    assert cases["stella_4_1_1_accumulator"]["nonzero_columns"] == []
    lotka = cases["package_lotka_volterra"]
    assert lotka["review_category"] == "flow_limiting_semantic_difference"
    predation = next(
        column for column in lotka["columns"] if column["reference_column"] == "predation"
    )
    assert predation["max_absolute_error"] == 1652.2784644028288
    assert predation["max_absolute_error_time"] == 41.0

    output_json = tmp_path / "numeric.json"
    output_markdown = tmp_path / "numeric.md"
    write_numeric_fidelity_report(report, output_json, output_markdown)
    assert json.loads(output_json.read_text(encoding="utf-8")) == report
    assert output_markdown.read_text(encoding="utf-8") == (
        render_numeric_fidelity_markdown(report) + "\n"
    )
    assert "not a tolerated parity result" in output_markdown.read_text(encoding="utf-8")
