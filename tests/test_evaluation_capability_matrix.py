"""Tests for generated capability evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.capability_matrix import (
    generate_capability_matrix,
    render_capability_markdown,
    write_capability_matrix,
)


def test_capability_matrix_distinguishes_supported_and_preserved_only(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pysd")

    report = generate_capability_matrix()
    rows = {row["id"]: row for row in report["fixtures"]}

    assert report["claim_scope"] == "retained_fixtures_only"
    assert len(rows) == 12
    assert rows["sdx_sir"]["strict_import"] == "supported"
    assert rows["sdx_sir"]["stella_numeric_evidence"] == "pinned_upstream_export"
    assert rows["sdx_array_a2a"]["strict_import"] == "rejected_unsupported"
    assert rows["sdx_array_a2a"]["unsupported_xml_preserved"] is True
    assert rows["sdx_array_a2a"]["pysd_simulation"] is False
    assert rows["package_sir"]["strict_import"] == "supported"
    assert rows["package_sir"]["stella_numeric_evidence"] == "local_desktop_export"
    assert rows["package_sir"]["desktop_open_run_save"] is True
    assert rows["stella_4_1_1_accumulator"]["desktop_open_run_save"] is True

    output_json = tmp_path / "matrix.json"
    output_markdown = tmp_path / "matrix.md"
    write_capability_matrix(report, output_json, output_markdown)
    assert json.loads(output_json.read_text(encoding="utf-8")) == report
    assert output_markdown.read_text(encoding="utf-8") == (
        render_capability_markdown(report) + "\n"
    )
    assert "compatibility with every Stella or XMILE model" in output_markdown.read_text(
        encoding="utf-8"
    )
