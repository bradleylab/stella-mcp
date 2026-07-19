"""Tests for pinned external-corpus numeric parity generation."""

from __future__ import annotations

import json

import pytest

from evaluation import external_parity


def test_external_parity_wires_manifest_policy_and_provenance(tmp_path, monkeypatch):
    calls = []

    def fake_report(model_path, stella_path, pysd_path, columns, **kwargs):
        calls.append((model_path, stella_path, pysd_path, columns, kwargs))
        pysd_path.write_text("time,value\n0,1\n", encoding="utf-8")
        return {"schema_version": 2, "comparison": {"points": 1}}

    monkeypatch.setattr(external_parity, "generate_desktop_parity_report", fake_report)

    records = external_parity.generate_external_parity_reports(
        tmp_path,
        version_label="0.13.0",
        selected_ids={"sdx_sir"},
    )

    assert len(records) == 1
    _, stella_path, pysd_path, columns, kwargs = calls[0]
    assert stella_path.name == "output_stella1006.csv"
    assert pysd_path.name == "0.13.0-sdx_sir-pysd.csv"
    assert columns == [
        ("infectious", "infectious"),
        ("recovered", "recovered"),
        ("susceptible", "susceptible"),
    ]
    assert kwargs["stella_application"] == "Stella"
    assert kwargs["stella_version"] == "10.0.6 for Windows"
    assert kwargs["time_alignment"] == "rounded_reference_labels"
    assert kwargs["candidate_decimal_places"] == 3
    saved = json.loads(records[0]["report_path"].read_text(encoding="utf-8"))
    assert saved["fixture"]["id"] == "sdx_sir"


def test_external_parity_rejects_unknown_selection(tmp_path):
    with pytest.raises(ValueError, match="Unknown numeric external fixture"):
        external_parity.generate_external_parity_reports(
            tmp_path,
            version_label="0.13.0",
            selected_ids={"missing"},
        )


def test_external_parity_rejects_unsafe_version_label(tmp_path):
    with pytest.raises(ValueError, match="version_label"):
        external_parity.generate_external_parity_reports(
            tmp_path,
            version_label="../outside",
        )
