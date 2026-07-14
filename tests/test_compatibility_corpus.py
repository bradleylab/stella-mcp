"""Compatibility corpus tests for real-world-style XMILE variants."""

import json
from pathlib import Path

import pytest

from stella_mcp.xmile import parse_stmx

_CORPUS_DIR = Path(__file__).resolve().parent / "fixtures" / "compat_corpus"
_MANIFEST_PATH = _CORPUS_DIR / "manifest.json"
_MANIFEST = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
_CASES = _MANIFEST["fixtures"]


def _semantic_signature(model):
    """Return the supported model semantics that a round-trip must retain."""
    return {
        "stocks": sorted(
            (key, item.initial_value, item.units, item.non_negative)
            for key, item in model.stocks.items()
        ),
        "flows": sorted(
            (key, item.equation, item.units, item.from_stock, item.to_stock)
            for key, item in model.flows.items()
        ),
        "auxs": sorted(
            (key, item.equation, item.units)
            for key, item in model.auxs.items()
        ),
        "connectors": sorted(
            (item.from_var, item.to_var, item.angle, tuple(item.points))
            for item in model.connectors
        ),
        "modules": sorted(
            (key, tuple(sorted(item.members))) for key, item in model.modules.items()
        ),
        "sim_specs": (
            model.sim_specs.start,
            model.sim_specs.stop,
            model.sim_specs.dt,
            model.sim_specs.method,
            model.sim_specs.time_units,
        ),
    }


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["file"])
def test_compatibility_corpus_round_trip(case, tmp_path):
    """Each corpus fixture should be importable and round-trip safely."""
    source_path = _CORPUS_DIR / case["file"]

    if case.get("strict_import", False):
        parse_stmx(str(source_path), compat_mode="strict")

    model = parse_stmx(str(source_path), compat_mode="permissive")

    expected_warnings = case.get("expected_warning_contains", [])
    if expected_warnings:
        warning_text = "\n".join(model.compatibility_warnings)
        for fragment in expected_warnings:
            assert fragment in warning_text

    xml = model.to_xml(auto_layout=False, compat_mode="permissive")

    for marker in case.get("preserve_markers", []):
        assert marker in xml

    roundtrip_path = tmp_path / f"roundtrip_{source_path.name}"
    roundtrip_path.write_text(xml, encoding="utf-8")
    roundtripped = parse_stmx(str(roundtrip_path), compat_mode="strict")
    assert _semantic_signature(roundtripped) == _semantic_signature(model)


def test_stella_saved_fixtures_record_desktop_acceptance_provenance():
    stella_cases = [case for case in _CASES if case["file"].startswith("stella_")]

    assert stella_cases
    for case in stella_cases:
        provenance = case["provenance"]
        assert provenance["application"] == "Stella Professional"
        assert provenance["application_version"] == "4.1.1"
        acceptance = provenance["acceptance"]
        assert acceptance["opened_without_repair"] is True
        assert acceptance["run_completed"] is True
