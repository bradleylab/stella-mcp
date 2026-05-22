"""Compatibility corpus tests for real-world-style XMILE variants."""

import json
from pathlib import Path

import pytest

from stella_mcp.xmile import parse_stmx

_CORPUS_DIR = Path(__file__).resolve().parent / "fixtures" / "compat_corpus"
_MANIFEST_PATH = _CORPUS_DIR / "manifest.json"
_MANIFEST = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
_CASES = _MANIFEST["fixtures"]


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
    parse_stmx(str(roundtrip_path), compat_mode="strict")
