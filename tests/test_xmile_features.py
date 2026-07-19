"""Unsupported XMILE feature detection, preservation, and failure behavior."""

from __future__ import annotations

import asyncio
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from evaluation.model_fidelity import unsupported_xml_signature
from stella_mcp import server as server_mod
from stella_mcp.analysis import compare_scenarios, sensitivity_analysis
from stella_mcp.calibrate import calibrate
from stella_mcp.simulate import run_simulation
from stella_mcp.xmile import parse_stmx
from stella_mcp.xmile_features import (
    UnsupportedModelFeatureError,
    detect_xmile_features,
)

_CORPUS = Path(__file__).parent / "fixtures" / "external_corpus" / "sdxorg"
_ARRAY = _CORPUS / "samples" / "arrays" / "a2a" / "a2a.stmx"
_ARRAY_GF = _CORPUS / "samples" / "arrays" / "non-a2a" / "non-a2a-gf.stmx"
_NESTED = (
    _CORPUS / "samples" / "bpowers-hares_and_lynxes_modules" / "model.stmx"
)
_SIR = _CORPUS / "samples" / "SIR" / "SIR.stmx"


def test_detect_xmile_features_distinguishes_groups_from_module_instances():
    root = ET.fromstring(
        """
        <xmile>
          <header><smile uses_arrays="1"/></header>
          <dimensions><dim name="Region"/></dimensions>
          <model>
            <variables>
              <aux name="Demand"><dimensions><dim name="Region"/></dimensions></aux>
              <group name="Logical group"/>
              <module name="Nested instance"/>
            </variables>
          </model>
          <model name="Nested definition"><variables/></model>
        </xmile>
        """
    )

    report = detect_xmile_features(root)

    assert report.preserved_only_codes == (
        "xmile.arrays",
        "xmile.module_instances",
        "xmile.nested_models",
    )
    assert [finding.count for finding in report.preserved_only] == [3, 1, 1]


@pytest.mark.parametrize(
    ("path", "expected_codes"),
    [
        (_ARRAY, ("xmile.arrays",)),
        (_NESTED, ("xmile.module_instances", "xmile.nested_models")),
    ],
)
def test_strict_import_rejects_preserved_only_features(path, expected_codes):
    with pytest.raises(UnsupportedModelFeatureError) as caught:
        parse_stmx(str(path), compat_mode="strict")

    assert tuple(caught.value.details["feature_codes"]) == expected_codes


@pytest.mark.parametrize("path", [_ARRAY, _ARRAY_GF, _NESTED])
def test_permissive_roundtrip_preserves_unsupported_feature_structure(path, tmp_path):
    original_root = ET.parse(path).getroot()
    model = parse_stmx(str(path), compat_mode="permissive")
    original_report = model.xmile_feature_report.to_dict()
    output = tmp_path / path.name
    output.write_text(
        model.to_xml(auto_layout=False, compat_mode="permissive"),
        encoding="utf-8",
    )

    roundtripped = parse_stmx(str(output), compat_mode="permissive")

    assert roundtripped.xmile_feature_report.to_dict() == original_report
    assert roundtripped.compatibility_warnings
    assert unsupported_xml_signature(ET.parse(output).getroot()) == (
        unsupported_xml_signature(original_root)
    )


def test_supported_external_model_remains_strict_roundtrip_safe(tmp_path):
    model = parse_stmx(str(_SIR), compat_mode="strict")
    output = tmp_path / "sir-roundtrip.stmx"
    output.write_text(model.to_xml(auto_layout=False, compat_mode="strict"), encoding="utf-8")

    roundtripped = parse_stmx(str(output), compat_mode="strict")

    assert roundtripped.xmile_feature_report.to_dict() == {
        "supported": True,
        "findings": [],
    }


def test_strict_export_rejects_permissively_loaded_unsupported_model():
    model = parse_stmx(str(_ARRAY), compat_mode="permissive")

    with pytest.raises(UnsupportedModelFeatureError, match="xmile.arrays"):
        model.to_xml(auto_layout=False, compat_mode="strict")


@pytest.mark.parametrize(
    ("tool_name", "operation"),
    [
        ("simulate", lambda model: run_simulation(model)),
        (
            "compare_scenarios",
            lambda model: compare_scenarios(
                model,
                scenarios=[{"name": "baseline", "overrides": {}}],
            ),
        ),
        (
            "sensitivity_analysis",
            lambda model: sensitivity_analysis(
                model,
                parameters=[{"name": "Price", "values": [8, 10]}],
                output={"variable": "Sales", "metric": "final"},
            ),
        ),
        (
            "calibrate",
            lambda model: calibrate(
                model,
                observations={"time": [1, 2], "targets": {"Sales": [9, 9]}},
                parameters=[{"name": "Price"}],
            ),
        ),
    ],
)
def test_pysd_backed_tools_reject_unsupported_models_before_import(
    monkeypatch, tool_name, operation
):
    model = parse_stmx(str(_ARRAY), compat_mode="permissive")
    monkeypatch.setattr(
        "stella_mcp.simulate._import_pysd",
        lambda: pytest.fail(f"PySD should not be imported by {tool_name}"),
    )

    with pytest.raises(UnsupportedModelFeatureError, match="xmile.arrays"):
        operation(model)


def test_calibrate_rejects_unsupported_model_before_numpy_import(monkeypatch):
    model = parse_stmx(str(_ARRAY), compat_mode="permissive")
    monkeypatch.setitem(sys.modules, "numpy", None)

    with pytest.raises(UnsupportedModelFeatureError, match="xmile.arrays"):
        calibrate(
            model,
            observations={"time": [1, 2], "targets": {"Sales": [9, 9]}},
            parameters=[{"name": "Price"}],
        )


def test_mcp_strict_import_returns_structured_compatibility_error():
    server_mod._clear_session_store()

    result = asyncio.run(
        server_mod.call_tool(
            "read_model",
            {"filepath": str(_ARRAY), "model_id": "array", "compat_mode": "strict"},
        )
    )

    assert result.isError is True
    assert result.structuredContent["error"] == {
        "code": "unsupported_model_feature",
        "message": "Unsupported XMILE model features: xmile.arrays",
        "category": "compatibility",
        "feature_codes": ["xmile.arrays"],
        "findings": [
            model_finding.to_dict()
            for model_finding in detect_xmile_features(ET.parse(_ARRAY).getroot()).findings
        ],
    }
