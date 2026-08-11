"""Behavioral gates for the pinned external XMILE compatibility corpus."""

from __future__ import annotations

from pathlib import Path

import pytest

from stella_mcp.simulate import run_simulation
from stella_mcp.xmile import parse_stmx
from stella_mcp.xmile_features import UnsupportedModelFeatureError
from tests.support.corpus_manifest import load_external_corpus_manifest
from tests.support.model_fidelity import compare_model_fidelity

_ROOT = Path(__file__).parent / "fixtures" / "external_corpus"
_MANIFEST = load_external_corpus_manifest()
_FIXTURES = _MANIFEST["fixtures"]


@pytest.mark.parametrize("fixture", _FIXTURES, ids=lambda fixture: fixture["id"])
def test_external_corpus_matches_compatibility_contract(fixture, tmp_path):
    path = _ROOT / fixture["model"]["path"]
    expectation = fixture["expect"]
    expected_codes = tuple(expectation["findings"])

    if expectation["strict_import"] == "supported":
        strict_model = parse_stmx(str(path), compat_mode="strict")
        assert strict_model.xmile_feature_report.preserved_only_codes == ()
    else:
        with pytest.raises(UnsupportedModelFeatureError) as caught:
            parse_stmx(str(path), compat_mode="strict")
        assert caught.value.details["feature_codes"] == list(expected_codes)

    model = parse_stmx(str(path), compat_mode="permissive")
    assert model.xmile_feature_report.preserved_only_codes == expected_codes
    output = tmp_path / f"{fixture['id']}.stmx"
    output.write_text(
        model.to_xml(auto_layout=False, compat_mode="permissive"),
        encoding="utf-8",
    )
    roundtripped = parse_stmx(str(output), compat_mode="permissive")
    assert roundtripped.xmile_feature_report.preserved_only_codes == expected_codes

    comparison = compare_model_fidelity(model, roundtripped)
    assert comparison["semantic_equal"] is True, comparison["semantic_changes"]

    if expectation["simulation"] == "unsupported":
        with pytest.raises(UnsupportedModelFeatureError):
            run_simulation(model)
