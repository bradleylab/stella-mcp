"""Semantic round-trip and structured-difference evaluation tests."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from stella_mcp.xmile import parse_stmx
from tests.support.corpus_manifest import load_external_corpus_manifest
from tests.support.model_fidelity import (
    compare_model_fidelity,
    model_semantic_signature,
    structured_diff,
)

_CORPUS_ROOT = Path(__file__).parent / "fixtures" / "external_corpus"


def _fixture_path(fixture):
    return _CORPUS_ROOT / fixture["model"]["path"]


def test_structured_diff_uses_stable_json_pointer_paths():
    before = {"a/b": {"tilde~key": 1}, "removed": [1]}
    after = {"a/b": {"tilde~key": 2}, "added": True}

    assert structured_diff(before, after) == [
        {"path": "/a~1b/tilde~0key", "kind": "changed", "before": 1, "after": 2},
        {"path": "/added", "kind": "added", "before": None, "after": True},
        {"path": "/removed", "kind": "removed", "before": [1], "after": None},
    ]


def test_model_semantic_signature_is_independent_of_uuid():
    manifest = load_external_corpus_manifest()
    fixture = manifest["fixtures"][0]
    model = parse_stmx(str(_fixture_path(fixture)), compat_mode="strict")
    copied = copy.deepcopy(model)
    copied.uuid = "different-metadata-uuid"

    assert model_semantic_signature(copied) == model_semantic_signature(model)
    comparison = compare_model_fidelity(model, copied)
    assert comparison["semantic_equal"] is True
    assert comparison["metadata_changes"] == [
        {
            "path": "/uuid",
            "kind": "changed",
            "before": model.uuid,
            "after": "different-metadata-uuid",
        }
    ]


def test_model_semantic_signature_covers_supported_view_semantics():
    manifest = load_external_corpus_manifest()
    fixture = next(item for item in manifest["fixtures"] if item["id"] == "sdx_teacup")
    before = parse_stmx(str(_fixture_path(fixture)), compat_mode="strict")
    after = copy.deepcopy(before)
    after.view_page_width += 1
    after.stocks["teacup_temperature"].label_side = "left"

    changes = compare_model_fidelity(before, after)["semantic_changes"]

    assert changes == [
        {
            "path": "/variables/stocks/0/label_side",
            "kind": "changed",
            "before": before.stocks["teacup_temperature"].label_side,
            "after": "left",
        },
        {
            "path": "/view/page_width",
            "kind": "changed",
            "before": before.view_page_width,
            "after": before.view_page_width + 1,
        },
    ]


@pytest.mark.parametrize("fixture_id", ["sdx_sir", "sdx_teacup"])
def test_supported_external_fixture_preserves_semantics(fixture_id, tmp_path):
    manifest = load_external_corpus_manifest()
    fixture = next(item for item in manifest["fixtures"] if item["id"] == fixture_id)
    model = parse_stmx(str(_fixture_path(fixture)), compat_mode="strict")
    output = tmp_path / f"{fixture_id}.stmx"
    output.write_text(model.to_xml(auto_layout=False, compat_mode="strict"), encoding="utf-8")
    roundtripped = parse_stmx(str(output), compat_mode="strict")

    comparison = compare_model_fidelity(model, roundtripped)

    assert comparison["semantic_equal"] is True, comparison["semantic_changes"]


def test_targeted_edit_changes_only_requested_supported_fields(tmp_path):
    manifest = load_external_corpus_manifest()
    fixture = next(item for item in manifest["fixtures"] if item["id"] == "sdx_teacup")
    original = parse_stmx(str(_fixture_path(fixture)), compat_mode="strict")
    edited = copy.deepcopy(original)
    edited.auxs["room_temperature"].equation = "70 + 0"
    edited.sim_specs.stop = 100
    output = tmp_path / "edited-teacup.stmx"
    output.write_text(edited.to_xml(auto_layout=False, compat_mode="strict"), encoding="utf-8")
    roundtripped = parse_stmx(str(output), compat_mode="strict")

    comparison = compare_model_fidelity(original, roundtripped)

    assert comparison["semantic_changes"] == [
        {
            "path": "/sim_specs/stop",
            "kind": "changed",
            "before": 30.0,
            "after": 100.0,
        },
        {
            "path": "/variables/auxiliaries/1/equation",
            "kind": "changed",
            "before": "70",
            "after": "70 + 0",
        },
    ]
