"""Integrity checks for the pinned external XMILE compatibility corpus."""

from __future__ import annotations

import copy

import pytest

from tests.support.corpus_manifest import (
    DEFAULT_EXTERNAL_CORPUS_MANIFEST,
    load_external_corpus_manifest,
    validate_external_corpus_manifest,
)


def test_external_corpus_manifest_and_pinned_files_are_valid():
    document = load_external_corpus_manifest()

    assert document["sources"]["sdxorg_test_models"]["license"] == "MIT"
    assert {
        fixture["desktop_acceptance_evidence"] for fixture in document["fixtures"]
    } == {"none"}
    assert [fixture["id"] for fixture in document["fixtures"]] == [
        "sdx_sir",
        "sdx_teacup",
        "sdx_array_a2a",
        "sdx_array_gf",
        "sdx_nested_modules",
    ]


def test_external_corpus_manifest_rejects_hash_drift():
    document = load_external_corpus_manifest()
    changed = copy.deepcopy(document)
    changed["fixtures"][0]["model"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_external_corpus_manifest(changed, DEFAULT_EXTERNAL_CORPUS_MANIFEST.parent)


def test_external_corpus_manifest_rejects_unsafe_paths():
    document = load_external_corpus_manifest()
    changed = copy.deepcopy(document)
    changed["fixtures"][0]["model"]["path"] = "../outside.stmx"

    with pytest.raises(ValueError, match="safe relative path"):
        validate_external_corpus_manifest(changed, DEFAULT_EXTERNAL_CORPUS_MANIFEST.parent)


def test_external_corpus_manifest_requires_findings_for_unsupported_cases():
    document = load_external_corpus_manifest()
    changed = copy.deepcopy(document)
    changed["fixtures"][2]["expect"]["findings"] = []

    with pytest.raises(ValueError, match="declares no finding codes"):
        validate_external_corpus_manifest(changed, DEFAULT_EXTERNAL_CORPUS_MANIFEST.parent)


def test_external_corpus_manifest_requires_desktop_evidence_origin():
    document = load_external_corpus_manifest()
    changed = copy.deepcopy(document)
    changed["fixtures"][0].pop("desktop_acceptance_evidence")

    with pytest.raises(ValueError, match="desktop_acceptance_evidence is unsupported"):
        validate_external_corpus_manifest(changed, DEFAULT_EXTERNAL_CORPUS_MANIFEST.parent)
