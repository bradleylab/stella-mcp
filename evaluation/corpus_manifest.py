"""Validation helpers for the pinned external XMILE compatibility corpus."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_EXTERNAL_CORPUS_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "external_corpus"
    / "manifest.json"
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_EXPECTATION_VALUES = {
    "strict_import": {"supported", "unsupported"},
    "permissive_import": {"supported", "preserved_only"},
    "permissive_roundtrip": {"supported_semantics", "preserved_only"},
    "simulation": {"supported", "unsupported"},
}
_ALIGNMENT_POLICIES = {"exact", "rounded_reference_labels"}
_DESKTOP_EVIDENCE_ORIGINS = {"none", "upstream", "local"}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a corpus artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_relative_path(value: Any, context: str) -> Path:
    raw = _require_string(value, context)
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{context} must be a safe relative path")
    return path


def _verify_artifact(root: Path, artifact: dict[str, Any], context: str) -> None:
    relative = _require_relative_path(artifact.get("path"), f"{context}.path")
    expected_hash = _require_string(artifact.get("sha256"), f"{context}.sha256")
    if _SHA256_PATTERN.fullmatch(expected_hash) is None:
        raise ValueError(f"{context}.sha256 must be a lowercase SHA-256 digest")
    _require_string(artifact.get("upstream_path"), f"{context}.upstream_path")
    path = root / relative
    if not path.is_file():
        raise ValueError(f"{context} file does not exist: {relative.as_posix()}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{context} SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )


def _validate_numeric(root: Path, numeric: Any, context: str) -> None:
    if not isinstance(numeric, dict):
        raise ValueError(f"{context} must be an object")
    output = numeric.get("stella_output")
    if not isinstance(output, dict):
        raise ValueError(f"{context}.stella_output must be an object")
    _verify_artifact(root, output, f"{context}.stella_output")
    _require_string(numeric.get("application"), f"{context}.application")
    _require_string(numeric.get("application_version"), f"{context}.application_version")
    _require_string(numeric.get("reference_time"), f"{context}.reference_time")
    _require_string(numeric.get("candidate_time"), f"{context}.candidate_time")

    alignment = numeric.get("time_alignment")
    if not isinstance(alignment, dict):
        raise ValueError(f"{context}.time_alignment must be an object")
    policy = alignment.get("policy")
    if policy not in _ALIGNMENT_POLICIES:
        raise ValueError(f"{context}.time_alignment.policy is unsupported: {policy!r}")
    if policy == "rounded_reference_labels":
        decimal_places = alignment.get("candidate_decimal_places")
        if not isinstance(decimal_places, int) or isinstance(decimal_places, bool):
            raise ValueError(
                f"{context}.time_alignment.candidate_decimal_places must be an integer"
            )
        if decimal_places < 0:
            raise ValueError(
                f"{context}.time_alignment.candidate_decimal_places must be >= 0"
            )

    columns = numeric.get("columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError(f"{context}.columns must be a non-empty array")
    for index, column in enumerate(columns):
        if not isinstance(column, dict):
            raise ValueError(f"{context}.columns[{index}] must be an object")
        _require_string(column.get("reference"), f"{context}.columns[{index}].reference")
        _require_string(column.get("candidate"), f"{context}.columns[{index}].candidate")


def validate_external_corpus_manifest(document: Any, root: Path) -> dict[str, Any]:
    """Validate manifest structure and every pinned local artifact."""
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("External corpus manifest requires schema_version 1")

    sources = document.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("External corpus manifest requires a non-empty sources object")
    for source_id, source in sources.items():
        _require_string(source_id, "source id")
        if not isinstance(source, dict):
            raise ValueError(f"sources.{source_id} must be an object")
        _require_string(source.get("repository"), f"sources.{source_id}.repository")
        commit = _require_string(source.get("commit"), f"sources.{source_id}.commit")
        if _COMMIT_PATTERN.fullmatch(commit) is None:
            raise ValueError(f"sources.{source_id}.commit must be a lowercase 40-character hash")
        _require_string(source.get("license"), f"sources.{source_id}.license")
        license_path = _require_relative_path(
            source.get("license_path"), f"sources.{source_id}.license_path"
        )
        license_hash = _require_string(
            source.get("license_sha256"), f"sources.{source_id}.license_sha256"
        )
        if _SHA256_PATTERN.fullmatch(license_hash) is None:
            raise ValueError(f"sources.{source_id}.license_sha256 must be a SHA-256 digest")
        actual_license = root / license_path
        if not actual_license.is_file():
            raise ValueError(f"sources.{source_id} license file does not exist: {license_path}")
        if sha256_file(actual_license) != license_hash:
            raise ValueError(f"sources.{source_id} license SHA-256 mismatch")
        attribution_path = _require_relative_path(
            source.get("attribution_path"), f"sources.{source_id}.attribution_path"
        )
        if not (root / attribution_path).is_file():
            raise ValueError(
                f"sources.{source_id} attribution file does not exist: {attribution_path}"
            )

    fixtures = document.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("External corpus manifest requires a non-empty fixtures array")
    seen_ids: set[str] = set()
    for index, fixture in enumerate(fixtures):
        context = f"fixtures[{index}]"
        if not isinstance(fixture, dict):
            raise ValueError(f"{context} must be an object")
        fixture_id = _require_string(fixture.get("id"), f"{context}.id")
        if fixture_id in seen_ids:
            raise ValueError(f"Duplicate external corpus fixture id: {fixture_id}")
        seen_ids.add(fixture_id)
        source_id = _require_string(fixture.get("source"), f"{context}.source")
        if source_id not in sources:
            raise ValueError(f"{context}.source is unknown: {source_id}")

        model = fixture.get("model")
        if not isinstance(model, dict):
            raise ValueError(f"{context}.model must be an object")
        _verify_artifact(root, model, f"{context}.model")

        constructs = fixture.get("constructs")
        if (
            not isinstance(constructs, list)
            or not constructs
            or any(not isinstance(item, str) or not item for item in constructs)
            or len(constructs) != len(set(constructs))
        ):
            raise ValueError(f"{context}.constructs must be unique non-empty strings")

        desktop_origin = fixture.get("desktop_acceptance_evidence")
        if desktop_origin not in _DESKTOP_EVIDENCE_ORIGINS:
            raise ValueError(
                f"{context}.desktop_acceptance_evidence is unsupported: "
                f"{desktop_origin!r}"
            )

        expect = fixture.get("expect")
        if not isinstance(expect, dict):
            raise ValueError(f"{context}.expect must be an object")
        for field, allowed in _EXPECTATION_VALUES.items():
            if expect.get(field) not in allowed:
                raise ValueError(f"{context}.expect.{field} is unsupported: {expect.get(field)!r}")
        findings = expect.get("findings")
        if not isinstance(findings, list) or any(
            not isinstance(item, str) or not item for item in findings
        ):
            raise ValueError(f"{context}.expect.findings must be an array of strings")
        unsupported = (
            expect["strict_import"] == "unsupported"
            or expect["simulation"] == "unsupported"
        )
        if unsupported and not findings:
            raise ValueError(f"{context} is unsupported but declares no finding codes")

        numeric = fixture.get("numeric")
        if numeric is not None:
            if expect["simulation"] != "supported":
                raise ValueError(f"{context} has numeric evidence but simulation is unsupported")
            _validate_numeric(root, numeric, f"{context}.numeric")

    return document


def load_external_corpus_manifest(
    path: Path = DEFAULT_EXTERNAL_CORPUS_MANIFEST,
) -> dict[str, Any]:
    """Load and validate the external corpus manifest and its pinned files."""
    document = json.loads(path.read_text(encoding="utf-8"))
    return validate_external_corpus_manifest(document, path.parent)
