"""Generate the evidence-backed Stella MCP capability matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from stella_mcp.simulate import run_simulation
from stella_mcp.xmile import parse_stmx
from stella_mcp.xmile_features import UnsupportedModelFeatureError

from .corpus_manifest import DEFAULT_EXTERNAL_CORPUS_MANIFEST, load_external_corpus_manifest
from .model_fidelity import compare_model_fidelity, unsupported_xml_signature

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"
DEFAULT_RELEASE = "0.13.0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return document


def _verify_artifact(record: dict[str, Any], *, root: Path = PROJECT_ROOT) -> Path:
    path = root / record["path"]
    if not path.is_file():
        raise FileNotFoundError(f"Retained evidence file does not exist: {path}")
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"Retained evidence hash mismatch: {path}")
    return path


def _numeric_report(
    report_path: Path,
    *,
    expected_schema: int,
) -> dict[str, Any] | None:
    if not report_path.is_file():
        return None
    report = _load_json(report_path)
    if report.get("schema_version") != expected_schema:
        raise ValueError(f"Unexpected parity report schema: {report_path}")
    for record in report["artifacts"].values():
        _verify_artifact(record)
    columns = report.get("comparison", {}).get("columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError(f"Parity report has no compared columns: {report_path}")
    return report


def _external_rows(
    manifest_path: Path,
    results_dir: Path,
    release: str,
) -> list[dict[str, Any]]:
    document = load_external_corpus_manifest(manifest_path)
    rows = []
    for fixture in document["fixtures"]:
        model_path = manifest_path.parent / fixture["model"]["path"]
        original_xml_signature = unsupported_xml_signature(ET.parse(model_path).getroot())
        expected_codes = tuple(fixture["expect"]["findings"])
        permissive = parse_stmx(str(model_path), compat_mode="permissive")
        with tempfile.TemporaryDirectory() as temporary_dir:
            roundtrip_path = Path(temporary_dir) / "roundtrip.stmx"
            roundtrip_path.write_text(
                permissive.to_xml(auto_layout=False, compat_mode="permissive"),
                encoding="utf-8",
            )
            roundtripped = parse_stmx(str(roundtrip_path), compat_mode="permissive")
            roundtrip_xml_signature = unsupported_xml_signature(
                ET.parse(roundtrip_path).getroot()
            )
        comparison = compare_model_fidelity(permissive, roundtripped)
        actual_codes = roundtripped.xmile_feature_report.preserved_only_codes
        if actual_codes != expected_codes:
            raise ValueError(f"Feature preservation changed for {fixture['id']}")

        strict_supported = not expected_codes
        if strict_supported:
            parse_stmx(str(model_path), compat_mode="strict")
        else:
            try:
                parse_stmx(str(model_path), compat_mode="strict")
            except UnsupportedModelFeatureError as exc:
                if tuple(exc.details["feature_codes"]) != expected_codes:
                    raise ValueError(f"Strict feature codes changed for {fixture['id']}") from exc
            else:
                raise ValueError(f"Strict import accepted unsupported fixture {fixture['id']}")

        if fixture["expect"]["simulation"] == "supported":
            simulation = run_simulation(permissive)
            pysd_supported = True
            guard_verified = True
            backend = simulation["backend"]
        else:
            try:
                run_simulation(permissive)
            except UnsupportedModelFeatureError:
                pysd_supported = False
                guard_verified = True
                backend = None
            else:
                raise ValueError(f"Unsupported fixture reached PySD: {fixture['id']}")

        parity_path = results_dir / f"{release}-{fixture['id']}-parity.json"
        numeric = _numeric_report(parity_path, expected_schema=2)
        rows.append(
            {
                "id": fixture["id"],
                "source": "pinned_external_corpus",
                "constructs": fixture["constructs"],
                "permissive_parse": True,
                "strict_import": "supported" if strict_supported else "rejected_unsupported",
                "supported_semantics_preserved": not comparison["semantic_changes"],
                "semantic_changes": comparison["semantic_changes"],
                "unsupported_xml_preserved": (
                    original_xml_signature == roundtrip_xml_signature
                    if expected_codes
                    else None
                ),
                "feature_codes": list(expected_codes),
                "pysd_simulation": pysd_supported,
                "simulation_guard_verified": guard_verified,
                "simulation_backend": backend,
                "stella_numeric_evidence": (
                    "pinned_upstream_export" if numeric is not None else None
                ),
                "numeric_report": (
                    parity_path.relative_to(PROJECT_ROOT).as_posix()
                    if numeric is not None
                    else None
                ),
                "desktop_open_run_save": False,
                "desktop_record": None,
            }
        )
    return rows


def _desktop_records(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    document = _load_json(path)
    if document.get("schema_version") != 1:
        raise ValueError(f"Unsupported desktop evidence schema: {path}")
    records = {}
    for record in document.get("cases", []):
        if record["id"] in records:
            raise ValueError(f"Duplicate desktop evidence id: {record['id']}")
        _verify_artifact(record["stella_csv"])
        saved_path = _verify_artifact(record["stella_saved_model"])
        parse_stmx(str(saved_path), compat_mode="strict")
        records[record["id"]] = record
    return records


def _package_rows(
    candidate_manifest_path: Path,
    results_dir: Path,
    release: str,
    desktop_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest = _load_json(candidate_manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("release") != release:
        raise ValueError(f"Unexpected desktop candidate manifest: {candidate_manifest_path}")
    rows = []
    for case in manifest["cases"]:
        model_path = _verify_artifact(case["model"], root=results_dir)
        _verify_artifact(case["pysd_csv"], root=results_dir)
        permissive = parse_stmx(str(model_path), compat_mode="permissive")
        parse_stmx(str(model_path), compat_mode="strict")
        desktop = desktop_records.get(case["id"])
        desktop_accepted = bool(
            desktop
            and desktop.get("opened_without_repair") is True
            and desktop.get("run_completed") is True
            and desktop.get("saved_reimport_strict") is True
            and desktop.get("visual_review") == "pass"
            and desktop.get("identifier_renames") == []
            and desktop.get("computational_changes") == []
        )
        parity_path = results_dir / f"{release}-{case['id']}-parity.json"
        numeric = _numeric_report(parity_path, expected_schema=2)
        rows.append(
            {
                "id": case["id"],
                "source": "package_generated",
                "constructs": case["constructs"],
                "permissive_parse": True,
                "strict_import": "supported",
                "supported_semantics_preserved": not case["semantic_diff"],
                "semantic_changes": case["semantic_diff"],
                "unsupported_xml_preserved": None,
                "feature_codes": [],
                "pysd_simulation": True,
                "simulation_guard_verified": (
                    case["simulation"]["backend"]["unsupported_feature_preflight"]["status"]
                    == "passed"
                ),
                "simulation_backend": case["simulation"]["backend"],
                "stella_numeric_evidence": (
                    "local_desktop_export" if numeric is not None else None
                ),
                "numeric_report": (
                    parity_path.relative_to(PROJECT_ROOT).as_posix()
                    if numeric is not None
                    else None
                ),
                "desktop_open_run_save": desktop_accepted,
                "desktop_record": desktop,
            }
        )
        if permissive.xmile_feature_report.preserved_only_codes:
            raise ValueError(f"Package candidate has unsupported features: {case['id']}")
    return rows


def _accumulator_row(results_dir: Path) -> dict[str, Any]:
    report_path = results_dir / "0.12.0-accumulator-parity.json"
    report = _numeric_report(report_path, expected_schema=1)
    assert report is not None
    model_path = _verify_artifact(report["artifacts"]["model"])
    model = parse_stmx(str(model_path), compat_mode="strict")
    simulation = run_simulation(model)
    return {
        "id": "stella_4_1_1_accumulator",
        "source": "retained_local_desktop",
        "constructs": ["scalar", "constant_flow"],
        "permissive_parse": True,
        "strict_import": "supported",
        "supported_semantics_preserved": True,
        "semantic_changes": [],
        "unsupported_xml_preserved": None,
        "feature_codes": [],
        "pysd_simulation": True,
        "simulation_guard_verified": True,
        "simulation_backend": simulation["backend"],
        "stella_numeric_evidence": "local_desktop_export",
        "numeric_report": report_path.relative_to(PROJECT_ROOT).as_posix(),
        "desktop_open_run_save": True,
        "desktop_record": "tests/fixtures/compat_corpus/manifest.json",
    }


def render_capability_markdown(report: dict[str, Any]) -> str:
    """Render the capability matrix from its machine-readable JSON record."""
    lines = [
        f"# Stella MCP {report['release']} Capability Matrix",
        "",
        "This matrix reports retained fixture evidence. It is not a claim of compatibility "
        "with every Stella or XMILE model.",
        "",
        "| Fixture | Source | Permissive parse | Strict import | Supported semantics | "
        "Unsupported XML | PySD simulation | Stella numeric | Desktop open/run/save |",
        "|---|---|---:|---|---:|---:|---:|---|---:|",
    ]
    for row in report["fixtures"]:
        unsupported = row["unsupported_xml_preserved"]
        lines.append(
            f"| `{row['id']}` | {row['source']} | {_yes_no(row['permissive_parse'])} | "
            f"{row['strict_import']} | {_yes_no(row['supported_semantics_preserved'])} | "
            f"{_yes_no(unsupported)} | {_yes_no(row['pysd_simulation'])} | "
            f"{row['stella_numeric_evidence'] or 'none'} | "
            f"{_yes_no(row['desktop_open_run_save'])} |"
        )
    lines.extend(
        [
            "",
            "`N/A` means the fixture contains no preserved-only XML construct. Pinned upstream "
            "numeric exports are kept distinct from locally generated Stella desktop evidence.",
        ]
    )
    return "\n".join(lines)


def _yes_no(value: bool | None) -> str:
    if value is None:
        return "N/A"
    return "yes" if value else "no"


def generate_capability_matrix(
    *,
    release: str = DEFAULT_RELEASE,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    external_manifest_path: Path = DEFAULT_EXTERNAL_CORPUS_MANIFEST,
    candidate_manifest_path: Path | None = None,
    desktop_evidence_path: Path | None = None,
) -> dict[str, Any]:
    """Build the capability matrix from retained manifests and reports."""
    results_dir = results_dir.resolve()
    if candidate_manifest_path is None:
        candidate_manifest_path = results_dir / f"{release}-desktop-candidates.json"
    if desktop_evidence_path is None:
        desktop_evidence_path = results_dir / f"{release}-desktop-evidence.json"
    desktop = _desktop_records(desktop_evidence_path)
    fixtures = [
        *_external_rows(external_manifest_path, results_dir, release),
        *_package_rows(candidate_manifest_path, results_dir, release, desktop),
        _accumulator_row(results_dir),
    ]
    fixtures.sort(key=lambda row: row["id"])
    return {
        "schema_version": 1,
        "release": release,
        "claim_scope": "retained_fixtures_only",
        "fixtures": fixtures,
    }


def write_capability_matrix(
    report: dict[str, Any], output_json: Path, output_markdown: Path
) -> None:
    """Write synchronized JSON and generated Markdown capability evidence."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output_markdown.write_text(render_capability_markdown(report) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_RESULTS_DIR / f"{DEFAULT_RELEASE}-capability-matrix.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=PROJECT_ROOT / "docs/evaluation/0.13.0-capability-matrix.md",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = generate_capability_matrix(release=args.release, results_dir=args.results_dir)
    expected_json = json.dumps(report, indent=2) + "\n"
    expected_markdown = render_capability_markdown(report) + "\n"
    if args.check:
        if not args.output_json.is_file() or not args.output_markdown.is_file():
            return 1
        return int(
            args.output_json.read_text(encoding="utf-8") != expected_json
            or args.output_markdown.read_text(encoding="utf-8") != expected_markdown
        )
    write_capability_matrix(report, args.output_json, args.output_markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
