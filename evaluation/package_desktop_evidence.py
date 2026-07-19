"""Validate local Stella desktop artifacts and generate package parity evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from stella_mcp.xmile import StellaModel, parse_stmx

from .desktop_parity import generate_desktop_parity_report
from .model_fidelity import structured_diff

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = "0.13.0"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"
DEFAULT_OBSERVATIONS = Path(__file__).with_name("desktop_observations_0.13.0.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_record(path: Path) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "sha256": _sha256(path),
    }


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return document


_QUOTED_IDENTIFIER = re.compile(r'"([^"\\]+)"')


def _graphical_function(value: Any, *, effective: bool) -> dict[str, Any] | None:
    if value is None:
        return None
    document = asdict(value)
    if effective and document["gf_type"] is None:
        document["gf_type"] = "continuous"
    return document


def _equation(model: StellaModel, value: str, *, effective: bool) -> str:
    if not effective:
        return value
    variable_keys = model.stocks.keys() | model.flows.keys() | model.auxs.keys()

    def replace_identifier(match: re.Match[str]) -> str:
        normalized = model._normalize_name(match.group(1))
        return normalized if normalized in variable_keys else match.group(0)

    return _QUOTED_IDENTIFIER.sub(replace_identifier, value)


def _computational_signature(
    model: StellaModel,
    *,
    effective: bool,
) -> dict[str, Any]:
    return {
        "sim_specs": {
            "start": model.sim_specs.start,
            "stop": model.sim_specs.stop,
            "dt": model.sim_specs.dt,
            "method": model.sim_specs.method,
            "time_units": model.sim_specs.time_units,
        },
        "stocks": {
            key: {
                "name": stock.name,
                "initial_value": _equation(model, stock.initial_value, effective=effective),
                "units": stock.units,
                "inflows": sorted(stock.inflows),
                "outflows": sorted(stock.outflows),
                "non_negative": stock.non_negative,
            }
            for key, stock in sorted(model.stocks.items())
        },
        "flows": {
            key: {
                "name": flow.name,
                "equation": _equation(model, flow.equation, effective=effective),
                "units": flow.units,
                "from_stock": flow.from_stock,
                "to_stock": flow.to_stock,
                "non_negative": flow.non_negative,
                "graphical_function": _graphical_function(
                    flow.graphical_function, effective=effective
                ),
            }
            for key, flow in sorted(model.flows.items())
        },
        "auxiliaries": {
            key: {
                "name": auxiliary.name,
                "equation": _equation(model, auxiliary.equation, effective=effective),
                "units": auxiliary.units,
                "graphical_function": _graphical_function(
                    auxiliary.graphical_function, effective=effective
                ),
            }
            for key, auxiliary in sorted(model.auxs.items())
        },
    }


def _validate_observations(document: dict[str, Any], release: str) -> dict[str, Any]:
    if document.get("schema_version") != 1 or document.get("release") != release:
        raise ValueError("Unexpected desktop observations schema or release")
    for field in ("application", "application_version", "export_date"):
        if not isinstance(document.get(field), str) or not document[field]:
            raise ValueError(f"Desktop observations require {field}")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Desktop observations require cases")
    by_id = {}
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in by_id:
            raise ValueError("Desktop observation ids must be non-empty and unique")
        for field in ("opened_without_repair", "run_completed"):
            if not isinstance(case.get(field), bool):
                raise ValueError(f"Desktop observation {case_id} requires boolean {field}")
        final_time = case.get("run_final_time")
        if isinstance(final_time, bool) or not isinstance(final_time, (int, float)):
            raise ValueError(f"Desktop observation {case_id} requires numeric run_final_time")
        if case.get("visual_review") not in {"pass", "fail"}:
            raise ValueError(f"Desktop observation {case_id} has invalid visual_review")
        if not isinstance(case.get("notes"), str) or not case["notes"]:
            raise ValueError(f"Desktop observation {case_id} requires notes")
        by_id[case_id] = case
    return by_id


def render_desktop_evidence_markdown(report: dict[str, Any]) -> str:
    """Render the validated desktop acceptance manifest as Markdown."""
    lines = [
        f"# Stella MCP {report['release']} Desktop Acceptance",
        "",
        f"- Application: {report['application']} {report['application_version']}",
        f"- Export date: {report['export_date']}",
        "- CSV policy: all model variables at every saved model time; no interpolation",
        "",
        "| Case | Open | Run | Final time | Strict re-import | Identifier renames | "
        "Computational changes | Serialization rewrites | Visual |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for case in report["cases"]:
        renames = json.dumps(case["identifier_renames"], sort_keys=True)
        lines.append(
            f"| `{case['id']}` | {_yes_no(case['opened_without_repair'])} | "
            f"{_yes_no(case['run_completed'])} | {case['run_final_time']} | "
            f"{_yes_no(case['saved_reimport_strict'])} | `{renames}` | "
            f"`{json.dumps(case['computational_changes'], sort_keys=True)}` | "
            f"{len(case['serialization_changes'])} | "
            f"{case['visual_review']} |"
        )
    lines.extend(["", "## Operator Notes", ""])
    for case in report["cases"]:
        lines.append(f"- `{case['id']}`: {case['notes']}")
    lines.extend(["", "## Serialization Review", ""])
    changed_cases = [case for case in report["cases"] if case["serialization_changes"]]
    if not changed_cases:
        lines.append("No Stella serialization rewrites were observed.")
    for case in changed_cases:
        for change in case["serialization_changes"]:
            lines.append(
                f"- `{case['id']}{change['path']}`: "
                f"`{json.dumps(change['before'])}` -> `{json.dumps(change['after'])}`. "
                "The effective computational signature is unchanged."
            )
    return "\n".join(lines)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def generate_package_desktop_evidence(
    *,
    release: str = DEFAULT_RELEASE,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    observations_path: Path = DEFAULT_OBSERVATIONS,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate desktop files and write per-case numeric parity reports."""
    results_dir = results_dir.resolve()
    output_dir = (output_dir or results_dir).resolve()
    candidate_manifest = _load_json(results_dir / f"{release}-desktop-candidates.json")
    if candidate_manifest.get("schema_version") != 1:
        raise ValueError("Unexpected desktop candidate manifest schema")
    observations = _load_json(observations_path)
    observations_by_id = _validate_observations(observations, release)
    candidate_ids = {case["id"] for case in candidate_manifest["cases"]}
    if set(observations_by_id) != candidate_ids:
        raise ValueError("Desktop observations must exactly cover candidate ids")

    evidence_dir = results_dir / f"{release}-desktop-evidence"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for case in candidate_manifest["cases"]:
        case_id = case["id"]
        observation = observations_by_id[case_id]
        model_path = results_dir / case["model"]["path"]
        if _sha256(model_path) != case["model"]["sha256"]:
            raise ValueError(f"Candidate model hash mismatch: {case_id}")
        stella_csv = evidence_dir / f"{case_id}-stella.csv"
        stella_saved = evidence_dir / f"{case_id}-stella-saved.stmx"
        if not stella_csv.is_file() or stella_csv.stat().st_size == 0:
            raise FileNotFoundError(f"Missing Stella CSV: {stella_csv}")
        if not stella_saved.is_file() or stella_saved.stat().st_size == 0:
            raise FileNotFoundError(f"Missing Stella-saved model: {stella_saved}")

        candidate_model = parse_stmx(str(model_path), compat_mode="strict")
        saved_model = parse_stmx(str(stella_saved), compat_mode="strict")
        candidate_names = {
            kind: sorted(registry)
            for kind, registry in (
                ("stocks", candidate_model.stocks),
                ("flows", candidate_model.flows),
                ("auxiliaries", candidate_model.auxs),
            )
        }
        saved_names = {
            kind: sorted(registry)
            for kind, registry in (
                ("stocks", saved_model.stocks),
                ("flows", saved_model.flows),
                ("auxiliaries", saved_model.auxs),
            )
        }
        identifier_renames = structured_diff(candidate_names, saved_names)
        serialization_changes = structured_diff(
            _computational_signature(candidate_model, effective=False),
            _computational_signature(saved_model, effective=False),
        )
        computational_changes = structured_diff(
            _computational_signature(candidate_model, effective=True),
            _computational_signature(saved_model, effective=True),
        )
        if identifier_renames:
            raise ValueError(f"Stella renamed identifiers in {case_id}: {identifier_renames}")
        if computational_changes:
            raise ValueError(
                f"Stella changed computational semantics in {case_id}: {computational_changes}"
            )
        if observation["run_final_time"] != case["simulation"]["sim_specs"]["stop"]:
            raise ValueError(f"Observed final time does not match candidate: {case_id}")

        pysd_output = output_dir / f"{release}-{case_id}-pysd.csv"
        parity_path = output_dir / f"{release}-{case_id}-parity.json"
        parity = generate_desktop_parity_report(
            model_path,
            stella_csv,
            pysd_output,
            [(column["pysd"], column["stella"]) for column in case["columns"]],
            stella_application=observations["application"],
            stella_version=observations["application_version"],
            pysd_time=case["time_columns"]["pysd"],
            stella_time=case["time_columns"]["stella"],
        )
        record = {
            **observation,
            "source_model": _artifact_record(model_path),
            "stella_csv": _artifact_record(stella_csv),
            "stella_saved_model": _artifact_record(stella_saved),
            "saved_reimport_strict": True,
            "identifier_renames": identifier_renames,
            "serialization_changes": serialization_changes,
            "computational_changes": computational_changes,
            "parity_report": _portable_path(parity_path),
        }
        parity["desktop_evidence"] = record
        parity_path.write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")
        records.append(record)

    report = {
        "schema_version": 1,
        "release": release,
        "application": observations["application"],
        "application_version": observations["application_version"],
        "export_date": observations["export_date"],
        "cases": records,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = (args.output_dir or args.results_dir).resolve()
    report = generate_package_desktop_evidence(
        release=args.release,
        results_dir=args.results_dir,
        observations_path=args.observations,
        output_dir=output_dir,
    )
    manifest_path = output_dir / f"{args.release}-desktop-evidence.json"
    markdown_path = PROJECT_ROOT / f"docs/evaluation/{args.release}-desktop-acceptance.md"
    manifest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_desktop_evidence_markdown(report) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
