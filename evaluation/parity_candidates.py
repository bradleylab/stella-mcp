"""Generate reproducible package models for Stella desktop parity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stella_mcp.simulate import run_simulation
from stella_mcp.templates import load_template_model
from stella_mcp.xmile import GraphicalFunction, StellaModel, parse_stmx

from .model_fidelity import compare_model_fidelity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LABEL_PATTERN = re.compile(r"[0-9A-Za-z][0-9A-Za-z._-]*")
_GF_UUID_NAMESPACE = "https://github.com/bradleylab/stella-mcp/parity/scalar-gf"
_CANDIDATE_UUID_BASE = "https://github.com/bradleylab/stella-mcp/parity/candidate/"

_BUILTIN_CASES = {
    "package_carbon_cycle_2box": ("carbon_cycle_2box", ["scalar", "two_stock"]),
    "package_exponential_growth": ("exponential_growth", ["scalar", "feedback"]),
    "package_lotka_volterra": ("lotka_volterra", ["scalar", "nonlinear"]),
    "package_nutrient_box_2box": ("nutrient_box_2box", ["scalar", "two_stock"]),
    "package_sir": ("sir", ["scalar", "nonlinear"]),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_scalar_graphical_function() -> StellaModel:
    """Build the fixed scalar graphical-function desktop parity case."""
    model = StellaModel("Scalar Graphical Function")
    model.uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, _GF_UUID_NAMESPACE))
    model.set_sim_specs(start=0, stop=10, dt=1, method="Euler", time_units="Years")
    model.add_stock("Accumulated response", "0", x=430, y=250)
    model.add_aux(
        "seasonal multiplier",
        "GRAPH(TIME)",
        x=130,
        y=120,
        graphical_function=GraphicalFunction(
            xpts=[0.0, 5.0, 10.0],
            ypts=[0.0, 1.0, 0.0],
            yscale=(0.0, 1.0),
            gf_type="continuous",
        ),
    )
    model.add_flow(
        "response input",
        '"seasonal multiplier"',
        to_stock="Accumulated response",
        x=280,
        y=250,
    )
    model.sync_connectors_from_equations()
    return model


def _case_builders() -> dict[str, tuple[Callable[[], StellaModel], list[str], str]]:
    cases: dict[str, tuple[Callable[[], StellaModel], list[str], str]] = {}
    for case_id, (template_name, constructs) in _BUILTIN_CASES.items():
        cases[case_id] = (
            lambda name=template_name: load_template_model(name)[1],
            constructs,
            f"stella_mcp/builtin_templates/{template_name}.stmx",
        )
    cases["package_scalar_graphical_function"] = (
        build_scalar_graphical_function,
        ["scalar", "graphical_function"],
        "evaluation.parity_candidates:build_scalar_graphical_function",
    )
    return cases


def _variable_names(model: StellaModel) -> list[str]:
    return [
        variable.name
        for registry in (model.stocks, model.flows, model.auxs)
        for _, variable in sorted(registry.items())
    ]


def generate_parity_candidates(output_dir: Path, *, version_label: str) -> dict[str, Any]:
    """Write candidate STMX/PySD files and a deterministic manifest."""
    if _LABEL_PATTERN.fullmatch(version_label) is None:
        raise ValueError("version_label must contain only letters, numbers, dot, dash, underscore")

    output_dir = output_dir.resolve()
    candidate_dir = output_dir / f"{version_label}-desktop-candidates"
    if candidate_dir.exists():
        raise FileExistsError(f"Candidate directory already exists: {candidate_dir}")
    candidate_dir.mkdir(parents=True)

    records = []
    for case_id, (builder, constructs, source) in sorted(_case_builders().items()):
        model = builder()
        model.uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, _CANDIDATE_UUID_BASE + case_id))
        model_path = candidate_dir / f"{case_id}.stmx"
        pysd_path = candidate_dir / f"{case_id}-pysd.csv"
        model_path.write_text(
            model.to_xml(auto_layout=False, compat_mode="strict"),
            encoding="utf-8",
        )
        imported = parse_stmx(str(model_path), compat_mode="strict")
        with tempfile.TemporaryDirectory(dir=candidate_dir) as temporary_dir:
            roundtrip_path = Path(temporary_dir) / "roundtrip.stmx"
            roundtrip_path.write_text(
                imported.to_xml(auto_layout=False, compat_mode="strict"),
                encoding="utf-8",
            )
            roundtripped = parse_stmx(str(roundtrip_path), compat_mode="strict")
        fidelity = compare_model_fidelity(imported, roundtripped)
        if fidelity["semantic_changes"]:
            raise RuntimeError(f"Candidate {case_id} changed semantics during strict round-trip")
        simulation = run_simulation(imported, save_results_csv=str(pysd_path))
        variables = _variable_names(imported)
        records.append(
            {
                "id": case_id,
                "source": source,
                "constructs": constructs,
                "model": {
                    "path": model_path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256(model_path),
                },
                "pysd_csv": {
                    "path": pysd_path.relative_to(output_dir).as_posix(),
                    "sha256": _sha256(pysd_path),
                },
                "time_columns": {
                    "pysd": "time",
                    "stella": imported.sim_specs.time_units,
                },
                "columns": [
                    {"pysd": variable, "stella": variable} for variable in variables
                ],
                "simulation": {
                    "backend": simulation["backend"],
                    "sim_specs": simulation["sim_specs"],
                    "warnings": simulation["warnings"],
                },
                "semantic_diff": fidelity["semantic_changes"],
                "uuid_changed": bool(fidelity["metadata_changes"]),
            }
        )

    manifest = {
        "schema_version": 1,
        "release": version_label,
        "generator": "evaluation.parity_candidates",
        "cases": records,
    }
    manifest_path = output_dir / f"{version_label}-desktop-candidates.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/evaluation")
    parser.add_argument("--version-label", required=True)
    args = parser.parse_args()
    generate_parity_candidates(args.output_dir, version_label=args.version_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
