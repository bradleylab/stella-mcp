"""Generate reproducible Stella-to-PySD numeric comparison evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from stella_mcp.simulate import run_simulation
from stella_mcp.xmile import parse_stmx

from .compare_runs import compare_csv_runs

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def generate_desktop_parity_report(
    model_path: Path,
    stella_csv_path: Path,
    pysd_csv_path: Path,
    column_pairs: list[tuple[str, str]],
    *,
    stella_version: str,
    stella_application: str = "Stella Professional",
    pysd_time: str = "time",
    stella_time: str = "time",
    time_alignment: str = "exact",
    candidate_decimal_places: int | None = None,
) -> dict[str, Any]:
    """Run PySD and compare its CSV with an existing Stella export.

    Parameters
    ----------
    model_path : Path
        XMILE model used for the PySD run.
    stella_csv_path : Path
        Existing CSV exported from Stella.
    pysd_csv_path : Path
        Destination for the generated PySD CSV.
    column_pairs : list of tuple of str
        Explicit ``(PySD, Stella)`` column mappings.
    stella_version : str
        Version of Stella that produced ``stella_csv_path``.
    pysd_time : str
        Time-column name in the generated PySD CSV.
    stella_time : str
        Time-column name in the Stella CSV.
    """
    model_path = model_path.resolve()
    stella_csv_path = stella_csv_path.resolve()
    pysd_csv_path = pysd_csv_path.resolve()

    if not stella_version.strip():
        raise ValueError("stella_version must be non-empty")
    if not stella_application.strip():
        raise ValueError("stella_application must be non-empty")
    if not stella_csv_path.is_file():
        raise FileNotFoundError(f"Stella CSV does not exist: {stella_csv_path}")
    if pysd_csv_path in {model_path, stella_csv_path}:
        raise ValueError("PySD output must not overwrite the model or Stella CSV")

    pysd_csv_path.parent.mkdir(parents=True, exist_ok=True)
    model = parse_stmx(str(model_path), compat_mode="strict")
    simulation = run_simulation(model, save_results_csv=str(pysd_csv_path))
    comparison = compare_csv_runs(
        pysd_csv_path,
        stella_csv_path,
        column_pairs,
        reference_time=pysd_time,
        candidate_time=stella_time,
        time_alignment=time_alignment,
        candidate_decimal_places=candidate_decimal_places,
    )
    comparison["reference"] = _portable_path(pysd_csv_path)
    comparison["candidate"] = _portable_path(stella_csv_path)

    return {
        "schema_version": 2,
        "engines": {
            "pysd": {"version": package_version("pysd")},
            "stella": {
                "application": stella_application,
                "version": stella_version,
            },
        },
        "artifacts": {
            "model": {
                "path": _portable_path(model_path),
                "sha256": _sha256(model_path),
            },
            "pysd_csv": {
                "path": _portable_path(pysd_csv_path),
                "sha256": _sha256(pysd_csv_path),
            },
            "stella_csv": {
                "path": _portable_path(stella_csv_path),
                "sha256": _sha256(stella_csv_path),
            },
        },
        "simulation": {
            "backend": simulation["backend"],
            "sim_specs": simulation["sim_specs"],
            "warnings": simulation["warnings"],
        },
        "comparison": comparison,
    }


def _parse_column_pair(value: str) -> tuple[str, str]:
    pysd_column, separator, stella_column = value.partition("=")
    if not separator or not pysd_column or not stella_column:
        raise argparse.ArgumentTypeError("column mappings must be PYSD=STELLA")
    return pysd_column, stella_column


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("stella_csv", type=Path)
    parser.add_argument("--pysd-output", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--stella-version", required=True)
    parser.add_argument("--stella-application", default="Stella Professional")
    parser.add_argument("--column", action="append", type=_parse_column_pair, required=True)
    parser.add_argument("--pysd-time", default="time")
    parser.add_argument("--stella-time", default="time")
    parser.add_argument(
        "--time-alignment",
        choices=["exact", "rounded_reference_labels"],
        default="exact",
    )
    parser.add_argument("--candidate-decimal-places", type=int)
    args = parser.parse_args()

    output_json = args.output_json.resolve()
    protected_paths = {args.model.resolve(), args.stella_csv.resolve(), args.pysd_output.resolve()}
    if output_json in protected_paths:
        parser.error("JSON output must not overwrite an input or the PySD CSV")

    report = generate_desktop_parity_report(
        args.model,
        args.stella_csv,
        args.pysd_output,
        args.column,
        stella_version=args.stella_version,
        stella_application=args.stella_application,
        pysd_time=args.pysd_time,
        stella_time=args.stella_time,
        time_alignment=args.time_alignment,
        candidate_decimal_places=args.candidate_decimal_places,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
