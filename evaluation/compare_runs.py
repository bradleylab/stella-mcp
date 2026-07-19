"""Compare two simulation CSV files without interpolation or a pass threshold."""

from __future__ import annotations

import argparse
import csv
import json
import math
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any


def _read_columns(
    path: Path, columns: list[str]
) -> tuple[dict[str, list[float]], dict[str, list[str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(f"CSV has duplicate column names: {path}")
        missing = [column for column in columns if column not in fieldnames]
        if missing:
            raise ValueError(f"Missing columns in {path}: {', '.join(missing)}")

        values = {column: [] for column in columns}
        raw_values = {column: [] for column in columns}
        for row_number, row in enumerate(reader, start=2):
            for column in columns:
                raw = row[column]
                try:
                    value = float(raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Non-numeric value in {path}, row {row_number}, column {column}: {raw!r}"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite value in {path}, row {row_number}, column {column}: {raw!r}"
                    )
                values[column].append(value)
                raw_values[column].append(raw)

    if not values[columns[0]]:
        raise ValueError(f"CSV has no data rows: {path}")
    return values, raw_values


def _check_strictly_increasing(values: list[float], label: str) -> None:
    for index, (previous, current) in enumerate(zip(values, values[1:], strict=False), start=2):
        if current <= previous:
            raise ValueError(
                f"{label} time grid is not strictly increasing at data row {index}: "
                f"{previous} then {current}"
            )


def _align_time_grids(
    reference_values: list[float],
    candidate_values: list[float],
    reference_raw: list[str],
    candidate_raw: list[str],
    *,
    policy: str,
    candidate_decimal_places: int | None,
) -> float:
    if len(reference_values) != len(candidate_values):
        raise ValueError(
            "Time grids have different lengths: "
            f"{len(reference_values)} reference rows, {len(candidate_values)} candidate rows"
        )
    _check_strictly_increasing(reference_values, "Reference")
    _check_strictly_increasing(candidate_values, "Candidate")

    if policy == "exact":
        for index, (reference_value, candidate_value) in enumerate(
            zip(reference_values, candidate_values, strict=True)
        ):
            if reference_value != candidate_value:
                raise ValueError(
                    "Time grids differ at data row "
                    f"{index + 1}: {reference_value} != {candidate_value}"
                )
        return max(
            abs(reference_value - candidate_value)
            for reference_value, candidate_value in zip(
                reference_values, candidate_values, strict=True
            )
        )

    if policy != "rounded_reference_labels":
        raise ValueError(f"Unknown time alignment policy: {policy}")
    if (
        not isinstance(candidate_decimal_places, int)
        or isinstance(candidate_decimal_places, bool)
        or candidate_decimal_places < 0
    ):
        raise ValueError(
            "rounded_reference_labels requires candidate_decimal_places >= 0"
        )

    quantum = Decimal(1).scaleb(-candidate_decimal_places)
    max_difference = Decimal(0)
    for index, (reference_text, candidate_text) in enumerate(
        zip(reference_raw, candidate_raw, strict=True), start=1
    ):
        try:
            reference_decimal = Decimal(reference_text)
            candidate_decimal = Decimal(candidate_text)
        except InvalidOperation as exc:
            raise ValueError(f"Invalid decimal time label at data row {index}") from exc
        if not reference_decimal.is_finite() or not candidate_decimal.is_finite():
            raise ValueError(f"Non-finite decimal time label at data row {index}")
        if candidate_decimal.as_tuple().exponent != -candidate_decimal_places:
            raise ValueError(
                f"Candidate time label at data row {index} does not have "
                f"{candidate_decimal_places} decimal places: {candidate_text!r}"
            )
        expected = reference_decimal.quantize(quantum, rounding=ROUND_HALF_UP)
        if candidate_decimal != expected:
            raise ValueError(
                "Rounded time grids differ at data row "
                f"{index}: expected {expected}, got {candidate_decimal}"
            )
        max_difference = max(max_difference, abs(reference_decimal - candidate_decimal))
    return float(max_difference)


def compare_csv_runs(
    reference_path: Path,
    candidate_path: Path,
    column_pairs: list[tuple[str, str]],
    *,
    reference_time: str = "time",
    candidate_time: str = "time",
    time_alignment: str = "exact",
    candidate_decimal_places: int | None = None,
) -> dict[str, Any]:
    """Return raw discrepancy metrics for runs on an identical time grid.

    Relative error uses ``abs(reference - candidate) / max(abs(reference),
    abs(candidate))``. A pair of zero values has zero relative error. The
    function deliberately does not interpolate or apply a tolerance.
    """
    if not column_pairs:
        raise ValueError("At least one column pair is required")

    reference_columns = [reference_time, *(pair[0] for pair in column_pairs)]
    candidate_columns = [candidate_time, *(pair[1] for pair in column_pairs)]
    reference, reference_raw = _read_columns(reference_path, reference_columns)
    candidate, candidate_raw = _read_columns(candidate_path, candidate_columns)
    reference_times = reference[reference_time]
    candidate_times = candidate[candidate_time]
    max_time_label_difference = _align_time_grids(
        reference_times,
        candidate_times,
        reference_raw[reference_time],
        candidate_raw[candidate_time],
        policy=time_alignment,
        candidate_decimal_places=candidate_decimal_places,
    )

    comparisons = []
    for reference_name, candidate_name in column_pairs:
        rows = []
        for time_value, reference_value, candidate_value in zip(
            reference_times,
            reference[reference_name],
            candidate[candidate_name],
            strict=True,
        ):
            absolute_error = abs(reference_value - candidate_value)
            scale = max(abs(reference_value), abs(candidate_value))
            relative_error = 0.0 if scale == 0 else absolute_error / scale
            rows.append(
                {
                    "time": time_value,
                    "reference": reference_value,
                    "candidate": candidate_value,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                }
            )

        max_absolute = max(rows, key=lambda row: row["absolute_error"])
        max_relative = max(rows, key=lambda row: row["relative_error"])
        comparisons.append(
            {
                "reference_column": reference_name,
                "candidate_column": candidate_name,
                "max_absolute_error": max_absolute["absolute_error"],
                "max_absolute_error_time": max_absolute["time"],
                "max_relative_error": max_relative["relative_error"],
                "max_relative_error_time": max_relative["time"],
            }
        )

    return {
        "schema_version": 2,
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "time_columns": {"reference": reference_time, "candidate": candidate_time},
        "points": len(reference_times),
        "comparison_policy": {
            "time_alignment": time_alignment,
            "candidate_decimal_places": candidate_decimal_places,
            "interpolation": "none",
            "max_time_label_difference": max_time_label_difference,
            "relative_error_denominator": "max(abs(reference), abs(candidate))",
            "pass_threshold": None,
        },
        "columns": comparisons,
    }


def _parse_pair(value: str) -> tuple[str, str]:
    reference, separator, candidate = value.partition("=")
    if not separator or not reference or not candidate:
        raise argparse.ArgumentTypeError("column mappings must be REFERENCE=CANDIDATE")
    return reference, candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--column", action="append", type=_parse_pair, required=True)
    parser.add_argument("--reference-time", default="time")
    parser.add_argument("--candidate-time", default="time")
    parser.add_argument(
        "--time-alignment",
        choices=["exact", "rounded_reference_labels"],
        default="exact",
    )
    parser.add_argument("--candidate-decimal-places", type=int)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    result = compare_csv_runs(
        args.reference,
        args.candidate,
        args.column,
        reference_time=args.reference_time,
        candidate_time=args.candidate_time,
        time_alignment=args.time_alignment,
        candidate_decimal_places=args.candidate_decimal_places,
    )
    output = json.dumps(result, indent=2) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
