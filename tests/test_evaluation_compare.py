"""Tests for simulation CSV comparison evidence."""

from pathlib import Path

import pytest

from evaluation.compare_runs import compare_csv_runs


def _write_csv(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_compare_csv_runs_reports_raw_errors_without_threshold(tmp_path: Path) -> None:
    reference = tmp_path / "pysd.csv"
    candidate = tmp_path / "stella.csv"
    _write_csv(reference, "time,Accumulator\n0,0\n1,1\n2,2\n")
    _write_csv(candidate, "Time,Stock\n0,0\n1,1.5\n2,2\n")

    result = compare_csv_runs(
        reference,
        candidate,
        [("Accumulator", "Stock")],
        candidate_time="Time",
    )

    assert result["points"] == 3
    assert result["schema_version"] == 2
    assert result["comparison_policy"]["time_alignment"] == "exact"
    assert result["comparison_policy"]["interpolation"] == "none"
    assert result["comparison_policy"]["pass_threshold"] is None
    assert result["columns"] == [
        {
            "reference_column": "Accumulator",
            "candidate_column": "Stock",
            "max_absolute_error": 0.5,
            "max_absolute_error_time": 1.0,
            "max_relative_error": 1 / 3,
            "max_relative_error_time": 1.0,
        }
    ]


def test_compare_csv_runs_rejects_different_time_grids(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(reference, "time,value\n0,1\n1,2\n")
    _write_csv(candidate, "time,value\n0,1\n2,2\n")

    with pytest.raises(ValueError, match="Time grids differ"):
        compare_csv_runs(reference, candidate, [("value", "value")])


@pytest.mark.parametrize(
    ("candidate_rows", "message"),
    [
        ("0,1\n", "different lengths"),
        ("0,1\n2,2\n1,3\n", "not strictly increasing"),
        ("0,1\nnot-a-time,2\n", "Non-numeric value"),
        ("0,1\n1,nan\n", "Non-finite value"),
    ],
)
def test_compare_csv_runs_rejects_invalid_grid_or_values(
    tmp_path: Path, candidate_rows: str, message: str
) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(reference, "time,value\n0,1\n1,2\n2,3\n")
    _write_csv(candidate, f"time,value\n{candidate_rows}")

    with pytest.raises(ValueError, match=message):
        compare_csv_runs(reference, candidate, [("value", "value")])


def test_compare_csv_runs_accepts_declared_rounded_reference_labels(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(reference, "time,value\n0.0,1\n0.03125,2\n0.0625,3\n")
    _write_csv(candidate, "Time,value\n0.000,1\n0.031,2\n0.063,3\n")

    result = compare_csv_runs(
        reference,
        candidate,
        [("value", "value")],
        candidate_time="Time",
        time_alignment="rounded_reference_labels",
        candidate_decimal_places=3,
    )

    assert result["comparison_policy"] == {
        "time_alignment": "rounded_reference_labels",
        "candidate_decimal_places": 3,
        "interpolation": "none",
        "max_time_label_difference": 0.0005,
        "relative_error_denominator": "max(abs(reference), abs(candidate))",
        "pass_threshold": None,
    }


@pytest.mark.parametrize(
    ("candidate_times", "message"),
    [
        (["0.00", "0.03", "0.06"], "does not have 3 decimal places"),
        (["0.000", "0.032", "0.063"], "Rounded time grids differ"),
        (["0.000", "0.031", "0.031"], "not strictly increasing"),
    ],
)
def test_compare_csv_runs_rejects_invalid_rounded_time_labels(
    tmp_path: Path, candidate_times: list[str], message: str
) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(reference, "time,value\n0.0,1\n0.03125,2\n0.0625,3\n")
    rows = "\n".join(f"{time},1" for time in candidate_times)
    _write_csv(candidate, f"Time,value\n{rows}\n")

    with pytest.raises(ValueError, match=message):
        compare_csv_runs(
            reference,
            candidate,
            [("value", "value")],
            candidate_time="Time",
            time_alignment="rounded_reference_labels",
            candidate_decimal_places=3,
        )
