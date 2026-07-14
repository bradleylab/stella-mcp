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
