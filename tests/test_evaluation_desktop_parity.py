"""Tests for reproducible Stella-to-PySD comparison evidence."""

import hashlib
from pathlib import Path
from typing import Any

import pytest

from evaluation import desktop_parity


def test_generate_desktop_parity_report_records_reproducibility_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model.stmx"
    stella_csv = tmp_path / "stella.csv"
    pysd_csv = tmp_path / "pysd.csv"
    model_path.write_text("<xmile />\n", encoding="utf-8")
    stella_csv.write_text("Years,Stock\n0,0\n1,1\n", encoding="utf-8")

    def fake_parse(filepath: str, compat_mode: str) -> object:
        assert filepath == str(model_path)
        assert compat_mode == "strict"
        return object()

    def fake_run(model: object, *, save_results_csv: str) -> dict[str, Any]:
        assert save_results_csv == str(pysd_csv)
        Path(save_results_csv).write_text("time,Stock\n0,0\n1,1\n", encoding="utf-8")
        return {
            "backend": {
                "name": "PySD",
                "version": "3.14.3",
                "actual_integration_method": "Euler",
            },
            "sim_specs": {
                "start": 0,
                "stop": 1,
                "dt": 1,
                "method": "Euler",
                "time_units": "Years",
            },
            "warnings": [],
        }

    monkeypatch.setattr(desktop_parity, "parse_stmx", fake_parse)
    monkeypatch.setattr(desktop_parity, "run_simulation", fake_run)
    monkeypatch.setattr(desktop_parity, "package_version", lambda name: "3.14.3")

    report = desktop_parity.generate_desktop_parity_report(
        model_path,
        stella_csv,
        pysd_csv,
        [("Stock", "Stock")],
        stella_version="4.1.1",
        stella_time="Years",
    )

    assert report["schema_version"] == 2
    assert report["engines"] == {
        "pysd": {"version": "3.14.3"},
        "stella": {"application": "Stella Professional", "version": "4.1.1"},
    }
    assert report["artifacts"]["model"] == {
        "path": "model.stmx",
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    assert report["artifacts"]["stella_csv"]["path"] == "stella.csv"
    assert report["artifacts"]["pysd_csv"]["path"] == "pysd.csv"
    assert report["simulation"]["warnings"] == []
    assert report["simulation"]["backend"]["name"] == "PySD"
    assert report["comparison"]["points"] == 2
    assert report["comparison"]["comparison_policy"]["pass_threshold"] is None
    assert report["comparison"]["columns"][0]["max_absolute_error"] == 0.0


def test_generate_desktop_parity_report_requires_stella_provenance(tmp_path: Path) -> None:
    stella_csv = tmp_path / "stella.csv"
    stella_csv.write_text("time,value\n0,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="stella_version"):
        desktop_parity.generate_desktop_parity_report(
            tmp_path / "model.stmx",
            stella_csv,
            tmp_path / "pysd.csv",
            [("value", "value")],
            stella_version=" ",
        )


def test_generate_desktop_parity_report_protects_stella_export(tmp_path: Path) -> None:
    model_path = tmp_path / "model.stmx"
    stella_csv = tmp_path / "stella.csv"
    model_path.write_text("<xmile />\n", encoding="utf-8")
    stella_csv.write_text("time,value\n0,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not overwrite"):
        desktop_parity.generate_desktop_parity_report(
            model_path,
            stella_csv,
            stella_csv,
            [("value", "value")],
            stella_version="4.1.1",
        )
