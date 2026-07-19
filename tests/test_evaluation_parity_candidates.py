"""Tests for deterministic package-generated desktop parity candidates."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from evaluation.parity_candidates import generate_parity_candidates
from stella_mcp.xmile import parse_stmx


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generate_parity_candidates_is_complete_and_reproducible(tmp_path: Path) -> None:
    pytest.importorskip("pysd")
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = generate_parity_candidates(first_dir, version_label="0.13.0")
    second = generate_parity_candidates(second_dir, version_label="0.13.0")

    assert first == second
    assert first["schema_version"] == 1
    assert {case["id"] for case in first["cases"]} == {
        "package_carbon_cycle_2box",
        "package_exponential_growth",
        "package_lotka_volterra",
        "package_nutrient_box_2box",
        "package_scalar_graphical_function",
        "package_sir",
    }
    for case in first["cases"]:
        model_path = first_dir / case["model"]["path"]
        pysd_path = first_dir / case["pysd_csv"]["path"]
        assert case["model"]["sha256"] == _sha256(model_path)
        assert case["pysd_csv"]["sha256"] == _sha256(pysd_path)
        assert case["columns"]
        assert case["semantic_diff"] == []
        parse_stmx(str(model_path), compat_mode="strict")


def test_generate_parity_candidates_rejects_existing_destination(tmp_path: Path) -> None:
    pytest.importorskip("pysd")
    generate_parity_candidates(tmp_path, version_label="0.13.0")

    with pytest.raises(FileExistsError, match="already exists"):
        generate_parity_candidates(tmp_path, version_label="0.13.0")
