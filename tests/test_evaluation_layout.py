"""Tests for the reproducible layout evaluation harness."""

import json
import math
from pathlib import Path

from evaluation.layout_fixtures import fixture_builders, template_models
from evaluation.layout_runner import _canonicalize_report_value, run_layout_evaluation
from stella_mcp.layout_quality import (
    ROUTE_BEND_CAP,
    ROUTE_LENGTH_MANHATTAN_MULTIPLIER,
    analyze_layout,
)
from stella_mcp.validator import validate_model

BASELINE_REPORT = (
    Path(__file__).parents[1]
    / "docs"
    / "evaluation"
    / "layout-baseline-0.12"
    / "layout-report.json"
)
RELEASE_REPORT = (
    Path(__file__).parents[1]
    / "docs"
    / "evaluation"
    / "layout-0.13"
    / "layout-report.json"
)


def test_route_limits_are_derived_from_the_phase_one_baseline():
    baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    routes = [
        points
        for case in baseline["cases"].values()
        for points in case["routes"].values()
        if len(points) >= 2
    ]
    finite_ratios = []
    for points in routes:
        manhattan = abs(points[-1][0] - points[0][0]) + abs(
            points[-1][1] - points[0][1]
        )
        if manhattan:
            length = sum(
                math.dist(start, end)
                for start, end in zip(points, points[1:], strict=False)
            )
            finite_ratios.append(length / manhattan)

    baseline_bend_maximum = max(len(points) - 2 for points in routes)
    baseline_ratio_quarter_ceiling = math.ceil(max(finite_ratios) * 4) / 4

    assert baseline_bend_maximum == 2
    assert ROUTE_BEND_CAP == baseline_bend_maximum + 2
    assert ROUTE_LENGTH_MANHATTAN_MULTIPLIER == baseline_ratio_quarter_ceiling == 4.5


def test_layout_fixture_catalog_covers_specified_topologies():
    assert set(fixture_builders()) == {
        "chain",
        "dense_planar",
        "disconnected",
        "fanout",
        "feedback",
        "long_labels",
        "mixed_pins",
        "nonplanar",
        "special_flows",
    }
    assert set(template_models()) == {
        "template_carbon_cycle_2box",
        "template_exponential_growth",
        "template_lotka_volterra",
        "template_nutrient_box_2box",
        "template_sir",
    }


def test_every_fixture_can_be_laid_out_and_analyzed():
    models = template_models()
    models.update({name: builder() for name, builder in fixture_builders().items()})

    for name, model in models.items():
        assert not [error for error in validate_model(model) if error.severity == "error"], name
        model._auto_layout()
        assert analyze_layout(model).missing_positions == (), name


def test_layout_evaluation_is_deterministic_and_writes_artifacts(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = run_layout_evaluation(first_dir)
    second = run_layout_evaluation(second_dir)

    assert first == second
    assert json.loads(RELEASE_REPORT.read_text(encoding="utf-8")) == first
    assert json.loads((first_dir / "layout-report.json").read_text()) == first
    assert {
        path.name: path.read_bytes() for path in sorted(first_dir.glob("*.stmx"))
    } == {path.name: path.read_bytes() for path in sorted(second_dir.glob("*.stmx"))}
    assert (first_dir / "fanout.svg").exists()
    assert (first_dir / "fanout.stmx").exists()
    assert (first_dir / "incremental_before.svg").exists()
    assert first["cases"]["fanout"]["layout"]["warnings"] == []
    assert [
        warning["code"]
        for warning in first["cases"]["nonplanar"]["layout"]["warnings"]
    ] == ["layout.unavoidable_crossing"]
    assert first["incremental"]["before"]["elements"] == 4
    assert first["incremental"]["after"]["elements"] == 7
    assert first["incremental"]["before"]["position_sources"]["first"] == "user"
    assert first["incremental"]["after"]["position_sources"]["first"] == "user"
    assert first["incremental"]["before"]["position_sources"]["second"] == "auto"
    assert first["incremental"]["after"]["position_sources"]["second"] == "auto"
    assert (
        first["incremental"]["before"]["positions"]["first"]
        == first["incremental"]["after"]["positions"]["first"]
    )
    assert first["incremental"]["moved_elements"] > 0
    assert first["incremental"]["total_displacement"] > 0
    assert first["incremental"]["before"]["routes"]["connector:1"]


def test_layout_report_float_canonicalization_is_platform_stable():
    first = {"length": 925.6016734282955, "bounds": (32.400000000000006, 32.0)}
    second = {"length": 925.6016734282956, "bounds": (32.4, 32.0)}

    assert _canonicalize_report_value(first) == _canonicalize_report_value(second)
