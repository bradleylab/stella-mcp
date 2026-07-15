"""Tests for deterministic layout-quality analysis."""

import pytest

from stella_mcp.layout_quality import (
    LayoutBox,
    LayoutViewport,
    analyze_layout,
    estimate_label_box,
)
from stella_mcp.xmile import StellaModel


def _clean_linear_model() -> StellaModel:
    model = StellaModel("Clean")
    model.add_stock("A", "100", x=100, y=200)
    model.add_stock("B", "0", x=300, y=200)
    flow = model.add_flow("transfer", "rate", from_stock="A", to_stock="B", x=200, y=200)
    flow.points = [(122.5, 200), (277.5, 200)]
    model.add_aux("rate", "1", x=200, y=100)
    model.add_connector("rate", "transfer")
    return model


def test_analyze_clean_linear_model_has_no_hard_violations():
    metrics = analyze_layout(_clean_linear_model())

    assert metrics.missing_positions == ()
    assert metrics.glyph_overlaps == ()
    assert metrics.label_glyph_overlaps == ()
    assert metrics.label_label_overlaps == ()
    assert metrics.flow_glyph_crossings == ()
    assert metrics.connector_glyph_crossings == ()
    assert metrics.connector_flow_crossings == ()
    assert metrics.avoidable_route_detours == ()
    assert metrics.backward_flow_edges == ()
    assert metrics.page_overflow == (0.0, 0.0, 0.0, 0.0)
    assert metrics.total_flow_length == 155.0
    assert metrics.maximum_flow_length == 155.0
    assert metrics.total_connector_length == 100.0


def test_analyzer_detects_route_through_unrelated_stock():
    model = StellaModel("Blocked")
    model.add_stock("A", "100", x=100, y=200)
    model.add_stock("B", "0", x=300, y=200)
    model.add_stock("blocker", "0", x=200, y=200)
    flow = model.add_flow("transfer", "1", from_stock="A", to_stock="B", x=160, y=200)
    flow.points = [(122.5, 200), (277.5, 200)]

    metrics = analyze_layout(model)

    assert ("transfer", "blocker") in metrics.flow_glyph_crossings


def test_analyzer_distinguishes_crossings_from_endpoint_touches():
    model = _clean_linear_model()
    model.add_aux("crossing source", "1", x=150, y=150)
    model.add_aux("crossing target", "1", x=250, y=250)
    crossing = model.add_connector("crossing source", "crossing target")
    crossing.points = [(150, 150), (250, 250)]

    metrics = analyze_layout(model)

    assert (str(crossing.uid), "transfer") in metrics.connector_flow_crossings
    assert ("1", "transfer") not in metrics.connector_flow_crossings


def test_connector_attachment_to_target_flow_is_not_a_crossing():
    model = StellaModel("Expected attachment")
    model.add_stock("stock", "0", x=300, y=200)
    flow = model.add_flow("flow", "control", to_stock="stock", x=200, y=200)
    model.add_aux("control", "1", x=200, y=100)
    connector = model.add_connector("control", "flow")
    flow.points = [(100.0, 200.0), (300.0, 200.0)]
    connector.points = [(200.0, 100.0), (200.0, 200.0)]

    metrics = analyze_layout(model)

    assert metrics.connector_flow_crossings == ()


def test_unrelated_route_touch_is_a_crossing():
    model = StellaModel("Touch")
    model.add_aux("a", "1", x=20, y=20)
    model.add_aux("b", "1", x=100, y=20)
    model.add_aux("c", "1", x=60, y=60)
    model.add_aux("d", "1", x=60, y=100)
    first = model.add_connector("a", "b")
    second = model.add_connector("c", "d")
    first.points = [(38, 20), (82, 20)]
    second.points = [(60, 20), (60, 82)]

    metrics = analyze_layout(model)

    assert metrics.connector_connector_crossings == ((str(first.uid), str(second.uid)),)


def test_analyzer_reports_an_avoidable_route_detour():
    model = StellaModel("Avoidable detour")
    model.add_aux("source", "1", x=100, y=100)
    model.add_aux("target", "1", x=300, y=100)
    connector = model.add_connector("source", "target")
    connector.points = [(118, 100), (118, 60), (282, 60), (282, 100)]

    metrics = analyze_layout(model)

    assert metrics.avoidable_route_detours == (f"connector:{connector.uid}",)


def test_analyzer_reports_backward_edges_outside_feedback_components():
    model = StellaModel("Direction")
    model.add_stock("left", "0", x=100, y=100)
    model.add_stock("right", "100", x=300, y=100)
    flow = model.add_flow("backward", "1", from_stock="right", to_stock="left", x=200, y=100)
    flow.points = [(277.5, 100), (122.5, 100)]

    assert analyze_layout(model).backward_flow_edges == ("backward",)

    feedback = model.add_flow("feedback", "1", from_stock="left", to_stock="right", x=200, y=140)
    feedback.points = [(122.5, 110), (277.5, 110)]
    assert analyze_layout(model).backward_flow_edges == ()


def test_analyzer_checks_reference_geometry_without_mutating_model():
    model = _clean_linear_model()
    before = model.stocks["A"].x, model.stocks["A"].y

    metrics = analyze_layout(
        model,
        pinned_reference={"A": (101, 200)},
        locked_route_reference={"flow:transfer": ((0, 0), (1, 1))},
    )

    assert metrics.pinned_position_movements == ("A",)
    assert metrics.locked_route_movements == ("flow:transfer",)
    assert (model.stocks["A"].x, model.stocks["A"].y) == before


def test_label_estimate_and_viewport_are_derived_from_declared_geometry():
    glyph = LayoutBox("stock", "stock", 100, 100, 40, 20, locked=True)
    label = estimate_label_box(glyph, "abcd", "bottom", font_points=9)
    viewport = LayoutViewport(page_width=768, page_height=596, columns=2, rows=1)

    assert label.width == pytest.approx(28.8)
    assert label.height == 12.0
    assert label.top == 111.0
    assert glyph.locked is True
    assert viewport.width == 1536
    assert viewport.height == 596
    assert viewport.bounds == (0.0, 0.0, 1536, 596)
