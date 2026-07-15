"""End-to-end tests for the staged layout pipeline."""

import math

from evaluation.layout_fixtures import (
    build_chain,
    build_dense_planar,
    build_fanout,
    build_feedback,
    build_mixed_pins,
    build_nonplanar,
    build_special_flows,
    fixture_builders,
    template_models,
)
from stella_mcp import layout_pipeline
from stella_mcp.layout_quality import (
    ROUTE_BEND_CAP,
    ROUTE_LENGTH_MANHATTAN_MULTIPLIER,
)
from stella_mcp.xmile import StellaModel


def _hard_counts(metrics):
    return {
        "missing": len(metrics.missing_positions),
        "glyph": len(metrics.glyph_overlaps),
        "label_glyph": len(metrics.label_glyph_overlaps),
        "label_label": len(metrics.label_label_overlaps),
        "flow_glyph": len(metrics.flow_glyph_crossings),
        "connector_glyph": len(metrics.connector_glyph_crossings),
        "flow_label": len(metrics.flow_label_crossings),
        "connector_label": len(metrics.connector_label_crossings),
        "connector_flow": len(metrics.connector_flow_crossings),
        "connector_connector": len(metrics.connector_connector_crossings),
        "flow_flow": len(metrics.flow_flow_crossings),
        "flow_shared": len(metrics.flow_shared_segments),
        "self_intersections": len(metrics.route_self_intersections),
        "repeated": len(metrics.repeated_route_points),
        "redundant": len(metrics.redundant_route_points),
        "avoidable_detours": len(metrics.avoidable_route_detours),
        "backward": len(metrics.backward_flow_edges),
        "pinned": len(metrics.pinned_position_movements),
        "locked": len(metrics.locked_route_movements),
    }


def _assert_route_caps(model):
    routes = [flow.points for flow in model.flows.values()]
    routes.extend(connector.points for connector in model.connectors)
    for points in routes:
        assert len(points) >= 2
        assert len(points) - 2 <= ROUTE_BEND_CAP
        length = sum(math.dist(start, end) for start, end in zip(points, points[1:], strict=False))
        manhattan = abs(points[-1][0] - points[0][0]) + abs(points[-1][1] - points[0][1])
        assert manhattan > 0
        assert length <= ROUTE_LENGTH_MANHATTAN_MULTIPLIER * manhattan


def _assert_clean_planar_layout(model):
    first = model._auto_layout()
    second = model._auto_layout()

    assert first == second
    assert _hard_counts(first.metrics) == {key: 0 for key in _hard_counts(first.metrics)}
    assert first.metrics.page_overflow == (0.0, 0.0, 0.0, 0.0)
    assert first.warnings == ()
    _assert_route_caps(model)


def test_chain_is_straight_deterministic_and_fits_two_by_one_pages():
    model = build_chain()

    first = model._auto_layout()
    second = model._auto_layout()

    assert first == second
    assert _hard_counts(first.metrics) == {key: 0 for key in _hard_counts(first.metrics)}
    assert first.metrics.total_bend_count == 0
    assert (model.view_page_columns, model.view_page_rows) == (2, 1)


def test_fanout_uses_distinct_routes_inside_one_page():
    model = build_fanout()

    result = model._auto_layout()

    assert _hard_counts(result.metrics) == {key: 0 for key in _hard_counts(result.metrics)}
    assert (model.view_page_columns, model.view_page_rows) == (1, 1)
    assert model.stocks["hub"].height == 252
    assert all(
        flow.points[0][0] == model.stocks["hub"].x + model.stocks["hub"].width / 2
        for flow in model.flows.values()
    )
    assert result.metrics.avoidable_route_detours == ()
    assert len({tuple(flow.points) for flow in model.flows.values()}) == len(model.flows)
    assert all(
        start[0] == end[0] or start[1] == end[1]
        for flow in model.flows.values()
        for start, end in zip(flow.points, flow.points[1:], strict=False)
    )


def test_feedback_flows_use_stella_compatible_routes_without_crossings():
    model = build_feedback()

    result = model._auto_layout()

    assert result.metrics.avoidable_route_detours == ()
    assert all(
        start[0] == end[0] or start[1] == end[1]
        for flow in model.flows.values()
        for start, end in zip(flow.points, flow.points[1:], strict=False)
    )
    assert result.metrics.connector_flow_crossings == ()
    assert result.metrics.connector_connector_crossings == ()


def test_dense_planar_auxiliary_chain_has_no_glyph_or_label_overlap():
    result = build_dense_planar()._auto_layout()

    assert result.metrics.glyph_overlaps == ()
    assert result.metrics.label_glyph_overlaps == ()
    assert result.metrics.label_label_overlaps == ()


def test_nonplanar_graph_reports_deterministic_unavoidable_crossing():
    model = build_nonplanar()
    result = model._auto_layout()

    counts = _hard_counts(result.metrics)
    assert counts.pop("connector_connector") == 1
    assert counts == {key: 0 for key in counts}
    assert result.metrics.page_overflow == (0.0, 0.0, 0.0, 0.0)
    assert [warning.code for warning in result.warnings] == ["layout.unavoidable_crossing"]
    _assert_route_caps(model)


def test_mixed_pins_are_preserved_exactly():
    model = build_mixed_pins()

    result = model._auto_layout()

    assert model.stocks["pinned_source"].x == 120.5
    assert model.stocks["pinned_source"].y == 280.25
    assert model.stocks["pinned_destination"].x == 620.5
    assert model.stocks["pinned_destination"].y == 280.25
    assert result.metrics.pinned_position_movements == ()


def test_special_flow_forms_receive_normalized_routes():
    model = build_special_flows()

    result = model._auto_layout()

    assert all(len(flow.points) >= 2 for flow in model.flows.values())
    assert result.metrics.route_self_intersections == ()
    assert result.metrics.repeated_route_points == ()
    assert result.metrics.redundant_route_points == ()


def test_all_planar_fixtures_and_templates_meet_release_gates():
    models = template_models()
    models.update(
        {name: builder() for name, builder in fixture_builders().items() if name != "nonplanar"}
    )

    for name, model in sorted(models.items()):
        try:
            _assert_clean_planar_layout(model)
        except AssertionError as error:
            raise AssertionError(f"layout release gate failed for {name}") from error


def test_fixture_specific_page_grid_gates():
    templates = template_models()
    for name, model in sorted(templates.items()):
        model._auto_layout()
        assert (model.view_page_columns, model.view_page_rows) == (1, 1), name

    expected = {
        "chain": (2, 1),
        "fanout": (1, 1),
        "feedback": (1, 1),
    }
    builders = fixture_builders()
    for name, page_grid in expected.items():
        model = builders[name]()
        model._auto_layout()
        assert (model.view_page_columns, model.view_page_rows) == page_grid, name


def test_legacy_violation_resolver_preserves_authored_geometry_and_locked_routes():
    model = StellaModel("Authored")
    model.add_stock("source", "100", x=100.25, y=200.75)
    model.add_stock("destination", "0", x=400.25, y=200.75)
    flow = model.add_flow(
        "transfer",
        "1",
        from_stock="source",
        to_stock="destination",
        x=250.25,
        y=160.75,
    )
    flow.points = [(122.75, 180.75), (250.25, 140.75), (377.75, 180.75)]
    flow.points_locked = True
    model.add_aux("control", "1", x=250.25, y=80.75)
    connector = model.add_connector("control", "transfer")
    connector.points = [(250.25, 98.75), (250.25, 150.75)]
    connector.points_locked = True
    original_positions = {
        name: (element.x, element.y)
        for registry in (model.stocks, model.flows, model.auxs)
        for name, element in registry.items()
    }
    original_flow = tuple(flow.points)
    original_connector = tuple(connector.points)

    result = model._resolve_layout_violations()

    assert {
        name: (element.x, element.y)
        for registry in (model.stocks, model.flows, model.auxs)
        for name, element in registry.items()
    } == original_positions
    assert tuple(flow.points) == original_flow
    assert tuple(connector.points) == original_connector
    assert result.metrics.pinned_position_movements == ()
    assert result.metrics.locked_route_movements == ()


def test_locked_connector_is_reserved_before_unlocked_flow_routing():
    model = StellaModel("Locked route priority")
    model.add_stock("A", "100", x=100, y=200)
    model.add_stock("B", "0", x=500, y=200)
    flow = model.add_flow("transfer", "1", from_stock="A", to_stock="B")
    model.add_aux("upper", "1", x=300, y=100)
    model.add_aux("lower", "1", x=300, y=300)
    connector = model.add_connector("upper", "lower")
    connector.points = [(300.0, 118.0), (300.0, 282.0)]
    connector.points_locked = True
    original_route = tuple(connector.points)

    result = model._auto_layout()

    assert tuple(connector.points) == original_route
    assert result.metrics.locked_route_movements == ()
    assert result.metrics.connector_flow_crossings == ()
    assert len(flow.points) > 2


def test_authored_label_is_a_fixed_routing_obstacle():
    model = StellaModel("Authored label obstacle")
    model.add_stock("A", "100", x=100, y=200)
    model.add_stock("B", "0", x=500, y=200)
    flow = model.add_flow("transfer", "1", from_stock="A", to_stock="B")
    obstacle = model.add_aux("authored obstacle label", "1", x=300, y=170)
    obstacle.label_side = "bottom"

    result = model._auto_layout()

    assert len(flow.points) > 2
    assert result.metrics.flow_label_crossings == ()
    assert result.metrics.label_glyph_overlaps == ()
    assert result.warnings == ()


def test_locked_route_prevents_page_shift_and_reports_unreachable_negative_bounds():
    model = StellaModel("Locked page geometry")
    model.add_aux("a", "1")
    model.add_aux("b", "1")
    connector = model.add_connector("a", "b")
    connector.points = [(-10.0, 100.0), (50.0, 100.0)]
    connector.points_locked = True

    result = model._auto_layout()

    assert connector.points == [(-10.0, 100.0), (50.0, 100.0)]
    assert result.metrics.page_overflow[0] == 10.0
    assert "layout.page_overflow" in {warning.code for warning in result.warnings}


def test_locked_route_conflict_is_reported_without_rewriting_the_route():
    model = StellaModel("Locked route conflict")
    model.add_aux("a", "1", x=100, y=200)
    model.add_aux("b", "1", x=500, y=200)
    model.add_stock("blocker", "0", x=300, y=200)
    connector = model.add_connector("a", "b")
    connector.points = [(118.0, 200.0), (482.0, 200.0)]
    connector.points_locked = True
    original_route = tuple(connector.points)

    result = model._auto_layout()

    assert tuple(connector.points) == original_route
    assert result.metrics.connector_glyph_crossings == ((str(connector.uid), "blocker"),)
    assert "layout.locked_route_conflict" in {warning.code for warning in result.warnings}


def test_overlapping_pinned_glyphs_emit_pinned_conflict():
    model = StellaModel("Pinned overlap")
    model.add_stock("a", "1", x=200, y=200)
    model.add_stock("b", "1", x=200, y=200)

    result = model._auto_layout()

    assert result.metrics.glyph_overlaps == (("a", "b"),)
    assert "layout.pinned_conflict" in {warning.code for warning in result.warnings}


def test_label_sides_are_selected_in_normalized_name_order(monkeypatch):
    model = StellaModel("Label order")
    model.add_aux("z label", "1", x=300, y=100)
    model.add_aux("a label", "1", x=100, y=100)
    observed: list[str] = []
    original = layout_pipeline.estimate_label_box

    def record_order(glyph, *args, **kwargs):
        if glyph.name not in observed:
            observed.append(glyph.name)
        return original(glyph, *args, **kwargs)

    monkeypatch.setattr(layout_pipeline, "estimate_label_box", record_order)

    layout_pipeline._choose_label_sides(model)

    assert observed == ["a_label", "z_label"]
