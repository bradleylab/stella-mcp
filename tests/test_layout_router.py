"""Tests for boundary ports and obstacle-aware polyline routing."""

from evaluation.layout_fixtures import build_chain, build_fanout
from stella_mcp.layout import segment_intersects_box
from stella_mcp.layout_graph import place_stock_backbone
from stella_mcp.layout_quality import LayoutBox
from stella_mcp.layout_router import (
    _visibility_route,
    allocate_flow_ports,
    assign_provisional_flow_routes,
    normalize_route,
    point_at_half_length,
    route_between,
)


def test_normalize_route_removes_duplicates_and_collinear_points():
    assert normalize_route([(0, 0), (0, 0), (5, 0), (10, 0), (10, 10)]) == (
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 10.0),
    )


def test_half_length_point_uses_polyline_distance():
    assert point_at_half_length([(0, 0), (10, 0), (10, 30)]) == (10.0, 10.0)


def test_router_uses_direct_route_when_unobstructed():
    route, fallback = route_between((10, 20), (110, 20))

    assert route == ((10.0, 20.0), (110.0, 20.0))
    assert fallback is False


def test_router_can_require_stella_compatible_orthogonal_route():
    route, fallback = route_between((10, 20), (110, 80), orthogonal_only=True)

    assert fallback is False
    assert len(route) >= 3
    assert all(
        start[0] == end[0] or start[1] == end[1]
        for start, end in zip(route, route[1:], strict=False)
    )


def test_router_avoids_obstacle_with_bounded_bends():
    obstacle = LayoutBox("block", "stock", 60, 20, 30, 30)

    route, _ = route_between((10, 20), (110, 20), (obstacle,))

    assert len(route) - 2 <= 4
    assert all(
        not segment_intersects_box(start, end, obstacle.as_bounding_box())
        for start, end in zip(route, route[1:], strict=False)
    )


def test_router_avoids_existing_route_when_a_clean_corridor_exists():
    existing = (((60.0, 0.0), (60.0, 100.0)),)

    route, fallback = route_between(
        (10, 50),
        (110, 50),
        existing_routes=existing,
    )

    assert fallback is False
    assert route != ((10.0, 50.0), (110.0, 50.0))


def test_visibility_search_prioritizes_length_before_bend_count():
    obstacles = (
        LayoutBox("upper left", "stock", 40, 0, 10, 102),
        LayoutBox("lower middle", "stock", 100, 100, 10, 102),
        LayoutBox("upper right", "stock", 160, 0, 10, 102),
    )

    route = _visibility_route((0.0, 50.0), (200.0, 50.0), obstacles, ())

    assert route is not None
    assert len(route) - 2 == 6
    assert sum(
        ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
        for start, end in zip(route, route[1:], strict=False)
    ) == 338.0


def test_fanout_allocates_distinct_hub_ports():
    model = build_fanout()
    graph, _ = place_stock_backbone(model)

    ports = allocate_flow_ports(model, graph)

    assert len({ports[f"flow_{index}"][0] for index in range(8)}) == 8
    hub_right = float(
        round(model.stocks["hub"].x + model.stocks["hub"].width / 2)
    )
    assert {ports[f"flow_{index}"][0][0] for index in range(8)} == {hub_right}
    assert all(
        ports[f"flow_{index}"][1][0]
        == round(
            model.stocks[f"destination_{index}"].x
            - model.stocks[f"destination_{index}"].width / 2
        )
        for index in range(8)
    )


def test_chain_provisional_routes_are_straight_with_centered_valves():
    model = build_chain()
    graph, _ = place_stock_backbone(model)

    assign_provisional_flow_routes(model, graph)

    assert all(len(flow.points) == 2 for flow in model.flows.values())
    assert all(flow.points[0][1] == flow.points[1][1] for flow in model.flows.values())
    assert all(flow.y == flow.points[0][1] for flow in model.flows.values())
