"""Staged deterministic layout pipeline for Stella stock-flow diagrams."""

from __future__ import annotations

import math

from stella_mcp.layout import (
    _SEPARATION_GAP,
    SegmentIntersection,
    segment_intersection_kind,
    segment_intersects_box,
)
from stella_mcp.layout_graph import MARGIN, DirectedStockGraph, place_stock_backbone
from stella_mcp.layout_quality import (
    AUX_RADIUS,
    FLOW_VALVE_SIZE,
    LayoutBox,
    LayoutMetrics,
    LayoutResult,
    LayoutRoute,
    LayoutViewport,
    LayoutWarning,
    analyze_layout,
    direct_segment_is_clear,
    estimate_label_box,
    label_font_points,
)
from stella_mcp.layout_router import (
    allocate_flow_ports,
    assign_provisional_flow_routes,
    boundary_port,
    element_box,
    fanout_channel_routes,
    is_orthogonal_route,
    normalize_route,
    point_at_half_length,
    route_between,
    route_segments,
    score_route,
)

Point = tuple[float, float]
LABEL_SIDES = ("top", "bottom", "left", "right")
MAX_AUX_RINGS = 12
MAX_ROUTING_PASSES = 3


def _pinned_positions(model) -> dict[str, Point]:
    result: dict[str, Point] = {}
    for registry in (model.stocks, model.flows, model.auxs):
        for name, element in registry.items():
            if (
                element.position_source == "user"
                and element.x is not None
                and element.y is not None
            ):
                result[name] = (element.x, element.y)
    return result


def _locked_routes(model) -> dict[str, tuple[Point, ...]]:
    result = {
        f"flow:{name}": tuple(flow.points)
        for name, flow in model.flows.items()
        if flow.points_locked
    }
    result.update(
        {
            f"connector:{connector.uid}": tuple(connector.points)
            for connector in model.connectors
            if connector.points_locked
        }
    )
    return result


def _fill_or_set_position(element, position: Point) -> None:
    """Fill missing authored axes or set a complete generated position."""
    if element.position_source == "user":
        if element.x is None:
            element.x = position[0]
        if element.y is None:
            element.y = position[1]
        return
    element.x, element.y = position
    element.position_source = "auto"


def _all_glyph_boxes(model) -> dict[str, LayoutBox]:
    boxes: dict[str, LayoutBox] = {}
    for name in sorted((*model.stocks, *model.auxs, *model.flows)):
        box = element_box(model, name)
        if box is not None:
            boxes[name] = box
    return boxes


def _authored_label_obstacles(
    model,
    excluded_names: set[str] | None = None,
    clearance: float = 0.0,
) -> tuple[LayoutBox, ...]:
    """Return fixed label boxes supplied with authored element geometry."""
    excluded = excluded_names or set()
    registries = {**model.stocks, **model.flows, **model.auxs}
    glyphs = _all_glyph_boxes(model)
    labels = tuple(
        estimate_label_box(
            glyphs[name],
            model._display_name(element.name),
            element.label_side,
            label_font_points(model, name),
        )
        for name, element in sorted(registries.items())
        if name not in excluded
        and name in glyphs
        and element.position_source == "user"
        and element.label_side in LABEL_SIDES
    )
    return tuple(
        LayoutBox(
            label.name,
            label.kind,
            label.x,
            label.y,
            label.width + 2 * clearance,
            label.height + 2 * clearance,
            True,
        )
        for label in labels
    )


def _geometric_median(points: list[Point]) -> Point:
    if len(points) == 1:
        return points[0]
    current = (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )
    for _ in range(50):
        distances = [math.dist(current, point) for point in points]
        if any(distance == 0 for distance in distances):
            current = points[distances.index(0)]
            break
        weights = [1 / distance for distance in distances]
        updated = (
            sum(weight * point[0] for weight, point in zip(weights, points, strict=True))
            / sum(weights),
            sum(weight * point[1] for weight, point in zip(weights, points, strict=True))
            / sum(weights),
        )
        if math.dist(current, updated) < 0.01:
            current = updated
            break
        current = updated
    return float(round(current[0])), float(round(current[1]))


def _aux_order(model) -> tuple[str, ...]:
    targets = {
        name: tuple(
            sorted(
                connector.to_var
                for connector in model.connectors
                if connector.from_var == name and connector.to_var in model.auxs
            )
        )
        for name in model.auxs
    }
    visited: set[str] = set()
    visiting: set[str] = set()
    order: list[str] = []

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            return
        visiting.add(name)
        for target in targets[name]:
            visit(target)
        visiting.remove(name)
        visited.add(name)
        order.append(name)

    for name in sorted(model.auxs):
        visit(name)
    return tuple(order)


def _between_on_circle(value: float, start: float, end: float) -> bool:
    return (value - start) % (2 * math.pi) < (end - start) % (2 * math.pi)


def _circular_chords_cross(first: tuple[float, float], second: tuple[float, float]) -> bool:
    first_start, first_end = first
    second_start, second_end = second
    return _between_on_circle(second_start, first_start, first_end) != _between_on_circle(
        second_end, first_start, first_end
    ) and _between_on_circle(first_start, second_start, second_end) != _between_on_circle(
        first_end, second_start, second_end
    )


def _aux_radial_preferences(model, graph: DirectedStockGraph) -> dict[str, bool]:
    component_by_node = graph.component_map()
    flow_targets = {
        connector.from_var: connector.to_var
        for connector in model.connectors
        if connector.from_var in model.auxs and connector.to_var in model.flows
    }
    chords_by_component: dict[int, dict[str, tuple[float, float]]] = {}
    for connector in model.connectors:
        if connector.from_var not in model.stocks or connector.to_var not in flow_targets:
            continue
        flow = model.flows[flow_targets[connector.to_var]]
        component_id = component_by_node.get(connector.from_var)
        if (
            component_id is None
            or len(graph.components[component_id]) < 3
            or flow.from_stock not in graph.components[component_id]
            or flow.to_stock not in graph.components[component_id]
            or flow.x is None
            or flow.y is None
        ):
            continue
        members = graph.components[component_id]
        center = (
            sum(model.stocks[name].x for name in members) / len(members),
            sum(model.stocks[name].y for name in members) / len(members),
        )
        source = model.stocks[connector.from_var]
        chords_by_component.setdefault(component_id, {})[connector.to_var] = (
            math.atan2(source.y - center[1], source.x - center[0]) % (2 * math.pi),
            math.atan2(flow.y - center[1], flow.x - center[0]) % (2 * math.pi),
        )

    preferences: dict[str, bool] = {}
    for chords in chords_by_component.values():
        conflicts: dict[str, set[str]] = {name: set() for name in chords}
        names = sorted(chords)
        for index, first in enumerate(names):
            for second in names[index + 1 :]:
                if _circular_chords_cross(chords[first], chords[second]):
                    conflicts[first].add(second)
                    conflicts[second].add(first)
        colors: dict[str, bool] = {}
        valid = True
        for root in names:
            if root in colors or not conflicts[root]:
                continue
            colors[root] = False
            pending = [root]
            while pending:
                current = pending.pop()
                for neighbor in sorted(conflicts[current]):
                    expected = not colors[current]
                    if neighbor in colors and colors[neighbor] != expected:
                        valid = False
                        break
                    if neighbor not in colors:
                        colors[neighbor] = expected
                        pending.append(neighbor)
                if not valid:
                    break
        if valid:
            preferences.update(colors)

    return preferences


def _candidate_aux_positions(center: Point, clearance: float, ring: int) -> tuple[Point, ...]:
    radius = clearance + ring * _SEPARATION_GAP
    diagonal = float(round(radius / math.sqrt(2)))
    x, y = center
    offsets = (
        (0.0, -radius),
        (radius, 0.0),
        (0.0, radius),
        (-radius, 0.0),
        (diagonal, -diagonal),
        (diagonal, diagonal),
        (-diagonal, diagonal),
        (-diagonal, -diagonal),
    )
    return tuple((float(round(x + dx)), float(round(y + dy))) for dx, dy in offsets)


def _connector_segments_to_targets(
    position: Point, targets: list[Point]
) -> tuple[tuple[Point, Point], ...]:
    return tuple((position, target) for target in targets)


def _aux_candidate_score(
    name: str,
    position: Point,
    target_names: set[str],
    targets: list[Point],
    obstacles: dict[str, LayoutBox],
    provisional_routes: tuple[tuple[str, tuple[Point, ...]], ...],
    prior_connector_segments: tuple[tuple[Point, Point], ...],
    layout_center: Point,
    prefer_outward: bool,
    *,
    route_conflicts_are_hard: bool = True,
) -> tuple[int, int, float, float, Point]:
    box = LayoutBox(name, "aux", position[0], position[1], AUX_RADIUS * 2, AUX_RADIUS * 2)
    hard = sum(
        box.as_bounding_box().intersects(obstacle.as_bounding_box(), margin=_SEPARATION_GAP)
        for obstacle_name, obstacle in obstacles.items()
        if obstacle_name != name
    )
    segments = _connector_segments_to_targets(position, targets)
    route_conflicts = sum(
        segment_intersects_box(start, end, obstacle.as_bounding_box())
        for start, end in segments
        for obstacle_name, obstacle in obstacles.items()
        if obstacle_name != name and obstacle_name not in target_names
    )
    route_conflicts += sum(
        segment_intersects_box(start, end, box.as_bounding_box())
        for _, route in provisional_routes
        for start, end in route_segments(route)
    )
    crossings = 0
    for start, end in segments:
        for route_name, route in provisional_routes:
            if route_name in target_names:
                continue
            for other_start, other_end in route_segments(route):
                if segment_intersection_kind(start, end, other_start, other_end) in {
                    SegmentIntersection.CROSS,
                    SegmentIntersection.OVERLAP,
                }:
                    crossings += 1
        for other_start, other_end in prior_connector_segments:
            if segment_intersection_kind(start, end, other_start, other_end) in {
                SegmentIntersection.CROSS,
                SegmentIntersection.OVERLAP,
            }:
                crossings += 1
    length = sum(math.dist(position, target) for target in targets)
    radial_distance = math.dist(position, layout_center)
    outward_tiebreak = (-radial_distance if prefer_outward else radial_distance) if targets else 0.0
    hard_score = hard + route_conflicts if route_conflicts_are_hard else hard
    return hard_score, crossings, length, outward_tiebreak, position


def _fallback_aux_center(model, index: int) -> Point:
    boxes = [box for name in (*model.stocks, *model.flows) if (box := element_box(model, name))]
    bottom = max((box.bottom for box in boxes), default=MARGIN)
    columns = max(1, int((model.view_page_width - 2 * MARGIN) // 100))
    return (
        MARGIN + 50.0 + (index % columns) * 100.0,
        bottom + 70.0 + (index // columns) * 80.0,
    )


def _provisional_stock_control_routes(
    model,
) -> tuple[tuple[str, tuple[Point, ...]], ...]:
    """Reserve cross-component stock-control corridors before placing auxiliaries."""
    routes: list[tuple[str, tuple[Point, ...]]] = []
    for connector in sorted(model.connectors, key=lambda item: item.uid):
        target_flow = model.flows.get(connector.to_var)
        if (
            connector.from_var not in model.stocks
            or target_flow is None
            or connector.from_var in {target_flow.from_stock, target_flow.to_stock}
        ):
            continue
        existing_routes = tuple(
            tuple(flow.points)
            for name, flow in sorted(model.flows.items())
            if name != connector.to_var
        ) + tuple(route for _, route in routes)
        route, _ = _route_connector(model, connector, existing_routes)
        if route:
            routes.append((f"stock-control:{connector.uid}", route))
    return tuple(routes)


def _place_auxiliaries(
    model,
    radial_preferences: dict[str, bool] | None = None,
    only_names: set[str] | None = None,
    *,
    preferred_sources: dict[str, set[str]] | None = None,
) -> tuple[LayoutWarning, ...]:
    warnings: list[LayoutWarning] = []
    moving_names = set(model.auxs) if only_names is None else only_names
    previous_positions = {name: (model.auxs[name].x, model.auxs[name].y) for name in moving_names}
    for name in moving_names:
        aux = model.auxs[name]
        if aux.position_source == "auto":
            aux.x = None
            aux.y = None
    obstacles = {
        name: box
        for name, box in _all_glyph_boxes(model).items()
        if name not in model.auxs
        or model.auxs[name].position_source == "user"
        or name not in moving_names
    }
    provisional_routes = tuple(
        (name, tuple(flow.points)) for name, flow in model.flows.items() if flow.points
    ) + _provisional_stock_control_routes(model)
    prior_segments: list[tuple[Point, Point]] = []
    stock_centers = [
        (stock.x, stock.y)
        for stock in model.stocks.values()
        if stock.x is not None and stock.y is not None
    ]
    layout_center = (
        _geometric_median(stock_centers)
        if stock_centers
        else (model.view_page_width / 2, model.view_page_height / 2)
    )
    orphan_index = 0
    for name in _aux_order(model):
        aux = model.auxs[name]
        if name not in moving_names:
            continue
        if aux.position_source == "user" and aux.x is not None and aux.y is not None:
            obstacles[name] = element_box(model, name)  # type: ignore[assignment]
            continue
        target_names = {
            connector.to_var for connector in model.connectors if connector.from_var == name
        }
        targets = [element_box(model, target_name) for target_name in sorted(target_names)]
        target_boxes = [box for box in targets if box is not None]
        target_points = [(box.x, box.y) for box in target_boxes]
        connected_names = set(target_names)
        connected_points = list(target_points)
        source_names = (preferred_sources or {}).get(name, set())
        if source_names:
            source_boxes = {
                connector.from_var: box
                for connector in model.connectors
                if connector.to_var == name
                and connector.from_var in source_names
                and (box := element_box(model, connector.from_var)) is not None
            }
            connected_names.update(source_boxes)
            connected_points.extend((box.x, box.y) for _, box in sorted(source_boxes.items()))
        if target_points:
            center = _geometric_median(target_points)
            clearance = max(max(box.width, box.height) / 2 for box in target_boxes)
            clearance += AUX_RADIUS + _SEPARATION_GAP
        else:
            center = _fallback_aux_center(model, orphan_index)
            orphan_index += 1
            clearance = 0.0

        best: tuple[tuple[int, int, float, float, Point], Point] | None = None
        for ring in range(MAX_AUX_RINGS):
            raw_candidates = (
                _candidate_aux_positions(center, clearance, ring) if target_points else (center,)
            )
            candidates = tuple(
                dict.fromkeys(
                    (
                        aux.x
                        if aux.position_source == "user" and aux.x is not None
                        else candidate[0],
                        aux.y
                        if aux.position_source == "user" and aux.y is not None
                        else candidate[1],
                    )
                    for candidate in raw_candidates
                    if name not in (preferred_sources or {})
                    or candidate != previous_positions[name]
                )
            )
            scored = [
                (
                    _aux_candidate_score(
                        name,
                        candidate,
                        connected_names,
                        connected_points,
                        obstacles,
                        provisional_routes,
                        tuple(prior_segments),
                        layout_center,
                        (radial_preferences or {}).get(name, False),
                        route_conflicts_are_hard=name not in (preferred_sources or {}),
                    ),
                    candidate,
                )
                for candidate in candidates
            ]
            local_best = min(scored)
            best = min(best, local_best) if best is not None else local_best
            if local_best[0][:2] == (0, 0):
                best = local_best
                break
        assert best is not None
        _fill_or_set_position(aux, best[1])
        obstacles[name] = element_box(model, name)  # type: ignore[assignment]
        prior_segments.extend(_connector_segments_to_targets(best[1], target_points))
        if best[0][0] > 0:
            warnings.append(
                LayoutWarning(
                    "layout.placement_exhausted",
                    "No conflict-free auxiliary position was available.",
                    (name,),
                )
            )
    return tuple(warnings)


def _conflicted_auxiliary_sources(
    model,
) -> dict[str, set[str]]:
    eligible = {
        connector.from_var
        for connector in model.connectors
        if connector.from_var in model.auxs and connector.to_var in model.flows
    }
    metrics = analyze_layout(model)
    conflicted = {name for pair in metrics.glyph_overlaps for name in pair if name in eligible}
    conflicted.update(
        glyph_name
        for _, glyph_name in (
            *metrics.flow_glyph_crossings,
            *metrics.connector_glyph_crossings,
        )
        if glyph_name in eligible
    )
    connector_by_uid = {str(connector.uid): connector for connector in model.connectors}
    conflicted_connector_ids = {
        connector_name for connector_name, _ in metrics.connector_flow_crossings
    }
    conflicted_connector_ids.update(
        connector_name for pair in metrics.connector_connector_crossings for connector_name in pair
    )
    preferred_sources: dict[str, set[str]] = {}
    for connector_id in sorted(conflicted_connector_ids):
        connector = connector_by_uid.get(connector_id)
        if connector is None:
            continue
        if connector.from_var in eligible:
            conflicted.add(connector.from_var)
        if connector.to_var in eligible:
            conflicted.add(connector.to_var)
            preferred_sources.setdefault(connector.to_var, set()).add(connector.from_var)
    for name in sorted(eligible & conflicted):
        preferred_sources.setdefault(name, set())
    return preferred_sources


def _route_flows(
    model,
    graph: DirectedStockGraph,
) -> tuple[tuple[tuple[Point, ...], ...], bool]:
    ports = allocate_flow_ports(model, graph)
    channel_routes = fanout_channel_routes(model, graph, ports)
    order = _unlocked_flow_order(model, graph)
    routed: list[tuple[Point, ...]] = [
        tuple(flow.points)
        for _, flow in sorted(model.flows.items())
        if flow.points_locked and flow.points
    ]
    routed.extend(
        tuple(connector.points)
        for connector in sorted(model.connectors, key=lambda item: item.uid)
        if connector.points_locked and connector.points
    )
    used_fallback = False
    for index, name in enumerate(order):
        pending_stubs = _reserved_flow_port_stubs(
            model,
            ports,
            set(order[index + 1 :]),
        )
        route, fallback = _route_one_flow(
            model,
            name,
            ports,
            channel_routes,
            (*routed, *pending_stubs),
        )
        used_fallback |= fallback
        flow = model.flows[name]
        flow.points = list(route)
        _fill_or_set_position(flow, point_at_half_length(route))
        routed.append(route)
    return (
        tuple(tuple(flow.points) for _, flow in sorted(model.flows.items())),
        used_fallback,
    )


def _unlocked_flow_order(model, graph: DirectedStockGraph) -> tuple[str, ...]:
    """Return the specified deterministic routing order for unlocked flows."""
    component_by_node = graph.component_map()
    ranks = graph.rank_map()
    return tuple(
        sorted(
            (
                name
                for name, flow in model.flows.items()
                if not (flow.points_locked and flow.points)
            ),
            key=lambda name: (
                ranks.get(component_by_node.get(model.flows[name].from_stock, -1), -1),
                ranks.get(component_by_node.get(model.flows[name].to_stock, -1), math.inf),
                name,
            ),
        )
    )


def _port_stub(model, owner_name: str | None, port: Point | None) -> tuple[Point, ...]:
    if owner_name is None or port is None:
        return ()
    box = element_box(model, owner_name)
    if box is None:
        return ()
    dx = port[0] - box.x
    dy = port[1] - box.y
    if abs(dx) >= abs(dy):
        direction = 1.0 if dx >= 0 else -1.0
        lead = (port[0] + direction * _SEPARATION_GAP, port[1])
    else:
        direction = 1.0 if dy >= 0 else -1.0
        lead = (port[0], port[1] + direction * _SEPARATION_GAP)
    return normalize_route((port, lead))


def _reserved_flow_port_stubs(
    model,
    ports: dict[str, tuple[Point | None, Point | None]],
    names: set[str],
) -> tuple[tuple[Point, ...], ...]:
    """Reserve the outward leads of flows that have not been routed yet."""
    stubs: list[tuple[Point, ...]] = []
    for name in sorted(names):
        flow = model.flows[name]
        start, end = ports[name]
        for stub in (
            _port_stub(model, flow.from_stock, start),
            _port_stub(model, flow.to_stock, end),
        ):
            if stub:
                stubs.append(stub)
    return tuple(stubs)


def _route_one_flow(
    model,
    name: str,
    ports: dict[str, tuple[Point | None, Point | None]],
    channel_routes: dict[str, tuple[Point, ...]],
    existing_routes: tuple[tuple[Point, ...], ...],
) -> tuple[tuple[Point, ...], bool]:
    flow = model.flows[name]
    if flow.from_stock == flow.to_stock and flow.from_stock is not None:
        return normalize_route(flow.points), False
    start, end = ports[name]
    if start is None or end is None:
        return normalize_route(flow.points), False
    excluded = {flow.from_stock, flow.to_stock, name}
    obstacles = tuple(
        box
        for obstacle_name, box in _all_glyph_boxes(model).items()
        if obstacle_name not in excluded
    ) + _authored_label_obstacles(
        model,
        excluded,
        FLOW_VALVE_SIZE / 2 + _SEPARATION_GAP,
    )
    direct = normalize_route((start, end))
    if is_orthogonal_route(direct) and direct_segment_is_clear(
        start, end, obstacles, existing_routes
    ):
        return direct, False
    preferred = channel_routes.get(name)
    if (
        preferred is not None
        and is_orthogonal_route(preferred)
        and score_route(preferred, obstacles, existing_routes)[:2] == (0, 0)
    ):
        return preferred, False
    return route_between(
        start,
        end,
        obstacles,
        existing_routes,
        orthogonal_only=True,
    )


def _all_other_routes_for_flow(model, flow_name: str) -> tuple[tuple[Point, ...], ...]:
    return tuple(
        tuple(flow.points)
        for name, flow in sorted(model.flows.items())
        if name != flow_name and flow.points
    ) + tuple(
        tuple(connector.points)
        for connector in sorted(model.connectors, key=lambda item: item.uid)
        if connector.to_var != flow_name and connector.points
    )


def _route_connectors(
    model,
) -> tuple[bool, tuple[tuple[Point, ...], ...]]:
    connector_routes: list[tuple[Point, ...]] = [
        tuple(connector.points)
        for connector in sorted(model.connectors, key=lambda item: item.uid)
        if connector.points_locked and connector.points
    ]
    used_fallback = False
    for connector in sorted(model.connectors, key=lambda item: item.uid):
        if connector.points_locked and connector.points:
            continue
        existing_routes = tuple(
            tuple(flow.points)
            for name, flow in sorted(model.flows.items())
            if name != connector.to_var
        ) + tuple(connector_routes)
        route, fallback = _route_connector(model, connector, existing_routes)
        used_fallback |= fallback
        connector.points = list(route)
        connector_routes.append(route)
    return used_fallback, tuple(
        tuple(connector.points) for connector in sorted(model.connectors, key=lambda item: item.uid)
    )


def _route_connector(
    model,
    connector,
    existing_routes,
    extra_obstacles: tuple[LayoutBox, ...] = (),
) -> tuple[tuple[Point, ...], bool]:
    source = element_box(model, connector.from_var)
    target = element_box(model, connector.to_var)
    if source is None or target is None:
        return (), False
    attached_flow = model.flows.get(connector.to_var)
    follows_attached_flow = (
        attached_flow is not None
        and connector.from_var in {attached_flow.from_stock, attached_flow.to_stock}
        and len(attached_flow.points) >= 2
    )
    excluded = {connector.from_var, connector.to_var}
    obstacles = tuple(
        box
        for obstacle_name, box in _all_glyph_boxes(model).items()
        if obstacle_name not in excluded
    ) + _authored_label_obstacles(model, excluded, _SEPARATION_GAP)
    obstacles += tuple(
        box
        for box in extra_obstacles
        if box.kind != "label" or box.name.removeprefix("label:") not in excluded
    )
    port_pairs = tuple(
        (source_boundary, source_lead, target_boundary, target_lead)
        for source_boundary, source_lead in _connector_port_options(source)
        for target_boundary, target_lead in _connector_port_options(target)
    )
    direct_existing_routes = tuple(existing_routes)
    if attached_flow is not None and attached_flow.points:
        direct_existing_routes += (tuple(attached_flow.points),)
    direct_candidates = list(
        (source_boundary, target_boundary)
        for source_boundary, source_lead, target_boundary, target_lead in port_pairs
        if _port_pair_faces(
            source_boundary,
            source_lead,
            target_boundary,
            target_lead,
        )
        and direct_segment_is_clear(
            source_boundary,
            target_boundary,
            obstacles,
            direct_existing_routes,
        )
    )
    attached_route: tuple[Point, ...] = ()
    if follows_attached_flow:
        attached_route = _attached_stock_connector_route(
            attached_flow,
            connector.from_var == attached_flow.from_stock,
            target,
        )
        attached_direct = (attached_route[0], attached_route[-1])
        if direct_segment_is_clear(
            *attached_direct,
            obstacles,
            direct_existing_routes,
        ):
            direct_candidates.append(attached_direct)
    if direct_candidates:
        return min(
            direct_candidates,
            key=lambda route: (math.dist(*route), route),
        ), False
    if follows_attached_flow:
        return attached_route, False
    candidates: list[
        tuple[
            tuple[int, int, float, int, tuple[Point, ...]],
            tuple[Point, ...],
            Point,
            Point,
            Point,
            Point,
        ]
    ] = []
    for source_boundary, source_lead, target_boundary, target_lead in port_pairs:
        cores = [
            (source_lead, target_lead),
            (source_lead, (source_lead[0], target_lead[1]), target_lead),
            (source_lead, (target_lead[0], source_lead[1]), target_lead),
        ]
        for obstacle in extra_obstacles:
            if not segment_intersects_box(
                source_lead,
                target_lead,
                obstacle.as_bounding_box(),
            ):
                continue
            left, right, top, bottom = _strict_outer_axes(obstacle)
            cores.extend(
                (
                    (source_lead, (left, source_lead[1]), (left, target_lead[1]), target_lead),
                    (
                        source_lead,
                        (right, source_lead[1]),
                        (right, target_lead[1]),
                        target_lead,
                    ),
                    (source_lead, (source_lead[0], top), (target_lead[0], top), target_lead),
                    (
                        source_lead,
                        (source_lead[0], bottom),
                        (target_lead[0], bottom),
                        target_lead,
                    ),
                )
            )
        for core in cores:
            route = normalize_route((source_boundary, *core, target_boundary))
            candidates.append(
                (
                    score_route(route, obstacles, existing_routes),
                    route,
                    source_boundary,
                    source_lead,
                    target_boundary,
                    target_lead,
                )
            )
    score, route, source_boundary, source_lead, target_boundary, target_lead = min(candidates)
    if score[:2] == (0, 0):
        return route, False

    pair_scores = []
    for source_boundary, source_lead, target_boundary, target_lead in port_pairs:
        pair_score = min(
            candidate[0]
            for candidate in candidates
            if candidate[2:]
            == (
                source_boundary,
                source_lead,
                target_boundary,
                target_lead,
            )
        )
        pair_scores.append(
            (
                pair_score,
                source_boundary,
                source_lead,
                target_boundary,
                target_lead,
            )
        )
    for _, source_boundary, source_lead, target_boundary, target_lead in sorted(pair_scores)[:4]:
        core, _ = route_between(
            source_lead,
            target_lead,
            obstacles,
            existing_routes,
            allow_visibility=False,
        )
        expanded_route = normalize_route((source_boundary, *core, target_boundary))
        candidates.append(
            (
                score_route(expanded_route, obstacles, existing_routes),
                expanded_route,
                source_boundary,
                source_lead,
                target_boundary,
                target_lead,
            )
        )
    score, route, source_boundary, source_lead, target_boundary, target_lead = min(candidates)
    if score[:2] == (0, 0):
        return route, False
    core, _ = route_between(source_lead, target_lead, obstacles, existing_routes)
    fallback_route = normalize_route((source_boundary, *core, target_boundary))
    return min(
        (route, fallback_route),
        key=lambda points: score_route(points, obstacles, existing_routes),
    ), True


def _connector_port_options(box: LayoutBox) -> tuple[tuple[Point, Point], ...]:
    options = (
        ((box.left, box.y), (box.left - _SEPARATION_GAP, box.y)),
        ((box.right, box.y), (box.right + _SEPARATION_GAP, box.y)),
        ((box.x, box.top), (box.x, box.top - _SEPARATION_GAP)),
        ((box.x, box.bottom), (box.x, box.bottom + _SEPARATION_GAP)),
    )
    return tuple(
        (
            (float(round(boundary[0])), float(round(boundary[1]))),
            (float(round(lead[0])), float(round(lead[1]))),
        )
        for boundary, lead in options
    )


def _port_pair_faces(
    source_boundary: Point,
    source_lead: Point,
    target_boundary: Point,
    target_lead: Point,
) -> bool:
    """Return whether a direct segment leaves and enters its endpoint boxes."""
    travel = (
        target_boundary[0] - source_boundary[0],
        target_boundary[1] - source_boundary[1],
    )
    source_outward = (
        source_lead[0] - source_boundary[0],
        source_lead[1] - source_boundary[1],
    )
    target_outward = (
        target_lead[0] - target_boundary[0],
        target_lead[1] - target_boundary[1],
    )
    source_dot = sum(a * b for a, b in zip(travel, source_outward, strict=True))
    target_dot = sum(a * b for a, b in zip(travel, target_outward, strict=True))
    return source_dot > 0 and target_dot < 0


def _strict_outer_axes(box: LayoutBox) -> tuple[float, float, float, float]:
    """Return whole-pixel axes immediately outside a rectangular obstacle."""

    def before(value: float) -> float:
        floor = math.floor(value)
        return float(floor if floor < value else floor - 1)

    def after(value: float) -> float:
        ceiling = math.ceil(value)
        return float(ceiling if ceiling > value else ceiling + 1)

    return before(box.left), after(box.right), before(box.top), after(box.bottom)


def _attached_stock_connector_route(
    flow, starts_at_source: bool, target: LayoutBox
) -> tuple[Point, ...]:
    """Follow an attached flow pipe from its stock port to the valve boundary."""
    points = tuple(flow.points if starts_at_source else reversed(flow.points))
    half_length = sum(math.dist(start, end) for start, end in route_segments(points)) / 2
    route = [points[0]]
    remaining = half_length
    for start, end in route_segments(points):
        length = math.dist(start, end)
        if remaining <= length:
            ratio = remaining / length if length else 0.0
            valve_center = (
                float(round(start[0] + ratio * (end[0] - start[0]))),
                float(round(start[1] + ratio * (end[1] - start[1]))),
            )
            route.append(boundary_port(target, start).point)
            if route[-1] == route[-2] and route[-1] != valve_center:
                route[-1] = valve_center
            break
        route.append(end)
        remaining -= length
    return normalize_route(route)


def _routing_score(model) -> tuple[object, ...]:
    metrics = analyze_layout(model)
    registries = {**model.stocks, **model.flows, **model.auxs}
    authored_labels = {
        name
        for name, element in registries.items()
        if element.position_source == "user" and element.label_side in LABEL_SIDES
    }
    authored_label_crossings = sum(
        label_name in authored_labels
        for _, label_name in (
            *metrics.flow_label_crossings,
            *metrics.connector_label_crossings,
        )
    )
    coordinates = tuple(
        ("flow", name, tuple(flow.points)) for name, flow in sorted(model.flows.items())
    ) + tuple(
        ("connector", str(connector.uid), tuple(connector.points))
        for connector in sorted(model.connectors, key=lambda item: item.uid)
    )
    return (
        len(metrics.missing_positions),
        len(metrics.glyph_overlaps),
        len(metrics.flow_glyph_crossings) + len(metrics.connector_glyph_crossings),
        authored_label_crossings,
        len(metrics.route_self_intersections),
        len(metrics.repeated_route_points),
        len(metrics.redundant_route_points),
        len(metrics.avoidable_route_detours),
        len(metrics.connector_flow_crossings),
        len(metrics.connector_connector_crossings),
        len(metrics.flow_flow_crossings),
        len(metrics.flow_shared_segments),
        metrics.total_flow_length + metrics.total_connector_length,
        metrics.total_bend_count,
        coordinates,
    )


def _route_everything(model, graph: DirectedStockGraph) -> tuple[LayoutWarning, ...]:
    _route_flows(model, graph)
    _route_connectors(model)
    ports = allocate_flow_ports(model, graph)
    channel_routes = fanout_channel_routes(model, graph, ports)
    hit_cap = False
    for pass_index in range(MAX_ROUTING_PASSES - 1):
        accepted = False
        for name in _unlocked_flow_order(model, graph):
            flow = model.flows[name]
            old_route = tuple(flow.points)
            old_position = (flow.x, flow.y, flow.position_source)
            before = _routing_score(model)
            route, _ = _route_one_flow(
                model,
                name,
                ports,
                channel_routes,
                _all_other_routes_for_flow(model, name),
            )
            flow.points = list(route)
            _fill_or_set_position(flow, point_at_half_length(route))
            if _routing_score(model) < before:
                accepted = True
            else:
                flow.points = list(old_route)
                flow.x, flow.y, flow.position_source = old_position
        for connector in sorted(model.connectors, key=lambda item: item.uid):
            if connector.points_locked:
                continue
            old_route = tuple(connector.points)
            before = _routing_score(model)
            existing_routes = tuple(
                tuple(flow.points)
                for name, flow in sorted(model.flows.items())
                if name != connector.to_var
            ) + tuple(
                tuple(other.points)
                for other in sorted(model.connectors, key=lambda item: item.uid)
                if other.uid != connector.uid and other.points
            )
            route, _ = _route_connector(model, connector, existing_routes)
            connector.points = list(route)
            if _routing_score(model) < before:
                accepted = True
            else:
                connector.points = list(old_route)
        if not accepted:
            break
        hit_cap = pass_index == MAX_ROUTING_PASSES - 2
    if hit_cap:
        return (
            LayoutWarning(
                "layout.routing_fallback",
                "Routing reached the bounded strict-improvement pass cap.",
            ),
        )
    return ()


def _label_route_conflicts(model, name: str, box: LayoutBox) -> int:
    count = 0
    for flow_name, flow in model.flows.items():
        if flow_name == name:
            continue
        for start, end in route_segments(flow.points):
            if segment_intersects_box(start, end, box.as_bounding_box()):
                count += 1
    for connector in model.connectors:
        if name in {connector.from_var, connector.to_var}:
            continue
        for start, end in route_segments(connector.points):
            if segment_intersects_box(start, end, box.as_bounding_box()):
                count += 1
    return count


def _choose_label_sides(model) -> tuple[LayoutWarning, ...]:
    glyphs = _all_glyph_boxes(model)
    chosen: dict[str, LayoutBox] = {}
    warnings: list[LayoutWarning] = []
    registries = {**model.stocks, **model.flows, **model.auxs}

    def candidate_sides(name: str) -> tuple[str, ...]:
        element = registries[name]
        if element.position_source == "user" and element.label_side in LABEL_SIDES:
            return (element.label_side,)
        if name in model.stocks:
            preferred = "top"
        elif name in model.auxs and element.y is not None:
            preferred = "top" if element.y > model.view_page_height / 2 else "bottom"
        else:
            preferred = "bottom"
        return (preferred,) + tuple(side for side in LABEL_SIDES if side != preferred)

    for name in sorted(registries):
        scored = []
        candidates = candidate_sides(name)
        for preference, side in enumerate(candidates):
            box = estimate_label_box(
                glyphs[name],
                model._display_name(registries[name].name),
                side,
                label_font_points(model, name),
            )
            glyph_overlaps = sum(
                box.as_bounding_box().intersects(glyph.as_bounding_box())
                for glyph_name, glyph in glyphs.items()
                if glyph_name != name
            )
            label_overlaps = sum(
                box.as_bounding_box().intersects(other.as_bounding_box())
                for other in chosen.values()
            )
            scored.append(
                (
                    (
                        glyph_overlaps + label_overlaps,
                        _label_route_conflicts(model, name, box),
                        preference,
                        side,
                    ),
                    side,
                    box,
                )
            )
        score, side, box = min(scored)
        registries[name].label_side = side
        chosen[name] = box
        if score[0] > 0 or score[1] > 0:
            warnings.append(
                LayoutWarning(
                    "layout.label_conflict",
                    "No label side avoided every glyph, label, and route.",
                    (name,),
                )
            )
    return tuple(warnings)


def _selected_label_boxes(model) -> dict[str, LayoutBox]:
    glyphs = _all_glyph_boxes(model)
    return {
        name: estimate_label_box(
            glyphs[name],
            model._display_name(element.name),
            element.label_side or "bottom",
            label_font_points(model, name),
        )
        for name, element in sorted({**model.stocks, **model.flows, **model.auxs}.items())
    }


def _violation_score(metrics: LayoutMetrics) -> tuple[int, ...]:
    return (
        len(metrics.missing_positions),
        len(metrics.glyph_overlaps),
        len(metrics.label_glyph_overlaps),
        len(metrics.label_label_overlaps),
        len(metrics.flow_glyph_crossings),
        len(metrics.connector_glyph_crossings),
        len(metrics.flow_label_crossings),
        len(metrics.connector_label_crossings),
        len(metrics.connector_flow_crossings),
        len(metrics.connector_connector_crossings),
        len(metrics.flow_flow_crossings),
        len(metrics.flow_shared_segments),
        len(metrics.route_self_intersections),
        len(metrics.repeated_route_points),
        len(metrics.redundant_route_points),
        len(metrics.avoidable_route_detours),
        len(metrics.backward_flow_edges),
    )


def _repair_connector_label_crossings(model) -> bool:
    """Reroute only connectors whose selected label boxes block their path."""
    changed = False
    for _ in range(MAX_ROUTING_PASSES):
        metrics = analyze_layout(model)
        conflicted_uids = {uid for uid, _ in metrics.connector_label_crossings}
        if not conflicted_uids:
            break
        accepted = False
        label_obstacles = tuple(_selected_label_boxes(model).values())
        for connector in sorted(model.connectors, key=lambda item: item.uid):
            if str(connector.uid) not in conflicted_uids or connector.points_locked:
                continue
            original = tuple(connector.points)
            before = _violation_score(analyze_layout(model))
            existing_routes = tuple(
                tuple(flow.points)
                for name, flow in sorted(model.flows.items())
                if name != connector.to_var
            ) + tuple(
                tuple(other.points)
                for other in sorted(model.connectors, key=lambda item: item.uid)
                if other.uid != connector.uid and other.points
            )
            route, _ = _route_connector(
                model,
                connector,
                existing_routes,
                label_obstacles,
            )
            connector.points = list(route)
            after = _violation_score(analyze_layout(model))
            if after < before:
                accepted = True
                changed = True
            else:
                connector.points = list(original)
        if not accepted:
            break
    return changed


def _label_side_snapshot(model) -> dict[str, str | None]:
    return {
        name: element.label_side
        for name, element in {**model.stocks, **model.flows, **model.auxs}.items()
    }


def _restore_label_sides(model, snapshot: dict[str, str | None]) -> None:
    for name, element in {**model.stocks, **model.flows, **model.auxs}.items():
        element.label_side = snapshot[name]


def _reroute_incident_connectors(model, aux_name: str) -> None:
    incident = [
        connector
        for connector in sorted(model.connectors, key=lambda item: item.uid)
        if aux_name in {connector.from_var, connector.to_var} and not connector.points_locked
    ]
    for connector in incident:
        existing_routes = tuple(
            tuple(flow.points)
            for name, flow in sorted(model.flows.items())
            if name != connector.to_var
        ) + tuple(
            tuple(other.points)
            for other in sorted(model.connectors, key=lambda item: item.uid)
            if other.uid != connector.uid and other.points
        )
        route, _ = _route_connector(model, connector, existing_routes)
        connector.points = list(route)


def _retry_one_label_blocker(model) -> bool:
    baseline_metrics = analyze_layout(model)
    baseline_score = _violation_score(baseline_metrics)
    conflicted_labels = {
        label_name
        for _, label_name in (
            *baseline_metrics.flow_label_crossings,
            *baseline_metrics.connector_label_crossings,
        )
    }
    conflicted_labels.update(label_name for label_name, _ in baseline_metrics.label_glyph_overlaps)
    conflicted_labels.update(
        name for pair in baseline_metrics.label_label_overlaps for name in pair
    )
    for label_name in sorted(conflicted_labels):
        target = element_box(model, label_name)
        if target is None:
            continue
        blockers = sorted(
            {
                connector.from_var
                for connector in model.connectors
                if connector.to_var == label_name
                and connector.from_var in model.auxs
                and model.auxs[connector.from_var].position_source == "auto"
            }
        )
        for blocker_name in blockers:
            blocker = model.auxs[blocker_name]
            original_position = (blocker.x, blocker.y)
            original_routes = {
                connector.uid: tuple(connector.points)
                for connector in model.connectors
                if blocker_name in {connector.from_var, connector.to_var}
            }
            original_labels = _label_side_snapshot(model)
            clearance = max(target.width, target.height) / 2 + AUX_RADIUS + _SEPARATION_GAP
            best: (
                tuple[
                    tuple[int, ...],
                    Point,
                    dict[int, tuple[Point, ...]],
                    dict[str, str | None],
                ]
                | None
            ) = None
            for ring in range(MAX_AUX_RINGS):
                for candidate in _candidate_aux_positions((target.x, target.y), clearance, ring):
                    blocker.x, blocker.y = candidate
                    _reroute_incident_connectors(model, blocker_name)
                    _choose_label_sides(model)
                    score = _violation_score(analyze_layout(model))
                    record = (
                        score,
                        candidate,
                        {
                            connector.uid: tuple(connector.points)
                            for connector in model.connectors
                            if blocker_name in {connector.from_var, connector.to_var}
                        },
                        _label_side_snapshot(model),
                    )
                    if not any(score):
                        return True
                    best = min(best, record) if best is not None else record
                    blocker.x, blocker.y = original_position
                    for connector in model.connectors:
                        if connector.uid in original_routes:
                            connector.points = list(original_routes[connector.uid])
                    _restore_label_sides(model, original_labels)
            if best is not None and best[0] < baseline_score:
                _, position, routes, labels = best
                blocker.x, blocker.y = position
                for connector in model.connectors:
                    if connector.uid in routes:
                        connector.points = list(routes[connector.uid])
                _restore_label_sides(model, labels)
                return True
    return False


def _retry_label_blockers(model) -> bool:
    changed = False
    for _ in range(max(1, len(model.auxs))):
        if not _retry_one_label_blocker(model):
            break
        changed = True
        if not any(_violation_score(analyze_layout(model))):
            break
    return changed


def _shift_unpinned_layout(model, dx: float, dy: float) -> None:
    dx = float(math.ceil(dx))
    dy = float(math.ceil(dy))
    for registry in (model.stocks, model.flows, model.auxs):
        for element in registry.values():
            if element.x is not None:
                element.x = float(round(element.x + dx))
            if element.y is not None:
                element.y = float(round(element.y + dy))
    for flow in model.flows.values():
        if not flow.points_locked:
            flow.points = [(float(round(x + dx)), float(round(y + dy))) for x, y in flow.points]
    for connector in model.connectors:
        if not connector.points_locked:
            connector.points = [
                (float(round(x + dx)), float(round(y + dy))) for x, y in connector.points
            ]
    for module in model.modules.values():
        if module.x is not None:
            module.x += dx
        if module.y is not None:
            module.y += dy


def _select_page_geometry(model, has_pins: bool) -> tuple[LayoutWarning, ...]:
    metrics = analyze_layout(model)
    left, top, right, bottom = metrics.bounds
    warnings: list[LayoutWarning] = []
    if not has_pins:
        dx = math.ceil(MARGIN - left)
        dy = math.ceil(MARGIN - top)
        if right - left <= model.view_page_width:
            dx = min(dx, math.floor(model.view_page_width - right))
        if bottom - top <= model.view_page_height:
            dy = min(dy, math.floor(model.view_page_height - bottom))
        dx = max(dx, math.ceil(-left))
        dy = max(dy, math.ceil(-top))
        _shift_unpinned_layout(model, dx, dy)
        metrics = analyze_layout(model)
        _, _, right, bottom = metrics.bounds
    elif left < 0 or top < 0:
        warnings.append(
            LayoutWarning(
                "layout.page_overflow",
                "Pinned geometry extends beyond the reachable page grid.",
            )
        )
    required_columns = max(1, math.ceil(right / model.view_page_width))
    required_rows = max(1, math.ceil(bottom / model.view_page_height))
    if has_pins:
        model.view_page_columns = max(model.view_page_columns, required_columns)
        model.view_page_rows = max(model.view_page_rows, required_rows)
    else:
        model.view_page_columns = required_columns
        model.view_page_rows = required_rows
    return tuple(warnings)


def _result_from_model(
    model,
    metrics: LayoutMetrics,
    warnings: tuple[LayoutWarning, ...],
) -> LayoutResult:
    positions = tuple(
        (name, (element.x, element.y))
        for name, element in sorted({**model.stocks, **model.flows, **model.auxs}.items())
        if element.x is not None and element.y is not None
    )
    flow_routes = tuple(
        LayoutRoute(
            name,
            "flow",
            tuple(flow.points),
            tuple(item for item in (flow.from_stock, flow.to_stock) if item),
            flow.points_locked,
        )
        for name, flow in sorted(model.flows.items())
    )
    connector_routes = tuple(
        LayoutRoute(
            str(connector.uid),
            "connector",
            tuple(connector.points),
            (connector.from_var, connector.to_var),
            connector.points_locked,
        )
        for connector in sorted(model.connectors, key=lambda item: item.uid)
    )
    label_sides = tuple(
        (name, element.label_side or "bottom")
        for name, element in sorted({**model.stocks, **model.flows, **model.auxs}.items())
    )
    return LayoutResult(
        positions=positions,
        flow_routes=flow_routes,
        connector_routes=connector_routes,
        label_sides=label_sides,
        viewport=LayoutViewport(
            model.view_page_width,
            model.view_page_height,
            model.view_page_columns,
            model.view_page_rows,
        ),
        metrics=metrics,
        warnings=warnings,
    )


def _locked_route_conflicts(model, metrics: LayoutMetrics) -> tuple[str, ...]:
    locked_flows = {name for name, flow in model.flows.items() if flow.points_locked}
    locked_connectors = {
        str(connector.uid) for connector in model.connectors if connector.points_locked
    }
    conflicts = set(metrics.locked_route_movements)
    for flow_name, _ in (*metrics.flow_glyph_crossings, *metrics.flow_label_crossings):
        if flow_name in locked_flows:
            conflicts.add(f"flow:{flow_name}")
    for connector_name, _ in (
        *metrics.connector_glyph_crossings,
        *metrics.connector_label_crossings,
    ):
        if connector_name in locked_connectors:
            conflicts.add(f"connector:{connector_name}")
    for connector_name, flow_name in metrics.connector_flow_crossings:
        if connector_name in locked_connectors:
            conflicts.add(f"connector:{connector_name}")
        if flow_name in locked_flows:
            conflicts.add(f"flow:{flow_name}")
    for first, second in metrics.connector_connector_crossings:
        for connector_name in (first, second):
            if connector_name in locked_connectors:
                conflicts.add(f"connector:{connector_name}")
    for first, second in (*metrics.flow_flow_crossings, *metrics.flow_shared_segments):
        for flow_name in (first, second):
            if flow_name in locked_flows:
                conflicts.add(f"flow:{flow_name}")
    locked_keys = {
        *(f"flow:{name}" for name in locked_flows),
        *(f"connector:{name}" for name in locked_connectors),
    }
    conflicts.update(
        route_name
        for route_name in (
            *metrics.route_self_intersections,
            *metrics.repeated_route_points,
            *metrics.redundant_route_points,
        )
        if route_name in locked_keys
    )
    return tuple(sorted(conflicts))


def _pinned_conflicts(
    pinned_names: set[str],
    metrics: LayoutMetrics,
    warnings: list[LayoutWarning],
) -> tuple[str, ...]:
    conflicts = {
        element
        for warning in warnings
        if warning.code == "layout.pinned_conflict"
        for element in warning.elements
    }
    conflicts.update(metrics.pinned_position_movements)
    pair_metrics = (
        *metrics.glyph_overlaps,
        *metrics.label_glyph_overlaps,
        *metrics.label_label_overlaps,
        *metrics.flow_glyph_crossings,
        *metrics.connector_glyph_crossings,
        *metrics.flow_label_crossings,
        *metrics.connector_label_crossings,
    )
    conflicts.update(
        name
        for pair in pair_metrics
        if any(name in pinned_names for name in pair)
        for name in pair
        if name in pinned_names
    )
    return tuple(sorted(conflicts))


def run_layout_pipeline(model) -> LayoutResult:
    """Run the complete deterministic placement, routing, and validation pipeline."""
    authored_names = {
        name
        for registry in (model.stocks, model.flows, model.auxs)
        for name, element in registry.items()
        if element.position_source == "user" and (element.x is not None or element.y is not None)
    }
    pinned = _pinned_positions(model)
    locked = _locked_routes(model)
    model._calculate_stock_sizes()
    graph, backbone_warnings = place_stock_backbone(model)
    assign_provisional_flow_routes(model, graph)
    radial_preferences = _aux_radial_preferences(model, graph)
    placement_warnings = _place_auxiliaries(model, radial_preferences)
    routing_warnings = _route_everything(model, graph)
    retry_warnings: tuple[LayoutWarning, ...] = ()
    preferred_sources = _conflicted_auxiliary_sources(model)
    for name, sources in sorted(preferred_sources.items()):
        before_retry = _routing_score(model)
        aux_snapshot = {
            name: (aux.x, aux.y, aux.position_source) for name, aux in model.auxs.items()
        }
        flow_snapshot = {
            name: (tuple(flow.points), flow.x, flow.y, flow.position_source)
            for name, flow in model.flows.items()
        }
        connector_snapshot = {
            connector.uid: tuple(connector.points) for connector in model.connectors
        }
        proposed_retry_warnings = _place_auxiliaries(
            model,
            radial_preferences,
            {name},
            preferred_sources={name: sources},
        )
        proposed_routing_warnings = _route_everything(model, graph)
        if _routing_score(model) < before_retry:
            retry_warnings += proposed_retry_warnings
            routing_warnings += proposed_routing_warnings
        else:
            for name, (x, y, source) in aux_snapshot.items():
                model.auxs[name].x = x
                model.auxs[name].y = y
                model.auxs[name].position_source = source
            for name, (points, x, y, source) in flow_snapshot.items():
                model.flows[name].points = list(points)
                model.flows[name].x = x
                model.flows[name].y = y
                model.flows[name].position_source = source
            for connector in model.connectors:
                connector.points = list(connector_snapshot[connector.uid])
    label_warnings = _choose_label_sides(model)
    if _retry_label_blockers(model):
        label_warnings = _choose_label_sides(model)
    if _repair_connector_label_crossings(model):
        label_warnings = _choose_label_sides(model)
    if model.modules:
        model.auto_place_module_boxes()
    page_warnings = _select_page_geometry(model, bool(authored_names or locked))
    model._calculate_connector_angles(force=True)
    metrics = analyze_layout(
        model,
        pinned_reference=pinned,
        locked_route_reference=locked,
    )
    warnings = list(
        backbone_warnings
        + placement_warnings
        + routing_warnings
        + retry_warnings
        + label_warnings
        + page_warnings
    )
    pinned_conflicts = _pinned_conflicts(authored_names, metrics, warnings)
    if pinned_conflicts:
        warnings = [warning for warning in warnings if warning.code != "layout.pinned_conflict"]
        warnings.append(
            LayoutWarning(
                "layout.pinned_conflict",
                "Pinned geometry prevents a conflict-free layout.",
                pinned_conflicts,
            )
        )
    crossing_elements = {
        *metrics.connector_flow_crossings,
        *metrics.connector_connector_crossings,
        *metrics.flow_flow_crossings,
        *metrics.flow_shared_segments,
    }
    if crossing_elements:
        names = tuple(sorted({name for pair in crossing_elements for name in pair}))
        warnings.append(
            LayoutWarning(
                "layout.unavoidable_crossing",
                "At least one line crossing remained after deterministic routing.",
                names,
            )
        )
    locked_conflicts = _locked_route_conflicts(model, metrics)
    if locked_conflicts:
        warnings.append(
            LayoutWarning(
                "layout.locked_route_conflict",
                "Locked route geometry conflicts with another visual element or route.",
                locked_conflicts,
            )
        )
    if not (
        metrics.glyph_overlaps or metrics.flow_glyph_crossings or metrics.connector_glyph_crossings
    ):
        warnings = [warning for warning in warnings if warning.code != "layout.placement_exhausted"]
    warnings = sorted(set(warnings), key=lambda warning: (warning.code, warning.elements))
    result = _result_from_model(model, metrics, tuple(warnings))
    model.last_layout_result = result
    model.last_layout_metrics = metrics
    model.layout_warnings = list(result.warnings)
    return result
