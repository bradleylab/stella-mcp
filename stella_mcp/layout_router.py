"""Deterministic boundary ports and obstacle-aware polyline routing."""

from __future__ import annotations

import heapq
import math
from collections import defaultdict

from stella_mcp.layout import (
    _SEPARATION_GAP,
    SegmentIntersection,
    segment_intersection_kind,
    segment_intersects_box,
)
from stella_mcp.layout_graph import DirectedStockGraph
from stella_mcp.layout_quality import LayoutBox, LayoutPort, Point

MAX_ENUMERATED_BENDS = 4


def route_segments(points: tuple[Point, ...] | list[Point]) -> tuple[tuple[Point, Point], ...]:
    """Return adjacent segments for one route."""
    return tuple(zip(points, points[1:], strict=False))


def is_orthogonal_route(points: tuple[Point, ...] | list[Point]) -> bool:
    """Return whether every route segment is horizontal or vertical."""
    return all(
        start[0] == end[0] or start[1] == end[1]
        for start, end in route_segments(points)
    )


def normalize_route(points: tuple[Point, ...] | list[Point]) -> tuple[Point, ...]:
    """Snap, deduplicate, and remove collinear interior route points."""
    snapped: list[Point] = []
    for x, y in points:
        point = (float(round(x)), float(round(y)))
        if not snapped or point != snapped[-1]:
            snapped.append(point)
    changed = True
    while changed and len(snapped) >= 3:
        changed = False
        reduced = [snapped[0]]
        for index, point in enumerate(snapped[1:-1], start=1):
            previous = reduced[-1]
            following = snapped[index + 1]
            cross = (point[0] - previous[0]) * (following[1] - point[1]) - (
                point[1] - previous[1]
            ) * (following[0] - point[0])
            if cross == 0:
                changed = True
                continue
            reduced.append(point)
        reduced.append(snapped[-1])
        snapped = reduced
    return tuple(snapped)


def point_at_half_length(points: tuple[Point, ...] | list[Point]) -> Point:
    """Return the whole-pixel point halfway along a polyline."""
    segments = route_segments(points)
    total = sum(math.dist(start, end) for start, end in segments)
    if total == 0:
        return points[0] if points else (0.0, 0.0)
    remaining = total / 2
    for start, end in segments:
        length = math.dist(start, end)
        if remaining <= length:
            ratio = remaining / length if length else 0.0
            return (
                float(round(start[0] + ratio * (end[0] - start[0]))),
                float(round(start[1] + ratio * (end[1] - start[1]))),
            )
        remaining -= length
    return points[-1]


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


def element_box(model, name: str) -> LayoutBox | None:
    """Return the visual glyph box for any positioned element."""
    if name in model.stocks:
        stock = model.stocks[name]
        if stock.x is not None and stock.y is not None:
            return LayoutBox(
                name,
                "stock",
                stock.x,
                stock.y,
                stock.width,
                stock.height,
                locked=stock.position_source == "user",
            )
    if name in model.auxs:
        aux = model.auxs[name]
        if aux.x is not None and aux.y is not None:
            return LayoutBox(
                name,
                "aux",
                aux.x,
                aux.y,
                36.0,
                36.0,
                locked=aux.position_source == "user",
            )
    if name in model.flows:
        flow = model.flows[name]
        if flow.x is not None and flow.y is not None:
            return LayoutBox(
                name,
                "flow",
                flow.x,
                flow.y,
                20.0,
                20.0,
                locked=flow.position_source == "user",
            )
    return None


def boundary_port(box: LayoutBox, target: Point) -> LayoutPort:
    """Return the box boundary point facing a target coordinate."""
    dx = target[0] - box.x
    dy = target[1] - box.y
    if abs(dx) / max(box.width, 1.0) >= abs(dy) / max(box.height, 1.0):
        direction = "right" if dx >= 0 else "left"
        point = (box.right if dx >= 0 else box.left, box.y)
    else:
        direction = "bottom" if dy >= 0 else "top"
        point = (box.x, box.bottom if dy >= 0 else box.top)
    return LayoutPort(box.name, (float(round(point[0])), float(round(point[1]))), direction)


def _expanded(box: LayoutBox, gap: float = _SEPARATION_GAP) -> LayoutBox:
    return LayoutBox(
        box.name,
        box.kind,
        box.x,
        box.y,
        box.width + 2 * gap,
        box.height + 2 * gap,
        locked=box.locked,
    )


def _point_inside(point: Point, box: LayoutBox) -> bool:
    return box.left < point[0] < box.right and box.top < point[1] < box.bottom


def _axis_segment_crosses_interior(start: Point, end: Point, box: LayoutBox) -> bool:
    if start[0] == end[0]:
        x = start[0]
        low, high = sorted((start[1], end[1]))
        return box.left < x < box.right and max(low, box.top) < min(high, box.bottom)
    if start[1] == end[1]:
        y = start[1]
        low, high = sorted((start[0], end[0]))
        return box.top < y < box.bottom and max(low, box.left) < min(high, box.right)
    return segment_intersects_box(start, end, box.as_bounding_box())


def score_route(
    points: tuple[Point, ...],
    obstacles: tuple[LayoutBox, ...],
    existing_routes: tuple[tuple[Point, ...], ...],
) -> tuple[int, int, float, int, tuple[Point, ...]]:
    segments = route_segments(points)
    obstacle_boxes = tuple(obstacle.as_bounding_box() for obstacle in obstacles)
    existing_segments = tuple(
        segment for existing in existing_routes for segment in route_segments(existing)
    )
    hard = sum(
        any(
            segment_intersects_box(start, end, obstacle_box)
            for start, end in segments
        )
        for obstacle_box in obstacle_boxes
    )
    crossings = 0
    for start, end in segments:
        for other_start, other_end in existing_segments:
            kind = segment_intersection_kind(start, end, other_start, other_end)
            if kind is not SegmentIntersection.NONE:
                crossings += 1
    length = sum(math.dist(start, end) for start, end in segments)
    bends = max(0, len(points) - 2)
    return hard, crossings, length, bends, points


def _enumerated_candidates(
    start: Point,
    end: Point,
    obstacles: tuple[LayoutBox, ...],
    existing_routes: tuple[tuple[Point, ...], ...],
    *,
    orthogonal_only: bool = False,
) -> tuple[tuple[Point, ...], ...]:
    sx, sy = start
    ex, ey = end
    axis_routes = tuple(
        route
        for route in existing_routes
        if any(
            segment_intersection_kind(start, end, other_start, other_end)
            is not SegmentIntersection.NONE
            for other_start, other_end in route_segments(route)
        )
    )
    mid_x = float(round((sx + ex) / 2))
    mid_y = float(round((sy + ey) / 2))
    candidates: list[tuple[Point, ...]] = [
        (start, (sx, ey), end),
        (start, (ex, sy), end),
        (start, (mid_x, sy), (mid_x, ey), end),
        (start, (sx, mid_y), (ex, mid_y), end),
    ]
    if not orthogonal_only or sx == ex or sy == ey:
        candidates.insert(0, (start, end))
    x_axes = sorted(
        {
            float(round(axis))
            for axis in (
                *(
                    value
                    for box in obstacles
                    for value in (box.left - _SEPARATION_GAP, box.right + _SEPARATION_GAP)
                ),
                *(
                    point[0] + offset
                    for route in axis_routes
                    for point in route
                    for offset in (-_SEPARATION_GAP, _SEPARATION_GAP)
                ),
            )
        }
    )
    y_axes = sorted(
        {
            float(round(axis))
            for axis in (
                *(
                    value
                    for box in obstacles
                    for value in (box.top - _SEPARATION_GAP, box.bottom + _SEPARATION_GAP)
                ),
                *(
                    point[1] + offset
                    for route in axis_routes
                    for point in route
                    for offset in (-_SEPARATION_GAP, _SEPARATION_GAP)
                ),
            )
        }
    )
    for axis in x_axes:
        candidates.append((start, (axis, sy), (axis, ey), end))
    for axis in y_axes:
        candidates.append((start, (sx, axis), (ex, axis), end))
    if x_axes and y_axes:
        for x_axis, y_axis in (
            (x_axes[0], y_axes[0]),
            (x_axes[0], y_axes[-1]),
            (x_axes[-1], y_axes[0]),
            (x_axes[-1], y_axes[-1]),
        ):
            candidates.append((start, (x_axis, sy), (x_axis, y_axis), (ex, y_axis), end))
    return tuple(dict.fromkeys(normalize_route(candidate) for candidate in candidates))


def _visibility_route(
    start: Point,
    end: Point,
    obstacles: tuple[LayoutBox, ...],
    existing_routes: tuple[tuple[Point, ...], ...],
) -> tuple[Point, ...] | None:
    expanded = tuple(
        box
        if _point_inside(start, _expanded(box)) or _point_inside(end, _expanded(box))
        else _expanded(box)
        for box in obstacles
    )
    axis_routes = tuple(
        route
        for route in existing_routes
        if any(
            segment_intersection_kind(start, end, other_start, other_end)
            is not SegmentIntersection.NONE
            for other_start, other_end in route_segments(route)
        )
    )
    x_axes = sorted(
        {
            float(round(axis))
            for axis in (
                start[0],
                end[0],
                *(value for box in expanded for value in (box.left, box.right)),
                *(
                    point[0] + offset
                    for route in axis_routes
                    for point in (route[0], route[-1])
                    for offset in (-_SEPARATION_GAP, 0.0, _SEPARATION_GAP)
                ),
            )
        }
    )
    y_axes = sorted(
        {
            float(round(axis))
            for axis in (
                start[1],
                end[1],
                *(value for box in expanded for value in (box.top, box.bottom)),
                *(
                    point[1] + offset
                    for route in axis_routes
                    for point in (route[0], route[-1])
                    for offset in (-_SEPARATION_GAP, 0.0, _SEPARATION_GAP)
                ),
            )
        }
    )
    vertices = {
        (float(round(x)), float(round(y)))
        for x in x_axes
        for y in y_axes
        if not any(_point_inside((x, y), box) for box in expanded)
    }
    vertices.update((start, end))
    adjacency: dict[Point, list[Point]] = defaultdict(list)
    existing_segments = tuple(
        segment for route in existing_routes for segment in route_segments(route)
    )
    for x in x_axes:
        points = sorted((point for point in vertices if point[0] == x), key=lambda point: point[1])
        for first, second in zip(points, points[1:], strict=False):
            if not any(_axis_segment_crosses_interior(first, second, box) for box in expanded):
                adjacency[first].append(second)
                adjacency[second].append(first)
    for y in y_axes:
        points = sorted((point for point in vertices if point[1] == y), key=lambda point: point[0])
        for first, second in zip(points, points[1:], strict=False):
            if not any(_axis_segment_crosses_interior(first, second, box) for box in expanded):
                adjacency[first].append(second)
                adjacency[second].append(first)

    queue: list[tuple[int, float, int, tuple[Point, ...], Point, str]] = [
        (0, 0.0, 0, (start,), start, "")
    ]
    best: dict[tuple[Point, str], tuple[int, float, int, tuple[Point, ...]]] = {}
    while queue:
        crossings, length, bends, path, point, direction = heapq.heappop(queue)
        if point == end:
            return normalize_route(path)
        state = (point, direction)
        cost = (crossings, length, bends, path)
        if best.get(state, (math.inf, math.inf, math.inf, ())) <= cost:
            continue
        best[state] = cost
        for neighbor in sorted(adjacency[point]):
            next_direction = "h" if neighbor[1] == point[1] else "v"
            next_bends = bends + int(bool(direction) and next_direction != direction)
            next_crossings = crossings + sum(
                segment_intersection_kind(point, neighbor, other_start, other_end)
                is not SegmentIntersection.NONE
                for other_start, other_end in existing_segments
            )
            heapq.heappush(
                queue,
                (
                    next_crossings,
                    length + math.dist(point, neighbor),
                    next_bends,
                    path + (neighbor,),
                    neighbor,
                    next_direction,
                ),
            )
    return None


def route_between(
    start: Point,
    end: Point,
    obstacles: tuple[LayoutBox, ...] = (),
    existing_routes: tuple[tuple[Point, ...], ...] = (),
    *,
    allow_visibility: bool = True,
    orthogonal_only: bool = False,
) -> tuple[tuple[Point, ...], bool]:
    """Choose the lexicographically best enumerated or visibility route."""
    start = (float(round(start[0])), float(round(start[1])))
    end = (float(round(end[0])), float(round(end[1])))
    candidates = _enumerated_candidates(
        start,
        end,
        obstacles,
        existing_routes,
        orthogonal_only=orthogonal_only,
    )
    best = min(candidates, key=lambda points: score_route(points, obstacles, existing_routes))
    if score_route(best, obstacles, existing_routes)[:2] == (0, 0):
        return best, False
    if not allow_visibility:
        return best, False
    fallback = _visibility_route(start, end, obstacles, existing_routes)
    if fallback is None:
        return best, True
    return min(
        (best, fallback),
        key=lambda points: score_route(points, obstacles, existing_routes),
    ), True


def _distributed_ports(
    model, stock_name: str, flow_names: list[str], side: str
) -> dict[str, Point]:
    stock = model.stocks[stock_name]
    assert stock.x is not None and stock.y is not None
    result: dict[str, Point] = {}
    for index, flow_name in enumerate(flow_names):
        fraction = (index + 1) / (len(flow_names) + 1)
        if side in {"left", "right"}:
            x = stock.x + (stock.width / 2 if side == "right" else -stock.width / 2)
            y = stock.y - stock.height / 2 + fraction * stock.height
        else:
            x = stock.x - stock.width / 2 + fraction * stock.width
            y = stock.y + (stock.height / 2 if side == "bottom" else -stock.height / 2)
        result[flow_name] = (float(round(x)), float(round(y)))
    return result


def allocate_flow_ports(
    model, graph: DirectedStockGraph
) -> dict[str, tuple[Point | None, Point | None]]:
    """Allocate distinct stock-boundary ports for every incident flow."""
    component_by_node = graph.component_map()
    ranks = graph.rank_map()
    source_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    target_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for name, flow in sorted(model.flows.items()):
        if flow.from_stock == flow.to_stock:
            continue
        source = element_box(model, flow.from_stock) if flow.from_stock else None
        target = element_box(model, flow.to_stock) if flow.to_stock else None
        source_component = component_by_node.get(flow.from_stock)
        target_component = component_by_node.get(flow.to_stock)
        follows_rank_direction = (
            source_component is not None
            and target_component is not None
            and source_component != target_component
            and ranks[target_component] > ranks[source_component]
        )
        if source is not None:
            source_side = (
                "right"
                if follows_rank_direction
                else boundary_port(source, (target.x, target.y)).direction
                if target is not None
                else "right"
            )
            source_groups[(flow.from_stock, source_side)].append(name)
        if target is not None:
            target_side = (
                "left"
                if follows_rank_direction
                else boundary_port(target, (source.x, source.y)).direction
                if source is not None
                else "left"
            )
            target_groups[(flow.to_stock, target_side)].append(name)

    source_ports: dict[str, Point] = {}
    target_ports: dict[str, Point] = {}
    for (stock_name, side), flow_names in source_groups.items():
        flow_names.sort(
            key=lambda name: (
                ranks.get(component_by_node.get(model.flows[name].to_stock, -1), math.inf),
                (
                    model.stocks[model.flows[name].to_stock].y
                    if side in {"left", "right"}
                    else model.stocks[model.flows[name].to_stock].x
                )
                if model.flows[name].to_stock in model.stocks
                else math.inf,
                name,
            )
        )
        source_ports.update(_distributed_ports(model, stock_name, flow_names, side))
    for (stock_name, side), flow_names in target_groups.items():
        flow_names.sort(
            key=lambda name: (
                ranks.get(component_by_node.get(model.flows[name].from_stock, -1), -math.inf),
                (
                    model.stocks[model.flows[name].from_stock].y
                    if side in {"left", "right"}
                    else model.stocks[model.flows[name].from_stock].x
                )
                if model.flows[name].from_stock in model.stocks
                else -math.inf,
                name,
            )
        )
        target_ports.update(_distributed_ports(model, stock_name, flow_names, side))

    return {name: (source_ports.get(name), target_ports.get(name)) for name in sorted(model.flows)}


def fanout_channel_routes(
    model,
    graph: DirectedStockGraph,
    ports: dict[str, tuple[Point | None, Point | None]],
) -> dict[str, tuple[Point, ...]]:
    """Assign crossing-free orthogonal channels to directed high-degree fan-outs."""
    component_by_node = graph.component_map()
    ranks = graph.rank_map()
    groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    for name, flow in sorted(model.flows.items()):
        source_component = component_by_node.get(flow.from_stock)
        target_component = component_by_node.get(flow.to_stock)
        start, end = ports[name]
        if (
            source_component is None
            or target_component is None
            or source_component == target_component
            or ranks[target_component] <= ranks[source_component]
            or start is None
            or end is None
        ):
            continue
        groups[(flow.from_stock, ranks[target_component])].append(name)

    routes: dict[str, tuple[Point, ...]] = {}
    for names in groups.values():
        if len(names) < 3:
            continue
        ordered = sorted(names, key=lambda name: (ports[name][1][1], name))
        upward = [name for name in ordered if ports[name][1][1] < ports[name][0][1]]
        downward = [name for name in ordered if ports[name][1][1] >= ports[name][0][1]]
        for partition, reverse_slots in ((upward, False), (downward, True)):
            if not partition:
                continue
            source_x = ports[partition[0]][0][0]
            target_x = min(ports[name][1][0] for name in partition)
            span = target_x - source_x
            if span <= 2 * _SEPARATION_GAP:
                continue
            for index, name in enumerate(partition):
                slot = len(partition) - index if reverse_slots else index + 1
                channel_x = float(round(source_x + span * slot / (len(partition) + 1)))
                start, end = ports[name]
                routes[name] = normalize_route(
                    (start, (channel_x, start[1]), (channel_x, end[1]), end)
                )
    return routes


def assign_provisional_flow_routes(model, graph: DirectedStockGraph) -> None:
    """Assign typed provisional flow routes and half-length valves."""
    ports = allocate_flow_ports(model, graph)
    channel_routes = fanout_channel_routes(model, graph, ports)
    positioned_y = [stock.y for stock in model.stocks.values() if stock.y is not None]
    orphan_y = (max(positioned_y) + 120.0) if positioned_y else 100.0
    orphan_index = 0
    for name, flow in sorted(model.flows.items()):
        if flow.points_locked and flow.points:
            route = tuple(flow.points)
        elif flow.from_stock == flow.to_stock and flow.from_stock in model.stocks:
            stock = model.stocks[flow.from_stock]
            assert stock.x is not None and stock.y is not None
            top = float(round(stock.y - stock.height / 2))
            extent = float(round(max(stock.width, stock.height) / 2 + _SEPARATION_GAP))
            right_port = float(round(stock.x + stock.width / 4))
            left_port = float(round(stock.x - stock.width / 4))
            route = normalize_route(
                (
                    (right_port, top),
                    (right_port, top - extent),
                    (left_port, top - extent),
                    (left_port, top),
                )
            )
        elif name in channel_routes:
            preferred = channel_routes[name]
            start, end = ports[name]
            if is_orthogonal_route(preferred):
                route = preferred
            elif start is not None and end is not None:
                route, _ = route_between(start, end, orthogonal_only=True)
            else:
                route = preferred
        else:
            start, end = ports[name]
            if start is None and end is None:
                x = 100.0 + orphan_index * 120.0
                route = ((x, orphan_y), (x + 60.0, orphan_y))
                orphan_index += 1
            elif start is None:
                route = ((end[0] - 140.0, end[1]), end)
            elif end is None:
                route = (start, (start[0] + 140.0, start[1]))
            else:
                route, _ = route_between(start, end, orthogonal_only=True)
        flow.points = list(route)
        _fill_or_set_position(flow, point_at_half_length(route))
