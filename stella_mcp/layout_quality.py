"""Deterministic layout metrics and result types for Stella diagrams."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

from stella_mcp.layout import (
    BoundingBox,
    SegmentIntersection,
    segment_intersection_kind,
    segment_intersects_box,
)
from stella_mcp.model_types import (
    AUX_RADIUS,
    DEFAULT_VIEW_FONT_POINTS,
    DEFAULT_VIEW_PAGE_COLUMNS,
    DEFAULT_VIEW_PAGE_HEIGHT,
    DEFAULT_VIEW_PAGE_ROWS,
    DEFAULT_VIEW_PAGE_WIDTH,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from stella_mcp.xmile import StellaModel

Point = tuple[float, float]
LabelSide = Literal["top", "bottom", "left", "right"]

DEFAULT_PAGE_WIDTH = DEFAULT_VIEW_PAGE_WIDTH
DEFAULT_PAGE_HEIGHT = DEFAULT_VIEW_PAGE_HEIGHT
DEFAULT_PAGE_COLUMNS = DEFAULT_VIEW_PAGE_COLUMNS
DEFAULT_PAGE_ROWS = DEFAULT_VIEW_PAGE_ROWS
DEFAULT_FONT_POINTS = DEFAULT_VIEW_FONT_POINTS
CSS_PIXELS_PER_POINT = 96.0 / 72.0
LABEL_WIDTH_EM = 0.6
LABEL_GAP = 1.0
FLOW_VALVE_SIZE = 20.0
# Derived before release gating from the retained Phase 1 baseline. Its largest
# finite route ratio is 4.4971846166, rounded up to the next quarter unit.
ROUTE_LENGTH_MANHATTAN_MULTIPLIER = 4.5
# The baseline's routed points use at most two bends. One obstacle-detour pair
# adds at most two more without permitting a second avoidable detour.
ROUTE_BEND_CAP = 4


@dataclass(frozen=True)
class LayoutViewport:
    """Stella page dimensions and grid extent."""

    page_width: float = DEFAULT_PAGE_WIDTH
    page_height: float = DEFAULT_PAGE_HEIGHT
    columns: int = DEFAULT_PAGE_COLUMNS
    rows: int = DEFAULT_PAGE_ROWS

    @property
    def width(self) -> float:
        return self.page_width * self.columns

    @property
    def height(self) -> float:
        return self.page_height * self.rows

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return the complete declared page-grid bounds."""
        return (0.0, 0.0, self.width, self.height)


@dataclass(frozen=True)
class LayoutBox:
    """Named rectangular obstacle in layout coordinates."""

    name: str
    kind: str
    x: float
    y: float
    width: float
    height: float
    locked: bool = False

    @property
    def left(self) -> float:
        return self.x - self.width / 2

    @property
    def right(self) -> float:
        return self.x + self.width / 2

    @property
    def top(self) -> float:
        return self.y - self.height / 2

    @property
    def bottom(self) -> float:
        return self.y + self.height / 2

    def as_bounding_box(self) -> BoundingBox:
        return BoundingBox(self.x, self.y, self.width, self.height)


@dataclass(frozen=True)
class LayoutPort:
    """Attachment point and outward direction on an element boundary."""

    owner: str
    point: Point
    direction: LabelSide


@dataclass(frozen=True)
class LayoutRoute:
    """Ordered points for a flow or connector route."""

    name: str
    kind: Literal["flow", "connector"]
    points: tuple[Point, ...]
    endpoints: tuple[str, ...] = ()
    locked: bool = False


@dataclass(frozen=True)
class LayoutWarning:
    """Stable warning emitted when a clean layout cannot be produced."""

    code: str
    message: str
    elements: tuple[str, ...] = ()


@dataclass(frozen=True)
class LayoutMetrics:
    """Complete deterministic quality record for one laid-out model."""

    missing_positions: tuple[str, ...] = ()
    glyph_overlaps: tuple[tuple[str, str], ...] = ()
    label_glyph_overlaps: tuple[tuple[str, str], ...] = ()
    label_label_overlaps: tuple[tuple[str, str], ...] = ()
    flow_glyph_crossings: tuple[tuple[str, str], ...] = ()
    connector_glyph_crossings: tuple[tuple[str, str], ...] = ()
    flow_label_crossings: tuple[tuple[str, str], ...] = ()
    connector_label_crossings: tuple[tuple[str, str], ...] = ()
    connector_flow_crossings: tuple[tuple[str, str], ...] = ()
    connector_connector_crossings: tuple[tuple[str, str], ...] = ()
    flow_flow_crossings: tuple[tuple[str, str], ...] = ()
    flow_shared_segments: tuple[tuple[str, str], ...] = ()
    route_self_intersections: tuple[str, ...] = ()
    repeated_route_points: tuple[str, ...] = ()
    redundant_route_points: tuple[str, ...] = ()
    avoidable_route_detours: tuple[str, ...] = ()
    backward_flow_edges: tuple[str, ...] = ()
    pinned_position_movements: tuple[str, ...] = ()
    locked_route_movements: tuple[str, ...] = ()
    total_flow_length: float = 0.0
    maximum_flow_length: float = 0.0
    total_connector_length: float = 0.0
    maximum_connector_length: float = 0.0
    total_bend_count: int = 0
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    page_overflow: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable metrics record."""
        return asdict(self)


@dataclass(frozen=True)
class LayoutResult:
    """Pure output of the complete layout pipeline."""

    positions: tuple[tuple[str, Point], ...]
    flow_routes: tuple[LayoutRoute, ...]
    connector_routes: tuple[LayoutRoute, ...]
    label_sides: tuple[tuple[str, LabelSide], ...]
    viewport: LayoutViewport
    metrics: LayoutMetrics
    warnings: tuple[LayoutWarning, ...] = ()


def layout_report_to_dict(result: LayoutResult | None) -> dict[str, object] | None:
    """Return the stable MCP/evaluator representation of a layout report."""
    if result is None:
        return None
    return {
        "viewport": asdict(result.viewport),
        "metrics": asdict(result.metrics),
        "warnings": [asdict(warning) for warning in result.warnings],
    }


def layout_warning_suffix(result: LayoutResult | None) -> str:
    """Return a concise deterministic suffix for non-clean layout reports."""
    if result is None or not result.warnings:
        return ""
    codes = ", ".join(warning.code for warning in result.warnings)
    return f" (layout warnings: {len(result.warnings)}; {codes})"


def _viewport(model: StellaModel) -> LayoutViewport:
    return LayoutViewport(
        page_width=float(getattr(model, "view_page_width", DEFAULT_PAGE_WIDTH)),
        page_height=float(getattr(model, "view_page_height", DEFAULT_PAGE_HEIGHT)),
        columns=int(getattr(model, "view_page_columns", DEFAULT_PAGE_COLUMNS)),
        rows=int(getattr(model, "view_page_rows", DEFAULT_PAGE_ROWS)),
    )


def _glyph_boxes(model: StellaModel) -> tuple[dict[str, LayoutBox], tuple[str, ...]]:
    boxes: dict[str, LayoutBox] = {}
    missing: list[str] = []
    for name, stock in sorted(model.stocks.items()):
        if stock.x is None or stock.y is None:
            missing.append(name)
            continue
        boxes[name] = LayoutBox(
            name,
            "stock",
            stock.x,
            stock.y,
            stock.width,
            stock.height,
            locked=stock.position_source == "user",
        )
    for name, aux in sorted(model.auxs.items()):
        if aux.x is None or aux.y is None:
            missing.append(name)
            continue
        boxes[name] = LayoutBox(
            name,
            "aux",
            aux.x,
            aux.y,
            AUX_RADIUS * 2,
            AUX_RADIUS * 2,
            locked=aux.position_source == "user",
        )
    for name, flow in sorted(model.flows.items()):
        if flow.x is None or flow.y is None:
            missing.append(name)
            continue
        boxes[name] = LayoutBox(
            name,
            "flow",
            flow.x,
            flow.y,
            FLOW_VALVE_SIZE,
            FLOW_VALVE_SIZE,
            locked=flow.position_source == "user",
        )
    return boxes, tuple(sorted(missing))


def _element_label_side(element: object) -> LabelSide:
    side = getattr(element, "label_side", None)
    return side if side in {"top", "bottom", "left", "right"} else "bottom"


def estimate_label_box(
    glyph: LayoutBox,
    display_name: str,
    side: LabelSide,
    font_points: float = DEFAULT_FONT_POINTS,
) -> LayoutBox:
    """Estimate a dependency-free label box from the view font and text."""
    font_pixels = font_points * CSS_PIXELS_PER_POINT
    width = max(font_pixels, len(display_name) * font_pixels * LABEL_WIDTH_EM)
    height = font_pixels
    if side == "top":
        x = glyph.x
        y = glyph.top - LABEL_GAP - height / 2
    elif side == "left":
        x = glyph.left - LABEL_GAP - width / 2
        y = glyph.y
    elif side == "right":
        x = glyph.right + LABEL_GAP + width / 2
        y = glyph.y
    else:
        x = glyph.x
        y = glyph.bottom + LABEL_GAP + height / 2
    return LayoutBox(f"label:{glyph.name}", "label", x, y, width, height)


def label_font_points(model: StellaModel, name: str) -> float:
    """Return the imported view-style font size for an element kind."""
    if name in model.stocks:
        return model.view_stock_font_points
    if name in model.flows:
        return model.view_flow_font_points
    if name in model.auxs:
        return model.view_aux_font_points
    return DEFAULT_FONT_POINTS


def _label_boxes(model: StellaModel, glyphs: Mapping[str, LayoutBox]) -> dict[str, LayoutBox]:
    result: dict[str, LayoutBox] = {}
    registries = (model.stocks, model.auxs, model.flows)
    for registry in registries:
        for name, element in sorted(registry.items()):
            glyph = glyphs.get(name)
            if glyph is None:
                continue
            result[name] = estimate_label_box(
                glyph,
                model._display_name(element.name),
                _element_label_side(element),
                label_font_points(model, name),
            )
    return result


def _flow_routes(model: StellaModel) -> tuple[LayoutRoute, ...]:
    routes: list[LayoutRoute] = []
    for name, flow in sorted(model.flows.items()):
        endpoints = tuple(
            endpoint for endpoint in (flow.from_stock, flow.to_stock) if endpoint is not None
        )
        routes.append(
            LayoutRoute(
                name=name,
                kind="flow",
                points=tuple(flow.points),
                endpoints=endpoints,
                locked=flow.points_locked,
            )
        )
    return tuple(routes)


def _connector_routes(
    model: StellaModel,
    glyphs: Mapping[str, LayoutBox],
) -> tuple[LayoutRoute, ...]:
    routes: list[LayoutRoute] = []
    for connector in sorted(model.connectors, key=lambda item: item.uid):
        points = tuple(connector.points)
        if not points:
            source = glyphs.get(connector.from_var)
            target = glyphs.get(connector.to_var)
            if source is not None and target is not None:
                points = ((source.x, source.y), (target.x, target.y))
        routes.append(
            LayoutRoute(
                name=str(connector.uid),
                kind="connector",
                points=points,
                endpoints=(connector.from_var, connector.to_var),
                locked=connector.points_locked,
            )
        )
    return tuple(routes)


def _segments(route: LayoutRoute) -> tuple[tuple[Point, Point], ...]:
    return tuple(zip(route.points, route.points[1:], strict=False))


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    cross = (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (
        point[0] - start[0]
    )
    return cross == 0 and (
        min(start[0], end[0]) <= point[0] <= max(start[0], end[0])
        and min(start[1], end[1]) <= point[1] <= max(start[1], end[1])
    )


def direct_segment_is_clear(
    start: Point,
    end: Point,
    obstacles: tuple[LayoutBox, ...],
    existing_routes: tuple[tuple[Point, ...], ...],
) -> bool:
    """Return whether a direct route is obstacle-free and shares no segment."""
    if start == end:
        return False
    if any(
        segment_intersects_box(start, end, obstacle.as_bounding_box()) for obstacle in obstacles
    ):
        return False
    for route in existing_routes:
        for other_start, other_end in zip(route, route[1:], strict=False):
            kind = segment_intersection_kind(start, end, other_start, other_end)
            if kind is SegmentIntersection.NONE:
                continue
            endpoint_touch = kind is SegmentIntersection.TOUCH and (
                _point_on_segment(start, other_start, other_end)
                or _point_on_segment(end, other_start, other_end)
            )
            if not endpoint_touch:
                return False
    return True


def _route_length(route: LayoutRoute) -> float:
    return sum(math.dist(start, end) for start, end in _segments(route))


def _route_has_self_intersection(route: LayoutRoute) -> bool:
    segments = _segments(route)
    for index, first in enumerate(segments):
        for other_index, second in enumerate(segments[index + 2 :], start=index + 2):
            if (
                index == 0
                and other_index == len(segments) - 1
                and route.points[0] == route.points[-1]
            ):
                continue
            kind = segment_intersection_kind(*first, *second)
            if kind in {SegmentIntersection.CROSS, SegmentIntersection.OVERLAP}:
                return True
    return False


def _has_repeated_points(route: LayoutRoute) -> bool:
    return any(
        first == second for first, second in zip(route.points, route.points[1:], strict=False)
    )


def _has_redundant_points(route: LayoutRoute) -> bool:
    for first, middle, last in zip(
        route.points,
        route.points[1:],
        route.points[2:],
        strict=False,
    ):
        if segment_intersection_kind(first, last, middle, middle) is SegmentIntersection.TOUCH:
            return True
    return False


def _segment_touch_points(first: tuple[Point, Point], second: tuple[Point, Point]) -> set[Point]:
    return {
        point
        for point in (*first, *second)
        if segment_intersection_kind(point, point, *first) is SegmentIntersection.TOUCH
        and segment_intersection_kind(point, point, *second) is SegmentIntersection.TOUCH
    }


def _route_intersection(first: LayoutRoute, second: LayoutRoute) -> SegmentIntersection:
    strongest = SegmentIntersection.NONE
    shared_elements = set(first.endpoints).intersection(second.endpoints)
    shared_outer_points = (
        {first.points[0], first.points[-1]}.intersection({second.points[0], second.points[-1]})
        if first.points and second.points and shared_elements
        else set()
    )
    for first_segment in _segments(first):
        for second_segment in _segments(second):
            kind = segment_intersection_kind(*first_segment, *second_segment)
            if kind is SegmentIntersection.OVERLAP:
                return kind
            if kind is SegmentIntersection.CROSS:
                strongest = kind
            elif kind is SegmentIntersection.TOUCH:
                touch_points = _segment_touch_points(first_segment, second_segment)
                if not touch_points or not touch_points.issubset(shared_outer_points):
                    strongest = SegmentIntersection.TOUCH
    return strongest


def _stock_scc_ids(model: StellaModel) -> dict[str, int]:
    adjacency: dict[str, list[str]] = {name: [] for name in model.stocks}
    for flow in model.flows.values():
        if flow.from_stock in adjacency and flow.to_stock in adjacency:
            adjacency[flow.from_stock].append(flow.to_stock)
    for neighbors in adjacency.values():
        neighbors.sort()

    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    component_ids: dict[str, int] = {}

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for neighbor in adjacency[node]:
            if neighbor not in indices:
                visit(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])
        if lowlinks[node] != indices[node]:
            return
        component_id = len(set(component_ids.values()))
        while stack:
            member = stack.pop()
            on_stack.remove(member)
            component_ids[member] = component_id
            if member == node:
                break

    for stock_name in sorted(adjacency):
        if stock_name not in indices:
            visit(stock_name)
    return component_ids


def _bounds(
    glyphs: Mapping[str, LayoutBox],
    labels: Mapping[str, LayoutBox],
    routes: tuple[LayoutRoute, ...],
) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for box in (*glyphs.values(), *labels.values()):
        xs.extend((box.left, box.right))
        ys.extend((box.top, box.bottom))
    for route in routes:
        xs.extend(point[0] for point in route.points)
        ys.extend(point[1] for point in route.points)
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def analyze_layout(
    model: StellaModel,
    *,
    pinned_reference: Mapping[str, Point] | None = None,
    locked_route_reference: Mapping[str, tuple[Point, ...]] | None = None,
) -> LayoutMetrics:
    """Analyze one laid-out model without changing it."""
    glyphs, missing = _glyph_boxes(model)
    labels = _label_boxes(model, glyphs)
    flow_routes = _flow_routes(model)
    connector_routes = _connector_routes(model, glyphs)
    all_routes = flow_routes + connector_routes

    glyph_names = sorted(glyphs)
    glyph_overlaps = tuple(
        (first, second)
        for index, first in enumerate(glyph_names)
        for second in glyph_names[index + 1 :]
        if glyphs[first].as_bounding_box().intersects(glyphs[second].as_bounding_box())
    )
    label_glyph_overlaps = tuple(
        (label_name, glyph_name)
        for label_name in sorted(labels)
        for glyph_name in glyph_names
        if label_name != glyph_name
        and labels[label_name].as_bounding_box().intersects(glyphs[glyph_name].as_bounding_box())
    )
    label_names = sorted(labels)
    label_label_overlaps = tuple(
        (first, second)
        for index, first in enumerate(label_names)
        for second in label_names[index + 1 :]
        if labels[first].as_bounding_box().intersects(labels[second].as_bounding_box())
    )

    flow_glyph_crossings = tuple(
        (route.name, glyph_name)
        for route in flow_routes
        for glyph_name in glyph_names
        if glyph_name != route.name
        and glyph_name not in route.endpoints
        and any(
            segment_intersects_box(start, end, glyphs[glyph_name].as_bounding_box())
            for start, end in _segments(route)
        )
    )
    connector_glyph_crossings = tuple(
        (route.name, glyph_name)
        for route in connector_routes
        for glyph_name in glyph_names
        if glyph_name not in route.endpoints
        and any(
            segment_intersects_box(start, end, glyphs[glyph_name].as_bounding_box())
            for start, end in _segments(route)
        )
    )
    flow_label_crossings = tuple(
        (route.name, label_name)
        for route in flow_routes
        for label_name in label_names
        if label_name != route.name
        and label_name not in route.endpoints
        and any(
            segment_intersects_box(start, end, labels[label_name].as_bounding_box())
            for start, end in _segments(route)
        )
    )
    connector_label_crossings = tuple(
        (route.name, label_name)
        for route in connector_routes
        for label_name in label_names
        if label_name not in route.endpoints
        and any(
            segment_intersects_box(start, end, labels[label_name].as_bounding_box())
            for start, end in _segments(route)
        )
    )

    connector_flow_crossings = tuple(
        (connector.name, flow.name)
        for connector in connector_routes
        for flow in flow_routes
        if flow.name not in connector.endpoints
        if _route_intersection(connector, flow) is not SegmentIntersection.NONE
    )
    connector_connector_crossings = tuple(
        (first.name, second.name)
        for index, first in enumerate(connector_routes)
        for second in connector_routes[index + 1 :]
        if _route_intersection(first, second) is not SegmentIntersection.NONE
    )
    flow_pairs = tuple(
        (first, second, _route_intersection(first, second))
        for index, first in enumerate(flow_routes)
        for second in flow_routes[index + 1 :]
    )
    flow_flow_crossings = tuple(
        (first.name, second.name)
        for first, second, kind in flow_pairs
        if kind in {SegmentIntersection.CROSS, SegmentIntersection.TOUCH}
    )
    flow_shared_segments = tuple(
        (first.name, second.name)
        for first, second, kind in flow_pairs
        if kind is SegmentIntersection.OVERLAP
    )

    self_intersections = tuple(
        f"{route.kind}:{route.name}" for route in all_routes if _route_has_self_intersection(route)
    )
    repeated = tuple(
        f"{route.kind}:{route.name}" for route in all_routes if _has_repeated_points(route)
    )
    redundant = tuple(
        f"{route.kind}:{route.name}" for route in all_routes if _has_redundant_points(route)
    )
    avoidable_detours = tuple(
        f"{route.kind}:{route.name}"
        for route in all_routes
        if len(route.points) > 2
        and len(set(route.endpoints)) == len(route.endpoints)
        and not route.locked
        and (
            route.kind != "flow"
            or route.points[0][0] == route.points[-1][0]
            or route.points[0][1] == route.points[-1][1]
        )
        and direct_segment_is_clear(
            route.points[0],
            route.points[-1],
            tuple(box for name, box in glyphs.items() if name not in {*route.endpoints, route.name})
            + tuple(
                box
                for name, box in labels.items()
                if name not in {*route.endpoints, route.name}
            ),
            tuple(other.points for other in all_routes if other is not route),
        )
    )

    component_ids = _stock_scc_ids(model)
    backward = tuple(
        name
        for name, flow in sorted(model.flows.items())
        if flow.from_stock in glyphs
        and flow.to_stock in glyphs
        and component_ids.get(flow.from_stock) != component_ids.get(flow.to_stock)
        and glyphs[flow.to_stock].x < glyphs[flow.from_stock].x
    )

    pinned_movements: tuple[str, ...] = ()
    if pinned_reference:
        pinned_movements = tuple(
            name
            for name, expected in sorted(pinned_reference.items())
            if name not in glyphs or (glyphs[name].x, glyphs[name].y) != expected
        )
    locked_movements: tuple[str, ...] = ()
    if locked_route_reference:
        routes_by_key = {f"{route.kind}:{route.name}": route.points for route in all_routes}
        locked_movements = tuple(
            name
            for name, expected in sorted(locked_route_reference.items())
            if routes_by_key.get(name) != expected
        )

    flow_lengths = tuple(_route_length(route) for route in flow_routes)
    connector_lengths = tuple(_route_length(route) for route in connector_routes)
    bend_count = sum(max(0, len(route.points) - 2) for route in all_routes)
    bounds = _bounds(glyphs, labels, all_routes)
    viewport = _viewport(model)
    overflow = (
        max(0.0, -bounds[0]),
        max(0.0, -bounds[1]),
        max(0.0, bounds[2] - viewport.width),
        max(0.0, bounds[3] - viewport.height),
    )

    return LayoutMetrics(
        missing_positions=missing,
        glyph_overlaps=glyph_overlaps,
        label_glyph_overlaps=label_glyph_overlaps,
        label_label_overlaps=label_label_overlaps,
        flow_glyph_crossings=flow_glyph_crossings,
        connector_glyph_crossings=connector_glyph_crossings,
        flow_label_crossings=flow_label_crossings,
        connector_label_crossings=connector_label_crossings,
        connector_flow_crossings=connector_flow_crossings,
        connector_connector_crossings=connector_connector_crossings,
        flow_flow_crossings=flow_flow_crossings,
        flow_shared_segments=flow_shared_segments,
        route_self_intersections=self_intersections,
        repeated_route_points=repeated,
        redundant_route_points=redundant,
        avoidable_route_detours=avoidable_detours,
        backward_flow_edges=backward,
        pinned_position_movements=pinned_movements,
        locked_route_movements=locked_movements,
        total_flow_length=sum(flow_lengths),
        maximum_flow_length=max(flow_lengths, default=0.0),
        total_connector_length=sum(connector_lengths),
        maximum_connector_length=max(connector_lengths, default=0.0),
        total_bend_count=bend_count,
        bounds=bounds,
        page_overflow=overflow,
    )
