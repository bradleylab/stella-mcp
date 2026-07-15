"""Directed stock-flow graph construction and deterministic backbone placement."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from statistics import median

from stella_mcp.layout import (
    _SEPARATION_GAP,
    DEFAULT_IDEAL_EDGE_LENGTH,
    SegmentIntersection,
    segment_intersection_kind,
)
from stella_mcp.layout_quality import (
    AUX_RADIUS,
    CSS_PIXELS_PER_POINT,
    FLOW_VALVE_SIZE,
    LABEL_WIDTH_EM,
    LayoutWarning,
    label_font_points,
)

MARGIN = 32.0


@dataclass(frozen=True)
class StockEdge:
    """One typed flow edge in the stock graph."""

    flow_name: str
    source: str | None
    target: str | None

    @property
    def kind(self) -> str:
        if self.source is None and self.target is None:
            return "orphan"
        if self.source is None:
            return "source_only"
        if self.target is None:
            return "destination_only"
        if self.source == self.target:
            return "self_loop"
        return "stock_to_stock"


@dataclass(frozen=True)
class DirectedStockGraph:
    """Deterministic stock graph and its condensation metadata."""

    nodes: tuple[str, ...]
    edges: tuple[StockEdge, ...]
    components: tuple[tuple[str, ...], ...]
    component_by_node: tuple[tuple[str, int], ...]
    ranks: tuple[tuple[int, int], ...]
    weak_components: tuple[tuple[int, ...], ...]

    def component_map(self) -> dict[str, int]:
        return dict(self.component_by_node)

    def rank_map(self) -> dict[int, int]:
        return dict(self.ranks)


def _strong_components(
    nodes: tuple[str, ...], adjacency: dict[str, tuple[str, ...]]
) -> tuple[tuple[str, ...], ...]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    found: list[tuple[str, ...]] = []

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
        members: list[str] = []
        while stack:
            member = stack.pop()
            on_stack.remove(member)
            members.append(member)
            if member == node:
                break
        found.append(tuple(sorted(members)))

    for node in nodes:
        if node not in indices:
            visit(node)
    return tuple(sorted(found, key=lambda members: members[0]))


def build_stock_graph(model) -> DirectedStockGraph:
    """Build SCC, rank, and weak-component metadata from model flows."""
    nodes = tuple(sorted(model.stocks))
    edges = tuple(
        StockEdge(name, flow.from_stock, flow.to_stock)
        for name, flow in sorted(model.flows.items())
    )
    adjacency_lists: dict[str, list[str]] = {node: [] for node in nodes}
    for edge in edges:
        if edge.kind == "stock_to_stock":
            adjacency_lists[edge.source].append(edge.target)  # type: ignore[index]
    adjacency = {node: tuple(sorted(set(neighbors))) for node, neighbors in adjacency_lists.items()}
    components = _strong_components(nodes, adjacency)
    component_by_node = {
        member: component_id
        for component_id, members in enumerate(components)
        for member in members
    }

    dag_out: dict[int, set[int]] = {component_id: set() for component_id in range(len(components))}
    dag_in: dict[int, set[int]] = {component_id: set() for component_id in range(len(components))}
    for edge in edges:
        if edge.kind != "stock_to_stock":
            continue
        source_id = component_by_node[edge.source]  # type: ignore[index]
        target_id = component_by_node[edge.target]  # type: ignore[index]
        if source_id == target_id:
            continue
        dag_out[source_id].add(target_id)
        dag_in[target_id].add(source_id)

    indegree = {component_id: len(parents) for component_id, parents in dag_in.items()}
    ready = deque(sorted(component_id for component_id, degree in indegree.items() if degree == 0))
    ranks = {component_id: 0 for component_id in range(len(components))}
    while ready:
        component_id = ready.popleft()
        for target_id in sorted(dag_out[component_id]):
            ranks[target_id] = max(ranks[target_id], ranks[component_id] + 1)
            indegree[target_id] -= 1
            if indegree[target_id] == 0:
                ready.append(target_id)

    undirected: dict[int, set[int]] = {
        component_id: set() for component_id in range(len(components))
    }
    for source_id, targets in dag_out.items():
        for target_id in targets:
            undirected[source_id].add(target_id)
            undirected[target_id].add(source_id)
    weak: list[tuple[int, ...]] = []
    visited: set[int] = set()
    for component_id in sorted(undirected):
        if component_id in visited:
            continue
        pending = [component_id]
        members: list[int] = []
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            members.append(current)
            pending.extend(sorted(undirected[current] - visited, reverse=True))
        weak.append(tuple(sorted(members)))

    weak.sort(key=lambda ids: (-sum(len(components[item]) for item in ids), components[ids[0]][0]))
    return DirectedStockGraph(
        nodes=nodes,
        edges=edges,
        components=components,
        component_by_node=tuple(sorted(component_by_node.items())),
        ranks=tuple(sorted(ranks.items())),
        weak_components=tuple(weak),
    )


def _crossing_count(
    order: dict[int, tuple[int, ...]],
    rank_by_component: dict[int, int],
    dag_edges: tuple[tuple[int, int], ...],
) -> int:
    position = {
        component_id: index
        for rank in sorted(order)
        for index, component_id in enumerate(order[rank])
    }
    count = 0
    for index, (first_source, first_target) in enumerate(dag_edges):
        first_pair = (rank_by_component[first_source], rank_by_component[first_target])
        for second_source, second_target in dag_edges[index + 1 :]:
            if first_pair != (
                rank_by_component[second_source],
                rank_by_component[second_target],
            ):
                continue
            if (position[first_source] - position[second_source]) * (
                position[first_target] - position[second_target]
            ) < 0:
                count += 1
    return count


def _ordered_ranks(
    graph: DirectedStockGraph,
    weak_component: tuple[int, ...],
) -> dict[int, tuple[int, ...]]:
    rank_by_component = graph.rank_map()
    component_by_node = graph.component_map()
    block_name = {index: members[0] for index, members in enumerate(graph.components)}
    dag_edges = tuple(
        sorted(
            {
                (component_by_node[edge.source], component_by_node[edge.target])
                for edge in graph.edges
                if edge.kind == "stock_to_stock"
                and component_by_node[edge.source] != component_by_node[edge.target]
                and component_by_node[edge.source] in weak_component
            }
        )
    )
    ranks: dict[int, tuple[int, ...]] = {}
    for component_id in weak_component:
        ranks.setdefault(rank_by_component[component_id], ())
        ranks[rank_by_component[component_id]] += (component_id,)
    ranks = {
        rank: tuple(sorted(component_ids, key=lambda item: block_name[item]))
        for rank, component_ids in ranks.items()
    }

    parents: dict[int, tuple[int, ...]] = defaultdict(tuple)
    children: dict[int, tuple[int, ...]] = defaultdict(tuple)
    for source_id, target_id in dag_edges:
        children[source_id] += (target_id,)
        parents[target_id] += (source_id,)

    while True:
        start_score = _crossing_count(ranks, rank_by_component, dag_edges)
        candidate = dict(ranks)
        for sweep_ranks, neighbors in (
            (sorted(candidate), parents),
            (sorted(candidate, reverse=True), children),
        ):
            positions = {
                component_id: index
                for rank in candidate
                for index, component_id in enumerate(candidate[rank])
            }
            for rank in sweep_ranks:
                candidate[rank] = tuple(
                    sorted(
                        candidate[rank],
                        key=lambda component_id: (
                            sum(positions[item] for item in neighbors[component_id])
                            / len(neighbors[component_id])
                            if neighbors[component_id]
                            else positions[component_id],
                            block_name[component_id],
                        ),
                    )
                )
        if _crossing_count(candidate, rank_by_component, dag_edges) >= start_score:
            return ranks
        ranks = candidate


def _flow_label_width(model, flow_name: str) -> float:
    display = model._display_name(model.flows[flow_name].name)
    font_pixels = label_font_points(model, flow_name) * CSS_PIXELS_PER_POINT
    return len(display) * font_pixels * LABEL_WIDTH_EM


def _ring_dependency_layers(model, members: tuple[str, ...]) -> int:
    internal_flows = {
        name
        for name, flow in model.flows.items()
        if flow.from_stock in members
        and flow.to_stock in members
        and flow.from_stock != flow.to_stock
    }
    controllers = {
        connector.from_var
        for connector in model.connectors
        if connector.from_var in model.auxs and connector.to_var in internal_flows
    }
    cross_cycle_controllers = {
        connector.to_var
        for connector in model.connectors
        if connector.from_var in members and connector.to_var in controllers
    }
    incoming_aux = {
        name: tuple(
            connector.from_var
            for connector in model.connectors
            if connector.to_var == name and connector.from_var in model.auxs
        )
        for name in model.auxs
    }

    def depth(name: str, visiting: set[str]) -> int:
        if name in visiting:
            return 0
        parents = incoming_aux[name]
        if not parents:
            return 0
        return 1 + max(depth(parent, visiting | {name}) for parent in parents)

    return max(
        (1 + depth(controller, set()) for controller in cross_cycle_controllers),
        default=0,
    )


def _ring_radius(model, members: tuple[str, ...]) -> float:
    max_extent = max(max(model.stocks[name].width, model.stocks[name].height) for name in members)
    internal_flows = tuple(
        name
        for name, flow in model.flows.items()
        if flow.from_stock in members
        and flow.to_stock in members
        and flow.from_stock != flow.to_stock
    )
    flow_label_width = max(
        (_flow_label_width(model, flow_name) for flow_name in internal_flows),
        default=0.0,
    )
    dependency_clearance = _ring_dependency_layers(model, members) * (AUX_RADIUS + _SEPARATION_GAP)
    return max(
        DEFAULT_IDEAL_EDGE_LENGTH * 0.55,
        max_extent + flow_label_width + _SEPARATION_GAP + dependency_clearance,
        len(members) * (max_extent + _SEPARATION_GAP) / (2 * math.pi),
    )


def _ring_positions(
    model, members: tuple[str, ...], center: tuple[float, float]
) -> dict[str, tuple[float, float]]:
    radius = _ring_radius(model, members)
    internal_edges = [
        (flow.from_stock, flow.to_stock)
        for flow in model.flows.values()
        if flow.from_stock in members
        and flow.to_stock in members
        and flow.from_stock != flow.to_stock
    ]
    candidates: list[
        tuple[tuple[int, float, tuple[str, ...]], dict[str, tuple[float, float]]]
    ] = []
    for reverse in (False, True):
        order = tuple(reversed(members)) if reverse else members
        for rotation in range(len(order)):
            rotated = order[rotation:] + order[:rotation]
            positions = {
                name: (
                    center[0] + radius * math.cos(-math.pi / 2 + 2 * math.pi * index / len(order)),
                    center[1] + radius * math.sin(-math.pi / 2 + 2 * math.pi * index / len(order)),
                )
                for index, name in enumerate(rotated)
            }
            index_by_name = {name: index for index, name in enumerate(rotated)}
            length = sum(
                2
                * radius
                * math.sin(
                    math.pi
                    * min(
                        abs(index_by_name[source] - index_by_name[target]),
                        len(order) - abs(index_by_name[source] - index_by_name[target]),
                    )
                    / len(order)
                )
                for source, target in internal_edges
            )
            crossings = sum(
                segment_intersection_kind(
                    positions[first_source],
                    positions[first_target],
                    positions[second_source],
                    positions[second_target],
                )
                in {SegmentIntersection.CROSS, SegmentIntersection.OVERLAP}
                for edge_index, (first_source, first_target) in enumerate(internal_edges)
                for second_source, second_target in internal_edges[edge_index + 1 :]
                if {first_source, first_target}.isdisjoint({second_source, second_target})
            )
            candidates.append(((crossings, length, rotated), positions))
    return min(candidates, key=lambda item: item[0])[1]


def _component_positions(
    model,
    graph: DirectedStockGraph,
    weak_component: tuple[int, ...],
) -> dict[str, tuple[float, float]]:
    ordered = _ordered_ranks(graph, weak_component)
    stock_label_height = model.view_stock_font_points * CSS_PIXELS_PER_POINT
    rank_by_component = graph.rank_map()
    component_by_node = graph.component_map()
    block_sizes: dict[int, tuple[float, float]] = {}
    for component_id in weak_component:
        members = graph.components[component_id]
        if len(members) == 1:
            stock = model.stocks[members[0]]
            block_sizes[component_id] = (float(stock.width), float(stock.height))
        else:
            max_extent = max(
                max(model.stocks[name].width, model.stocks[name].height) for name in members
            )
            radius = _ring_radius(model, members)
            diameter = 2 * radius + max_extent
            block_sizes[component_id] = (diameter, diameter)

    rank_width = {
        rank: max(block_sizes[component_id][0] for component_id in component_ids)
        for rank, component_ids in ordered.items()
    }
    rank_x: dict[int, float] = {}
    previous_rank: int | None = None
    for rank in sorted(ordered):
        if previous_rank is None:
            rank_x[rank] = rank_width[rank] / 2
        else:
            rank_edges = tuple(
                edge
                for edge in graph.edges
                if edge.kind == "stock_to_stock"
                and component_by_node[edge.source] in ordered[previous_rank]
                and component_by_node[edge.target] in ordered[rank]
            )
            flow_width = max(
                (_flow_label_width(model, edge.flow_name) for edge in rank_edges),
                default=0.0,
            )
            spacing = max(
                DEFAULT_IDEAL_EDGE_LENGTH,
                rank_width[previous_rank] / 2 + rank_width[rank] / 2 + _SEPARATION_GAP,
                flow_width + FLOW_VALVE_SIZE + 2 * _SEPARATION_GAP,
            )
            if len(rank_edges) > 1:
                controller_clearance = FLOW_VALVE_SIZE / 2 + 2 * AUX_RADIUS + 2 * _SEPARATION_GAP
                spacing = max(
                    spacing,
                    2 * (controller_clearance + rank_width[rank] / 2)
                    + max(0.0, (rank_width[previous_rank] - rank_width[rank]) / 2),
                )
            rank_x[rank] = rank_x[previous_rank] + spacing
        previous_rank = rank

    rank_centers: dict[int, dict[int, float]] = {}
    maximum_span = 0.0
    for rank, component_ids in ordered.items():
        centers: dict[int, float] = {}
        cursor = 0.0
        previous_height = 0.0
        for index, component_id in enumerate(component_ids):
            height = block_sizes[component_id][1]
            if index == 0:
                cursor = height / 2
            else:
                cursor += (
                    previous_height / 2
                    + stock_label_height
                    + _SEPARATION_GAP
                    + height / 2
                )
            centers[component_id] = cursor
            previous_height = height
        span = cursor + previous_height / 2
        maximum_span = max(maximum_span, span)
        rank_centers[rank] = centers

    for rank, centers in rank_centers.items():
        rank_height = max(
            (center + block_sizes[component_id][1] / 2 for component_id, center in centers.items()),
            default=0.0,
        )
        shift = (maximum_span - rank_height) / 2
        rank_centers[rank] = {
            component_id: center + shift for component_id, center in centers.items()
        }

    # A single block per rank is a flow chain; align every rank on one pipe.
    if all(len(component_ids) == 1 for component_ids in ordered.values()):
        aligned_y = maximum_span / 2
        rank_centers = {
            rank: {component_ids[0]: aligned_y} for rank, component_ids in ordered.items()
        }
    else:
        # Align singleton parent/child ranks to the barycenter of their neighbors.
        for _ in range(2):
            for rank in sorted(ordered):
                for component_id in ordered[rank]:
                    neighbors = [
                        component_by_node[edge.target]
                        for edge in graph.edges
                        if edge.kind == "stock_to_stock"
                        and component_by_node[edge.source] == component_id
                        and component_by_node[edge.target] != component_id
                    ] + [
                        component_by_node[edge.source]
                        for edge in graph.edges
                        if edge.kind == "stock_to_stock"
                        and component_by_node[edge.target] == component_id
                        and component_by_node[edge.source] != component_id
                    ]
                    neighbor_y = []
                    for neighbor in neighbors:
                        neighbor_rank = rank_by_component[neighbor]
                        if neighbor_rank in rank_centers:
                            neighbor_y.append(rank_centers[neighbor_rank][neighbor])
                    if len(ordered[rank]) == 1 and neighbor_y:
                        rank_centers[rank][component_id] = sum(neighbor_y) / len(neighbor_y)

    positions: dict[str, tuple[float, float]] = {}
    for rank, component_ids in ordered.items():
        for component_id in component_ids:
            center = (rank_x[rank], rank_centers[rank][component_id])
            members = graph.components[component_id]
            if len(members) == 1:
                positions[members[0]] = center
            else:
                positions.update(_ring_positions(model, members, center))
    return positions


def _apply_pins(
    model, graph: DirectedStockGraph, positions: dict[str, tuple[float, float]]
) -> None:
    component_by_node = graph.component_map()
    rank_by_component = graph.rank_map()
    for weak_component in graph.weak_components:
        members = [
            name for component_id in weak_component for name in graph.components[component_id]
        ]
        pins = [
            name
            for name in members
            if model.stocks[name].position_source == "user"
            and model.stocks[name].x is not None
            and model.stocks[name].y is not None
        ]
        if not pins:
            continue
        x_offsets = [model.stocks[name].x - positions[name][0] for name in pins]
        y_offsets = [model.stocks[name].y - positions[name][1] for name in pins]
        rank_anchors: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for name in pins:
            rank = rank_by_component[component_by_node[name]]
            rank_anchors[rank].append((model.stocks[name].x, model.stocks[name].y))
        anchor_x = {
            rank: sum(x for x, _ in values) / len(values) for rank, values in rank_anchors.items()
        }
        sorted_anchor_ranks = sorted(anchor_x)

        for name in members:
            stock = model.stocks[name]
            if name in pins:
                positions[name] = (stock.x, stock.y)
                continue
            rank = rank_by_component[component_by_node[name]]
            left = max((item for item in sorted_anchor_ranks if item <= rank), default=None)
            right = min((item for item in sorted_anchor_ranks if item >= rank), default=None)
            if left is not None and right is not None and left != right:
                fraction = (rank - left) / (right - left)
                x = anchor_x[left] + fraction * (anchor_x[right] - anchor_x[left])
            else:
                x = positions[name][0] + median(x_offsets)
            y = positions[name][1] + median(y_offsets)
            positions[name] = (x, y)


def _bounds(
    model, names: list[str], positions: dict[str, tuple[float, float]]
) -> tuple[float, float, float, float]:
    left = min(positions[name][0] - model.stocks[name].width / 2 for name in names)
    top = min(positions[name][1] - model.stocks[name].height / 2 for name in names)
    right = max(positions[name][0] + model.stocks[name].width / 2 for name in names)
    stock_label_height = model.view_stock_font_points * CSS_PIXELS_PER_POINT
    bottom = max(
        positions[name][1]
        + model.stocks[name].height / 2
        + stock_label_height
        + _SEPARATION_GAP
        for name in names
    )
    if any(flow.from_stock is None and flow.to_stock in names for flow in model.flows.values()):
        left -= 140.0 + _SEPARATION_GAP
    if any(flow.from_stock in names and flow.to_stock is None for flow in model.flows.values()):
        right += 140.0 + _SEPARATION_GAP
    return left, top, right, bottom


def _dependency_horizontal_biases(
    model, graph: DirectedStockGraph
) -> dict[tuple[int, ...], int]:
    """Prefer row alignments that shorten controls between flow components."""
    weak_by_stock = {
        stock_name: weak_component
        for weak_component in graph.weak_components
        for component_id in weak_component
        for stock_name in graph.components[component_id]
    }
    biases: dict[tuple[int, ...], int] = defaultdict(int)
    for connector in model.connectors:
        if connector.from_var not in model.stocks or connector.to_var not in model.flows:
            continue
        flow = model.flows[connector.to_var]
        source_component = weak_by_stock[connector.from_var]
        if flow.from_stock is None and flow.to_stock is not None:
            left_component = source_component
            right_component = weak_by_stock[flow.to_stock]
        elif flow.to_stock is None and flow.from_stock is not None:
            left_component = weak_by_stock[flow.from_stock]
            right_component = source_component
        else:
            continue
        if left_component == right_component:
            continue
        biases[left_component] -= 1
        biases[right_component] += 1
    return dict(biases)


def _align_dependency_rows(
    model,
    graph: DirectedStockGraph,
    positions: dict[str, tuple[float, float]],
    packed_rows: list[list[tuple[int, ...]]],
) -> None:
    """Use free horizontal space in single-component rows to shorten controls."""
    biases = _dependency_horizontal_biases(model, graph)
    eligible = [
        row[0]
        for row in packed_rows
        if len(row) == 1 and row[0] in biases
    ]
    if not eligible:
        return
    minimum = min(biases[component] for component in eligible)
    maximum = max(biases[component] for component in eligible)
    if minimum == maximum:
        return

    page_width = float(model.view_page_width)
    for weak_component in eligible:
        names = [
            name
            for component_id in weak_component
            for name in graph.components[component_id]
        ]
        left, _, right, _ = _bounds(model, names, positions)
        width = right - left
        free_width = max(0.0, page_width - 2 * MARGIN - width)
        fraction = (biases[weak_component] - minimum) / (maximum - minimum)
        alignment_span = min(free_width, DEFAULT_IDEAL_EDGE_LENGTH)
        centered_offset = (free_width - alignment_span) / 2
        shift = MARGIN + centered_offset + fraction * alignment_span - left
        for name in names:
            x, y = positions[name]
            positions[name] = (x + shift, y)


def place_stock_backbone(model) -> tuple[DirectedStockGraph, tuple[LayoutWarning, ...]]:
    """Place stocks by flow direction while preserving authored coordinates."""
    graph = build_stock_graph(model)
    positions: dict[str, tuple[float, float]] = {}
    for weak_component in graph.weak_components:
        positions.update(_component_positions(model, graph, weak_component))
    _apply_pins(model, graph, positions)

    cursor_x = MARGIN
    cursor_y = MARGIN
    row_height = 0.0
    page_width = float(model.view_page_width)
    packed_rows: list[list[tuple[int, ...]]] = [[]]
    for weak_component in graph.weak_components:
        names = [name for component_id in weak_component for name in graph.components[component_id]]
        has_pin = any(model.stocks[name].position_source == "user" for name in names)
        if has_pin:
            continue
        left, top, right, bottom = _bounds(model, names, positions)
        width = right - left
        height = bottom - top
        if cursor_x > MARGIN and cursor_x + width > page_width - MARGIN:
            cursor_x = MARGIN
            cursor_y += row_height + DEFAULT_IDEAL_EDGE_LENGTH
            row_height = 0.0
            packed_rows.append([])
        dx = cursor_x - left
        dy = cursor_y - top
        for name in names:
            positions[name] = (positions[name][0] + dx, positions[name][1] + dy)
        packed_rows[-1].append(weak_component)
        cursor_x += width + DEFAULT_IDEAL_EDGE_LENGTH
        row_height = max(row_height, height)

    _align_dependency_rows(model, graph, positions, packed_rows)

    for name, (x, y) in positions.items():
        stock = model.stocks[name]
        if stock.position_source == "user":
            if stock.x is None:
                stock.x = float(round(x))
            if stock.y is None:
                stock.y = float(round(y))
            continue
        stock.x = float(round(x))
        stock.y = float(round(y))
        stock.position_source = "auto"

    warnings: list[LayoutWarning] = []
    component_by_node = graph.component_map()
    for edge in graph.edges:
        if edge.kind != "stock_to_stock":
            continue
        if component_by_node[edge.source] == component_by_node[edge.target]:
            continue
        source = model.stocks[edge.source]
        target = model.stocks[edge.target]
        if source.x is not None and target.x is not None and target.x <= source.x:
            if source.position_source == "user" or target.position_source == "user":
                warnings.append(
                    LayoutWarning(
                        "layout.pinned_conflict",
                        "Pinned stock positions oppose flow direction.",
                        tuple(sorted((edge.source, edge.target))),
                    )
                )
    return graph, tuple(sorted(set(warnings), key=lambda item: (item.code, item.elements)))
