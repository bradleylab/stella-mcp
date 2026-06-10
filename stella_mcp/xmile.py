"""XMILE XML generation and parsing for Stella .stmx files."""

import math
import re
import uuid
from dataclasses import dataclass, field
from html import escape
from typing import Optional

from stella_mcp.equation_parser import extract_variable_references
from stella_mcp.layout import (
    BoundingBox,
    force_directed_layout,
    segment_intersects_box,
    segments_intersect,
)

# XML namespaces
XMILE_NS = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"
ISEE_NS = "http://iseesystems.com/XMILE"

# Layout constants
AUX_RADIUS = 18  # Default aux circle radius in pixels



@dataclass
class Stock:
    """Represents a stock (reservoir) in the model."""
    name: str
    initial_value: str
    units: str = ""
    inflows: list[str] = field(default_factory=list)
    outflows: list[str] = field(default_factory=list)
    non_negative: bool = True
    x: float | None = None  # None means auto-position
    y: float | None = None  # None means auto-position
    width: int = 45  # Default stock width
    height: int = 35  # Default stock height
    size_locked: bool = False  # Preserve imported/user-defined size
    extra_attrs: dict[str, str] = field(default_factory=dict)
    extra_children_xml: list[str] = field(default_factory=list)
    view_extra_attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class Flow:
    """Represents a flow between stocks."""
    name: str
    equation: str
    units: str = ""
    from_stock: str | None = None  # None means external source
    to_stock: str | None = None    # None means external sink
    non_negative: bool = True
    x: float | None = None  # None means auto-position
    y: float | None = None  # None means auto-position
    points: list[tuple[float, float]] = field(default_factory=list)
    points_locked: bool = False  # Preserve imported/user-defined routing points
    graphical_function: Optional["GraphicalFunction"] = None
    extra_attrs: dict[str, str] = field(default_factory=dict)
    extra_children_xml: list[str] = field(default_factory=list)
    view_extra_attrs: dict[str, str] = field(default_factory=dict)
    view_extra_children_xml: list[str] = field(default_factory=list)


@dataclass
class Aux:
    """Represents an auxiliary variable."""
    name: str
    equation: str
    units: str = ""
    x: float | None = None  # None means auto-position
    y: float | None = None  # None means auto-position
    graphical_function: Optional["GraphicalFunction"] = None
    extra_attrs: dict[str, str] = field(default_factory=dict)
    extra_children_xml: list[str] = field(default_factory=list)
    view_extra_attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class GraphicalFunction:
    """Represents a graphical function (lookup table) definition."""
    ypts: list[float]
    xscale: tuple[float, float] | None = None
    xpts: list[float] | None = None
    yscale: tuple[float, float] | None = None
    gf_type: str | None = None


@dataclass
class Connector:
    """Represents a dependency connector between variables."""
    uid: int
    from_var: str
    to_var: str
    angle: float = 0
    angle_locked: bool = False  # Preserve imported/user-defined connector angle
    points: list[tuple[float, float]] = field(default_factory=list)
    points_locked: bool = False  # Preserve imported/user-defined routing points
    extra_attrs: dict[str, str] = field(default_factory=dict)
    extra_children_xml: list[str] = field(default_factory=list)


@dataclass
class Module:
    """Represents a logical module/group of model variables."""
    name: str
    members: list[str] = field(default_factory=list)
    x: float | None = None  # Module box center X in view
    y: float | None = None  # Module box center Y in view
    width: float | None = None
    height: float | None = None
    border_color: str | None = None
    background: str | None = None
    font_color: str | None = None
    font_size: str | None = None
    label_side: str | None = None
    extra_attrs: dict[str, str] = field(default_factory=dict)
    extra_children_xml: list[str] = field(default_factory=list)
    view_extra_attrs: dict[str, str] = field(default_factory=dict)
    view_extra_children_xml: list[str] = field(default_factory=list)


@dataclass
class SimSpecs:
    """Simulation specifications."""
    start: float = 0
    stop: float = 100
    dt: float = 0.25
    method: str = "Euler"
    time_units: str = "Years"
    extra_attrs: dict[str, str] = field(default_factory=dict)
    extra_children_xml: list[str] = field(default_factory=list)


class StellaModel:
    """Represents a complete Stella system dynamics model."""

    def __init__(self, name: str = "Untitled"):
        self.name = name
        self.uuid = str(uuid.uuid4())
        self.sim_specs = SimSpecs()
        self.stocks: dict[str, Stock] = {}
        self.flows: dict[str, Flow] = {}
        self.auxs: dict[str, Aux] = {}
        self.modules: dict[str, Module] = {}
        self.connectors: list[Connector] = []
        self._connector_uid = 0
        self.compatibility_warnings: list[str] = []
        self.last_export_warnings: list[str] = []
        self.header_extra_children_xml: list[str] = []
        self.model_extra_children_xml: list[str] = []
        self.views_extra_children_xml: list[str] = []
        self.view_extra_children_xml: list[str] = []
        self.view_extra_attrs: dict[str, str] = {}
        self.prefs_xml: str | None = None
        self.views_style_xml: str | None = None
        self.inner_view_style_xml: str | None = None
        self._export_ns_prefix_by_uri: dict[str, str] = {}

    @staticmethod
    def _validate_compat_mode(compat_mode: str) -> str:
        """Validate compatibility mode."""
        mode = str(compat_mode or "").strip().lower()
        if mode not in {"permissive", "strict"}:
            raise ValueError("compat_mode must be one of: permissive, strict")
        return mode

    @staticmethod
    def _xml_local_name(tag: str) -> str:
        """Extract local XML tag name from namespaced or plain tags."""
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    @staticmethod
    def _xml_attr_parts(attr_key: str) -> tuple[str | None, str]:
        """Split ElementTree attr key into (namespace_uri, local_name)."""
        if attr_key.startswith("{") and "}" in attr_key:
            namespace, local = attr_key[1:].split("}", 1)
            return namespace, local
        return None, attr_key

    def _xml_attr_name(self, attr_key: str) -> str:
        """Convert ElementTree attribute key to output-safe name."""
        namespace, local = self._xml_attr_parts(attr_key)
        if namespace is None or namespace == XMILE_NS:
            return local
        if namespace == ISEE_NS:
            return f"isee:{local}"
        prefix = self._export_ns_prefix_by_uri.get(namespace)
        if prefix:
            return f"{prefix}:{local}"
        # Fallback for robustness; prefix should normally be precomputed.
        return local

    def _iter_all_extra_attrs(self):
        """Iterate over all preserved extra-attribute dictionaries."""
        yield self.sim_specs.extra_attrs
        yield self.view_extra_attrs
        for stock in self.stocks.values():
            yield stock.extra_attrs
            yield stock.view_extra_attrs
        for flow in self.flows.values():
            yield flow.extra_attrs
            yield flow.view_extra_attrs
        for aux in self.auxs.values():
            yield aux.extra_attrs
            yield aux.view_extra_attrs
        for module in self.modules.values():
            yield module.extra_attrs
            yield module.view_extra_attrs
        for conn in self.connectors:
            yield conn.extra_attrs

    def _build_export_ns_prefixes(self) -> dict[str, str]:
        """Build deterministic XML namespace prefixes for unknown attr namespaces."""
        uris: set[str] = set()
        for attrs in self._iter_all_extra_attrs():
            for raw_key in attrs:
                namespace, _ = self._xml_attr_parts(raw_key)
                if namespace and namespace not in {XMILE_NS, ISEE_NS}:
                    uris.add(namespace)
        prefix_by_uri: dict[str, str] = {}
        for index, uri in enumerate(sorted(uris), start=1):
            prefix_by_uri[uri] = f"ns{index}"
        return prefix_by_uri

    def _format_extra_attrs(
        self,
        attrs: dict[str, str],
        reserved_names: set[str] | None = None,
    ) -> str:
        """Format preserved extra XML attrs while avoiding known fields."""
        if not attrs:
            return ""
        reserved = reserved_names or set()
        rendered: list[str] = []
        for raw_key in sorted(attrs):
            key = self._xml_attr_name(raw_key)
            if key in reserved:
                continue
            rendered.append(f'{key}="{escape(attrs[raw_key])}"')
        return (" " + " ".join(rendered)) if rendered else ""

    def _append_xml_fragment(self, lines: list[str], fragment: str, indent: str):
        """Append a preserved XML fragment with target indentation."""
        text = fragment.strip()
        if not text:
            return
        for line in text.splitlines():
            lines.append(f"{indent}{line}")

    def _next_connector_uid(self) -> int:
        """Get the next unique connector ID."""
        self._connector_uid += 1
        return self._connector_uid

    def _normalize_name(self, name: str) -> str:
        """Convert display name to internal name (spaces to underscores)."""
        return name.replace(" ", "_")

    def _display_name(self, name: str) -> str:
        """Convert internal name to display name (underscores to spaces)."""
        return name.replace("_", " ")

    def _extract_variable_refs(self, equation: str) -> set[str]:
        """Extract variable names referenced in an equation.

        Returns normalized variable names (spaces converted to underscores).
        Filters out Stella built-in functions and keywords.
        """
        refs = extract_variable_references(equation)
        return {self._normalize_name(token) for token in refs}

    @staticmethod
    def _format_number(value: float) -> str:
        """Format numbers for XMILE with stable precision."""
        return f"{value:.12g}"

    def _dt_xml(self, dt: float | None = None) -> str:
        """Format dt for XMILE with compatibility-safe reciprocal usage.

        Stella commonly uses reciprocal dt when dt is an exact inverse integer
        (e.g., 0.25 -> reciprocal 4). For non-exact values, writing reciprocal
        with truncation can change dt on round-trip, so export plain dt instead.
        """
        dt = float(self.sim_specs.dt if dt is None else dt)
        if dt <= 0:
            raise ValueError("sim_specs.dt must be > 0")
        reciprocal = 1.0 / dt
        nearest = round(reciprocal)
        if dt < 1.0 and abs(reciprocal - nearest) < 1e-9 and nearest >= 1:
            return f'<dt reciprocal="true">{int(nearest)}</dt>'
        return f"<dt>{self._format_number(dt)}</dt>"

    def _build_dependency_graph(self) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        """Build bidirectional adjacency lists from connectors and flow-stock relationships.

        Returns:
            (outgoing, incoming) where:
            - outgoing[node] = set of nodes this node connects TO
            - incoming[node] = set of nodes that connect TO this node
        """
        from collections import defaultdict

        outgoing: dict[str, set[str]] = defaultdict(set)
        incoming: dict[str, set[str]] = defaultdict(set)

        # Initialize all elements as nodes
        for name in self.stocks:
            outgoing.setdefault(name, set())
            incoming.setdefault(name, set())
        for name in self.flows:
            outgoing.setdefault(name, set())
            incoming.setdefault(name, set())
        for name in self.auxs:
            outgoing.setdefault(name, set())
            incoming.setdefault(name, set())

        # Add connector edges
        for conn in self.connectors:
            from_var = conn.from_var
            to_var = conn.to_var
            if from_var in outgoing and to_var in incoming:
                outgoing[from_var].add(to_var)
                incoming[to_var].add(from_var)

        # Add implicit flow-stock edges
        for name, flow in self.flows.items():
            if flow.from_stock and flow.from_stock in self.stocks:
                outgoing[flow.from_stock].add(name)
                incoming[name].add(flow.from_stock)
            if flow.to_stock and flow.to_stock in self.stocks:
                outgoing[name].add(flow.to_stock)
                incoming[flow.to_stock].add(name)

        return dict(outgoing), dict(incoming)

    def _find_subsystems(self, outgoing: dict[str, set[str]], incoming: dict[str, set[str]]) -> list[set[str]]:
        """Find connected components (subsystems) in the graph.

        Returns list of node sets, sorted by size (largest first).
        """
        all_nodes = set(self.stocks) | set(self.flows) | set(self.auxs)
        visited: set[str] = set()
        subsystems: list[set[str]] = []

        # Build undirected graph for component detection
        undirected: dict[str, set[str]] = {node: set() for node in all_nodes}
        for node, neighbors in outgoing.items():
            for neighbor in neighbors:
                if neighbor in undirected:
                    undirected[node].add(neighbor)
                    undirected[neighbor].add(node)
        for node, neighbors in incoming.items():
            for neighbor in neighbors:
                if neighbor in undirected:
                    undirected[node].add(neighbor)
                    undirected[neighbor].add(node)

        def dfs(node: str, component: set[str]):
            if node in visited:
                return
            visited.add(node)
            component.add(node)
            for neighbor in undirected.get(node, set()):
                dfs(neighbor, component)

        for node in sorted(all_nodes):  # Sorted for determinism
            if node not in visited:
                component: set[str] = set()
                dfs(node, component)
                if component:
                    subsystems.append(component)

        return sorted(subsystems, key=len, reverse=True)

    def _position_subsystem(
        self,
        subsystem: set[str],
        outgoing: dict[str, set[str]],
        incoming: dict[str, set[str]],
    ) -> tuple[float, float, float, float]:
        """Position all elements in a subsystem using force-directed layout.

        Returns bounding box (min_x, min_y, max_x, max_y).
        """
        # Collect nodes that participate in FR (stocks + auxs, not flows)
        nodes: list[str] = sorted(
            name for name in subsystem if name in self.stocks or name in self.auxs
        )

        if not nodes:
            return (0, 0, 0, 0)

        # Collect fixed (user-specified) positions
        fixed_positions: dict[str, tuple[float, float]] = {}
        for name in nodes:
            if name in self.stocks:
                s = self.stocks[name]
                if s.x is not None and s.y is not None:
                    fixed_positions[name] = (s.x, s.y)
            elif name in self.auxs:
                a = self.auxs[name]
                if a.x is not None and a.y is not None:
                    fixed_positions[name] = (a.x, a.y)

        # Build edges with weights
        FLOW_WEIGHT = 2.0
        CONNECTOR_WEIGHT = 1.5

        edges: list[tuple[str, str, float]] = []

        # Flow-stock edges (strong attraction)
        for flow_name, flow in self.flows.items():
            if flow_name not in subsystem:
                continue
            if flow.from_stock and flow.from_stock in self.stocks and flow.from_stock in subsystem:
                if flow.to_stock and flow.to_stock in self.stocks and flow.to_stock in subsystem:
                    # Direct stock-to-stock edge via flow
                    edges.append((flow.from_stock, flow.to_stock, FLOW_WEIGHT))

        # Connector edges (weaker attraction)
        for conn in self.connectors:
            if conn.from_var in subsystem and conn.to_var in subsystem:
                src = conn.from_var
                tgt = conn.to_var
                # If target is a flow, redirect edge to the flow's stocks
                if tgt in self.flows:
                    flow = self.flows[tgt]
                    if flow.from_stock and flow.from_stock in subsystem and src in nodes:
                        edges.append((src, flow.from_stock, CONNECTOR_WEIGHT))
                    if flow.to_stock and flow.to_stock in subsystem and src in nodes:
                        edges.append((src, flow.to_stock, CONNECTOR_WEIGHT))
                elif src in nodes and tgt in nodes:
                    edges.append((src, tgt, CONNECTOR_WEIGHT))

        # Run force-directed layout
        positions = force_directed_layout(nodes, edges, fixed_positions)

        # Apply positions to stocks and auxs
        for name, (x, y) in positions.items():
            if name in fixed_positions:
                continue
            if name in self.stocks:
                self.stocks[name].x = x
                self.stocks[name].y = y
            elif name in self.auxs:
                self.auxs[name].x = x
                self.auxs[name].y = y

        # Position flows at midpoint between their stocks
        for flow_name in subsystem:
            if flow_name not in self.flows:
                continue
            flow = self.flows[flow_name]
            if flow.x is not None and flow.y is not None:
                continue

            from_stock = self.stocks.get(flow.from_stock) if flow.from_stock else None
            to_stock = self.stocks.get(flow.to_stock) if flow.to_stock else None

            if from_stock and to_stock:
                fx = from_stock.x if from_stock.x is not None else 0
                fy = from_stock.y if from_stock.y is not None else 0
                tx = to_stock.x if to_stock.x is not None else 0
                ty = to_stock.y if to_stock.y is not None else 0
                flow.x = (fx + tx) / 2
                flow.y = (fy + ty) / 2
            elif from_stock:
                flow.x = (from_stock.x or 0) + 90
                flow.y = from_stock.y or 0
            elif to_stock:
                flow.x = (to_stock.x or 0) - 90
                flow.y = to_stock.y or 0

        # Calculate bounding box
        all_x: list[float] = []
        all_y: list[float] = []

        for name in subsystem:
            if name in self.stocks and self.stocks[name].x is not None:
                all_x.append(self.stocks[name].x)  # type: ignore
                all_y.append(self.stocks[name].y)  # type: ignore
            if name in self.flows and self.flows[name].x is not None:
                all_x.append(self.flows[name].x)  # type: ignore
                all_y.append(self.flows[name].y)  # type: ignore
            if name in self.auxs and self.auxs[name].x is not None:
                all_x.append(self.auxs[name].x)  # type: ignore
                all_y.append(self.auxs[name].y)  # type: ignore

        if all_x and all_y:
            return (min(all_x), min(all_y), max(all_x), max(all_y))
        return (0, 0, 100, 100)

    def _arrange_subsystems(
        self,
        subsystems: list[set[str]],
        bounds: list[tuple[float, float, float, float]],
        gap: float
    ):
        """Arrange subsystems: largest stays in place, smaller ones offset to the right."""
        if len(subsystems) <= 1:
            return

        # First subsystem (largest) stays in place
        # Offset subsequent subsystems to the right
        current_x = bounds[0][2] + gap  # max_x of first + gap

        for i, subsystem in enumerate(subsystems[1:], start=1):
            min_x = bounds[i][0]
            max_x = bounds[i][2]
            offset_x = current_x - min_x

            # Shift all elements in this subsystem
            for name in subsystem:
                if name in self.stocks and self.stocks[name].x is not None:
                    self.stocks[name].x += offset_x
                if name in self.flows and self.flows[name].x is not None:
                    self.flows[name].x += offset_x
                if name in self.auxs and self.auxs[name].x is not None:
                    self.auxs[name].x += offset_x

            current_x = current_x + (max_x - min_x) + gap

    def add_stock(
        self,
        name: str,
        initial_value: str,
        units: str = "",
        inflows: list[str] | None = None,
        outflows: list[str] | None = None,
        non_negative: bool = True,
        x: float | None = None,
        y: float | None = None
    ) -> Stock:
        """Add a stock to the model."""
        self._validate_new_variable_name(name)
        stock = Stock(
            name=name,
            initial_value=initial_value,
            units=units,
            inflows=[self._normalize_name(f) for f in (inflows or [])],
            outflows=[self._normalize_name(f) for f in (outflows or [])],
            non_negative=non_negative,
            x=x,
            y=y
        )
        self.stocks[self._normalize_name(name)] = stock
        return stock

    def add_flow(
        self,
        name: str,
        equation: str,
        units: str = "",
        from_stock: str | None = None,
        to_stock: str | None = None,
        non_negative: bool = True,
        x: float | None = None,
        y: float | None = None,
        graphical_function: GraphicalFunction | None = None
    ) -> Flow:
        """Add a flow to the model."""
        self._validate_new_variable_name(name)
        from_key = self._normalize_name(from_stock) if from_stock else None
        to_key = self._normalize_name(to_stock) if to_stock else None
        if from_key is not None and from_key not in self.stocks:
            raise ValueError(f"from_stock '{from_stock}' is not a known stock")
        if to_key is not None and to_key not in self.stocks:
            raise ValueError(f"to_stock '{to_stock}' is not a known stock")

        flow = Flow(
            name=name,
            equation=equation,
            units=units,
            from_stock=from_key,
            to_stock=to_key,
            non_negative=non_negative,
            x=x,
            y=y,
            graphical_function=graphical_function,
        )
        flow_key = self._normalize_name(name)
        self.flows[flow_key] = flow

        # Update stock inflows/outflows
        if from_key is not None and flow_key not in self.stocks[from_key].outflows:
            self.stocks[from_key].outflows.append(flow_key)

        if to_key is not None and flow_key not in self.stocks[to_key].inflows:
            self.stocks[to_key].inflows.append(flow_key)

        return flow

    def add_aux(
        self,
        name: str,
        equation: str,
        units: str = "",
        x: float | None = None,
        y: float | None = None,
        graphical_function: GraphicalFunction | None = None
    ) -> Aux:
        """Add an auxiliary variable to the model."""
        self._validate_new_variable_name(name)
        aux = Aux(
            name=name,
            equation=equation,
            units=units,
            x=x,
            y=y,
            graphical_function=graphical_function,
        )
        self.auxs[self._normalize_name(name)] = aux
        return aux

    def add_connector(self, from_var: str, to_var: str) -> Connector:
        """Add a connector (dependency) between variables."""
        norm_from = self._normalize_name(from_var)
        norm_to = self._normalize_name(to_var)
        if not self._has_variable(norm_from):
            raise ValueError(f"Connector source '{from_var}' is not a known variable")
        if not self._has_variable(norm_to):
            raise ValueError(f"Connector target '{to_var}' is not a known variable")
        connector = Connector(
            uid=self._next_connector_uid(),
            from_var=norm_from,
            to_var=norm_to
        )
        self.connectors.append(connector)
        return connector

    def sync_connectors_from_equations(self) -> dict[str, int]:
        """Add missing connectors for equation references on flows and auxiliaries."""
        existing = {(conn.from_var, conn.to_var) for conn in self.connectors}
        added = 0
        already_present = 0

        targets: list[tuple[str, str]] = []
        for name, flow in self.flows.items():
            for ref in sorted(self._extract_variable_refs(flow.equation)):
                if ref != name and self._has_variable(ref):
                    targets.append((ref, name))
        for name, aux in self.auxs.items():
            for ref in sorted(self._extract_variable_refs(aux.equation)):
                if ref != name and self._has_variable(ref):
                    targets.append((ref, name))

        for from_var, to_var in targets:
            if (from_var, to_var) in existing:
                already_present += 1
                continue
            self.add_connector(from_var, to_var)
            existing.add((from_var, to_var))
            added += 1

        return {"added": added, "existing": already_present}

    def _resolve_connector(
        self,
        connector_uid: int | None = None,
        from_var: str | None = None,
        to_var: str | None = None,
    ) -> Connector:
        """Resolve one connector by uid or endpoint pair."""
        if connector_uid is not None:
            try:
                uid = int(connector_uid)
            except (TypeError, ValueError) as exc:
                raise ValueError("connector_uid must be an integer") from exc
            matches = [conn for conn in self.connectors if conn.uid == uid]
            if not matches:
                raise ValueError(f"No connector found with uid={uid}")
            if len(matches) > 1:
                raise ValueError(f"Multiple connectors found with uid={uid}")
            connector = matches[0]
            if from_var is not None and self._normalize_name(from_var) != connector.from_var:
                raise ValueError(
                    f"Connector uid={uid} source is '{connector.from_var}', not '{from_var}'"
                )
            if to_var is not None and self._normalize_name(to_var) != connector.to_var:
                raise ValueError(
                    f"Connector uid={uid} target is '{connector.to_var}', not '{to_var}'"
                )
            return connector

        if from_var is None or to_var is None:
            raise ValueError("Provide connector_uid, or both from_var and to_var")

        norm_from = self._normalize_name(from_var)
        norm_to = self._normalize_name(to_var)
        matches = [
            conn for conn in self.connectors
            if conn.from_var == norm_from and conn.to_var == norm_to
        ]
        if not matches:
            raise ValueError(f"No connector found from '{from_var}' to '{to_var}'")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple connectors found from '{from_var}' to '{to_var}'; "
                "specify connector_uid"
            )
        return matches[0]

    def set_connector_routing(
        self,
        connector_uid: int | None = None,
        from_var: str | None = None,
        to_var: str | None = None,
        angle: float | None = None,
        angle_locked: bool | None = None,
        points: list[tuple[float, float]] | None = None,
        points_locked: bool | None = None,
    ) -> Connector:
        """Set connector visual routing metadata (angle and optional waypoints)."""
        if angle is None and angle_locked is None and points is None and points_locked is None:
            raise ValueError("Provide at least one of angle, angle_locked, points, or points_locked")

        connector = self._resolve_connector(
            connector_uid=connector_uid,
            from_var=from_var,
            to_var=to_var,
        )

        if angle is not None:
            parsed_angle = float(angle)
            if not math.isfinite(parsed_angle):
                raise ValueError("connector angle must be a finite number")
            connector.angle = parsed_angle
            if angle_locked is None:
                connector.angle_locked = True

        if angle_locked is not None:
            connector.angle_locked = bool(angle_locked)

        if points is not None:
            parsed_points: list[tuple[float, float]] = []
            for index, point in enumerate(points):
                px = float(point[0])
                py = float(point[1])
                if not (math.isfinite(px) and math.isfinite(py)):
                    raise ValueError(
                        f"connector points[{index}] must contain finite coordinates"
                    )
                parsed_points.append((px, py))
            connector.points = parsed_points
            if points_locked is None:
                connector.points_locked = bool(parsed_points)

        if points_locked is not None:
            connector.points_locked = bool(points_locked)

        return connector

    def _has_variable(self, name: str) -> bool:
        """Check if a normalized variable name exists in the model."""
        return name in self.stocks or name in self.flows or name in self.auxs

    def _variable_kind(self, norm_name: str) -> str | None:
        """Get variable kind by normalized name."""
        if norm_name in self.stocks:
            return "stock"
        if norm_name in self.flows:
            return "flow"
        if norm_name in self.auxs:
            return "aux"
        return None

    def _validate_new_variable_name(self, name: str):
        """Ensure a new variable name is valid and does not collide."""
        norm_name = self._normalize_name(name)
        if not norm_name:
            raise ValueError("Variable name cannot be empty")
        if self._has_variable(norm_name):
            raise ValueError(f"Variable '{name}' already exists")

    def _replace_equation_identifier(self, equation: str, old_name: str, new_name: str) -> str:
        """Replace exact identifier tokens in equations."""
        if not equation or old_name == new_name:
            return equation
        pattern = re.compile(rf"\b{re.escape(old_name)}\b")
        return pattern.sub(new_name, equation)

    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        """Deduplicate while preserving order."""
        deduped: list[str] = []
        for item in items:
            if item not in deduped:
                deduped.append(item)
        return deduped

    def _replace_variable_everywhere(self, old_norm: str, new_norm: str):
        """Replace a variable identifier across model relationships and equations."""
        for flow in self.flows.values():
            if flow.from_stock == old_norm:
                flow.from_stock = new_norm
            if flow.to_stock == old_norm:
                flow.to_stock = new_norm

            flow.equation = self._replace_equation_identifier(flow.equation, old_norm, new_norm)

        for stock in self.stocks.values():
            stock.inflows = self._dedupe_preserve_order([
                new_norm if flow_name == old_norm else flow_name for flow_name in stock.inflows
            ])
            stock.outflows = self._dedupe_preserve_order([
                new_norm if flow_name == old_norm else flow_name for flow_name in stock.outflows
            ])
            stock.initial_value = self._replace_equation_identifier(stock.initial_value, old_norm, new_norm)

        for aux in self.auxs.values():
            aux.equation = self._replace_equation_identifier(aux.equation, old_norm, new_norm)

        for connector in self.connectors:
            if connector.from_var == old_norm:
                connector.from_var = new_norm
            if connector.to_var == old_norm:
                connector.to_var = new_norm

        for module in self.modules.values():
            module.members = self._dedupe_preserve_order([
                new_norm if member == old_norm else member for member in module.members
            ])

    def rename_variable(self, old_name: str, new_name: str) -> tuple[str, str]:
        """Rename a stock/flow/aux and update all dependent references."""
        old_norm = self._normalize_name(old_name)
        new_norm = self._normalize_name(new_name)
        kind = self._variable_kind(old_norm)
        if kind is None:
            raise ValueError(f"Variable '{old_name}' does not exist")
        if not new_norm:
            raise ValueError("new_name cannot be empty")
        if old_norm != new_norm and self._has_variable(new_norm):
            raise ValueError(f"Variable '{new_name}' already exists")

        if kind == "stock":
            var = self.stocks.pop(old_norm)
            var.name = new_name
            self.stocks[new_norm] = var
        elif kind == "flow":
            var = self.flows.pop(old_norm)
            var.name = new_name
            self.flows[new_norm] = var
        else:
            var = self.auxs.pop(old_norm)
            var.name = new_name
            self.auxs[new_norm] = var

        self._replace_variable_everywhere(old_norm, new_norm)
        return kind, new_norm

    def _equation_reference_sites(self, norm_name: str) -> list[str]:
        """Find equations (or initial values) referencing a variable."""
        sites: list[str] = []
        for flow in self.flows.values():
            refs = self._extract_variable_refs(flow.equation)
            if norm_name in refs:
                sites.append(f"flow '{flow.name}'")
        for aux in self.auxs.values():
            refs = self._extract_variable_refs(aux.equation)
            if norm_name in refs:
                sites.append(f"aux '{aux.name}'")
        for stock in self.stocks.values():
            refs = self._extract_variable_refs(stock.initial_value)
            if norm_name in refs:
                sites.append(f"stock '{stock.name}'")
        return sites

    def delete_variable(self, name: str, force: bool = False) -> dict[str, int | str]:
        """Delete a stock/flow/aux while keeping references consistent.

        Rules:
        - equation references must be removed manually before deletion
        - connected stock->flow structural links require force=True for stock deletion
        """
        norm_name = self._normalize_name(name)
        kind = self._variable_kind(norm_name)
        if kind is None:
            raise ValueError(f"Variable '{name}' does not exist")

        refs = self._equation_reference_sites(norm_name)
        if refs:
            ref_str = ", ".join(refs[:3])
            suffix = "..." if len(refs) > 3 else ""
            raise ValueError(
                f"Cannot delete variable '{name}' because it is referenced in equations: {ref_str}{suffix}"
            )

        detached_flows = 0
        if kind == "stock":
            connected_flows = sorted({
                *self.stocks[norm_name].inflows,
                *self.stocks[norm_name].outflows,
            })
            if connected_flows and not force:
                display_flows = ", ".join(self._display_name(f) for f in connected_flows[:4])
                suffix = "..." if len(connected_flows) > 4 else ""
                raise ValueError(
                    f"Cannot delete stock '{name}' with connected flows ({display_flows}{suffix}); use force=true to detach flows first"
                )
            for flow_name in connected_flows:
                flow = self.flows.get(flow_name)
                if flow is None:
                    continue
                if flow.from_stock == norm_name:
                    flow.from_stock = None
                if flow.to_stock == norm_name:
                    flow.to_stock = None
                detached_flows += 1
            del self.stocks[norm_name]
        elif kind == "flow":
            for stock in self.stocks.values():
                stock.inflows = [fname for fname in stock.inflows if fname != norm_name]
                stock.outflows = [fname for fname in stock.outflows if fname != norm_name]
            del self.flows[norm_name]
        else:
            del self.auxs[norm_name]

        connectors_before = len(self.connectors)
        self.connectors = [
            conn for conn in self.connectors
            if conn.from_var != norm_name and conn.to_var != norm_name
        ]
        removed_connectors = connectors_before - len(self.connectors)

        removed_module_memberships = 0
        for module in self.modules.values():
            before = len(module.members)
            module.members = [member for member in module.members if member != norm_name]
            removed_module_memberships += before - len(module.members)

        return {
            "kind": kind,
            "removed_connectors": removed_connectors,
            "removed_module_memberships": removed_module_memberships,
            "detached_flows": detached_flows,
        }

    def set_sim_specs(
        self,
        start: float | None = None,
        stop: float | None = None,
        dt: float | None = None,
        method: str | None = None,
        time_units: str | None = None,
    ) -> SimSpecs:
        """Update simulation specs while preserving omitted fields."""
        new_start = self.sim_specs.start if start is None else float(start)
        new_stop = self.sim_specs.stop if stop is None else float(stop)
        new_dt = self.sim_specs.dt if dt is None else float(dt)
        if new_dt <= 0:
            raise ValueError("dt must be > 0")
        if new_stop <= new_start:
            raise ValueError("stop must be greater than start")
        self.sim_specs.start = new_start
        self.sim_specs.stop = new_stop
        self.sim_specs.dt = new_dt
        if method is not None:
            self.sim_specs.method = str(method)
        if time_units is not None:
            self.sim_specs.time_units = str(time_units)
        return self.sim_specs

    def update_stock(
        self,
        name: str,
        initial_value: str | None = None,
        units: str | None = None,
        non_negative: bool | None = None,
        x: float | None = None,
        y: float | None = None,
    ) -> Stock:
        """Update stock fields while preserving relationships."""
        norm_name = self._normalize_name(name)
        stock = self.stocks.get(norm_name)
        if stock is None:
            raise ValueError(f"Stock '{name}' does not exist")
        if initial_value is not None:
            stock.initial_value = str(initial_value)
        if units is not None:
            stock.units = str(units)
        if non_negative is not None:
            stock.non_negative = bool(non_negative)
        if x is not None:
            stock.x = float(x)
        if y is not None:
            stock.y = float(y)
        return stock

    def update_flow(
        self,
        name: str,
        equation: str | None = None,
        units: str | None = None,
        non_negative: bool | None = None,
        x: float | None = None,
        y: float | None = None,
        graphical_function: GraphicalFunction | None = None,
    ) -> Flow:
        """Update flow fields while preserving structural stock links."""
        norm_name = self._normalize_name(name)
        flow = self.flows.get(norm_name)
        if flow is None:
            raise ValueError(f"Flow '{name}' does not exist")
        if equation is not None:
            flow.equation = str(equation)
        if units is not None:
            flow.units = str(units)
        if non_negative is not None:
            flow.non_negative = bool(non_negative)
        if x is not None:
            flow.x = float(x)
        if y is not None:
            flow.y = float(y)
        if graphical_function is not None:
            flow.graphical_function = graphical_function
        return flow

    def update_aux(
        self,
        name: str,
        equation: str | None = None,
        units: str | None = None,
        x: float | None = None,
        y: float | None = None,
        graphical_function: GraphicalFunction | None = None,
    ) -> Aux:
        """Update auxiliary fields."""
        norm_name = self._normalize_name(name)
        aux = self.auxs.get(norm_name)
        if aux is None:
            raise ValueError(f"Auxiliary '{name}' does not exist")
        if equation is not None:
            aux.equation = str(equation)
        if units is not None:
            aux.units = str(units)
        if x is not None:
            aux.x = float(x)
        if y is not None:
            aux.y = float(y)
        if graphical_function is not None:
            aux.graphical_function = graphical_function
        return aux

    def create_module(self, name: str, members: list[str] | None = None) -> Module:
        """Create a logical module/group."""
        norm_name = self._normalize_name(name)
        if norm_name in self.modules:
            raise ValueError(f"Module '{name}' already exists")

        normalized_members: list[str] = []
        for member in members or []:
            norm_member = self._normalize_name(member)
            if not self._has_variable(norm_member):
                raise ValueError(f"Module member '{member}' is not a known stock, flow, or auxiliary")
            if norm_member not in normalized_members:
                normalized_members.append(norm_member)

        module = Module(name=name, members=normalized_members)
        self.modules[norm_name] = module
        return module

    def add_to_module(self, module_name: str, members: list[str]) -> Module:
        """Add variables to an existing module."""
        norm_module_name = self._normalize_name(module_name)
        module = self.modules.get(norm_module_name)
        if module is None:
            raise ValueError(f"Module '{module_name}' does not exist")

        for member in members:
            norm_member = self._normalize_name(member)
            if not self._has_variable(norm_member):
                raise ValueError(f"Module member '{member}' is not a known stock, flow, or auxiliary")
            if norm_member not in module.members:
                module.members.append(norm_member)
        return module

    def remove_from_module(self, module_name: str, members: list[str]) -> Module:
        """Remove variables from an existing module."""
        norm_module_name = self._normalize_name(module_name)
        module = self.modules.get(norm_module_name)
        if module is None:
            raise ValueError(f"Module '{module_name}' does not exist")

        remove_set = {self._normalize_name(member) for member in members}
        module.members = [member for member in module.members if member not in remove_set]
        return module

    def rename_module(self, module_name: str, new_name: str) -> Module:
        """Rename an existing module."""
        old_norm = self._normalize_name(module_name)
        new_norm = self._normalize_name(new_name)
        module = self.modules.get(old_norm)
        if module is None:
            raise ValueError(f"Module '{module_name}' does not exist")
        if new_norm != old_norm and new_norm in self.modules:
            raise ValueError(f"Module '{new_name}' already exists")

        module.name = new_name
        if new_norm != old_norm:
            self.modules[new_norm] = module
            del self.modules[old_norm]
        return module

    def delete_module(self, module_name: str) -> Module:
        """Delete a module."""
        norm_module_name = self._normalize_name(module_name)
        module = self.modules.get(norm_module_name)
        if module is None:
            raise ValueError(f"Module '{module_name}' does not exist")
        del self.modules[norm_module_name]
        return module

    def set_module_view(
        self,
        module_name: str,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> Module:
        """Set explicit view box geometry for a module."""
        norm_module_name = self._normalize_name(module_name)
        module = self.modules.get(norm_module_name)
        if module is None:
            raise ValueError(f"Module '{module_name}' does not exist")
        if width <= 0 or height <= 0:
            raise ValueError("Module width and height must be > 0")
        module.x = float(x)
        module.y = float(y)
        module.width = float(width)
        module.height = float(height)
        return module

    def set_module_style(
        self,
        module_name: str,
        border_color: str | None = None,
        background: str | None = None,
        font_color: str | None = None,
        font_size: str | None = None,
        label_side: str | None = None,
    ) -> Module:
        """Set display style for a module box in the view."""
        norm_module_name = self._normalize_name(module_name)
        module = self.modules.get(norm_module_name)
        if module is None:
            raise ValueError(f"Module '{module_name}' does not exist")

        if all(
            value is None
            for value in (border_color, background, font_color, font_size, label_side)
        ):
            raise ValueError("At least one module style field must be provided")

        if label_side is not None and label_side not in {"top", "bottom", "left", "right"}:
            raise ValueError("module label_side must be one of: top, bottom, left, right")

        if border_color is not None:
            module.border_color = border_color
        if background is not None:
            module.background = background
        if font_color is not None:
            module.font_color = font_color
        if font_size is not None:
            module.font_size = font_size
        if label_side is not None:
            module.label_side = label_side

        return module

    def _member_bounds(self, member: str) -> tuple[float, float, float, float] | None:
        """Get member bounds as (left, top, right, bottom)."""
        if member in self.stocks:
            stock = self.stocks[member]
            if stock.x is None or stock.y is None:
                return None
            half_w = stock.width / 2
            half_h = stock.height / 2
            return (stock.x - half_w, stock.y - half_h, stock.x + half_w, stock.y + half_h)

        if member in self.auxs:
            aux = self.auxs[member]
            if aux.x is None or aux.y is None:
                return None
            return (
                aux.x - AUX_RADIUS,
                aux.y - AUX_RADIUS,
                aux.x + AUX_RADIUS,
                aux.y + AUX_RADIUS,
            )

        if member in self.flows:
            flow = self.flows[member]
            if flow.x is None or flow.y is None:
                return None
            left = flow.x - 10
            right = flow.x + 10
            top = flow.y - 10
            bottom = flow.y + 10
            if flow.points:
                xs = [p[0] for p in flow.points]
                ys = [p[1] for p in flow.points]
                left = min(left, min(xs))
                right = max(right, max(xs))
                top = min(top, min(ys))
                bottom = max(bottom, max(ys))
            return (left, top, right, bottom)
        return None

    def auto_place_module_boxes(
        self,
        padding: float = 40.0,
        min_width: float = 180.0,
        min_height: float = 120.0,
        only_missing: bool = False,
    ):
        """Auto-place module boxes around member elements."""
        for module in self.modules.values():
            if only_missing and None not in (module.x, module.y, module.width, module.height):
                continue
            if not module.members:
                continue

            bounds: list[tuple[float, float, float, float]] = []
            for member in module.members:
                b = self._member_bounds(member)
                if b is not None:
                    bounds.append(b)
            if not bounds:
                continue

            left = min(b[0] for b in bounds) - padding
            top = min(b[1] for b in bounds) - padding
            right = max(b[2] for b in bounds) + padding
            bottom = max(b[3] for b in bounds) + padding

            width = max(min_width, right - left)
            height = max(min_height, bottom - top)
            module.width = width
            module.height = height
            module.x = left + width / 2
            module.y = top + height / 2

    def _calculate_stock_sizes(self):
        """Calculate appropriate width/height for each stock based on connectivity.

        Stocks with more flows get larger to allow visual separation of flow attachments.
        Maintains a pleasing aspect ratio (roughly 1.3:1 width:height).
        """
        MIN_WIDTH = 45
        MAX_WIDTH = 120
        MIN_HEIGHT = 35
        MAX_HEIGHT = 90
        FLOW_WIDTH_CONTRIBUTION = 15  # Extra width per flow beyond 2
        ASPECT_RATIO = 1.3  # width:height ratio

        for stock in self.stocks.values():
            if stock.size_locked:
                continue
            num_flows = len(stock.inflows) + len(stock.outflows)

            # Start at minimum, add width for extra flows
            width = MIN_WIDTH + max(0, num_flows - 2) * FLOW_WIDTH_CONTRIBUTION
            width = min(width, MAX_WIDTH)

            # Scale height to maintain aspect ratio
            height = int(width / ASPECT_RATIO)
            height = max(MIN_HEIGHT, min(height, MAX_HEIGHT))

            stock.width = width
            stock.height = height

    def _auto_layout(self):
        """Auto-arrange visual positions using force-directed layout.

        Uses connector relationships to position elements:
        1. Calculates stock sizes based on connectivity
        2. Builds dependency graph from connectors
        3. Detects subsystems (connected components)
        4. Positions elements via Fruchterman-Reingold force-directed layout
        5. Separates independent subsystems visually

        Always recalculates flow.points to ensure flows connect to stocks correctly.
        """
        # Calculate stock sizes first (affects flow attachment)
        self._calculate_stock_sizes()

        SUBSYSTEM_GAP = 250

        # Build dependency graph from connectors
        outgoing, incoming = self._build_dependency_graph()

        # Find subsystems (connected components)
        subsystems = self._find_subsystems(outgoing, incoming)

        # Position each subsystem
        subsystem_bounds: list[tuple[float, float, float, float]] = []

        for subsystem in subsystems:
            bounds = self._position_subsystem(subsystem, outgoing, incoming)
            subsystem_bounds.append(bounds)

        # Arrange subsystems relative to each other (largest centered, others offset)
        if len(subsystems) > 1 and len(subsystem_bounds) > 1:
            self._arrange_subsystems(subsystems, subsystem_bounds, SUBSYSTEM_GAP)

        # Always recalculate flow points to connect stocks at their actual positions
        self._recalculate_flow_points()

        # Calculate connector angles based on final positions
        self._calculate_connector_angles(force=True)

    def _calculate_flow_offset(self, index: int, total: int) -> float:
        """Calculate vertical offset for flow attachment point.

        When multiple flows share a stock endpoint, offset them vertically
        to prevent overlap. Centers the group around the stock center.

        Args:
            index: This flow's index in the group (0-based)
            total: Total number of flows in the group

        Returns:
            Vertical offset in pixels (positive = down, negative = up)
        """
        if total <= 1:
            return 0.0
        # Center the group: e.g., 3 flows -> offsets of -20, 0, +20
        return (index - (total - 1) / 2) * 20.0

    @staticmethod
    def _stock_attachment_point(
        stock_x: float, stock_y: float,
        half_w: float, half_h: float,
        target_x: float, target_y: float,
    ) -> tuple[float, float]:
        """Find the point on a stock's edge closest to a target point.

        Exits from the edge that faces the target (direction-aware).
        """
        dx = target_x - stock_x
        dy = target_y - stock_y

        if abs(dx) < 0.001 and abs(dy) < 0.001:
            return (stock_x + half_w, stock_y)  # default: right edge

        # Determine dominant direction
        # Compare aspect-ratio-adjusted deltas to pick edge
        if abs(dx) / max(half_w, 0.001) >= abs(dy) / max(half_h, 0.001):
            # Horizontal dominant
            if dx >= 0:
                return (stock_x + half_w, stock_y)  # right edge
            else:
                return (stock_x - half_w, stock_y)  # left edge
        else:
            # Vertical dominant
            if dy >= 0:
                return (stock_x, stock_y + half_h)  # bottom edge
            else:
                return (stock_x, stock_y - half_h)  # top edge

    def _recalculate_flow_points(self):
        """Recalculate flow.points to connect stocks at their actual positions.

        Direction-aware: exits/enters from the stock edge closest to the
        destination, supporting stocks at arbitrary angles (not just horizontal).
        Uses orthogonal routing for multiple flows from the same stock.
        """
        ROUTE_OFFSET = 40

        # Group flows by their source and destination stocks
        outflows_by_stock: dict[str, list[str]] = {}
        inflows_by_stock: dict[str, list[str]] = {}

        for name, flow in self.flows.items():
            if flow.from_stock:
                if flow.from_stock not in outflows_by_stock:
                    outflows_by_stock[flow.from_stock] = []
                outflows_by_stock[flow.from_stock].append(name)
            if flow.to_stock:
                if flow.to_stock not in inflows_by_stock:
                    inflows_by_stock[flow.to_stock] = []
                inflows_by_stock[flow.to_stock].append(name)

        # Sort flow lists for determinism
        for stock_name in outflows_by_stock:
            outflows_by_stock[stock_name].sort()
        for stock_name in inflows_by_stock:
            inflows_by_stock[stock_name].sort()

        for name, flow in self.flows.items():
            if flow.points_locked and flow.points:
                continue
            from_stock = self.stocks.get(flow.from_stock) if flow.from_stock else None
            to_stock = self.stocks.get(flow.to_stock) if flow.to_stock else None

            if from_stock and to_stock:
                from_x = from_stock.x if from_stock.x is not None else 0.0
                from_y = from_stock.y if from_stock.y is not None else 0.0
                to_x = to_stock.x if to_stock.x is not None else 0.0
                to_y = to_stock.y if to_stock.y is not None else 0.0

                from_hw = from_stock.width / 2
                from_hh = from_stock.height / 2
                to_hw = to_stock.width / 2
                to_hh = to_stock.height / 2

                # Direction-aware attachment points
                exit_pt = self._stock_attachment_point(from_x, from_y, from_hw, from_hh, to_x, to_y)
                entry_pt = self._stock_attachment_point(to_x, to_y, to_hw, to_hh, from_x, from_y)

                # Check if multiple outflows — use orthogonal routing
                outflows = outflows_by_stock.get(flow.from_stock, [name])
                total = len(outflows)

                if total == 1:
                    flow.points = [exit_pt, entry_pt]
                else:
                    # Multiple flows: orthogonal routing
                    flows_above: list[str] = []
                    flows_same: list[str] = []
                    flows_below: list[str] = []

                    for flow_name in outflows:
                        f = self.flows[flow_name]
                        dest = self.stocks.get(f.to_stock) if f.to_stock else None
                        if dest and dest.y is not None:
                            dest_y = dest.y
                            if dest_y < from_y - 20:
                                flows_above.append(flow_name)
                            elif dest_y > from_y + 20:
                                flows_below.append(flow_name)
                            else:
                                flows_same.append(flow_name)
                        else:
                            flows_same.append(flow_name)

                    if name in flows_above:
                        go_up = True
                        group_index = flows_above.index(name)
                    elif name in flows_below:
                        go_up = False
                        group_index = flows_below.index(name)
                    else:
                        same_index = flows_same.index(name)
                        if same_index == 0:
                            flow.points = [exit_pt, entry_pt]
                            continue
                        go_up = (same_index % 2 == 1)
                        group_index = (same_index - 1) // 2

                    offset = (group_index + 1) * ROUTE_OFFSET
                    route_y = from_y - from_hh - offset if go_up else from_y + from_hh + offset

                    exit_edge_y = from_y - from_hh if go_up else from_y + from_hh
                    entry_edge_y = to_y - to_hh if go_up else to_y + to_hh

                    flow.points = [
                        (exit_pt[0], exit_edge_y),
                        (exit_pt[0], route_y),
                        (entry_pt[0], route_y),
                        (entry_pt[0], entry_edge_y),
                    ]

            elif from_stock:
                # Source-only flow (external sink) — exit toward the right
                from_x = from_stock.x if from_stock.x is not None else 0.0
                from_y = from_stock.y if from_stock.y is not None else 0.0
                from_hw = from_stock.width / 2

                from_offset = 0.0
                if flow.from_stock and flow.from_stock in outflows_by_stock:
                    outflows = outflows_by_stock[flow.from_stock]
                    if len(outflows) > 1:
                        index = outflows.index(name)
                        from_offset = self._calculate_flow_offset(index, len(outflows))

                flow.points = [
                    (from_x + from_hw, from_y + from_offset),
                    (from_x + 160, from_y + from_offset),
                ]

            elif to_stock:
                # Sink-only flow (external source) — enter from the left
                to_x = to_stock.x if to_stock.x is not None else 0.0
                to_y = to_stock.y if to_stock.y is not None else 0.0
                to_hw = to_stock.width / 2

                to_offset = 0.0
                if flow.to_stock and flow.to_stock in inflows_by_stock:
                    inflows = inflows_by_stock[flow.to_stock]
                    if len(inflows) > 1:
                        index = inflows.index(name)
                        to_offset = self._calculate_flow_offset(index, len(inflows))

                flow.points = [
                    (to_x - 160, to_y + to_offset),
                    (to_x - to_hw, to_y + to_offset),
                ]

    def _calculate_connector_angles(self, force: bool = False):
        """Calculate connector angles based on source and target positions.

        Uses atan2 to compute the angle from source to target.
        Convention: degrees, 0 = right, counter-clockwise positive.
        Note: -dy because screen y-coordinates increase downward.
        """
        # Build position lookup for all elements
        positions: dict[str, tuple[float, float]] = {}
        for name, stock in self.stocks.items():
            if stock.x is not None and stock.y is not None:
                positions[name] = (stock.x, stock.y)
        for name, flow in self.flows.items():
            if flow.x is not None and flow.y is not None:
                positions[name] = (flow.x, flow.y)
        for name, aux in self.auxs.items():
            if aux.x is not None and aux.y is not None:
                positions[name] = (aux.x, aux.y)

        for conn in self.connectors:
            if conn.angle_locked and not force:
                continue
            from_pos = positions.get(conn.from_var)
            to_pos = positions.get(conn.to_var)

            if from_pos and to_pos:
                dx = to_pos[0] - from_pos[0]
                dy = to_pos[1] - from_pos[1]

                # Handle zero distance (same position)
                if abs(dx) < 0.001 and abs(dy) < 0.001:
                    conn.angle = 0
                else:
                    # -dy because y increases downward in screen coordinates
                    conn.angle = math.degrees(math.atan2(-dy, dx))

    # =========================================================================
    # Layout Collision/Crossing Detection and Resolution
    # =========================================================================

    def _get_element_box(self, name: str) -> BoundingBox | None:
        """Get bounding box for any model element."""
        if name in self.stocks:
            stock = self.stocks[name]
            if stock.x is not None and stock.y is not None:
                return BoundingBox(stock.x, stock.y, stock.width, stock.height)
        elif name in self.flows:
            flow = self.flows[name]
            if flow.x is not None and flow.y is not None:
                # Flow valve is roughly 20x20
                return BoundingBox(flow.x, flow.y, 20, 20)
        elif name in self.auxs:
            aux = self.auxs[name]
            if aux.x is not None and aux.y is not None:
                return BoundingBox(aux.x, aux.y, AUX_RADIUS * 2, AUX_RADIUS * 2)
        return None

    def _get_all_bounding_boxes(self) -> dict[str, BoundingBox]:
        """Get bounding boxes for all positioned elements."""
        boxes: dict[str, BoundingBox] = {}
        for name in self.stocks:
            box = self._get_element_box(name)
            if box:
                boxes[name] = box
        for name in self.auxs:
            box = self._get_element_box(name)
            if box:
                boxes[name] = box
        # Note: flows are not included as their position is the valve,
        # and flow lines are handled separately
        return boxes

    def _get_connector_segments(self) -> dict[int, tuple[tuple[float, float], tuple[float, float]]]:
        """Get line segments for all connectors (from source to target position)."""
        segments: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}

        # Build position lookup
        positions: dict[str, tuple[float, float]] = {}
        for name, stock in self.stocks.items():
            if stock.x is not None and stock.y is not None:
                positions[name] = (stock.x, stock.y)
        for name, flow in self.flows.items():
            if flow.x is not None and flow.y is not None:
                positions[name] = (flow.x, flow.y)
        for name, aux in self.auxs.items():
            if aux.x is not None and aux.y is not None:
                positions[name] = (aux.x, aux.y)

        for conn in self.connectors:
            from_pos = positions.get(conn.from_var)
            to_pos = positions.get(conn.to_var)
            if from_pos and to_pos:
                segments[conn.uid] = (from_pos, to_pos)

        return segments

    def _get_flow_segments(self) -> dict[str, list[tuple[tuple[float, float], tuple[float, float]]]]:
        """Get line segments for all flow paths."""
        segments: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]] = {}

        for name, flow in self.flows.items():
            if flow.points and len(flow.points) >= 2:
                flow_segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
                for i in range(len(flow.points) - 1):
                    flow_segs.append((flow.points[i], flow.points[i + 1]))
                segments[name] = flow_segs

        return segments

    def _detect_aux_collisions(self) -> list[tuple[str, str]]:
        """Detect pairs of auxs that overlap."""
        collisions: list[tuple[str, str]] = []
        aux_names = list(self.auxs.keys())

        for i, name1 in enumerate(aux_names):
            box1 = self._get_element_box(name1)
            if not box1:
                continue
            for name2 in aux_names[i + 1:]:
                box2 = self._get_element_box(name2)
                if box2 and box1.intersects(box2, margin=5):
                    collisions.append((name1, name2))

        return collisions

    def _detect_connector_flow_crossings(self) -> list[tuple[int, str]]:
        """Detect connectors that cross flow lines. Returns (connector_uid, flow_name) pairs.

        Note: A connector is expected to touch its target flow, so we skip checking
        if a connector crosses the flow it's connected TO.
        """
        crossings: list[tuple[int, str]] = []

        connector_segs = self._get_connector_segments()
        flow_segs = self._get_flow_segments()

        # Build map of connector uid -> target name
        conn_targets: dict[int, str] = {}
        for conn in self.connectors:
            conn_targets[conn.uid] = conn.to_var

        for conn_uid, (cp1, cp2) in connector_segs.items():
            target = conn_targets.get(conn_uid)
            for flow_name, segments in flow_segs.items():
                # Skip if this is the connector's target flow
                if flow_name == target:
                    continue
                for fp1, fp2 in segments:
                    if segments_intersect(cp1, cp2, fp1, fp2):
                        crossings.append((conn_uid, flow_name))
                        break  # One crossing per connector-flow pair is enough

        return crossings

    def _detect_flow_stock_crossings(self) -> list[tuple[str, str]]:
        """Detect flows that pass through stocks (not their source/dest). Returns (flow_name, stock_name) pairs."""
        crossings: list[tuple[str, str]] = []

        flow_segs = self._get_flow_segments()

        for flow_name, segments in flow_segs.items():
            flow = self.flows[flow_name]
            for stock_name in self.stocks:
                # Skip source and destination stocks
                if stock_name in (flow.from_stock, flow.to_stock):
                    continue

                box = self._get_element_box(stock_name)
                if not box:
                    continue

                for p1, p2 in segments:
                    if segment_intersects_box(p1, p2, box):
                        crossings.append((flow_name, stock_name))
                        break

        return crossings

    def _detect_connector_stock_crossings(self) -> list[tuple[int, str]]:
        """Detect connectors that pass through stocks. Returns (connector_uid, stock_name) pairs.

        Skips stocks that are the source or target of the connector.
        """
        crossings: list[tuple[int, str]] = []

        connector_segs = self._get_connector_segments()

        # Build map of connector uid -> (from_var, to_var)
        conn_endpoints: dict[int, tuple[str, str]] = {}
        for conn in self.connectors:
            conn_endpoints[conn.uid] = (conn.from_var, conn.to_var)

        for conn_uid, (cp1, cp2) in connector_segs.items():
            from_var, to_var = conn_endpoints.get(conn_uid, ("", ""))
            for stock_name in self.stocks:
                # Skip if this stock is the source or target of the connector
                if stock_name in (from_var, to_var):
                    continue

                box = self._get_element_box(stock_name)
                if not box:
                    continue

                if segment_intersects_box(cp1, cp2, box):
                    crossings.append((conn_uid, stock_name))

        return crossings

    def _separate_auxs(self, name1: str, name2: str):
        """Push two overlapping auxs apart."""
        aux1 = self.auxs.get(name1)
        aux2 = self.auxs.get(name2)

        if not aux1 or not aux2:
            return
        if aux1.x is None or aux1.y is None or aux2.x is None or aux2.y is None:
            return

        # Direction from aux1 to aux2
        dx = aux2.x - aux1.x
        dy = aux2.y - aux1.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.001:
            # Same position - push horizontally
            dx, dy = 1.0, 0.0
            dist = 1.0

        # Minimum distance is 2 * radius + margin
        min_dist = AUX_RADIUS * 2 + 10
        if dist >= min_dist:
            return  # Already separated

        # Push each aux half the needed distance
        push = (min_dist - dist) / 2 + 2
        aux1.x -= push * dx / dist
        aux1.y -= push * dy / dist
        aux2.x += push * dx / dist
        aux2.y += push * dy / dist

    def _reposition_aux_to_avoid_crossing(self, conn_uid: int, obstacle_name: str, obstacle_type: str = "flow"):
        """Move aux so its connector doesn't cross the specified obstacle.

        Args:
            conn_uid: The connector's unique ID
            obstacle_name: Name of the flow or stock to avoid
            obstacle_type: Either "flow" or "stock"
        """
        # Find the connector and its source aux
        conn = None
        for c in self.connectors:
            if c.uid == conn_uid:
                conn = c
                break

        if not conn:
            return

        aux = self.auxs.get(conn.from_var)
        if not aux or aux.x is None or aux.y is None:
            return

        # Get target position and size for proportional offsets
        target_pos: tuple[float, float] | None = None
        target_size = 45  # Default size for offset calculation
        if conn.to_var in self.stocks:
            stock = self.stocks[conn.to_var]
            if stock.x is not None and stock.y is not None:
                target_pos = (stock.x, stock.y)
                target_size = max(stock.width, stock.height)
        elif conn.to_var in self.flows:
            flow = self.flows[conn.to_var]
            if flow.x is not None and flow.y is not None:
                target_pos = (flow.x, flow.y)
                target_size = 20  # Flow valve size
        elif conn.to_var in self.auxs:
            other_aux = self.auxs[conn.to_var]
            if other_aux.x is not None and other_aux.y is not None:
                target_pos = (other_aux.x, other_aux.y)
                target_size = AUX_RADIUS * 2

        if not target_pos:
            return

        # Build obstacle check function based on type
        if obstacle_type == "flow":
            flow_segs = self._get_flow_segments().get(obstacle_name, [])
            if not flow_segs:
                return

            def crosses_obstacle(candidate: tuple[float, float]) -> bool:
                for fp1, fp2 in flow_segs:
                    if segments_intersect(candidate, target_pos, fp1, fp2):
                        return True
                return False
        else:  # stock
            stock_box = self._get_element_box(obstacle_name)
            if not stock_box:
                return

            def crosses_obstacle(candidate: tuple[float, float]) -> bool:
                return segment_intersects_box(candidate, target_pos, stock_box)

        # Calculate proportional offsets based on target element size
        base_offset = target_size + AUX_RADIUS + 20
        diag_offset = int(base_offset * 0.75)
        far_offset = int(base_offset * 1.5)

        offsets = [
            (0, -base_offset), (0, base_offset),  # above, below
            (-base_offset, 0), (base_offset, 0),  # left, right
            (-diag_offset, -diag_offset), (diag_offset, -diag_offset),  # diagonal up
            (-diag_offset, diag_offset), (diag_offset, diag_offset),  # diagonal down
            (0, -far_offset), (0, far_offset),  # further above/below
            (-far_offset, 0), (far_offset, 0),  # further left/right
        ]

        for dx, dy in offsets:
            candidate = (target_pos[0] + dx, target_pos[1] + dy)

            if not crosses_obstacle(candidate):
                aux.x, aux.y = candidate
                return

        # Fallback: keep current position (crossing unavoidable with simple repositioning)

    def _reroute_flow_around_stock(self, flow_name: str, stock_name: str):
        """Add waypoints to route flow around a stock it currently crosses.

        This is a best-effort fix - if the flow has already been modified multiple
        times (>8 points), we skip to avoid infinite loops.
        """
        flow = self.flows.get(flow_name)
        stock = self.stocks.get(stock_name)

        if not flow or not stock or not flow.points or len(flow.points) < 2:
            return
        if stock.x is None or stock.y is None:
            return

        # Guard against infinite rerouting - if flow already has many points, skip
        if len(flow.points) > 8:
            return

        box = BoundingBox(stock.x, stock.y, stock.width, stock.height)
        # Clearance proportional to stock size (minimum 20px, plus half the larger dimension)
        clearance = 20 + max(stock.width, stock.height) / 2

        # Find which segment intersects and modify
        for i in range(len(flow.points) - 1):
            p1, p2 = flow.points[i], flow.points[i + 1]

            if not segment_intersects_box(p1, p2, box):
                continue

            # Determine if this is a horizontal or vertical segment
            is_horizontal = abs(p1[1] - p2[1]) < abs(p1[0] - p2[0])

            if is_horizontal:
                # Route above or below the stock - pick the side further from current Y
                dist_above = abs(p1[1] - (stock.y - stock.height / 2))
                dist_below = abs(p1[1] - (stock.y + stock.height / 2))

                if dist_above > dist_below:
                    route_y = stock.y - stock.height / 2 - clearance
                else:
                    route_y = stock.y + stock.height / 2 + clearance

                # Insert waypoints to go around
                new_points = list(flow.points[:i + 1])
                new_points.append((p1[0], route_y))
                new_points.append((p2[0], route_y))
                new_points.extend(flow.points[i + 1:])
                flow.points = [(float(x), float(y)) for x, y in new_points]
            else:
                # Vertical segment - route left or right
                dist_left = abs(p1[0] - (stock.x - stock.width / 2))
                dist_right = abs(p1[0] - (stock.x + stock.width / 2))

                if dist_left > dist_right:
                    route_x = stock.x - stock.width / 2 - clearance
                else:
                    route_x = stock.x + stock.width / 2 + clearance

                new_points = list(flow.points[:i + 1])
                new_points.append((route_x, p1[1]))
                new_points.append((route_x, p2[1]))
                new_points.extend(flow.points[i + 1:])
                flow.points = [(float(x), float(y)) for x, y in new_points]

            return  # Only fix one intersection per call

    def _resolve_layout_violations(self, max_iterations: int = 10):
        """Iteratively resolve collisions and crossings in the layout."""
        # Track what we've already tried to fix to avoid infinite loops
        processed_flow_stock: set[tuple[str, str]] = set()
        processed_connector_flow: set[tuple[int, str]] = set()
        processed_connector_stock: set[tuple[int, str]] = set()

        for _iteration in range(max_iterations):
            # Detect all violations
            aux_collisions = self._detect_aux_collisions()
            connector_flow_crossings = self._detect_connector_flow_crossings()
            connector_stock_crossings = self._detect_connector_stock_crossings()
            flow_stock_crossings = self._detect_flow_stock_crossings()

            # Filter out already-processed items
            new_connector_flow = [c for c in connector_flow_crossings if c not in processed_connector_flow]
            new_connector_stock = [c for c in connector_stock_crossings if c not in processed_connector_stock]
            new_flow_stock = [c for c in flow_stock_crossings if c not in processed_flow_stock]

            if not aux_collisions and not new_connector_flow and not new_connector_stock and not new_flow_stock:
                return  # Layout is valid (or as good as we can make it)

            # Resolve aux collisions first (simplest)
            for name1, name2 in aux_collisions:
                self._separate_auxs(name1, name2)

            # Resolve connector-flow crossings by repositioning auxs
            for conn_uid, flow_name in new_connector_flow:
                self._reposition_aux_to_avoid_crossing(conn_uid, flow_name, "flow")
                processed_connector_flow.add((conn_uid, flow_name))

            # Resolve connector-stock crossings by repositioning auxs
            for conn_uid, stock_name in new_connector_stock:
                self._reposition_aux_to_avoid_crossing(conn_uid, stock_name, "stock")
                processed_connector_stock.add((conn_uid, stock_name))

            # Resolve flow-stock crossings by rerouting flows
            for flow_name, stock_name in new_flow_stock:
                self._reroute_flow_around_stock(flow_name, stock_name)
                processed_flow_stock.add((flow_name, stock_name))

            # Recalculate connector angles after moving auxs
            self._calculate_connector_angles(force=True)

        # If we hit max iterations, layout is best-effort (some violations may remain)

    def to_xml(
        self,
        auto_layout: bool = True,
        resolve_layout_violations: bool = False,
        compat_mode: str = "permissive",
    ) -> str:
        """Generate XMILE XML string for the model."""
        from .xmile_io import model_to_xml

        return model_to_xml(
            self,
            auto_layout=auto_layout,
            resolve_layout_violations=resolve_layout_violations,
            compat_mode=compat_mode,
        )

    def _add_view_styles_str(self, lines: list[str]):
        """Add the default view styles as strings."""
        lines.append('\t\t\t<style color="black" background="white" font_style="normal" font_weight="normal" text_decoration="none" text_align="center" vertical_text_align="center" font_color="black" font_family="Arial" font_size="10pt" padding="2" border_color="black" border_width="thin" border_style="none">')
        lines.append('\t\t\t\t<text_box color="black" background="white" text_align="left" vertical_text_align="top" font_size="12pt"/>')
        lines.append('\t\t\t</style>')

    def _add_inner_view_styles_str(self, lines: list[str]):
        """Add the inner view styles as strings."""
        lines.append('\t\t\t\t<style color="black" background="white" font_style="normal" font_weight="normal" text_decoration="none" text_align="center" vertical_text_align="center" font_color="black" font_family="Arial" font_size="10pt" padding="2" border_color="black" border_width="thin" border_style="none">')
        lines.append('\t\t\t\t\t<stock color="blue" background="white" font_color="blue" font_size="9pt" label_side="top">')
        lines.append('\t\t\t\t\t\t<shape type="rectangle" width="45" height="35"/>')
        lines.append('\t\t\t\t\t</stock>')
        lines.append('\t\t\t\t\t<flow color="blue" background="white" font_color="blue" font_size="9pt" label_side="bottom"/>')
        lines.append('\t\t\t\t\t<aux color="blue" background="white" font_color="blue" font_size="9pt" label_side="bottom">')
        lines.append('\t\t\t\t\t\t<shape type="circle" radius="18"/>')
        lines.append('\t\t\t\t\t</aux>')
        lines.append('\t\t\t\t\t<group color="#666666" background="#F5F5F5" font_color="black" font_size="9pt" label_side="top"/>')
        lines.append('\t\t\t\t\t<connector color="#FF007F" background="white" font_color="#FF007F" font_size="9pt" isee:thickness="1"/>')
        lines.append('\t\t\t\t</style>')

    def _format_point_list(self, points: list[float]) -> str:
        # XMILE defines point lists as comma-separated (the sep attribute can
        # override, but readers like Stella and PySD assume the spec default).
        return ",".join(f"{p:g}" for p in points)

    def _add_graphical_function_str(self, lines: list[str], gf: GraphicalFunction):
        attrs = f' type="{escape(gf.gf_type)}"' if gf.gf_type else ""
        lines.append(f'\t\t\t\t<gf{attrs}>')
        if gf.xpts is not None:
            lines.append(f'\t\t\t\t\t<xpts>{self._format_point_list(gf.xpts)}</xpts>')
        elif gf.xscale is not None:
            lines.append(f'\t\t\t\t\t<xscale min="{gf.xscale[0]:g}" max="{gf.xscale[1]:g}"/>')
        if gf.yscale is not None:
            lines.append(f'\t\t\t\t\t<yscale min="{gf.yscale[0]:g}" max="{gf.yscale[1]:g}"/>')
        lines.append(f'\t\t\t\t\t<ypts>{self._format_point_list(gf.ypts)}</ypts>')
        lines.append('\t\t\t\t</gf>')


def parse_stmx(filepath: str, compat_mode: str = "permissive") -> StellaModel:
    """Parse an existing .stmx file and return a StellaModel."""
    from .xmile_io import parse_stmx_file

    return parse_stmx_file(filepath, compat_mode=compat_mode)
