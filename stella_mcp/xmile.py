"""XMILE XML generation and parsing for Stella .stmx files."""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional
import uuid
from html import escape


# XML namespaces
XMILE_NS = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"
ISEE_NS = "http://iseesystems.com/XMILE"

# Stella/XMILE built-in functions (not variable names)
STELLA_FUNCTIONS = {
    'IF', 'THEN', 'ELSE', 'AND', 'OR', 'NOT',
    'MIN', 'MAX', 'ABS', 'SIN', 'COS', 'TAN',
    'EXP', 'LN', 'LOG', 'LOG10', 'SQRT', 'INT',
    'ROUND', 'MOD', 'TIME', 'DT', 'STARTTIME', 'STOPTIME',
    'DELAY', 'DELAY1', 'DELAY3', 'DELAYN',
    'SMOOTH', 'SMOOTH3', 'SMOOTHN', 'SMTH1', 'SMTH3', 'SMTHN',
    'TREND', 'FORCST', 'PULSE', 'STEP', 'RAMP',
    'RANDOM', 'NORMAL', 'POISSON', 'EXPRND',
    'PREVIOUS', 'INIT', 'SELF', 'SUM', 'MEAN',
    'GRAPH', 'LOOKUP', 'INTERPOLATE', 'HISTORY',
    'SAFEDIV', 'NPV', 'IRR', 'COUNTER',
    'TRUE', 'FALSE', 'PI', 'E', 'INF', 'NAN',
}


@dataclass
class Stock:
    """Represents a stock (reservoir) in the model."""
    name: str
    initial_value: str
    units: str = ""
    inflows: list[str] = field(default_factory=list)
    outflows: list[str] = field(default_factory=list)
    non_negative: bool = True
    x: Optional[float] = None  # None means auto-position
    y: Optional[float] = None  # None means auto-position


@dataclass
class Flow:
    """Represents a flow between stocks."""
    name: str
    equation: str
    units: str = ""
    from_stock: Optional[str] = None  # None means external source
    to_stock: Optional[str] = None    # None means external sink
    non_negative: bool = True
    x: Optional[float] = None  # None means auto-position
    y: Optional[float] = None  # None means auto-position
    points: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class Aux:
    """Represents an auxiliary variable."""
    name: str
    equation: str
    units: str = ""
    x: Optional[float] = None  # None means auto-position
    y: Optional[float] = None  # None means auto-position


@dataclass
class Connector:
    """Represents a dependency connector between variables."""
    uid: int
    from_var: str
    to_var: str
    angle: float = 0


@dataclass
class SimSpecs:
    """Simulation specifications."""
    start: float = 0
    stop: float = 100
    dt: float = 0.25
    method: str = "Euler"
    time_units: str = "Years"


class StellaModel:
    """Represents a complete Stella system dynamics model."""

    def __init__(self, name: str = "Untitled"):
        self.name = name
        self.uuid = str(uuid.uuid4())
        self.sim_specs = SimSpecs()
        self.stocks: dict[str, Stock] = {}
        self.flows: dict[str, Flow] = {}
        self.auxs: dict[str, Aux] = {}
        self.connectors: list[Connector] = []
        self._connector_uid = 0

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
        if not equation:
            return set()

        # Extract potential variable names (alphanumeric with underscores only)
        # Note: Stella allows spaces in variable names, but in equations they should
        # be written with underscores or quoted. We extract standard identifiers.
        tokens = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', equation)

        refs = set()
        for token in tokens:
            # Check if it's a function or keyword (case-insensitive)
            if token.upper() not in STELLA_FUNCTIONS:
                # Try to filter out pure numbers
                try:
                    float(token)
                except ValueError:
                    # Normalize (in case there are any spaces, though regex won't match them)
                    refs.add(self._normalize_name(token))

        return refs

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

    def _find_stock_chains(self, subsystem: set[str]) -> list[list[str]]:
        """Find linear chains of stocks connected by flows within a subsystem.

        Returns list of chains, where each chain is ordered by flow direction.
        """
        stocks_in_subsystem = [s for s in sorted(subsystem) if s in self.stocks]
        if not stocks_in_subsystem:
            return []

        # Build stock-to-stock adjacency via flows
        stock_adj: dict[str, list[str]] = {s: [] for s in stocks_in_subsystem}
        for name, flow in self.flows.items():
            if (flow.from_stock in stocks_in_subsystem and
                flow.to_stock in stocks_in_subsystem):
                if flow.to_stock not in stock_adj[flow.from_stock]:
                    stock_adj[flow.from_stock].append(flow.to_stock)

        # Find chain starting points (stocks with no incoming flow from within subsystem)
        has_incoming: set[str] = set()
        for sources in stock_adj.values():
            has_incoming.update(sources)

        start_stocks = [s for s in stocks_in_subsystem if s not in has_incoming]
        if not start_stocks:
            # Cycle - pick alphabetically first for determinism
            start_stocks = [stocks_in_subsystem[0]]

        # Trace chains
        chains: list[list[str]] = []
        visited: set[str] = set()

        for start in start_stocks:
            if start in visited:
                continue
            chain = []
            current: Optional[str] = start
            while current and current not in visited:
                visited.add(current)
                chain.append(current)
                # Follow first outgoing flow (sorted for determinism)
                next_stocks = sorted(stock_adj.get(current, []))
                unvisited_next = [s for s in next_stocks if s not in visited]
                current = unvisited_next[0] if unvisited_next else None
            if chain:
                chains.append(chain)

        # Add any remaining stocks not in chains
        for stock in stocks_in_subsystem:
            if stock not in visited:
                chains.append([stock])

        return chains

    def _find_aux_target_position(
        self,
        aux_name: str,
        outgoing: dict[str, set[str]]
    ) -> Optional[tuple[float, float]]:
        """Find the position where an aux should be placed (near its targets).

        Returns average position of all targets, or None if no targets have positions.
        """
        target_positions: list[tuple[float, float]] = []

        # Find what this aux connects to via connectors
        for conn in self.connectors:
            if conn.from_var == aux_name:
                target = conn.to_var
                if target in self.stocks:
                    stock = self.stocks[target]
                    if stock.x is not None and stock.y is not None:
                        target_positions.append((stock.x, stock.y))
                elif target in self.flows:
                    flow = self.flows[target]
                    if flow.x is not None and flow.y is not None:
                        target_positions.append((flow.x, flow.y))
                elif target in self.auxs:
                    aux = self.auxs[target]
                    if aux.x is not None and aux.y is not None:
                        target_positions.append((aux.x, aux.y))

        if target_positions:
            avg_x = sum(p[0] for p in target_positions) / len(target_positions)
            avg_y = sum(p[1] for p in target_positions) / len(target_positions)
            return (avg_x, avg_y)
        return None

    def _position_subsystem(
        self,
        subsystem: set[str],
        outgoing: dict[str, set[str]],
        incoming: dict[str, set[str]],
        start_x: float,
        stock_y: float,
        aux_y: float,
        stock_spacing: float,
        aux_spacing: float
    ) -> tuple[float, float, float, float]:
        """Position all elements in a subsystem. Returns bounding box (min_x, min_y, max_x, max_y)."""

        # Find stock chains
        chains = self._find_stock_chains(subsystem)

        # Position stocks in chains horizontally
        x = start_x
        for chain in chains:
            for stock_name in chain:
                stock = self.stocks[stock_name]
                if stock.x is None or stock.y is None:
                    stock.x = x
                    stock.y = stock_y
                x += stock_spacing

        # Position flows between their stocks
        flows_in_subsystem = [f for f in self.flows if f in subsystem]
        for flow_name in flows_in_subsystem:
            flow = self.flows[flow_name]
            if flow.x is not None and flow.y is not None:
                continue

            from_stock = self.stocks.get(flow.from_stock) if flow.from_stock else None
            to_stock = self.stocks.get(flow.to_stock) if flow.to_stock else None

            if from_stock and to_stock:
                from_x = from_stock.x if from_stock.x is not None else start_x
                to_x = to_stock.x if to_stock.x is not None else start_x
                from_y = from_stock.y if from_stock.y is not None else stock_y
                to_y = to_stock.y if to_stock.y is not None else stock_y
                flow.x = (from_x + to_x) / 2
                flow.y = (from_y + to_y) / 2
            elif from_stock:
                flow.x = (from_stock.x or start_x) + 90
                flow.y = from_stock.y or stock_y
            elif to_stock:
                flow.x = (to_stock.x or start_x) - 90
                flow.y = to_stock.y or stock_y

        # Position auxs near their targets
        auxs_in_subsystem = [a for a in sorted(subsystem) if a in self.auxs]
        positioned_auxs: set[str] = set()
        aux_x_offset = 0  # For auxs without clear targets

        # First pass: position auxs that have clear targets
        for aux_name in auxs_in_subsystem:
            aux = self.auxs[aux_name]
            if aux.x is not None and aux.y is not None:
                positioned_auxs.add(aux_name)
                continue

            target_pos = self._find_aux_target_position(aux_name, outgoing)
            if target_pos:
                aux.x = target_pos[0]
                aux.y = aux_y  # Above the target's y level
                positioned_auxs.add(aux_name)

        # Second pass: position remaining auxs (those without connector targets)
        for aux_name in auxs_in_subsystem:
            if aux_name in positioned_auxs:
                continue

            aux = self.auxs[aux_name]
            if aux.x is not None and aux.y is not None:
                continue

            # Check if this aux is referenced by any flow (it's a parameter for that flow)
            for flow_name, flow in self.flows.items():
                if flow_name not in subsystem:
                    continue
                flow_refs = self._extract_variable_refs(flow.equation)
                if aux_name in flow_refs:
                    # Position near the flow
                    if flow.x is not None:
                        aux.x = flow.x + aux_x_offset
                        aux.y = aux_y
                        aux_x_offset += aux_spacing
                        positioned_auxs.add(aux_name)
                        break

        # Third pass: any remaining auxs go in a row at start_x
        remaining_x = start_x
        for aux_name in auxs_in_subsystem:
            if aux_name in positioned_auxs:
                continue

            aux = self.auxs[aux_name]
            if aux.x is None or aux.y is None:
                aux.x = remaining_x
                aux.y = aux_y - 60  # Put orphan auxs above the main aux row
                remaining_x += aux_spacing

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
        return (start_x, aux_y, start_x + stock_spacing, stock_y)

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
        inflows: Optional[list[str]] = None,
        outflows: Optional[list[str]] = None,
        non_negative: bool = True,
        x: Optional[float] = None,
        y: Optional[float] = None
    ) -> Stock:
        """Add a stock to the model."""
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
        from_stock: Optional[str] = None,
        to_stock: Optional[str] = None,
        non_negative: bool = True,
        x: Optional[float] = None,
        y: Optional[float] = None
    ) -> Flow:
        """Add a flow to the model."""
        flow = Flow(
            name=name,
            equation=equation,
            units=units,
            from_stock=self._normalize_name(from_stock) if from_stock else None,
            to_stock=self._normalize_name(to_stock) if to_stock else None,
            non_negative=non_negative,
            x=x,
            y=y
        )
        self.flows[self._normalize_name(name)] = flow

        # Update stock inflows/outflows
        if from_stock:
            from_key = self._normalize_name(from_stock)
            if from_key in self.stocks:
                flow_key = self._normalize_name(name)
                if flow_key not in self.stocks[from_key].outflows:
                    self.stocks[from_key].outflows.append(flow_key)

        if to_stock:
            to_key = self._normalize_name(to_stock)
            if to_key in self.stocks:
                flow_key = self._normalize_name(name)
                if flow_key not in self.stocks[to_key].inflows:
                    self.stocks[to_key].inflows.append(flow_key)

        return flow

    def add_aux(
        self,
        name: str,
        equation: str,
        units: str = "",
        x: Optional[float] = None,
        y: Optional[float] = None
    ) -> Aux:
        """Add an auxiliary variable to the model."""
        aux = Aux(name=name, equation=equation, units=units, x=x, y=y)
        self.auxs[self._normalize_name(name)] = aux
        return aux

    def add_connector(self, from_var: str, to_var: str) -> Connector:
        """Add a connector (dependency) between variables."""
        connector = Connector(
            uid=self._next_connector_uid(),
            from_var=self._normalize_name(from_var),
            to_var=self._normalize_name(to_var)
        )
        self.connectors.append(connector)
        return connector

    def _auto_layout(self):
        """Auto-arrange visual positions using graph-based hierarchical layout.

        Uses connector relationships to position elements:
        1. Builds dependency graph from connectors
        2. Detects subsystems (connected components)
        3. Positions stocks in chains following flow direction
        4. Positions auxs near their connector targets
        5. Separates independent subsystems visually

        Falls back to simple row layout if no connectors exist.
        Always recalculates flow.points to ensure flows connect to stocks correctly.
        """
        # Layout constants
        STOCK_SPACING = 200
        AUX_SPACING = 80
        STOCK_Y = 300
        AUX_Y = 150
        START_X = 200
        SUBSYSTEM_GAP = 250

        # Build dependency graph from connectors
        outgoing, incoming = self._build_dependency_graph()

        # Find subsystems (connected components)
        subsystems = self._find_subsystems(outgoing, incoming)

        # Position each subsystem
        subsystem_bounds: list[tuple[float, float, float, float]] = []

        for subsystem in subsystems:
            bounds = self._position_subsystem(
                subsystem, outgoing, incoming,
                START_X, STOCK_Y, AUX_Y, STOCK_SPACING, AUX_SPACING
            )
            subsystem_bounds.append(bounds)

        # Arrange subsystems relative to each other (largest centered, others offset)
        if len(subsystems) > 1 and len(subsystem_bounds) > 1:
            self._arrange_subsystems(subsystems, subsystem_bounds, SUBSYSTEM_GAP)

        # Always recalculate flow points to connect stocks at their actual positions
        self._recalculate_flow_points(START_X, STOCK_Y)

    def _recalculate_flow_points(self, start_x: float, stock_y: float):
        """Recalculate flow.points to connect stocks at their actual positions."""
        for name, flow in self.flows.items():
            from_stock = self.stocks.get(flow.from_stock) if flow.from_stock else None
            to_stock = self.stocks.get(flow.to_stock) if flow.to_stock else None

            if from_stock and to_stock:
                from_x = from_stock.x if from_stock.x is not None else start_x
                from_y = from_stock.y if from_stock.y is not None else stock_y
                to_x = to_stock.x if to_stock.x is not None else start_x
                to_y = to_stock.y if to_stock.y is not None else stock_y

                flow.points = [
                    (from_x + 22.5, from_y),
                    (to_x - 22.5, to_y)
                ]
            elif from_stock:
                from_x = from_stock.x if from_stock.x is not None else start_x
                from_y = from_stock.y if from_stock.y is not None else stock_y

                flow.points = [
                    (from_x + 22.5, from_y),
                    (from_x + 160, from_y)
                ]
            elif to_stock:
                to_x = to_stock.x if to_stock.x is not None else start_x
                to_y = to_stock.y if to_stock.y is not None else stock_y

                flow.points = [
                    (to_x - 160, to_y),
                    (to_x - 22.5, to_y)
                ]

    def to_xml(self) -> str:
        """Generate XMILE XML string for the model."""
        self._auto_layout()

        lines = []
        lines.append('<?xml version="1.0" encoding="utf-8"?>')
        lines.append(f'<xmile version="1.0" xmlns="{XMILE_NS}" xmlns:isee="{ISEE_NS}">')

        # Header
        lines.append('\t<header>')
        lines.append('\t\t<smile version="1.0" namespace="std, isee"/>')
        lines.append(f'\t\t<name>{escape(self.name)}</name>')
        lines.append(f'\t\t<uuid>{self.uuid}</uuid>')
        lines.append('\t\t<vendor>isee systems, inc.</vendor>')
        lines.append('\t\t<product version="1.9.3" isee:build_number="1954" isee:saved_by_v1="true" lang="en">Stella Professional</product>')
        lines.append('\t</header>')

        # Sim specs
        if self.sim_specs.dt < 1:
            dt_str = f'<dt reciprocal="true">{int(1/self.sim_specs.dt)}</dt>'
        else:
            dt_str = f'<dt>{self.sim_specs.dt}</dt>'
        lines.append(f'\t<sim_specs isee:sim_duration="1.5" isee:simulation_delay="0.0015" isee:restore_on_start="false" method="{self.sim_specs.method}" time_units="{self.sim_specs.time_units}" isee:instantaneous_flows="false">')
        lines.append(f'\t\t<start>{self.sim_specs.start}</start>')
        lines.append(f'\t\t<stop>{self.sim_specs.stop}</stop>')
        lines.append(f'\t\t{dt_str}')
        lines.append('\t</sim_specs>')

        # Preferences
        lines.append('\t<isee:prefs show_module_prefix="true" live_update_on_drag="true" show_restore_buttons="false" layer="model" interface_scale_ui="true" interface_max_page_width="10000" interface_max_page_height="10000" interface_min_page_width="0" interface_min_page_height="0" saved_runs="5" keep="false" rifp="true"/>')

        # Model
        lines.append('\t<model>')
        lines.append('\t\t<variables>')

        # Stocks
        for name, stock in self.stocks.items():
            display = escape(self._display_name(stock.name))
            lines.append(f'\t\t\t<stock name="{display}">')
            lines.append(f'\t\t\t\t<eqn>{escape(stock.initial_value)}</eqn>')
            for inflow in stock.inflows:
                lines.append(f'\t\t\t\t<inflow>{inflow}</inflow>')
            for outflow in stock.outflows:
                lines.append(f'\t\t\t\t<outflow>{outflow}</outflow>')
            if stock.non_negative:
                lines.append('\t\t\t\t<non_negative/>')
            if stock.units:
                lines.append(f'\t\t\t\t<units>{escape(stock.units)}</units>')
            lines.append('\t\t\t</stock>')

        # Flows
        for name, flow in self.flows.items():
            display = escape(self._display_name(flow.name))
            lines.append(f'\t\t\t<flow name="{display}">')
            lines.append(f'\t\t\t\t<eqn>{escape(flow.equation)}</eqn>')
            if flow.non_negative:
                lines.append('\t\t\t\t<non_negative/>')
            if flow.units:
                lines.append(f'\t\t\t\t<units>{escape(flow.units)}</units>')
            lines.append('\t\t\t</flow>')

        # Auxiliaries
        for name, aux in self.auxs.items():
            display = escape(self._display_name(aux.name))
            lines.append(f'\t\t\t<aux name="{display}">')
            lines.append(f'\t\t\t\t<eqn>{escape(aux.equation)}</eqn>')
            if aux.units:
                lines.append(f'\t\t\t\t<units>{escape(aux.units)}</units>')
            lines.append('\t\t\t</aux>')

        lines.append('\t\t</variables>')

        # Views
        lines.append('\t\t<views>')
        self._add_view_styles_str(lines)

        # Main view
        lines.append('\t\t\t<view isee:show_pages="false" background="white" page_width="768" page_height="596" isee:page_cols="2" isee:page_rows="2" isee:popup_graphs_are_comparative="true" type="stock_flow">')
        self._add_inner_view_styles_str(lines)

        # Stock visuals (positions guaranteed by _auto_layout)
        for name, stock in self.stocks.items():
            display = escape(self._display_name(stock.name))
            sx = int(stock.x) if stock.x is not None else 0
            sy = int(stock.y) if stock.y is not None else 0
            lines.append(f'\t\t\t\t<stock x="{sx}" y="{sy}" name="{display}"/>')

        # Flow visuals (positions guaranteed by _auto_layout)
        for name, flow in self.flows.items():
            display = escape(self._display_name(flow.name))
            fx = flow.x if flow.x is not None else 0
            fy = int(flow.y) if flow.y is not None else 0
            if flow.points:
                lines.append(f'\t\t\t\t<flow x="{fx}" y="{fy}" name="{display}">')
                lines.append('\t\t\t\t\t<pts>')
                for px, py in flow.points:
                    lines.append(f'\t\t\t\t\t\t<pt x="{px}" y="{py}"/>')
                lines.append('\t\t\t\t\t</pts>')
                lines.append('\t\t\t\t</flow>')
            else:
                lines.append(f'\t\t\t\t<flow x="{fx}" y="{fy}" name="{display}"/>')

        # Aux visuals (positions guaranteed by _auto_layout)
        for name, aux in self.auxs.items():
            display = escape(self._display_name(aux.name))
            ax = int(aux.x) if aux.x is not None else 0
            ay = int(aux.y) if aux.y is not None else 0
            lines.append(f'\t\t\t\t<aux x="{ax}" y="{ay}" name="{display}"/>')

        # Connector visuals
        for conn in self.connectors:
            lines.append(f'\t\t\t\t<connector uid="{conn.uid}" angle="{conn.angle}">')
            lines.append(f'\t\t\t\t\t<from>{conn.from_var}</from>')
            lines.append(f'\t\t\t\t\t<to>{conn.to_var}</to>')
            lines.append('\t\t\t\t</connector>')

        lines.append('\t\t\t</view>')
        lines.append('\t\t</views>')
        lines.append('\t</model>')
        lines.append('</xmile>')

        return '\n'.join(lines)

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
        lines.append('\t\t\t\t\t<connector color="#FF007F" background="white" font_color="#FF007F" font_size="9pt" isee:thickness="1"/>')
        lines.append('\t\t\t\t</style>')


def parse_stmx(filepath: str) -> StellaModel:
    """Parse an existing .stmx file and return a StellaModel."""
    tree = ET.parse(filepath)
    root = tree.getroot()

    # Handle namespaces with full Clark notation
    xmile = f"{{{XMILE_NS}}}"
    isee = f"{{{ISEE_NS}}}"

    def find_elem(parent, *tags):
        """Find element trying both namespaced and non-namespaced tags."""
        for tag in tags:
            # Try with XMILE namespace
            elem = parent.find(f".//{xmile}{tag}")
            if elem is not None:
                return elem
            # Try without namespace
            elem = parent.find(f".//{tag}")
            if elem is not None:
                return elem
        return None

    def find_child(parent, tag):
        """Find direct child element."""
        elem = parent.find(f"{xmile}{tag}")
        if elem is None:
            elem = parent.find(tag)
        return elem

    def findall_children(parent, tag):
        """Find all direct children with given tag."""
        elems = parent.findall(f"{xmile}{tag}")
        if not elems:
            elems = parent.findall(tag)
        return elems

    # Get model name
    header = find_child(root, "header")
    name_elem = find_child(header, "name") if header is not None else None
    model_name = name_elem.text if name_elem is not None else "Untitled"
    model = StellaModel(name=model_name)

    # Parse sim_specs
    sim_specs = find_child(root, "sim_specs")
    if sim_specs is not None:
        start = find_child(sim_specs, "start")
        if start is not None and start.text:
            model.sim_specs.start = float(start.text)

        stop = find_child(sim_specs, "stop")
        if stop is not None and stop.text:
            model.sim_specs.stop = float(stop.text)

        dt = find_child(sim_specs, "dt")
        if dt is not None and dt.text:
            if dt.get("reciprocal") == "true":
                model.sim_specs.dt = 1.0 / float(dt.text)
            else:
                model.sim_specs.dt = float(dt.text)

        method = sim_specs.get("method")
        if method:
            model.sim_specs.method = method

        time_units = sim_specs.get("time_units")
        if time_units:
            model.sim_specs.time_units = time_units

    # Find variables section
    model_elem = find_child(root, "model")
    variables = find_child(model_elem, "variables") if model_elem is not None else None

    if variables is not None:
        # Parse stocks
        for stock_elem in findall_children(variables, "stock"):
            name = stock_elem.get("name")
            eqn = find_child(stock_elem, "eqn")
            initial_value = eqn.text if eqn is not None else "0"

            units_elem = find_child(stock_elem, "units")
            units = units_elem.text if units_elem is not None else ""

            inflows = [inf.text for inf in findall_children(stock_elem, "inflow") if inf.text]
            outflows = [outf.text for outf in findall_children(stock_elem, "outflow") if outf.text]

            non_negative = find_child(stock_elem, "non_negative") is not None

            stock = Stock(
                name=name,
                initial_value=initial_value,
                units=units,
                inflows=inflows,
                outflows=outflows,
                non_negative=non_negative
            )
            model.stocks[model._normalize_name(name)] = stock

        # Parse flows
        for flow_elem in findall_children(variables, "flow"):
            name = flow_elem.get("name")
            eqn = find_child(flow_elem, "eqn")
            equation = eqn.text if eqn is not None else "0"

            units_elem = find_child(flow_elem, "units")
            units = units_elem.text if units_elem is not None else ""

            non_negative = find_child(flow_elem, "non_negative") is not None

            flow = Flow(
                name=name,
                equation=equation,
                units=units,
                non_negative=non_negative
            )
            model.flows[model._normalize_name(name)] = flow

        # Parse auxiliaries
        for aux_elem in findall_children(variables, "aux"):
            name = aux_elem.get("name")
            eqn = find_child(aux_elem, "eqn")
            equation = eqn.text if eqn is not None else "0"

            units_elem = find_child(aux_elem, "units")
            units = units_elem.text if units_elem is not None else ""

            aux = Aux(name=name, equation=equation, units=units)
            model.auxs[model._normalize_name(name)] = aux

    # Determine flow from/to stocks based on stock inflows/outflows
    for stock_name, stock in model.stocks.items():
        for inflow in stock.inflows:
            norm_inflow = model._normalize_name(inflow)
            if norm_inflow in model.flows:
                model.flows[norm_inflow].to_stock = stock_name
        for outflow in stock.outflows:
            norm_outflow = model._normalize_name(outflow)
            if norm_outflow in model.flows:
                model.flows[norm_outflow].from_stock = stock_name

    # Parse visual positions and connectors from views
    views = find_child(model_elem, "views") if model_elem is not None else None
    view = find_child(views, "view") if views is not None else None

    if view is not None:
        # Extract stock positions from view
        for stock_elem in findall_children(view, "stock"):
            name = stock_elem.get("name")
            x_attr = stock_elem.get("x")
            y_attr = stock_elem.get("y")
            if name:
                norm_name = model._normalize_name(name)
                if norm_name in model.stocks:
                    if x_attr is not None:
                        model.stocks[norm_name].x = float(x_attr)
                    if y_attr is not None:
                        model.stocks[norm_name].y = float(y_attr)

        # Extract flow positions from view
        for flow_elem in findall_children(view, "flow"):
            name = flow_elem.get("name")
            x_attr = flow_elem.get("x")
            y_attr = flow_elem.get("y")
            if name:
                norm_name = model._normalize_name(name)
                if norm_name in model.flows:
                    if x_attr is not None:
                        model.flows[norm_name].x = float(x_attr)
                    if y_attr is not None:
                        model.flows[norm_name].y = float(y_attr)

        # Extract aux positions from view
        for aux_elem in findall_children(view, "aux"):
            name = aux_elem.get("name")
            x_attr = aux_elem.get("x")
            y_attr = aux_elem.get("y")
            if name:
                norm_name = model._normalize_name(name)
                if norm_name in model.auxs:
                    if x_attr is not None:
                        model.auxs[norm_name].x = float(x_attr)
                    if y_attr is not None:
                        model.auxs[norm_name].y = float(y_attr)

        # Extract connectors
        for conn_elem in findall_children(view, "connector"):
            uid = int(conn_elem.get("uid", 0))
            angle = float(conn_elem.get("angle", 0))

            from_elem = find_child(conn_elem, "from")
            to_elem = find_child(conn_elem, "to")

            if from_elem is not None and to_elem is not None and from_elem.text and to_elem.text:
                connector = Connector(
                    uid=uid,
                    from_var=from_elem.text,
                    to_var=to_elem.text,
                    angle=angle
                )
                model.connectors.append(connector)
                model._connector_uid = max(model._connector_uid, uid)

    return model
