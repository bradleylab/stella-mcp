"""Stella model state, lifecycle operations, and compatibility metadata."""

import math
import re
import uuid

from stella_mcp import layout_pipeline, model_layout, xmile_export
from stella_mcp.equation_parser import extract_variable_references
from stella_mcp.layout import BoundingBox
from stella_mcp.model_types import (
    DEFAULT_VIEW_FONT_POINTS,
    DEFAULT_VIEW_PAGE_COLUMNS,
    DEFAULT_VIEW_PAGE_HEIGHT,
    DEFAULT_VIEW_PAGE_ROWS,
    DEFAULT_VIEW_PAGE_WIDTH,
    Aux,
    Connector,
    Flow,
    GraphicalFunction,
    Module,
    SimSpecs,
    Stock,
)


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
        self.last_layout_result = None
        self.last_layout_metrics = None
        self.layout_warnings = []
        self.header_extra_children_xml: list[str] = []
        self.model_extra_children_xml: list[str] = []
        self.views_extra_children_xml: list[str] = []
        self.view_extra_children_xml: list[str] = []
        self.view_extra_attrs: dict[str, str] = {}
        self.view_page_width = DEFAULT_VIEW_PAGE_WIDTH
        self.view_page_height = DEFAULT_VIEW_PAGE_HEIGHT
        self.view_page_columns = DEFAULT_VIEW_PAGE_COLUMNS
        self.view_page_rows = DEFAULT_VIEW_PAGE_ROWS
        self.view_stock_font_points = DEFAULT_VIEW_FONT_POINTS
        self.view_flow_font_points = DEFAULT_VIEW_FONT_POINTS
        self.view_aux_font_points = DEFAULT_VIEW_FONT_POINTS
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
        return xmile_export._xml_local_name(tag)

    @staticmethod
    def _xml_attr_parts(attr_key: str) -> tuple[str | None, str]:
        """Split ElementTree attr key into (namespace_uri, local_name)."""
        return xmile_export._xml_attr_parts(attr_key)

    def _xml_attr_name(self, attr_key: str) -> str:
        """Convert ElementTree attribute key to output-safe name."""
        return xmile_export._xml_attr_name(self, attr_key)

    def _iter_all_extra_attrs(self):
        """Iterate over all preserved extra-attribute dictionaries."""
        return xmile_export._iter_all_extra_attrs(self)

    def _build_export_ns_prefixes(self) -> dict[str, str]:
        """Build deterministic XML namespace prefixes for unknown attr namespaces."""
        return xmile_export._build_export_ns_prefixes(self)

    def _format_extra_attrs(
        self,
        attrs: dict[str, str],
        reserved_names: set[str] | None = None,
    ) -> str:
        """Format preserved extra XML attrs while avoiding known fields."""
        return xmile_export._format_extra_attrs(self, attrs, reserved_names)

    def _append_xml_fragment(self, lines: list[str], fragment: str, indent: str):
        """Append a preserved XML fragment with target indentation."""
        return xmile_export._append_xml_fragment(self, lines, fragment, indent)

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
        return xmile_export._format_number(value)

    def _dt_xml(self, dt: float | None = None) -> str:
        """Format dt for XMILE with compatibility-safe reciprocal usage.

        Stella commonly uses reciprocal dt when dt is an exact inverse integer
        (e.g., 0.25 -> reciprocal 4). For non-exact values, writing reciprocal
        with truncation can change dt on round-trip, so export plain dt instead.
        """
        return xmile_export._dt_xml(self, dt)

    def _build_dependency_graph(self) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        """Build bidirectional adjacency lists from connectors and flow-stock relationships.

        Returns:
            (outgoing, incoming) where:
            - outgoing[node] = set of nodes this node connects TO
            - incoming[node] = set of nodes that connect TO this node
        """
        return model_layout._build_dependency_graph(self)

    def _find_subsystems(
        self, outgoing: dict[str, set[str]], incoming: dict[str, set[str]]
    ) -> list[set[str]]:
        """Find connected components (subsystems) in the graph.

        Returns list of node sets, sorted by size (largest first).
        """
        return model_layout._find_subsystems(self, outgoing, incoming)

    def _position_subsystem(
        self,
        subsystem: set[str],
        outgoing: dict[str, set[str]],
        incoming: dict[str, set[str]],
    ) -> tuple[float, float, float, float]:
        """Position all elements in a subsystem using force-directed layout.

        Returns bounding box (min_x, min_y, max_x, max_y).
        """
        return model_layout._position_subsystem(self, subsystem, outgoing, incoming)

    def _arrange_subsystems(
        self,
        subsystems: list[set[str]],
        bounds: list[tuple[float, float, float, float]],
        gap: float,
    ):
        """Arrange subsystems: largest stays in place, smaller ones offset to the right."""
        return model_layout._arrange_subsystems(self, subsystems, bounds, gap)

    def _snap_auto_geometry(self):
        """Snap generated positions and unlocked routes to whole pixels."""
        return model_layout._snap_auto_geometry(self)

    def add_stock(
        self,
        name: str,
        initial_value: str,
        units: str = "",
        inflows: list[str] | None = None,
        outflows: list[str] | None = None,
        non_negative: bool = True,
        x: float | None = None,
        y: float | None = None,
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
            y=y,
            position_source="user" if x is not None or y is not None else "auto",
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
        graphical_function: GraphicalFunction | None = None,
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
            position_source="user" if x is not None or y is not None else "auto",
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
        graphical_function: GraphicalFunction | None = None,
    ) -> Aux:
        """Add an auxiliary variable to the model."""
        self._validate_new_variable_name(name)
        aux = Aux(
            name=name,
            equation=equation,
            units=units,
            x=x,
            y=y,
            position_source="user" if x is not None or y is not None else "auto",
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
        connector = Connector(uid=self._next_connector_uid(), from_var=norm_from, to_var=norm_to)
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
            conn
            for conn in self.connectors
            if conn.from_var == norm_from and conn.to_var == norm_to
        ]
        if not matches:
            raise ValueError(f"No connector found from '{from_var}' to '{to_var}'")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple connectors found from '{from_var}' to '{to_var}'; specify connector_uid"
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
            raise ValueError(
                "Provide at least one of angle, angle_locked, points, or points_locked"
            )

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
                    raise ValueError(f"connector points[{index}] must contain finite coordinates")
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
            stock.inflows = self._dedupe_preserve_order(
                [new_norm if flow_name == old_norm else flow_name for flow_name in stock.inflows]
            )
            stock.outflows = self._dedupe_preserve_order(
                [new_norm if flow_name == old_norm else flow_name for flow_name in stock.outflows]
            )
            stock.initial_value = self._replace_equation_identifier(
                stock.initial_value, old_norm, new_norm
            )

        for aux in self.auxs.values():
            aux.equation = self._replace_equation_identifier(aux.equation, old_norm, new_norm)

        for connector in self.connectors:
            if connector.from_var == old_norm:
                connector.from_var = new_norm
            if connector.to_var == old_norm:
                connector.to_var = new_norm

        for module in self.modules.values():
            module.members = self._dedupe_preserve_order(
                [new_norm if member == old_norm else member for member in module.members]
            )

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
            connected_flows = sorted(
                {
                    *self.stocks[norm_name].inflows,
                    *self.stocks[norm_name].outflows,
                }
            )
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
            conn
            for conn in self.connectors
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
        if x is not None or y is not None:
            stock.position_source = "user"
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
        if x is not None or y is not None:
            flow.position_source = "user"
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
        if x is not None or y is not None:
            aux.position_source = "user"
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
                raise ValueError(
                    f"Module member '{member}' is not a known stock, flow, or auxiliary"
                )
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
                raise ValueError(
                    f"Module member '{member}' is not a known stock, flow, or auxiliary"
                )
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
            value is None for value in (border_color, background, font_color, font_size, label_side)
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
        return model_layout._member_bounds(self, member)

    def auto_place_module_boxes(
        self,
        padding: float = 40.0,
        min_width: float = 180.0,
        min_height: float = 120.0,
        only_missing: bool = False,
    ):
        """Auto-place module boxes around member elements."""
        return model_layout.auto_place_module_boxes(
            self, padding, min_width, min_height, only_missing
        )

    def _calculate_stock_sizes(self):
        """Calculate appropriate width/height for each stock based on connectivity.

        Stocks with more flows get larger to allow visual separation of flow attachments.
        Maintains a pleasing aspect ratio (roughly 1.3:1 width:height).
        """
        return model_layout._calculate_stock_sizes(self)

    def _auto_layout(self):
        """Run deterministic directed placement, routing, and validation."""
        return layout_pipeline.run_layout_pipeline(self)

    def _position_orphan_flows(self):
        """Place flows with no source or destination stock at a fallback spot.

        A flow with neither ``from_stock`` nor ``to_stock`` is never anchored
        by ``_position_subsystem``, so it would otherwise keep ``x/y == None``
        and block rendering and XMILE export. Lay such flows out in a row
        beneath the positioned elements with a short two-point segment; the
        renderer then draws source/sink clouds at both ends.
        """
        return model_layout._position_orphan_flows(self)

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
        return model_layout._calculate_flow_offset(self, index, total)

    @staticmethod
    def _stock_attachment_point(
        stock_x: float,
        stock_y: float,
        half_w: float,
        half_h: float,
        target_x: float,
        target_y: float,
    ) -> tuple[float, float]:
        """Find the point on a stock's edge closest to a target point.

        Exits from the edge that faces the target (direction-aware).
        """
        return model_layout._stock_attachment_point(
            stock_x, stock_y, half_w, half_h, target_x, target_y
        )

    def _recalculate_flow_points(self, *, only_missing: bool = False):
        """Recalculate flow.points to connect stocks at their actual positions.

        Direction-aware: exits/enters from the stock edge closest to the
        destination, supporting stocks at arbitrary angles (not just horizontal).
        Uses orthogonal routing for multiple flows from the same stock.
        """
        return model_layout._recalculate_flow_points(self, only_missing=only_missing)

    def _calculate_connector_angles(self, force: bool = False):
        """Calculate connector angles based on source and target positions.

        Uses atan2 to compute the angle from source to target.
        Convention: degrees, 0 = right, counter-clockwise positive.
        Note: -dy because screen y-coordinates increase downward.
        """
        return model_layout._calculate_connector_angles(self, force)

    # =========================================================================
    # Layout Collision/Crossing Detection and Resolution
    # =========================================================================

    def _get_element_box(self, name: str) -> BoundingBox | None:
        """Get bounding box for any model element."""
        return model_layout._get_element_box(self, name)

    def _get_all_bounding_boxes(self) -> dict[str, BoundingBox]:
        """Get bounding boxes for all positioned elements."""
        return model_layout._get_all_bounding_boxes(self)

    def _get_connector_segments(self) -> dict[int, tuple[tuple[float, float], tuple[float, float]]]:
        """Get line segments for all connectors (from source to target position)."""
        return model_layout._get_connector_segments(self)

    def _get_flow_segments(
        self,
    ) -> dict[str, list[tuple[tuple[float, float], tuple[float, float]]]]:
        """Get line segments for all flow paths."""
        return model_layout._get_flow_segments(self)

    def _detect_aux_collisions(self) -> list[tuple[str, str]]:
        """Detect pairs of auxs that overlap."""
        return model_layout._detect_aux_collisions(self)

    def _detect_connector_flow_crossings(self) -> list[tuple[int, str]]:
        """Detect connectors that cross flow lines. Returns (connector_uid, flow_name) pairs.

        Note: A connector is expected to touch its target flow, so we skip checking
        if a connector crosses the flow it's connected TO.
        """
        return model_layout._detect_connector_flow_crossings(self)

    def _detect_flow_stock_crossings(self) -> list[tuple[str, str]]:
        """Detect flows that pass through stocks (not their source/dest). Returns (flow_name, stock_name) pairs."""
        return model_layout._detect_flow_stock_crossings(self)

    def _detect_connector_stock_crossings(self) -> list[tuple[int, str]]:
        """Detect connectors that pass through stocks. Returns (connector_uid, stock_name) pairs.

        Skips stocks that are the source or target of the connector.
        """
        return model_layout._detect_connector_stock_crossings(self)

    def _separate_auxs(self, name1: str, name2: str):
        """Push two overlapping auxs apart."""
        return model_layout._separate_auxs(self, name1, name2)

    def _reposition_aux_to_avoid_crossing(
        self, conn_uid: int, obstacle_name: str, obstacle_type: str = "flow"
    ):
        """Move aux so its connector doesn't cross the specified obstacle.

        Args:
            conn_uid: The connector's unique ID
            obstacle_name: Name of the flow or stock to avoid
            obstacle_type: Either "flow" or "stock"
        """
        return model_layout._reposition_aux_to_avoid_crossing(
            self, conn_uid, obstacle_name, obstacle_type
        )

    def _reroute_flow_around_stock(self, flow_name: str, stock_name: str):
        """Add waypoints to route flow around a stock it currently crosses.

        This is a best-effort fix - if the flow has already been modified multiple
        times (>8 points), we skip to avoid infinite loops.
        """
        return model_layout._reroute_flow_around_stock(self, flow_name, stock_name)

    def _resolve_layout_violations(self, max_iterations: int = 10):
        """Iteratively resolve collisions and crossings in the layout."""
        return model_layout._resolve_layout_violations(self, max_iterations)

    def to_xml(
        self,
        auto_layout: bool = True,
        resolve_layout_violations: bool = False,
        compat_mode: str = "permissive",
    ) -> str:
        """Generate XMILE XML string for the model."""
        return xmile_export.model_to_xml(
            self,
            auto_layout=auto_layout,
            resolve_layout_violations=resolve_layout_violations,
            compat_mode=compat_mode,
        )

    def _add_view_styles_str(self, lines: list[str]):
        """Add the default view styles as strings."""
        return xmile_export._add_view_styles_str(self, lines)

    def _add_inner_view_styles_str(self, lines: list[str]):
        """Add the inner view styles as strings."""
        return xmile_export._add_inner_view_styles_str(self, lines)

    def _format_point_list(self, points: list[float]) -> str:
        # XMILE defines point lists as comma-separated (the sep attribute can
        # override, but readers like Stella and PySD assume the spec default).
        return xmile_export._format_point_list(self, points)

    def _add_graphical_function_str(self, lines: list[str], gf: GraphicalFunction):
        return xmile_export._add_graphical_function_str(self, lines, gf)
