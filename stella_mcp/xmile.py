"""XMILE XML generation and parsing for Stella .stmx files."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional
import uuid
from html import escape


# XML namespaces
XMILE_NS = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"
ISEE_NS = "http://iseesystems.com/XMILE"


@dataclass
class Stock:
    """Represents a stock (reservoir) in the model."""
    name: str
    initial_value: str
    units: str = ""
    inflows: list[str] = field(default_factory=list)
    outflows: list[str] = field(default_factory=list)
    non_negative: bool = True
    x: float = 0
    y: float = 0


@dataclass
class Flow:
    """Represents a flow between stocks."""
    name: str
    equation: str
    units: str = ""
    from_stock: Optional[str] = None  # None means external source
    to_stock: Optional[str] = None    # None means external sink
    non_negative: bool = True
    x: float = 0
    y: float = 0
    points: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class Aux:
    """Represents an auxiliary variable."""
    name: str
    equation: str
    units: str = ""
    x: float = 0
    y: float = 0


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

    def add_stock(
        self,
        name: str,
        initial_value: str,
        units: str = "",
        inflows: Optional[list[str]] = None,
        outflows: Optional[list[str]] = None,
        non_negative: bool = True
    ) -> Stock:
        """Add a stock to the model."""
        stock = Stock(
            name=name,
            initial_value=initial_value,
            units=units,
            inflows=[self._normalize_name(f) for f in (inflows or [])],
            outflows=[self._normalize_name(f) for f in (outflows or [])],
            non_negative=non_negative
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
        non_negative: bool = True
    ) -> Flow:
        """Add a flow to the model."""
        flow = Flow(
            name=name,
            equation=equation,
            units=units,
            from_stock=self._normalize_name(from_stock) if from_stock else None,
            to_stock=self._normalize_name(to_stock) if to_stock else None,
            non_negative=non_negative
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

    def add_aux(self, name: str, equation: str, units: str = "") -> Aux:
        """Add an auxiliary variable to the model."""
        aux = Aux(name=name, equation=equation, units=units)
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
        """Auto-arrange visual positions for model elements."""
        # Layout constants
        stock_spacing = 200
        aux_spacing = 80
        stock_y = 300
        aux_y = 150
        start_x = 200

        # Position stocks in a row
        x = start_x
        for name, stock in self.stocks.items():
            stock.x = x
            stock.y = stock_y
            x += stock_spacing

        # Position flows between their stocks
        for name, flow in self.flows.items():
            from_stock = self.stocks.get(flow.from_stock)
            to_stock = self.stocks.get(flow.to_stock)

            if from_stock and to_stock:
                # Flow between two stocks
                flow.x = (from_stock.x + to_stock.x) / 2
                flow.y = (from_stock.y + to_stock.y) / 2
                # Create flow points
                flow.points = [
                    (from_stock.x + 22.5, from_stock.y),  # Exit from stock
                    (to_stock.x - 22.5, to_stock.y)       # Enter to stock
                ]
            elif from_stock:
                # Flow from stock to external sink
                flow.x = from_stock.x + 90
                flow.y = from_stock.y
                flow.points = [
                    (from_stock.x + 22.5, from_stock.y),
                    (from_stock.x + 160, from_stock.y)
                ]
            elif to_stock:
                # Flow from external source to stock
                flow.x = to_stock.x - 90
                flow.y = to_stock.y
                flow.points = [
                    (to_stock.x - 160, to_stock.y),
                    (to_stock.x - 22.5, to_stock.y)
                ]
            else:
                # Orphan flow (shouldn't happen normally)
                flow.x = start_x
                flow.y = stock_y

        # Position auxiliary variables above stocks
        x = start_x
        for name, aux in self.auxs.items():
            aux.x = x
            aux.y = aux_y
            x += aux_spacing

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

        # Stock visuals
        for name, stock in self.stocks.items():
            display = escape(self._display_name(stock.name))
            lines.append(f'\t\t\t\t<stock x="{int(stock.x)}" y="{int(stock.y)}" name="{display}"/>')

        # Flow visuals
        for name, flow in self.flows.items():
            display = escape(self._display_name(flow.name))
            if flow.points:
                lines.append(f'\t\t\t\t<flow x="{flow.x}" y="{int(flow.y)}" name="{display}">')
                lines.append('\t\t\t\t\t<pts>')
                for px, py in flow.points:
                    lines.append(f'\t\t\t\t\t\t<pt x="{px}" y="{py}"/>')
                lines.append('\t\t\t\t\t</pts>')
                lines.append('\t\t\t\t</flow>')
            else:
                lines.append(f'\t\t\t\t<flow x="{flow.x}" y="{int(flow.y)}" name="{display}"/>')

        # Aux visuals
        for name, aux in self.auxs.items():
            display = escape(self._display_name(aux.name))
            lines.append(f'\t\t\t\t<aux x="{int(aux.x)}" y="{int(aux.y)}" name="{display}"/>')

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

    # Parse connectors from views (optional)
    views = find_child(model_elem, "views") if model_elem is not None else None
    view = find_child(views, "view") if views is not None else None

    if view is not None:
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
