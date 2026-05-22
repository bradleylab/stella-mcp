"""XMILE parse/export IO helpers split from xmile.py for maintainability."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from html import escape

from .xmile import (
    ISEE_NS,
    XMILE_NS,
    Aux,
    Connector,
    Flow,
    GraphicalFunction,
    Module,
    StellaModel,
    Stock,
)


def model_to_xml(
    model,
    auto_layout: bool = True,
    resolve_layout_violations: bool = False,
    compat_mode: str = "permissive",
) -> str:
    """Generate XMILE XML string for the model.

    Args:
        auto_layout: If True, run automatic layout before export.
        resolve_layout_violations: If True, run post-processing to reduce
            overlaps/crossings before export.
        compat_mode: "permissive" emits best-effort XML and records warnings;
            "strict" raises on compatibility issues.
    """
    mode = model._validate_compat_mode(compat_mode)
    export_warnings: list[str] = []

    def compat_issue(message: str):
        if mode == "strict":
            raise ValueError(message)
        export_warnings.append(message)

    if model.sim_specs.dt <= 0:
        compat_issue(
            f"sim_specs.dt={model.sim_specs.dt} is invalid; exporting with default dt=0.25"
        )

    for flow in model.flows.values():
        if flow.from_stock is not None and flow.from_stock not in model.stocks:
            compat_issue(
                f"Flow '{flow.name}' references missing from_stock '{flow.from_stock}'"
            )
        if flow.to_stock is not None and flow.to_stock not in model.stocks:
            compat_issue(
                f"Flow '{flow.name}' references missing to_stock '{flow.to_stock}'"
            )

    for stock in model.stocks.values():
        for inflow in stock.inflows:
            if inflow not in model.flows:
                compat_issue(
                    f"Stock '{stock.name}' references missing inflow '{inflow}'"
                )
        for outflow in stock.outflows:
            if outflow not in model.flows:
                compat_issue(
                    f"Stock '{stock.name}' references missing outflow '{outflow}'"
                )

    for connector in model.connectors:
        if connector.from_var not in model.stocks and connector.from_var not in model.flows and connector.from_var not in model.auxs:
            compat_issue(
                f"Connector uid={connector.uid} source '{connector.from_var}' is missing"
            )
        if connector.to_var not in model.stocks and connector.to_var not in model.flows and connector.to_var not in model.auxs:
            compat_issue(
                f"Connector uid={connector.uid} target '{connector.to_var}' is missing"
            )

    for module in model.modules.values():
        for member in module.members:
            if member not in model.stocks and member not in model.flows and member not in model.auxs:
                compat_issue(
                    f"Module '{module.name}' references missing member '{member}'"
                )

    model.last_export_warnings = export_warnings

    if auto_layout:
        model._auto_layout()
        if model.modules:
            model.auto_place_module_boxes(only_missing=True)
    else:
        # Even with fixed/manual positions, derive dependent visual metadata:
        # flow paths (when unlocked) and connector angles.
        model._recalculate_flow_points()
        model._calculate_connector_angles()
    if resolve_layout_violations:
        model._resolve_layout_violations()

    model._export_ns_prefix_by_uri = model._build_export_ns_prefixes()
    extra_ns_decls = " ".join(
        f'xmlns:{prefix}="{uri}"'
        for uri, prefix in sorted(model._export_ns_prefix_by_uri.items())
    )
    ns_suffix = f" {extra_ns_decls}" if extra_ns_decls else ""

    lines = []
    lines.append('<?xml version="1.0" encoding="utf-8"?>')
    lines.append(
        f'<xmile version="1.0" xmlns="{XMILE_NS}" xmlns:isee="{ISEE_NS}"{ns_suffix}>'
    )

    # Header
    lines.append('\t<header>')
    lines.append('\t\t<smile version="1.0" namespace="std, isee"/>')
    lines.append(f'\t\t<name>{escape(model.name)}</name>')
    lines.append(f'\t\t<uuid>{model.uuid}</uuid>')
    lines.append('\t\t<vendor>isee systems, inc.</vendor>')
    lines.append('\t\t<product version="1.9.3" isee:build_number="1954" isee:saved_by_v1="true" lang="en">Stella Professional</product>')
    for fragment in model.header_extra_children_xml:
        model._append_xml_fragment(lines, fragment, '\t\t')
    lines.append('\t</header>')

    # Sim specs
    export_dt = model.sim_specs.dt if model.sim_specs.dt > 0 else 0.25
    dt_str = model._dt_xml(export_dt)
    sim_attr_extra = model._format_extra_attrs(
        model.sim_specs.extra_attrs,
        reserved_names={
            "isee:sim_duration",
            "isee:simulation_delay",
            "isee:restore_on_start",
            "method",
            "time_units",
            "isee:instantaneous_flows",
        },
    )
    lines.append(
        f'\t<sim_specs isee:sim_duration="1.5" isee:simulation_delay="0.0015" '
        f'isee:restore_on_start="false" method="{escape(model.sim_specs.method)}" '
        f'time_units="{escape(model.sim_specs.time_units)}" isee:instantaneous_flows="false"{sim_attr_extra}>'
    )
    lines.append(f'\t\t<start>{model._format_number(model.sim_specs.start)}</start>')
    lines.append(f'\t\t<stop>{model._format_number(model.sim_specs.stop)}</stop>')
    lines.append(f'\t\t{dt_str}')
    for fragment in model.sim_specs.extra_children_xml:
        model._append_xml_fragment(lines, fragment, '\t\t')
    lines.append('\t</sim_specs>')

    # Preferences
    if model.prefs_xml:
        model._append_xml_fragment(lines, model.prefs_xml, '\t')
    else:
        lines.append('\t<isee:prefs show_module_prefix="true" live_update_on_drag="true" show_restore_buttons="false" layer="model" interface_scale_ui="true" interface_max_page_width="10000" interface_max_page_height="10000" interface_min_page_width="0" interface_min_page_height="0" saved_runs="5" keep="false" rifp="true"/>')

    # Model
    lines.append('\t<model>')
    lines.append('\t\t<variables>')

    # Stocks
    for name in sorted(model.stocks):
        stock = model.stocks[name]
        display = escape(model._display_name(stock.name))
        stock_extra_attrs = model._format_extra_attrs(
            stock.extra_attrs,
            reserved_names={"name"},
        )
        lines.append(f'\t\t\t<stock name="{display}"{stock_extra_attrs}>')
        lines.append(f'\t\t\t\t<eqn>{escape(stock.initial_value)}</eqn>')
        for inflow in stock.inflows:
            lines.append(f'\t\t\t\t<inflow>{escape(inflow)}</inflow>')
        for outflow in stock.outflows:
            lines.append(f'\t\t\t\t<outflow>{escape(outflow)}</outflow>')
        if stock.non_negative:
            lines.append('\t\t\t\t<non_negative/>')
        if stock.units:
            lines.append(f'\t\t\t\t<units>{escape(stock.units)}</units>')
        for fragment in stock.extra_children_xml:
            model._append_xml_fragment(lines, fragment, '\t\t\t\t')
        lines.append('\t\t\t</stock>')

    # Flows
    for name in sorted(model.flows):
        flow = model.flows[name]
        display = escape(model._display_name(flow.name))
        flow_extra_attrs = model._format_extra_attrs(
            flow.extra_attrs,
            reserved_names={"name"},
        )
        lines.append(f'\t\t\t<flow name="{display}"{flow_extra_attrs}>')
        lines.append(f'\t\t\t\t<eqn>{escape(flow.equation)}</eqn>')
        if flow.graphical_function is not None:
            model._add_graphical_function_str(lines, flow.graphical_function)
        if flow.non_negative:
            lines.append('\t\t\t\t<non_negative/>')
        if flow.units:
            lines.append(f'\t\t\t\t<units>{escape(flow.units)}</units>')
        for fragment in flow.extra_children_xml:
            model._append_xml_fragment(lines, fragment, '\t\t\t\t')
        lines.append('\t\t\t</flow>')

    # Auxiliaries
    for name in sorted(model.auxs):
        aux = model.auxs[name]
        display = escape(model._display_name(aux.name))
        aux_extra_attrs = model._format_extra_attrs(
            aux.extra_attrs,
            reserved_names={"name"},
        )
        lines.append(f'\t\t\t<aux name="{display}"{aux_extra_attrs}>')
        lines.append(f'\t\t\t\t<eqn>{escape(aux.equation)}</eqn>')
        if aux.graphical_function is not None:
            model._add_graphical_function_str(lines, aux.graphical_function)
        if aux.units:
            lines.append(f'\t\t\t\t<units>{escape(aux.units)}</units>')
        for fragment in aux.extra_children_xml:
            model._append_xml_fragment(lines, fragment, '\t\t\t\t')
        lines.append('\t\t\t</aux>')

    # Modules/groups
    for name in sorted(model.modules):
        module = model.modules[name]
        display = escape(model._display_name(module.name))
        module_extra_attrs = model._format_extra_attrs(
            module.extra_attrs,
            reserved_names={"name"},
        )
        lines.append(f'\t\t\t<group name="{display}"{module_extra_attrs}>')
        for member in sorted(module.members):
            member_display = escape(model._display_name(member))
            lines.append(f'\t\t\t\t<entity name="{member_display}"/>')
        for fragment in module.extra_children_xml:
            model._append_xml_fragment(lines, fragment, '\t\t\t\t')
        lines.append('\t\t\t</group>')

    lines.append('\t\t</variables>')

    # Views
    lines.append('\t\t<views>')
    if model.views_style_xml:
        model._append_xml_fragment(lines, model.views_style_xml, '\t\t\t')
    else:
        model._add_view_styles_str(lines)

    # Main view
    view_extra_attrs = model._format_extra_attrs(
        model.view_extra_attrs,
        reserved_names={
            "isee:show_pages",
            "background",
            "page_width",
            "page_height",
            "isee:page_cols",
            "isee:page_rows",
            "isee:popup_graphs_are_comparative",
            "type",
        },
    )
    lines.append(
        '\t\t\t<view isee:show_pages="false" background="white" page_width="768" '
        'page_height="596" isee:page_cols="2" isee:page_rows="2" '
        f'isee:popup_graphs_are_comparative="true" type="stock_flow"{view_extra_attrs}>'
    )
    if model.inner_view_style_xml:
        model._append_xml_fragment(lines, model.inner_view_style_xml, '\t\t\t\t')
    else:
        model._add_inner_view_styles_str(lines)

    # Module/group visuals (if view geometry is set)
    for module_name in sorted(model.modules):
        module = model.modules[module_name]
        has_geometry = None not in (module.x, module.y, module.width, module.height)
        has_style = any(
            value is not None
            for value in (
                module.border_color,
                module.background,
                module.font_color,
                module.font_size,
                module.label_side,
            )
        )
        has_view_extras = bool(module.view_extra_attrs or module.view_extra_children_xml)
        if not has_geometry and not has_style and not has_view_extras:
            continue
        display = escape(model._display_name(module.name))
        attrs = [f'name="{display}"']
        if module.x is not None:
            attrs.append(f'x="{int(module.x)}"')
        if module.y is not None:
            attrs.append(f'y="{int(module.y)}"')
        if module.width is not None:
            attrs.append(f'width="{int(module.width)}"')
        if module.height is not None:
            attrs.append(f'height="{int(module.height)}"')
        if module.border_color is not None:
            attrs.append(f'color="{escape(module.border_color)}"')
        if module.background is not None:
            attrs.append(f'background="{escape(module.background)}"')
        if module.font_color is not None:
            attrs.append(f'font_color="{escape(module.font_color)}"')
        if module.font_size is not None:
            attrs.append(f'font_size="{escape(module.font_size)}"')
        if module.label_side is not None:
            attrs.append(f'label_side="{escape(module.label_side)}"')
        module_view_extra_attrs = model._format_extra_attrs(
            module.view_extra_attrs,
            reserved_names={"x", "y", "width", "height", "name", "color", "background", "font_color", "font_size", "label_side"},
        )
        if module.view_extra_children_xml:
            lines.append(f'\t\t\t\t<group {" ".join(attrs)}{module_view_extra_attrs}>')
            for fragment in module.view_extra_children_xml:
                model._append_xml_fragment(lines, fragment, '\t\t\t\t\t')
            lines.append('\t\t\t\t</group>')
        else:
            lines.append(f'\t\t\t\t<group {" ".join(attrs)}{module_view_extra_attrs}/>')  # self-closing

    # Stock visuals (positions guaranteed by _auto_layout)
    for name in sorted(model.stocks):
        stock = model.stocks[name]
        display = escape(model._display_name(stock.name))
        sx = int(stock.x) if stock.x is not None else 0
        sy = int(stock.y) if stock.y is not None else 0
        stock_view_extra_attrs = model._format_extra_attrs(
            stock.view_extra_attrs,
            reserved_names={"x", "y", "width", "height", "name"},
        )
        lines.append(f'\t\t\t\t<stock x="{sx}" y="{sy}" width="{stock.width}" height="{stock.height}" name="{display}"{stock_view_extra_attrs}/>')

    # Flow visuals (positions guaranteed by _auto_layout)
    for name in sorted(model.flows):
        flow = model.flows[name]
        display = escape(model._display_name(flow.name))
        fx = flow.x if flow.x is not None else 0
        fy = int(flow.y) if flow.y is not None else 0
        flow_view_extra_attrs = model._format_extra_attrs(
            flow.view_extra_attrs,
            reserved_names={"x", "y", "name"},
        )
        if flow.points or flow.view_extra_children_xml:
            lines.append(f'\t\t\t\t<flow x="{fx}" y="{fy}" name="{display}"{flow_view_extra_attrs}>')
            if flow.points:
                lines.append('\t\t\t\t\t<pts>')
                for px, py in flow.points:
                    lines.append(f'\t\t\t\t\t\t<pt x="{px}" y="{py}"/>')
                lines.append('\t\t\t\t\t</pts>')
            for fragment in flow.view_extra_children_xml:
                model._append_xml_fragment(lines, fragment, '\t\t\t\t\t')
            lines.append('\t\t\t\t</flow>')
        else:
            lines.append(f'\t\t\t\t<flow x="{fx}" y="{fy}" name="{display}"{flow_view_extra_attrs}/>')

    # Aux visuals (positions guaranteed by _auto_layout)
    for name in sorted(model.auxs):
        aux = model.auxs[name]
        display = escape(model._display_name(aux.name))
        ax = int(aux.x) if aux.x is not None else 0
        ay = int(aux.y) if aux.y is not None else 0
        aux_view_extra_attrs = model._format_extra_attrs(
            aux.view_extra_attrs,
            reserved_names={"x", "y", "name"},
        )
        lines.append(f'\t\t\t\t<aux x="{ax}" y="{ay}" name="{display}"{aux_view_extra_attrs}/>')

    # Connector visuals
    for conn in sorted(model.connectors, key=lambda c: c.uid):
        conn_extra_attrs = model._format_extra_attrs(
            conn.extra_attrs,
            reserved_names={"uid", "angle"},
        )
        lines.append(f'\t\t\t\t<connector uid="{conn.uid}" angle="{conn.angle}"{conn_extra_attrs}>')
        lines.append(f'\t\t\t\t\t<from>{escape(conn.from_var)}</from>')
        lines.append(f'\t\t\t\t\t<to>{escape(conn.to_var)}</to>')
        if conn.points:
            lines.append('\t\t\t\t\t<pts>')
            for px, py in conn.points:
                lines.append(f'\t\t\t\t\t\t<pt x="{px}" y="{py}"/>')
            lines.append('\t\t\t\t\t</pts>')
        for fragment in conn.extra_children_xml:
            model._append_xml_fragment(lines, fragment, '\t\t\t\t\t')
        lines.append('\t\t\t\t</connector>')

    for fragment in model.view_extra_children_xml:
        model._append_xml_fragment(lines, fragment, '\t\t\t\t')
    lines.append('\t\t\t</view>')
    for fragment in model.views_extra_children_xml:
        model._append_xml_fragment(lines, fragment, '\t\t\t')
    lines.append('\t\t</views>')
    for fragment in model.model_extra_children_xml:
        model._append_xml_fragment(lines, fragment, '\t\t')
    lines.append('\t</model>')
    lines.append('</xmile>')

    return '\n'.join(lines)




def parse_stmx_file(filepath: str, compat_mode: str = "permissive") -> StellaModel:
    """Parse an existing .stmx file and return a StellaModel.

    Args:
        filepath: Path to .stmx file.
        compat_mode: "permissive" records compatibility warnings and keeps parsing.
            "strict" raises on compatibility issues.
    """
    mode = StellaModel._validate_compat_mode(compat_mode)
    compat_warnings: list[str] = []

    def compat_issue(message: str):
        if mode == "strict":
            raise ValueError(message)
        compat_warnings.append(message)

    tree = ET.parse(filepath)
    root = tree.getroot()

    # Handle namespaces with full Clark notation
    xmile = f"{{{XMILE_NS}}}"
    isee = f"{{{ISEE_NS}}}"

    def find_child(parent, tag):
        """Find direct child element."""
        elem = parent.find(f"{xmile}{tag}")
        if elem is None:
            elem = parent.find(f"{isee}{tag}")
        if elem is None:
            elem = parent.find(tag)
        return elem

    def findall_children(parent, tag):
        """Find all direct children with given tag."""
        elems = parent.findall(f"{xmile}{tag}")
        if not elems:
            elems = parent.findall(f"{isee}{tag}")
        if not elems:
            elems = parent.findall(tag)
        return elems

    def local_name(tag: str) -> str:
        return StellaModel._xml_local_name(tag)

    def collect_extra_attrs(elem: ET.Element, known_attr_names: set[str]) -> dict[str, str]:
        extras: dict[str, str] = {}
        for key, value in elem.attrib.items():
            if local_name(key) in known_attr_names:
                continue
            extras[key] = value
        return extras

    def collect_extra_children(elem: ET.Element, known_child_names: set[str]) -> list[str]:
        extras: list[str] = []
        for child in list(elem):
            if local_name(child.tag) in known_child_names:
                continue
            extras.append(ET.tostring(child, encoding="unicode"))
        return extras

    def parse_point_list(text: str | None, context: str) -> list[float]:
        if not text:
            return []
        values: list[float] = []
        for raw in text.split():
            try:
                values.append(float(raw))
            except ValueError:
                compat_issue(f"Invalid numeric value '{raw}' in {context}")
                return []
        return values

    def parse_optional_float(value: str | None, context: str) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            compat_issue(f"Invalid numeric value '{value}' for {context}")
            return None

    def parse_gf(elem: ET.Element | None, context: str) -> GraphicalFunction | None:
        if elem is None:
            return None
        gf_type = elem.get("type")
        xpts_elem = find_child(elem, "xpts")
        xscale_elem = find_child(elem, "xscale")
        yscale_elem = find_child(elem, "yscale")
        ypts_elem = find_child(elem, "ypts")

        xpts = parse_point_list(xpts_elem.text, f"{context}.xpts") if xpts_elem is not None else None
        xscale = None
        if xscale_elem is not None:
            min_val = parse_optional_float(xscale_elem.get("min"), f"{context}.xscale.min")
            max_val = parse_optional_float(xscale_elem.get("max"), f"{context}.xscale.max")
            if min_val is not None and max_val is not None:
                xscale = (min_val, max_val)
        yscale = None
        if yscale_elem is not None:
            min_val = parse_optional_float(yscale_elem.get("min"), f"{context}.yscale.min")
            max_val = parse_optional_float(yscale_elem.get("max"), f"{context}.yscale.max")
            if min_val is not None and max_val is not None:
                yscale = (min_val, max_val)
        ypts = parse_point_list(ypts_elem.text if ypts_elem is not None else None, f"{context}.ypts")
        if not ypts:
            compat_issue(f"Graphical function for {context} has empty/invalid ypts and was skipped")
            return None
        return GraphicalFunction(
            ypts=ypts,
            xscale=xscale,
            xpts=xpts,
            yscale=yscale,
            gf_type=gf_type if gf_type else None,
        )

    def parse_required_name(elem: ET.Element, context: str) -> tuple[str, str] | None:
        raw_name = elem.get("name")
        if raw_name is None or not raw_name.strip():
            compat_issue(f"{context} is missing required name attribute and was skipped")
            return None
        norm_name = raw_name.replace(" ", "_")
        return raw_name, norm_name

    # Get model name
    header = find_child(root, "header")
    name_elem = find_child(header, "name") if header is not None else None
    model_name = name_elem.text if name_elem is not None and name_elem.text else "Untitled"
    model = StellaModel(name=model_name)
    if header is not None:
        model.header_extra_children_xml = collect_extra_children(
            header,
            known_child_names={"smile", "name", "uuid", "vendor", "product"},
        )

    # Parse sim_specs
    sim_specs = find_child(root, "sim_specs")
    if sim_specs is not None:
        model.sim_specs.extra_attrs = collect_extra_attrs(
            sim_specs,
            known_attr_names={
                "sim_duration",
                "simulation_delay",
                "restore_on_start",
                "method",
                "time_units",
                "instantaneous_flows",
            },
        )
        model.sim_specs.extra_children_xml = collect_extra_children(
            sim_specs,
            known_child_names={"start", "stop", "dt"},
        )
        start = find_child(sim_specs, "start")
        if start is not None and start.text:
            parsed_start = parse_optional_float(start.text, "sim_specs.start")
            if parsed_start is not None:
                model.sim_specs.start = parsed_start

        stop = find_child(sim_specs, "stop")
        if stop is not None and stop.text:
            parsed_stop = parse_optional_float(stop.text, "sim_specs.stop")
            if parsed_stop is not None:
                model.sim_specs.stop = parsed_stop

        dt = find_child(sim_specs, "dt")
        if dt is not None and dt.text:
            parsed_dt = parse_optional_float(dt.text, "sim_specs.dt")
            if parsed_dt is not None:
                if dt.get("reciprocal") == "true":
                    if parsed_dt <= 0:
                        compat_issue(f"sim_specs.dt reciprocal value must be > 0, got {parsed_dt}")
                    else:
                        model.sim_specs.dt = 1.0 / parsed_dt
                elif parsed_dt <= 0:
                    compat_issue(f"sim_specs.dt must be > 0, got {parsed_dt}")
                else:
                    model.sim_specs.dt = parsed_dt

        method = sim_specs.get("method")
        if method:
            model.sim_specs.method = method

        time_units = sim_specs.get("time_units")
        if time_units:
            model.sim_specs.time_units = time_units

    prefs_elem = find_child(root, "prefs")
    if prefs_elem is not None:
        model.prefs_xml = ET.tostring(prefs_elem, encoding="unicode")

    # Find variables section
    model_elem = find_child(root, "model")
    variables = find_child(model_elem, "variables") if model_elem is not None else None

    def name_collides(norm_name: str) -> bool:
        return norm_name in model.stocks or norm_name in model.flows or norm_name in model.auxs

    if variables is not None:
        # Parse stocks
        for stock_elem in findall_children(variables, "stock"):
            parsed_name = parse_required_name(stock_elem, "stock")
            if parsed_name is None:
                continue
            name, norm_name = parsed_name
            if name_collides(norm_name):
                compat_issue(f"Duplicate variable name '{name}' encountered; later occurrence skipped")
                continue

            eqn = find_child(stock_elem, "eqn")
            initial_value = eqn.text if eqn is not None and eqn.text is not None else "0"
            units_elem = find_child(stock_elem, "units")
            units = units_elem.text if units_elem is not None and units_elem.text is not None else ""

            inflows = [inf.text for inf in findall_children(stock_elem, "inflow") if inf.text]
            outflows = [outf.text for outf in findall_children(stock_elem, "outflow") if outf.text]
            norm_inflows = [model._normalize_name(value) for value in inflows]
            norm_outflows = [model._normalize_name(value) for value in outflows]
            non_negative = find_child(stock_elem, "non_negative") is not None

            model.stocks[norm_name] = Stock(
                name=name,
                initial_value=initial_value,
                units=units,
                inflows=norm_inflows,
                outflows=norm_outflows,
                non_negative=non_negative,
                extra_attrs=collect_extra_attrs(stock_elem, {"name"}),
                extra_children_xml=collect_extra_children(
                    stock_elem,
                    {"eqn", "inflow", "outflow", "non_negative", "units"},
                ),
            )

        # Parse flows
        for flow_elem in findall_children(variables, "flow"):
            parsed_name = parse_required_name(flow_elem, "flow")
            if parsed_name is None:
                continue
            name, norm_name = parsed_name
            if name_collides(norm_name):
                compat_issue(f"Duplicate variable name '{name}' encountered; later occurrence skipped")
                continue

            eqn = find_child(flow_elem, "eqn")
            equation = eqn.text if eqn is not None and eqn.text is not None else "0"
            gf = parse_gf(find_child(flow_elem, "gf"), f"flow '{name}'")
            units_elem = find_child(flow_elem, "units")
            units = units_elem.text if units_elem is not None and units_elem.text is not None else ""
            non_negative = find_child(flow_elem, "non_negative") is not None

            model.flows[norm_name] = Flow(
                name=name,
                equation=equation,
                units=units,
                non_negative=non_negative,
                graphical_function=gf,
                extra_attrs=collect_extra_attrs(flow_elem, {"name"}),
                extra_children_xml=collect_extra_children(
                    flow_elem,
                    {"eqn", "gf", "non_negative", "units"},
                ),
            )

        # Parse auxiliaries
        for aux_elem in findall_children(variables, "aux"):
            parsed_name = parse_required_name(aux_elem, "aux")
            if parsed_name is None:
                continue
            name, norm_name = parsed_name
            if name_collides(norm_name):
                compat_issue(f"Duplicate variable name '{name}' encountered; later occurrence skipped")
                continue

            eqn = find_child(aux_elem, "eqn")
            equation = eqn.text if eqn is not None and eqn.text is not None else "0"
            gf = parse_gf(find_child(aux_elem, "gf"), f"aux '{name}'")
            units_elem = find_child(aux_elem, "units")
            units = units_elem.text if units_elem is not None and units_elem.text is not None else ""

            model.auxs[norm_name] = Aux(
                name=name,
                equation=equation,
                units=units,
                graphical_function=gf,
                extra_attrs=collect_extra_attrs(aux_elem, {"name"}),
                extra_children_xml=collect_extra_children(
                    aux_elem,
                    {"eqn", "gf", "units"},
                ),
            )

        # Parse modules/groups
        for group_elem in findall_children(variables, "group"):
            parsed_name = parse_required_name(group_elem, "group")
            if parsed_name is None:
                continue
            name, norm_name = parsed_name

            members: list[str] = []
            for entity_elem in findall_children(group_elem, "entity"):
                entity_name = entity_elem.get("name")
                if not entity_name:
                    compat_issue(f"Module '{name}' includes entity with missing name; skipped")
                    continue
                norm_member = model._normalize_name(entity_name)
                if norm_member not in members:
                    members.append(norm_member)

            if norm_name in model.modules:
                compat_issue(f"Duplicate module '{name}' encountered; merging members")
                existing = model.modules[norm_name]
                merged = existing.members + [m for m in members if m not in existing.members]
                existing.members = merged
                existing.extra_attrs.update(collect_extra_attrs(group_elem, {"name"}))
                existing.extra_children_xml.extend(
                    collect_extra_children(group_elem, {"entity"})
                )
            else:
                model.modules[norm_name] = Module(name=name, members=members)
                model.modules[norm_name].extra_attrs = collect_extra_attrs(group_elem, {"name"})
                model.modules[norm_name].extra_children_xml = collect_extra_children(
                    group_elem,
                    {"entity"},
                )

    # Determine flow from/to stocks based on stock inflows/outflows
    for stock_name, stock in model.stocks.items():
        for inflow in stock.inflows:
            norm_inflow = model._normalize_name(inflow)
            if norm_inflow in model.flows:
                model.flows[norm_inflow].to_stock = stock_name
            else:
                compat_issue(
                    f"Stock '{stock.name}' references inflow '{inflow}' that does not exist"
                )
        for outflow in stock.outflows:
            norm_outflow = model._normalize_name(outflow)
            if norm_outflow in model.flows:
                model.flows[norm_outflow].from_stock = stock_name
            else:
                compat_issue(
                    f"Stock '{stock.name}' references outflow '{outflow}' that does not exist"
                )

    # Parse visual positions and connectors from views
    if model_elem is not None:
        model.model_extra_children_xml = collect_extra_children(
            model_elem,
            known_child_names={"variables", "views"},
        )
    views = find_child(model_elem, "views") if model_elem is not None else None
    if views is not None:
        styles = findall_children(views, "style")
        view_elems = findall_children(views, "view")
        if styles:
            model.views_style_xml = ET.tostring(styles[0], encoding="unicode")
        model.views_extra_children_xml = collect_extra_children(
            views,
            known_child_names={"style", "view"},
        )
        if len(styles) > 1:
            model.views_extra_children_xml.extend(
                ET.tostring(style_elem, encoding="unicode")
                for style_elem in styles[1:]
            )
        if len(view_elems) > 1:
            model.views_extra_children_xml.extend(
                ET.tostring(view_elem, encoding="unicode")
                for view_elem in view_elems[1:]
            )
    view = find_child(views, "view") if views is not None else None

    if view is not None:
        inner_styles = findall_children(view, "style")
        if inner_styles:
            model.inner_view_style_xml = ET.tostring(inner_styles[0], encoding="unicode")
        model.view_extra_attrs = collect_extra_attrs(
            view,
            known_attr_names={
                "show_pages",
                "background",
                "page_width",
                "page_height",
                "page_cols",
                "page_rows",
                "popup_graphs_are_comparative",
                "type",
            },
        )
        model.view_extra_children_xml = collect_extra_children(
            view,
            known_child_names={"style", "stock", "flow", "aux", "group", "connector"},
        )
        if len(inner_styles) > 1:
            model.view_extra_children_xml.extend(
                ET.tostring(style_elem, encoding="unicode")
                for style_elem in inner_styles[1:]
            )

        # Extract stock positions from view
        for stock_elem in findall_children(view, "stock"):
            parsed_name = parse_required_name(stock_elem, "view.stock")
            if parsed_name is None:
                continue
            _, norm_name = parsed_name
            x_attr = parse_optional_float(stock_elem.get("x"), f"view.stock[{norm_name}].x")
            y_attr = parse_optional_float(stock_elem.get("y"), f"view.stock[{norm_name}].y")
            width_attr = parse_optional_float(stock_elem.get("width"), f"view.stock[{norm_name}].width")
            height_attr = parse_optional_float(stock_elem.get("height"), f"view.stock[{norm_name}].height")
            if norm_name in model.stocks:
                if x_attr is not None:
                    model.stocks[norm_name].x = x_attr
                if y_attr is not None:
                    model.stocks[norm_name].y = y_attr
                if width_attr is not None:
                    model.stocks[norm_name].width = int(width_attr)
                    model.stocks[norm_name].size_locked = True
                if height_attr is not None:
                    model.stocks[norm_name].height = int(height_attr)
                    model.stocks[norm_name].size_locked = True
                model.stocks[norm_name].view_extra_attrs = collect_extra_attrs(
                    stock_elem,
                    {"x", "y", "width", "height", "name"},
                )

        # Extract flow positions from view
        for flow_elem in findall_children(view, "flow"):
            parsed_name = parse_required_name(flow_elem, "view.flow")
            if parsed_name is None:
                continue
            _, norm_name = parsed_name
            x_attr = parse_optional_float(flow_elem.get("x"), f"view.flow[{norm_name}].x")
            y_attr = parse_optional_float(flow_elem.get("y"), f"view.flow[{norm_name}].y")
            if norm_name in model.flows:
                if x_attr is not None:
                    model.flows[norm_name].x = x_attr
                if y_attr is not None:
                    model.flows[norm_name].y = y_attr
                pts = find_child(flow_elem, "pts")
                if pts is not None:
                    points: list[tuple[float, float]] = []
                    for pt in findall_children(pts, "pt"):
                        px = parse_optional_float(pt.get("x"), f"view.flow[{norm_name}].pt.x")
                        py = parse_optional_float(pt.get("y"), f"view.flow[{norm_name}].pt.y")
                        if px is None or py is None:
                            continue
                        points.append((px, py))
                    if len(points) >= 2:
                        model.flows[norm_name].points = points
                        model.flows[norm_name].points_locked = True
                model.flows[norm_name].view_extra_attrs = collect_extra_attrs(
                    flow_elem,
                    {"x", "y", "name"},
                )
                model.flows[norm_name].view_extra_children_xml = collect_extra_children(
                    flow_elem,
                    {"pts"},
                )

        # Extract aux positions from view
        for aux_elem in findall_children(view, "aux"):
            parsed_name = parse_required_name(aux_elem, "view.aux")
            if parsed_name is None:
                continue
            _, norm_name = parsed_name
            x_attr = parse_optional_float(aux_elem.get("x"), f"view.aux[{norm_name}].x")
            y_attr = parse_optional_float(aux_elem.get("y"), f"view.aux[{norm_name}].y")
            if norm_name in model.auxs:
                if x_attr is not None:
                    model.auxs[norm_name].x = x_attr
                if y_attr is not None:
                    model.auxs[norm_name].y = y_attr
                model.auxs[norm_name].view_extra_attrs = collect_extra_attrs(
                    aux_elem,
                    {"x", "y", "name"},
                )

        # Extract module/group view geometry
        for group_elem in findall_children(view, "group"):
            parsed_name = parse_required_name(group_elem, "view.group")
            if parsed_name is None:
                continue
            name, norm_name = parsed_name
            module = model.modules.get(norm_name)
            if module is None:
                module = Module(name=name)
                model.modules[norm_name] = module

            x_attr = parse_optional_float(group_elem.get("x"), f"view.group[{norm_name}].x")
            y_attr = parse_optional_float(group_elem.get("y"), f"view.group[{norm_name}].y")
            width_attr = parse_optional_float(group_elem.get("width"), f"view.group[{norm_name}].width")
            height_attr = parse_optional_float(group_elem.get("height"), f"view.group[{norm_name}].height")
            color_attr = group_elem.get("color")
            border_color_attr = group_elem.get("border_color")
            background_attr = group_elem.get("background")
            font_color_attr = group_elem.get("font_color")
            font_size_attr = group_elem.get("font_size")
            label_side_attr = group_elem.get("label_side")
            if x_attr is not None:
                module.x = x_attr
            if y_attr is not None:
                module.y = y_attr
            if width_attr is not None:
                module.width = width_attr
            if height_attr is not None:
                module.height = height_attr
            if color_attr is not None:
                module.border_color = color_attr
            elif border_color_attr is not None:
                module.border_color = border_color_attr
            if background_attr is not None:
                module.background = background_attr
            if font_color_attr is not None:
                module.font_color = font_color_attr
            if font_size_attr is not None:
                module.font_size = font_size_attr
            if label_side_attr is not None:
                module.label_side = label_side_attr
            module.view_extra_attrs = collect_extra_attrs(
                group_elem,
                {"x", "y", "width", "height", "name", "color", "border_color", "background", "font_color", "font_size", "label_side"},
            )
            module.view_extra_children_xml = collect_extra_children(group_elem, set())

        # Extract connectors
        for conn_elem in findall_children(view, "connector"):
            uid_text = conn_elem.get("uid", "0")
            angle_text = conn_elem.get("angle", "0")
            try:
                uid = int(uid_text)
            except ValueError:
                compat_issue(f"Connector uid '{uid_text}' is invalid; using 0")
                uid = 0
            angle = parse_optional_float(angle_text, f"connector[{uid}].angle")
            if angle is None:
                angle = 0.0

            from_elem = find_child(conn_elem, "from")
            to_elem = find_child(conn_elem, "to")
            if from_elem is None or to_elem is None or not from_elem.text or not to_elem.text:
                compat_issue(f"Connector uid={uid} missing from/to endpoint; skipped")
                continue

            from_norm = model._normalize_name(from_elem.text)
            to_norm = model._normalize_name(to_elem.text)
            if not model._has_variable(from_norm):
                compat_issue(f"Connector uid={uid} source '{from_norm}' does not match a model variable")
            if not model._has_variable(to_norm):
                compat_issue(f"Connector uid={uid} target '{to_norm}' does not match a model variable")

            connector = Connector(
                uid=uid,
                from_var=from_norm,
                to_var=to_norm,
                angle=angle,
                angle_locked=True,
                extra_attrs=collect_extra_attrs(conn_elem, {"uid", "angle"}),
                extra_children_xml=collect_extra_children(conn_elem, {"from", "to", "pts"}),
            )
            pts_elem = find_child(conn_elem, "pts")
            if pts_elem is not None:
                points: list[tuple[float, float]] = []
                for pt_elem in findall_children(pts_elem, "pt"):
                    px = parse_optional_float(pt_elem.get("x"), f"connector[{uid}].pt.x")
                    py = parse_optional_float(pt_elem.get("y"), f"connector[{uid}].pt.y")
                    if px is None or py is None:
                        continue
                    points.append((px, py))
                if points:
                    connector.points = points
                    connector.points_locked = True
            model.connectors.append(connector)
            model._connector_uid = max(model._connector_uid, uid)

    model.compatibility_warnings = compat_warnings
    return model
