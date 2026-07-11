"""XMILE serialization and XML formatting helpers."""

from __future__ import annotations

import re
from html import escape

from .model_types import ISEE_NS, XMILE_NS, GraphicalFunction

# Stella accepts GRAPH(input), while XMILE stores only the input expression
# when a graphical-function definition is present.
_GRAPH_CALL = re.compile(r"^\s*GRAPH\s*\((.*)\)\s*$", re.IGNORECASE | re.DOTALL)


def _xml_local_name(tag: str) -> str:
    """Extract local XML tag name from namespaced or plain tags."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


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


def _add_view_styles_str(self, lines: list[str]):
    """Add the default view styles as strings."""
    lines.append(
        '\t\t\t<style color="black" background="white" font_style="normal" font_weight="normal" text_decoration="none" text_align="center" vertical_text_align="center" font_color="black" font_family="Arial" font_size="10pt" padding="2" border_color="black" border_width="thin" border_style="none">'
    )
    lines.append(
        '\t\t\t\t<text_box color="black" background="white" text_align="left" vertical_text_align="top" font_size="12pt"/>'
    )
    lines.append("\t\t\t</style>")


def _add_inner_view_styles_str(self, lines: list[str]):
    """Add the inner view styles as strings."""
    lines.append(
        '\t\t\t\t<style color="black" background="white" font_style="normal" font_weight="normal" text_decoration="none" text_align="center" vertical_text_align="center" font_color="black" font_family="Arial" font_size="10pt" padding="2" border_color="black" border_width="thin" border_style="none">'
    )
    lines.append(
        '\t\t\t\t\t<stock color="blue" background="white" font_color="blue" font_size="9pt" label_side="top">'
    )
    lines.append('\t\t\t\t\t\t<shape type="rectangle" width="45" height="35"/>')
    lines.append("\t\t\t\t\t</stock>")
    lines.append(
        '\t\t\t\t\t<flow color="blue" background="white" font_color="blue" font_size="9pt" label_side="bottom"/>'
    )
    lines.append(
        '\t\t\t\t\t<aux color="blue" background="white" font_color="blue" font_size="9pt" label_side="bottom">'
    )
    lines.append('\t\t\t\t\t\t<shape type="circle" radius="18"/>')
    lines.append("\t\t\t\t\t</aux>")
    lines.append(
        '\t\t\t\t\t<group color="#666666" background="#F5F5F5" font_color="black" font_size="9pt" label_side="top"/>'
    )
    lines.append(
        '\t\t\t\t\t<connector color="#FF007F" background="white" font_color="#FF007F" font_size="9pt" isee:thickness="1"/>'
    )
    lines.append("\t\t\t\t</style>")


def _format_point_list(self, points: list[float]) -> str:
    # XMILE defines point lists as comma-separated (the sep attribute can
    # override, but readers like Stella and PySD assume the spec default).
    return ",".join(f"{p:g}" for p in points)


def _add_graphical_function_str(self, lines: list[str], gf: GraphicalFunction):
    attrs = f' type="{escape(gf.gf_type)}"' if gf.gf_type else ""
    lines.append(f"\t\t\t\t<gf{attrs}>")
    if gf.xpts is not None:
        lines.append(f"\t\t\t\t\t<xpts>{self._format_point_list(gf.xpts)}</xpts>")
    elif gf.xscale is not None:
        lines.append(f'\t\t\t\t\t<xscale min="{gf.xscale[0]:g}" max="{gf.xscale[1]:g}"/>')
    if gf.yscale is not None:
        lines.append(f'\t\t\t\t\t<yscale min="{gf.yscale[0]:g}" max="{gf.yscale[1]:g}"/>')
    lines.append(f"\t\t\t\t\t<ypts>{self._format_point_list(gf.ypts)}</ypts>")
    lines.append("\t\t\t\t</gf>")


def gf_eqn_text(equation: str) -> str:
    """Equation text to export for a gf-bearing variable (spec form)."""
    match = _GRAPH_CALL.match(equation)
    return match.group(1).strip() if match else equation


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
            compat_issue(f"Flow '{flow.name}' references missing from_stock '{flow.from_stock}'")
        if flow.to_stock is not None and flow.to_stock not in model.stocks:
            compat_issue(f"Flow '{flow.name}' references missing to_stock '{flow.to_stock}'")

    for stock in model.stocks.values():
        for inflow in stock.inflows:
            if inflow not in model.flows:
                compat_issue(f"Stock '{stock.name}' references missing inflow '{inflow}'")
        for outflow in stock.outflows:
            if outflow not in model.flows:
                compat_issue(f"Stock '{stock.name}' references missing outflow '{outflow}'")

    for connector in model.connectors:
        if (
            connector.from_var not in model.stocks
            and connector.from_var not in model.flows
            and connector.from_var not in model.auxs
        ):
            compat_issue(f"Connector uid={connector.uid} source '{connector.from_var}' is missing")
        if (
            connector.to_var not in model.stocks
            and connector.to_var not in model.flows
            and connector.to_var not in model.auxs
        ):
            compat_issue(f"Connector uid={connector.uid} target '{connector.to_var}' is missing")

    for module in model.modules.values():
        for member in module.members:
            if (
                member not in model.stocks
                and member not in model.flows
                and member not in model.auxs
            ):
                compat_issue(f"Module '{module.name}' references missing member '{member}'")

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
        f'xmlns:{prefix}="{uri}"' for uri, prefix in sorted(model._export_ns_prefix_by_uri.items())
    )
    ns_suffix = f" {extra_ns_decls}" if extra_ns_decls else ""

    lines = []
    lines.append('<?xml version="1.0" encoding="utf-8"?>')
    lines.append(f'<xmile version="1.0" xmlns="{XMILE_NS}" xmlns:isee="{ISEE_NS}"{ns_suffix}>')

    # Header
    lines.append("\t<header>")
    lines.append('\t\t<smile version="1.0" namespace="std, isee"/>')
    lines.append(f"\t\t<name>{escape(model.name)}</name>")
    lines.append(f"\t\t<uuid>{model.uuid}</uuid>")
    lines.append("\t\t<vendor>isee systems, inc.</vendor>")
    lines.append(
        '\t\t<product version="1.9.3" isee:build_number="1954" isee:saved_by_v1="true" lang="en">Stella Professional</product>'
    )
    for fragment in model.header_extra_children_xml:
        model._append_xml_fragment(lines, fragment, "\t\t")
    lines.append("\t</header>")

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
    lines.append(f"\t\t<start>{model._format_number(model.sim_specs.start)}</start>")
    lines.append(f"\t\t<stop>{model._format_number(model.sim_specs.stop)}</stop>")
    lines.append(f"\t\t{dt_str}")
    for fragment in model.sim_specs.extra_children_xml:
        model._append_xml_fragment(lines, fragment, "\t\t")
    lines.append("\t</sim_specs>")

    # Preferences
    if model.prefs_xml:
        model._append_xml_fragment(lines, model.prefs_xml, "\t")
    else:
        lines.append(
            '\t<isee:prefs show_module_prefix="true" live_update_on_drag="true" show_restore_buttons="false" layer="model" interface_scale_ui="true" interface_max_page_width="10000" interface_max_page_height="10000" interface_min_page_width="0" interface_min_page_height="0" saved_runs="5" keep="false" rifp="true"/>'
        )

    # Model
    lines.append("\t<model>")
    lines.append("\t\t<variables>")

    # Stocks
    for name in sorted(model.stocks):
        stock = model.stocks[name]
        display = escape(model._display_name(stock.name))
        stock_extra_attrs = model._format_extra_attrs(
            stock.extra_attrs,
            reserved_names={"name"},
        )
        lines.append(f'\t\t\t<stock name="{display}"{stock_extra_attrs}>')
        lines.append(f"\t\t\t\t<eqn>{escape(stock.initial_value)}</eqn>")
        for inflow in stock.inflows:
            lines.append(f"\t\t\t\t<inflow>{escape(inflow)}</inflow>")
        for outflow in stock.outflows:
            lines.append(f"\t\t\t\t<outflow>{escape(outflow)}</outflow>")
        if stock.non_negative:
            lines.append("\t\t\t\t<non_negative/>")
        if stock.units:
            lines.append(f"\t\t\t\t<units>{escape(stock.units)}</units>")
        for fragment in stock.extra_children_xml:
            model._append_xml_fragment(lines, fragment, "\t\t\t\t")
        lines.append("\t\t\t</stock>")

    # Flows
    for name in sorted(model.flows):
        flow = model.flows[name]
        display = escape(model._display_name(flow.name))
        flow_extra_attrs = model._format_extra_attrs(
            flow.extra_attrs,
            reserved_names={"name"},
        )
        lines.append(f'\t\t\t<flow name="{display}"{flow_extra_attrs}>')
        if flow.graphical_function is not None:
            lines.append(f"\t\t\t\t<eqn>{escape(gf_eqn_text(flow.equation))}</eqn>")
            model._add_graphical_function_str(lines, flow.graphical_function)
        else:
            lines.append(f"\t\t\t\t<eqn>{escape(flow.equation)}</eqn>")
        if flow.non_negative:
            lines.append("\t\t\t\t<non_negative/>")
        if flow.units:
            lines.append(f"\t\t\t\t<units>{escape(flow.units)}</units>")
        for fragment in flow.extra_children_xml:
            model._append_xml_fragment(lines, fragment, "\t\t\t\t")
        lines.append("\t\t\t</flow>")

    # Auxiliaries
    for name in sorted(model.auxs):
        aux = model.auxs[name]
        display = escape(model._display_name(aux.name))
        aux_extra_attrs = model._format_extra_attrs(
            aux.extra_attrs,
            reserved_names={"name"},
        )
        lines.append(f'\t\t\t<aux name="{display}"{aux_extra_attrs}>')
        if aux.graphical_function is not None:
            lines.append(f"\t\t\t\t<eqn>{escape(gf_eqn_text(aux.equation))}</eqn>")
            model._add_graphical_function_str(lines, aux.graphical_function)
        else:
            lines.append(f"\t\t\t\t<eqn>{escape(aux.equation)}</eqn>")
        if aux.units:
            lines.append(f"\t\t\t\t<units>{escape(aux.units)}</units>")
        for fragment in aux.extra_children_xml:
            model._append_xml_fragment(lines, fragment, "\t\t\t\t")
        lines.append("\t\t\t</aux>")

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
            model._append_xml_fragment(lines, fragment, "\t\t\t\t")
        lines.append("\t\t\t</group>")

    lines.append("\t\t</variables>")

    # Views
    lines.append("\t\t<views>")
    if model.views_style_xml:
        model._append_xml_fragment(lines, model.views_style_xml, "\t\t\t")
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
        model._append_xml_fragment(lines, model.inner_view_style_xml, "\t\t\t\t")
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
            reserved_names={
                "x",
                "y",
                "width",
                "height",
                "name",
                "color",
                "background",
                "font_color",
                "font_size",
                "label_side",
            },
        )
        if module.view_extra_children_xml:
            lines.append(f"\t\t\t\t<group {' '.join(attrs)}{module_view_extra_attrs}>")
            for fragment in module.view_extra_children_xml:
                model._append_xml_fragment(lines, fragment, "\t\t\t\t\t")
            lines.append("\t\t\t\t</group>")
        else:
            lines.append(
                f"\t\t\t\t<group {' '.join(attrs)}{module_view_extra_attrs}/>"
            )  # self-closing

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
        lines.append(
            f'\t\t\t\t<stock x="{sx}" y="{sy}" width="{stock.width}" height="{stock.height}" name="{display}"{stock_view_extra_attrs}/>'
        )

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
            lines.append(
                f'\t\t\t\t<flow x="{fx}" y="{fy}" name="{display}"{flow_view_extra_attrs}>'
            )
            if flow.points:
                lines.append("\t\t\t\t\t<pts>")
                for px, py in flow.points:
                    lines.append(f'\t\t\t\t\t\t<pt x="{px}" y="{py}"/>')
                lines.append("\t\t\t\t\t</pts>")
            for fragment in flow.view_extra_children_xml:
                model._append_xml_fragment(lines, fragment, "\t\t\t\t\t")
            lines.append("\t\t\t\t</flow>")
        else:
            lines.append(
                f'\t\t\t\t<flow x="{fx}" y="{fy}" name="{display}"{flow_view_extra_attrs}/>'
            )

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
        lines.append(f"\t\t\t\t\t<from>{escape(conn.from_var)}</from>")
        lines.append(f"\t\t\t\t\t<to>{escape(conn.to_var)}</to>")
        if conn.points:
            lines.append("\t\t\t\t\t<pts>")
            for px, py in conn.points:
                lines.append(f'\t\t\t\t\t\t<pt x="{px}" y="{py}"/>')
            lines.append("\t\t\t\t\t</pts>")
        for fragment in conn.extra_children_xml:
            model._append_xml_fragment(lines, fragment, "\t\t\t\t\t")
        lines.append("\t\t\t\t</connector>")

    for fragment in model.view_extra_children_xml:
        model._append_xml_fragment(lines, fragment, "\t\t\t\t")
    lines.append("\t\t\t</view>")
    for fragment in model.views_extra_children_xml:
        model._append_xml_fragment(lines, fragment, "\t\t\t")
    lines.append("\t\t</views>")
    for fragment in model.model_extra_children_xml:
        model._append_xml_fragment(lines, fragment, "\t\t")
    lines.append("\t</model>")
    lines.append("</xmile>")

    return "\n".join(lines)
