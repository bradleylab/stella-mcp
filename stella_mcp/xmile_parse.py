"""XMILE parsing and compatibility-preservation logic."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from .equation_parser import (
    StellaReservedIdentifierError,
    find_stella_reserved_identifiers,
)
from .model import StellaModel
from .model_types import (
    ISEE_NS,
    XMILE_NS,
    Aux,
    Connector,
    Flow,
    GraphicalFunction,
    Module,
    Stock,
)
from .xmile_features import UnsupportedModelFeatureError, detect_xmile_features


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
    feature_report = detect_xmile_features(root)
    if mode == "strict" and feature_report.preserved_only:
        raise UnsupportedModelFeatureError(feature_report.preserved_only)
    compat_warnings.extend(finding.message for finding in feature_report.preserved_only)

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
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

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

    def parse_point_list(text: str | None, context: str, sep: str | None = None) -> list[float]:
        if not text:
            return []
        # XMILE point lists are comma-separated by default, with an optional
        # sep attribute; exports from older versions of this package used
        # spaces, so whitespace splitting is kept as the fallback.
        if sep:
            tokens = text.split(sep)
        elif "," in text:
            tokens = text.split(",")
        else:
            tokens = text.split()
        values: list[float] = []
        for raw in tokens:
            raw = raw.strip()
            if not raw:
                continue
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

    def parse_font_points(value: str | None, context: str) -> float | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        try:
            if normalized.endswith("pt"):
                points = float(normalized[:-2])
            elif normalized.endswith("px"):
                points = float(normalized[:-2]) * 72.0 / 96.0
            else:
                points = float(normalized)
        except ValueError:
            compat_issue(f"Invalid font size '{value}' for {context}")
            return None
        if points <= 0:
            compat_issue(f"Font size for {context} must be > 0, got {value}")
            return None
        return points

    def parse_gf(elem: ET.Element | None, context: str) -> GraphicalFunction | None:
        if elem is None:
            return None
        gf_type = elem.get("type")
        xpts_elem = find_child(elem, "xpts")
        xscale_elem = find_child(elem, "xscale")
        yscale_elem = find_child(elem, "yscale")
        ypts_elem = find_child(elem, "ypts")

        xpts = (
            parse_point_list(xpts_elem.text, f"{context}.xpts", sep=xpts_elem.get("sep"))
            if xpts_elem is not None
            else None
        )
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
        ypts = parse_point_list(
            ypts_elem.text if ypts_elem is not None else None,
            f"{context}.ypts",
            sep=ypts_elem.get("sep") if ypts_elem is not None else None,
        )
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
    model.xmile_feature_report = feature_report
    if header is not None:
        smile_elem = find_child(header, "smile")
        if smile_elem is not None:
            model.header_smile_extra_attrs = collect_extra_attrs(
                smile_elem,
                known_attr_names={"version", "namespace"},
            )
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
    model_elems = findall_children(root, "model")
    model_elem = model_elems[0] if model_elems else None
    model.root_extra_children_xml = collect_extra_children(
        root,
        known_child_names={"header", "sim_specs", "prefs", "model"},
    )
    model.additional_model_xml = [
        ET.tostring(extra_model, encoding="unicode") for extra_model in model_elems[1:]
    ]
    variables = find_child(model_elem, "variables") if model_elem is not None else None
    if variables is not None:
        model.variables_extra_children_xml = collect_extra_children(
            variables,
            known_child_names={"stock", "flow", "aux", "group"},
        )

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
                compat_issue(
                    f"Duplicate variable name '{name}' encountered; later occurrence skipped"
                )
                continue

            eqn = find_child(stock_elem, "eqn")
            initial_value = eqn.text if eqn is not None and eqn.text is not None else "0"
            units_elem = find_child(stock_elem, "units")
            units = (
                units_elem.text if units_elem is not None and units_elem.text is not None else ""
            )

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
                compat_issue(
                    f"Duplicate variable name '{name}' encountered; later occurrence skipped"
                )
                continue

            eqn = find_child(flow_elem, "eqn")
            equation = eqn.text if eqn is not None and eqn.text is not None else "0"
            gf = parse_gf(find_child(flow_elem, "gf"), f"flow '{name}'")
            units_elem = find_child(flow_elem, "units")
            units = (
                units_elem.text if units_elem is not None and units_elem.text is not None else ""
            )
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
                compat_issue(
                    f"Duplicate variable name '{name}' encountered; later occurrence skipped"
                )
                continue

            eqn = find_child(aux_elem, "eqn")
            equation = eqn.text if eqn is not None and eqn.text is not None else "0"
            gf = parse_gf(find_child(aux_elem, "gf"), f"aux '{name}'")
            units_elem = find_child(aux_elem, "units")
            units = (
                units_elem.text if units_elem is not None and units_elem.text is not None else ""
            )

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
                existing.extra_children_xml.extend(collect_extra_children(group_elem, {"entity"}))
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
                ET.tostring(style_elem, encoding="unicode") for style_elem in styles[1:]
            )
        if len(view_elems) > 1:
            model.views_extra_children_xml.extend(
                ET.tostring(view_elem, encoding="unicode") for view_elem in view_elems[1:]
            )
    view = find_child(views, "view") if views is not None else None

    if view is not None:
        page_width = parse_optional_float(view.get("page_width"), "view.page_width")
        page_height = parse_optional_float(view.get("page_height"), "view.page_height")
        page_columns = parse_optional_float(
            view.get(f"{isee}page_cols") or view.get("page_cols"),
            "view.page_cols",
        )
        page_rows = parse_optional_float(
            view.get(f"{isee}page_rows") or view.get("page_rows"),
            "view.page_rows",
        )
        if page_width is not None and page_width > 0:
            model.view_page_width = page_width
        elif page_width is not None:
            compat_issue(f"view.page_width must be > 0, got {page_width}")
        if page_height is not None and page_height > 0:
            model.view_page_height = page_height
        elif page_height is not None:
            compat_issue(f"view.page_height must be > 0, got {page_height}")
        if page_columns is not None and page_columns >= 1 and page_columns.is_integer():
            model.view_page_columns = int(page_columns)
        elif page_columns is not None:
            compat_issue(f"view.page_cols must be a positive integer, got {page_columns}")
        if page_rows is not None and page_rows >= 1 and page_rows.is_integer():
            model.view_page_rows = int(page_rows)
        elif page_rows is not None:
            compat_issue(f"view.page_rows must be a positive integer, got {page_rows}")

        inner_styles = findall_children(view, "style")
        if inner_styles:
            inner_style = inner_styles[0]
            model.inner_view_style_xml = ET.tostring(inner_style, encoding="unicode")
            for element_kind, attribute_name in (
                ("stock", "view_stock_font_points"),
                ("flow", "view_flow_font_points"),
                ("aux", "view_aux_font_points"),
            ):
                element_style = find_child(inner_style, element_kind)
                raw_font_size = (
                    element_style.get("font_size")
                    if element_style is not None and element_style.get("font_size") is not None
                    else inner_style.get("font_size")
                )
                font_points = parse_font_points(
                    raw_font_size,
                    f"view.style.{element_kind}.font_size",
                )
                if font_points is not None:
                    setattr(model, attribute_name, font_points)
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
                ET.tostring(style_elem, encoding="unicode") for style_elem in inner_styles[1:]
            )

        # Extract stock positions from view
        for stock_elem in findall_children(view, "stock"):
            parsed_name = parse_required_name(stock_elem, "view.stock")
            if parsed_name is None:
                continue
            _, norm_name = parsed_name
            x_attr = parse_optional_float(stock_elem.get("x"), f"view.stock[{norm_name}].x")
            y_attr = parse_optional_float(stock_elem.get("y"), f"view.stock[{norm_name}].y")
            width_attr = parse_optional_float(
                stock_elem.get("width"), f"view.stock[{norm_name}].width"
            )
            height_attr = parse_optional_float(
                stock_elem.get("height"), f"view.stock[{norm_name}].height"
            )
            if norm_name in model.stocks:
                # Stella stores stock x/y at the upper-left corner when that
                # dimension is explicit, but at the center when it is omitted.
                center_x = (
                    x_attr + width_attr / 2
                    if x_attr is not None and width_attr is not None
                    else x_attr
                )
                center_y = (
                    y_attr + height_attr / 2
                    if y_attr is not None and height_attr is not None
                    else y_attr
                )
                if x_attr is not None:
                    model.stocks[norm_name].x = center_x
                if y_attr is not None:
                    model.stocks[norm_name].y = center_y
                if x_attr is not None or y_attr is not None:
                    model.stocks[norm_name].position_source = "user"
                if width_attr is not None:
                    model.stocks[norm_name].width = int(width_attr)
                    model.stocks[norm_name].size_locked = True
                if height_attr is not None:
                    model.stocks[norm_name].height = int(height_attr)
                    model.stocks[norm_name].size_locked = True
                model.stocks[norm_name].view_extra_attrs = collect_extra_attrs(
                    stock_elem,
                    {"x", "y", "width", "height", "name", "label_side"},
                )
                label_side = stock_elem.get("label_side")
                if label_side in {"top", "bottom", "left", "right"}:
                    model.stocks[norm_name].label_side = label_side
                elif label_side is not None:
                    compat_issue(f"Invalid label_side '{label_side}' for view.stock[{norm_name}]")

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
                if x_attr is not None or y_attr is not None:
                    model.flows[norm_name].position_source = "user"
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
                    {"x", "y", "name", "label_side"},
                )
                label_side = flow_elem.get("label_side")
                if label_side in {"top", "bottom", "left", "right"}:
                    model.flows[norm_name].label_side = label_side
                elif label_side is not None:
                    compat_issue(f"Invalid label_side '{label_side}' for view.flow[{norm_name}]")
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
                if x_attr is not None or y_attr is not None:
                    model.auxs[norm_name].position_source = "user"
                model.auxs[norm_name].view_extra_attrs = collect_extra_attrs(
                    aux_elem,
                    {"x", "y", "name", "label_side"},
                )
                label_side = aux_elem.get("label_side")
                if label_side in {"top", "bottom", "left", "right"}:
                    model.auxs[norm_name].label_side = label_side
                elif label_side is not None:
                    compat_issue(f"Invalid label_side '{label_side}' for view.aux[{norm_name}]")

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
            width_attr = parse_optional_float(
                group_elem.get("width"), f"view.group[{norm_name}].width"
            )
            height_attr = parse_optional_float(
                group_elem.get("height"), f"view.group[{norm_name}].height"
            )
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
                {
                    "x",
                    "y",
                    "width",
                    "height",
                    "name",
                    "color",
                    "border_color",
                    "background",
                    "font_color",
                    "font_size",
                    "label_side",
                },
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
                compat_issue(
                    f"Connector uid={uid} source '{from_norm}' does not match a model variable"
                )
            if not model._has_variable(to_norm):
                compat_issue(
                    f"Connector uid={uid} target '{to_norm}' does not match a model variable"
                )

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

    reserved_identifiers = find_stella_reserved_identifiers(
        [
            *(stock.name for stock in model.stocks.values()),
            *(flow.name for flow in model.flows.values()),
            *(aux.name for aux in model.auxs.values()),
        ]
    )
    if reserved_identifiers:
        if mode == "strict":
            raise StellaReservedIdentifierError(reserved_identifiers)
        compat_warnings.extend(
            f"Variable '{name}' conflicts with a Stella/XMILE built-in and may be renamed "
            "by Stella"
            for name in reserved_identifiers
        )

    model.compatibility_warnings = compat_warnings
    return model
