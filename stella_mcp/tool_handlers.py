"""Tool handler registration for the Stella MCP server."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Protocol

from mcp.types import CallToolResult, TextContent

from .templates import (
    get_template_info,
    list_templates as list_available_templates,
    load_template_model,
    save_user_template,
)
from .validator import validate_model
from .xmile import GraphicalFunction, StellaModel, parse_stmx


ToolResponse = list[TextContent] | CallToolResult
ToolHandler = Callable[[dict[str, Any]], ToolResponse]


class SessionModelsLike(Protocol):
    """Minimal session model container needed by tool handlers."""

    models: dict[str, StellaModel]
    current_model_id: str | None


def register_tool_handlers(
    register: Callable[[str], Callable[[ToolHandler], ToolHandler]],
    *,
    get_model: Callable[[str | None], tuple[str, StellaModel]],
    set_current_model: Callable[[StellaModel, str | None], str],
    get_session_models: Callable[[], SessionModelsLike],
    build_graphical_function: Callable[[dict[str, Any] | None], GraphicalFunction | None],
    compat_warning_suffix: Callable[[list[str]], str],
) -> None:
    """Register all MCP tool handlers."""
    @register("create_model")
    def _handle_create_model(arguments: dict[str, Any]) -> ToolResponse:
        start = float(arguments.get("start", 0))
        stop = float(arguments.get("stop", 100))
        dt = float(arguments.get("dt", 0.25))
        if dt <= 0:
            raise ValueError("dt must be > 0")
        if stop <= start:
            raise ValueError("stop must be greater than start")

        model = StellaModel(name=arguments["name"])
        model.sim_specs.start = start
        model.sim_specs.stop = stop
        model.sim_specs.dt = dt
        model.sim_specs.method = arguments.get("method", "Euler")
        model.sim_specs.time_units = arguments.get("time_units", "Years")
        model_id = set_current_model(model, model_id=arguments.get("model_id"))
        return [TextContent(
            type="text",
            text=(
                f"Created model '{arguments['name']}' "
                f"(model_id={model_id}) with time range {start}-{stop}, dt={dt}"
            ),
        )]

    @register("add_stock")
    def _handle_add_stock(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        model.add_stock(
            name=arguments["name"],
            initial_value=arguments["initial_value"],
            units=arguments.get("units", ""),
            non_negative=arguments.get("non_negative", True),
            x=arguments.get("x"),
            y=arguments.get("y"),
        )
        pos_info = ""
        if arguments.get("x") is not None and arguments.get("y") is not None:
            pos_info = f" at position ({arguments['x']}, {arguments['y']})"
        return [TextContent(
            type="text",
            text=(
                f"Added stock '{arguments['name']}' to model_id={model_id} "
                f"with initial value {arguments['initial_value']}{pos_info}"
            ),
        )]

    @register("add_flow")
    def _handle_add_flow(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        model.add_flow(
            name=arguments["name"],
            equation=arguments["equation"],
            units=arguments.get("units", ""),
            from_stock=arguments.get("from_stock"),
            to_stock=arguments.get("to_stock"),
            non_negative=arguments.get("non_negative", True),
            x=arguments.get("x"),
            y=arguments.get("y"),
            graphical_function=build_graphical_function(arguments.get("graphical_function")),
        )
        flow_desc = []
        if arguments.get("from_stock"):
            flow_desc.append(f"from {arguments['from_stock']}")
        if arguments.get("to_stock"):
            flow_desc.append(f"to {arguments['to_stock']}")
        flow_str = " ".join(flow_desc) if flow_desc else "(external)"
        pos_info = ""
        if arguments.get("x") is not None and arguments.get("y") is not None:
            pos_info = f" at position ({arguments['x']}, {arguments['y']})"
        return [TextContent(
            type="text",
            text=f"Added flow '{arguments['name']}' to model_id={model_id} {flow_str}: {arguments['equation']}{pos_info}"
        )]

    @register("add_aux")
    def _handle_add_aux(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        model.add_aux(
            name=arguments["name"],
            equation=arguments["equation"],
            units=arguments.get("units", ""),
            x=arguments.get("x"),
            y=arguments.get("y"),
            graphical_function=build_graphical_function(arguments.get("graphical_function")),
        )
        pos_info = ""
        if arguments.get("x") is not None and arguments.get("y") is not None:
            pos_info = f" at position ({arguments['x']}, {arguments['y']})"
        return [TextContent(
            type="text",
            text=f"Added auxiliary '{arguments['name']}' to model_id={model_id} = {arguments['equation']}{pos_info}"
        )]

    @register("add_connector")
    def _handle_add_connector(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        model.add_connector(
            from_var=arguments["from_var"],
            to_var=arguments["to_var"],
        )
        return [TextContent(
            type="text",
            text=f"Added connector in model_id={model_id} from '{arguments['from_var']}' to '{arguments['to_var']}'"
        )]

    @register("set_connector_routing")
    def _handle_set_connector_routing(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        points_arg = arguments.get("points")
        points: list[tuple[float, float]] | None = None
        if points_arg is not None:
            if not isinstance(points_arg, list):
                raise ValueError("points must be an array of {x, y} objects")
            parsed_points: list[tuple[float, float]] = []
            for index, point in enumerate(points_arg):
                if not isinstance(point, dict):
                    raise ValueError(f"points[{index}] must be an object with x and y")
                if "x" not in point or "y" not in point:
                    raise ValueError(f"points[{index}] requires both x and y")
                px = float(point["x"])
                py = float(point["y"])
                if not (math.isfinite(px) and math.isfinite(py)):
                    raise ValueError(f"points[{index}] must contain finite x and y values")
                parsed_points.append((px, py))
            points = parsed_points

        angle_raw = arguments.get("angle")
        angle = float(angle_raw) if angle_raw is not None else None
        if angle is not None and not math.isfinite(angle):
            raise ValueError("angle must be a finite number")

        connector = model.set_connector_routing(
            connector_uid=arguments.get("connector_uid"),
            from_var=arguments.get("from_var"),
            to_var=arguments.get("to_var"),
            angle=angle,
            angle_locked=arguments.get("angle_locked"),
            points=points,
            points_locked=arguments.get("points_locked"),
        )
        return [TextContent(
            type="text",
            text=(
                f"Updated connector uid={connector.uid} in model_id={model_id}: "
                f"angle={connector.angle}, angle_locked={connector.angle_locked}, "
                f"points={len(connector.points)}, "
                f"points_locked={connector.points_locked}"
            ),
        )]

    @register("rename_variable")
    def _handle_rename_variable(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        kind, _ = model.rename_variable(
            old_name=arguments["old_name"],
            new_name=arguments["new_name"],
        )
        return [TextContent(
            type="text",
            text=(
                f"Renamed {kind} '{arguments['old_name']}' to '{arguments['new_name']}' "
                f"in model_id={model_id} and updated references"
            ),
        )]

    @register("delete_variable")
    def _handle_delete_variable(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        summary = model.delete_variable(
            name=arguments["name"],
            force=arguments.get("force", False),
        )
        return [TextContent(
            type="text",
            text=(
                f"Deleted {summary['kind']} '{arguments['name']}' from model_id={model_id}; "
                f"removed_connectors={summary['removed_connectors']}, "
                f"removed_module_memberships={summary['removed_module_memberships']}, "
                f"detached_flows={summary['detached_flows']}"
            ),
        )]

    @register("create_module")
    def _handle_create_module(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        module = model.create_module(
            name=arguments["name"],
            members=arguments.get("members"),
        )
        return [TextContent(
            type="text",
            text=f"Created module '{module.name}' in model_id={model_id} with {len(module.members)} members"
        )]

    @register("add_to_module")
    def _handle_add_to_module(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        module = model.add_to_module(
            module_name=arguments["module_name"],
            members=arguments["members"],
        )
        return [TextContent(
            type="text",
            text=(
                f"Added {len(arguments['members'])} members to module '{module.name}' "
                f"in model_id={model_id} (total members: {len(module.members)})"
            ),
        )]

    @register("remove_from_module")
    def _handle_remove_from_module(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        module = model.remove_from_module(
            module_name=arguments["module_name"],
            members=arguments["members"],
        )
        return [TextContent(
            type="text",
            text=(
                f"Removed up to {len(arguments['members'])} members from module '{module.name}' "
                f"in model_id={model_id} (total members: {len(module.members)})"
            ),
        )]

    @register("rename_module")
    def _handle_rename_module(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        module = model.rename_module(
            module_name=arguments["module_name"],
            new_name=arguments["new_name"],
        )
        return [TextContent(
            type="text",
            text=f"Renamed module '{arguments['module_name']}' to '{module.name}' in model_id={model_id}",
        )]

    @register("delete_module")
    def _handle_delete_module(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        module = model.delete_module(arguments["module_name"])
        return [TextContent(
            type="text",
            text=f"Deleted module '{module.name}' from model_id={model_id}",
        )]

    @register("set_module_view")
    def _handle_set_module_view(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        module = model.set_module_view(
            module_name=arguments["module_name"],
            x=arguments["x"],
            y=arguments["y"],
            width=arguments["width"],
            height=arguments["height"],
        )
        return [TextContent(
            type="text",
            text=(
                f"Set module view for '{module.name}' in model_id={model_id} "
                f"to center=({module.x}, {module.y}), size=({module.width}, {module.height})"
            ),
        )]

    @register("set_module_style")
    def _handle_set_module_style(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        module = model.set_module_style(
            module_name=arguments["module_name"],
            border_color=arguments.get("border_color"),
            background=arguments.get("background"),
            font_color=arguments.get("font_color"),
            font_size=arguments.get("font_size"),
            label_side=arguments.get("label_side"),
        )
        style_parts = []
        if module.border_color is not None:
            style_parts.append(f"border_color={module.border_color}")
        if module.background is not None:
            style_parts.append(f"background={module.background}")
        if module.font_color is not None:
            style_parts.append(f"font_color={module.font_color}")
        if module.font_size is not None:
            style_parts.append(f"font_size={module.font_size}")
        if module.label_side is not None:
            style_parts.append(f"label_side={module.label_side}")
        return [TextContent(
            type="text",
            text=(
                f"Set module style for '{module.name}' in model_id={model_id}: "
                + ", ".join(style_parts)
            ),
        )]

    @register("auto_place_module_boxes")
    def _handle_auto_place_module_boxes(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        model.auto_place_module_boxes(
            padding=arguments.get("padding", 40.0),
            min_width=arguments.get("min_width", 180.0),
            min_height=arguments.get("min_height", 120.0),
            only_missing=arguments.get("only_missing", False),
        )
        return [TextContent(
            type="text",
            text=f"Auto-placed module boxes in model_id={model_id} for {len(model.modules)} modules",
        )]

    @register("save_model")
    def _handle_save_model(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        filepath = Path(arguments["filepath"])
        if not filepath.suffix:
            filepath = filepath.with_suffix(".stmx")
        xml_content = model.to_xml(
            auto_layout=arguments.get("auto_layout", True),
            resolve_layout_violations=arguments.get("resolve_layout_violations", False),
            compat_mode=arguments.get("compat_mode", "permissive"),
        )
        filepath.write_text(xml_content, encoding="utf-8")
        warning_suffix = compat_warning_suffix(model.last_export_warnings)
        return [TextContent(
            type="text",
            text=f"Saved model_id={model_id} to {filepath}{warning_suffix}"
        )]

    @register("read_model")
    def _handle_read_model(arguments: dict[str, Any]) -> ToolResponse:
        filepath = Path(arguments["filepath"])
        model = parse_stmx(
            str(filepath),
            compat_mode=arguments.get("compat_mode", "permissive"),
        )
        model_id = set_current_model(
            model,
            model_id=arguments.get("model_id"),
        )
        n_stocks = len(model.stocks)
        n_flows = len(model.flows)
        n_aux = len(model.auxs)
        warning_suffix = compat_warning_suffix(model.compatibility_warnings)
        return [TextContent(
            type="text",
            text=(
                f"Loaded model '{model.name}' as model_id={model_id} "
                f"with {n_stocks} stocks, {n_flows} flows, {n_aux} auxiliaries"
                f"{warning_suffix}"
            ),
        )]

    @register("list_templates")
    def _handle_list_templates(arguments: dict[str, Any]) -> ToolResponse:
        templates = list_available_templates(
            source=arguments.get("source"),
            query=arguments.get("query"),
            tags=arguments.get("tags"),
        )
        if not templates:
            return [TextContent(type="text", text="No templates available.")]
        lines = ["Available templates:"]
        for info in templates:
            counts = f"{info.stocks}S/{info.flows}F/{info.auxiliaries}A"
            tags = ", ".join(info.tags) if info.tags else "-"
            lines.append(
                f"  - {info.name} [{info.source}] | title={info.title} | vars={counts} | tags={tags}"
            )
            if info.description:
                lines.append(f"    {info.description}")
        return [TextContent(type="text", text="\n".join(lines))]

    @register("get_template_info")
    def _handle_get_template_info(arguments: dict[str, Any]) -> ToolResponse:
        info = get_template_info(arguments["template_name"])
        tags = ", ".join(info.tags) if info.tags else "-"
        lines = [
            f"Template: {info.name}",
            f"source: {info.source}",
            f"title: {info.title}",
            f"description: {info.description or '-'}",
            f"tags: {tags}",
            f"variables: stocks={info.stocks}, flows={info.flows}, auxiliaries={info.auxiliaries}, modules={info.modules}",
            f"updated_at: {info.updated_at or '-'}",
            f"path: {info.path}",
        ]
        return [TextContent(type="text", text="\n".join(lines))]

    @register("load_template")
    def _handle_load_template(arguments: dict[str, Any]) -> ToolResponse:
        info, model = load_template_model(arguments["template_name"])
        model_id = set_current_model(model, model_id=arguments.get("model_id"))
        n_stocks = len(model.stocks)
        n_flows = len(model.flows)
        n_aux = len(model.auxs)
        return [TextContent(
            type="text",
            text=(
                f"Loaded template '{info.name}' [{info.source}] as model_id={model_id} "
                f"with {n_stocks} stocks, {n_flows} flows, {n_aux} auxiliaries"
            ),
        )]

    @register("save_as_template")
    def _handle_save_as_template(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        info = save_user_template(
            arguments["template_name"],
            model,
            overwrite=arguments.get("overwrite", False),
            description=arguments.get("description", ""),
            tags=arguments.get("tags"),
        )
        return [TextContent(
            type="text",
            text=f"Saved model_id={model_id} as template '{info.name}' at {info.path}",
        )]

    @register("list_models")
    def _handle_list_models(arguments: dict[str, Any]) -> ToolResponse:
        session_models = get_session_models()
        if not session_models.models:
            return [TextContent(type="text", text="No models created in this session.")]

        lines = ["Session models:"]
        for mid, model in sorted(session_models.models.items()):
            current = " (current)" if mid == session_models.current_model_id else ""
            lines.append(f"  - {mid}: {model.name}{current}")
        return [TextContent(type="text", text="\n".join(lines))]

    @register("list_modules")
    def _handle_list_modules(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        if not model.modules:
            return [TextContent(type="text", text=f"No modules in model_id={model_id}.")]

        lines = [f"Modules for model_id={model_id}:"]
        for module_name in sorted(model.modules):
            module = model.modules[module_name]
            members = (
                ", ".join(model._display_name(m) for m in sorted(module.members))
                if module.members else "(empty)"
            )
            style_parts = []
            if module.border_color is not None:
                style_parts.append(f"border_color={module.border_color}")
            if module.background is not None:
                style_parts.append(f"background={module.background}")
            if module.font_color is not None:
                style_parts.append(f"font_color={module.font_color}")
            if module.font_size is not None:
                style_parts.append(f"font_size={module.font_size}")
            if module.label_side is not None:
                style_parts.append(f"label_side={module.label_side}")
            style_suffix = f" | style=({', '.join(style_parts)})" if style_parts else ""
            if None not in (module.x, module.y, module.width, module.height):
                lines.append(
                    f"  - {module.name}: {members} | box=({module.x}, {module.y}, {module.width}, {module.height}){style_suffix}"
                )
            else:
                lines.append(f"  - {module.name}: {members}{style_suffix}")
        return [TextContent(type="text", text="\n".join(lines))]

    @register("list_connectors")
    def _handle_list_connectors(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        if not model.connectors:
            return [TextContent(type="text", text=f"No connectors in model_id={model_id}.")]

        lines = [f"Connectors for model_id={model_id}:"]
        for conn in sorted(model.connectors, key=lambda c: c.uid):
            from_display = model._display_name(conn.from_var)
            to_display = model._display_name(conn.to_var)
            line = (
                f"  - uid={conn.uid}: {from_display} -> {to_display} | "
                f"angle={conn.angle} (locked={conn.angle_locked}) | "
                f"points={len(conn.points)} (locked={conn.points_locked})"
            )
            if conn.points:
                preview = ", ".join(f"({x:g},{y:g})" for x, y in conn.points[:3])
                if len(conn.points) > 3:
                    preview += ", ..."
                line += f" | pts={preview}"
            lines.append(line)
        return [TextContent(type="text", text="\n".join(lines))]

    @register("validate_model")
    def _handle_validate_model(arguments: dict[str, Any]) -> ToolResponse:
        _, model = get_model(arguments.get("model_id"))
        errors = validate_model(model)
        if not errors:
            return [TextContent(type="text", text="Model validation passed with no errors or warnings.")]

        result_lines = ["Model validation results:"]
        for err in errors:
            prefix = "ERROR" if err.severity == "error" else "WARNING"
            result_lines.append(f"  [{prefix}] {err.category}: {err.message}")
        return [TextContent(type="text", text="\n".join(result_lines))]

    @register("list_variables")
    def _handle_list_variables(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        lines = [f"Model: {model.name}", ""]
        lines.insert(1, f"model_id: {model_id}")
        lines.insert(2, "")

        if model.stocks:
            lines.append("Stocks:")
            for name, stock in model.stocks.items():
                lines.append(f"  - {stock.name} = {stock.initial_value} [{stock.units}]")
            lines.append("")

        if model.flows:
            lines.append("Flows:")
            for name, flow in model.flows.items():
                from_str = flow.from_stock or "external"
                to_str = flow.to_stock or "external"
                lines.append(f"  - {flow.name}: {from_str} -> {to_str} = {flow.equation}")
            lines.append("")

        if model.auxs:
            lines.append("Auxiliaries:")
            for name, aux in model.auxs.items():
                lines.append(f"  - {aux.name} = {aux.equation} [{aux.units}]")
            lines.append("")

        if model.modules:
            lines.append("Modules:")
            for module_name in sorted(model.modules):
                module = model.modules[module_name]
                members = (
                    ", ".join(model._display_name(m) for m in sorted(module.members))
                    if module.members else "(empty)"
                )
                lines.append(f"  - {module.name}: {members}")

        return [TextContent(type="text", text="\n".join(lines))]

    @register("get_model_xml")
    def _handle_get_model_xml(arguments: dict[str, Any]) -> ToolResponse:
        _, model = get_model(arguments.get("model_id"))
        xml = model.to_xml(
            auto_layout=arguments.get("auto_layout", True),
            resolve_layout_violations=arguments.get("resolve_layout_violations", False),
            compat_mode=arguments.get("compat_mode", "permissive"),
        )
        # Truncate if too long
        if len(xml) > 10000:
            xml = xml[:10000] + "\n... (truncated)"
        output = [TextContent(type="text", text=xml)]
        if model.last_export_warnings:
            output.append(
                TextContent(
                    type="text",
                    text=(
                        f"Compatibility warnings ({len(model.last_export_warnings)}):\n"
                        + "\n".join(f"- {msg}" for msg in model.last_export_warnings[:5])
                    ),
                )
            )
        return output
