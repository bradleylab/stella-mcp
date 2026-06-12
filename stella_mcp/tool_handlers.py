"""Tool handler registration for the Stella MCP server."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from mcp.types import CallToolResult, TextContent

from .analysis import compare_scenarios, sensitivity_analysis
from .model_snapshot import (
    aux_to_dict,
    connector_to_dict,
    flow_to_dict,
    model_to_summary,
    module_to_dict,
    stock_to_dict,
    template_info_to_dict,
    validation_issue_to_dict,
)
from .render_svg import render_model_svg
from .simulate import run_simulation
from .templates import (
    get_template_info,
    load_template_model,
    save_user_template,
)
from .templates import (
    list_templates as list_available_templates,
)
from .tool_results import BatchItemError, success_result
from .validator import validate_model
from .xmile import GraphicalFunction, StellaModel, parse_stmx

ToolResponse = list[TextContent] | CallToolResult
ToolHandler = Callable[[dict[str, Any]], ToolResponse]


def _apply_batch_items(
    model: StellaModel,
    arguments: dict[str, Any],
    build_graphical_function: Callable[[dict[str, Any] | None], GraphicalFunction | None],
) -> dict[str, int]:
    """Apply batched variable/connector/module items to a model.

    Application order matters: auxs before flows so flow equations can
    reference parameters, flows after stocks because they attach to them.
    Raises BatchItemError naming the failing item; callers guarantee
    atomicity by applying to a scratch model and swapping on success.
    """
    added = {"stocks": 0, "flows": 0, "auxiliaries": 0, "connectors": 0, "modules": 0}

    def fail(stage: str, index: int, item: dict[str, Any], exc: Exception) -> BatchItemError:
        message = (
            f"missing required field {exc}" if isinstance(exc, KeyError) else str(exc)
        )
        name = item.get("name") if isinstance(item.get("name"), str) else item.get("to_var")
        return BatchItemError(stage, index, name, message)

    def name_field(item: dict[str, Any], field: str = "name") -> str:
        value = item[field]
        if not isinstance(value, str):
            raise ValueError(f"field '{field}' must be a string")
        return value

    def text_field(item: dict[str, Any], field: str) -> str:
        # Schema says string, but inputs are not schema-enforced at the
        # server; numbers are common for constant equations and unambiguous.
        # Anything else must fail HERE, before the item lands in the model —
        # a bad value that only explodes later (export, connector sync) would
        # break batch atomicity.
        value = item[field]
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"{value:g}" if isinstance(value, float) else str(value)
        raise ValueError(f"field '{field}' must be a string (or number)")

    for index, item in enumerate(arguments.get("stocks") or []):
        try:
            model.add_stock(
                name=name_field(item),
                initial_value=text_field(item, "initial_value"),
                units=item.get("units", ""),
                non_negative=item.get("non_negative", True),
                x=item.get("x"),
                y=item.get("y"),
            )
            added["stocks"] += 1
        except (KeyError, ValueError) as exc:
            raise fail("stocks", index, item, exc) from exc

    for index, item in enumerate(arguments.get("auxs") or []):
        try:
            model.add_aux(
                name=name_field(item),
                equation=text_field(item, "equation"),
                units=item.get("units", ""),
                x=item.get("x"),
                y=item.get("y"),
                graphical_function=build_graphical_function(item.get("graphical_function")),
            )
            added["auxiliaries"] += 1
        except (KeyError, ValueError) as exc:
            raise fail("auxs", index, item, exc) from exc

    for index, item in enumerate(arguments.get("flows") or []):
        try:
            model.add_flow(
                name=name_field(item),
                equation=text_field(item, "equation"),
                units=item.get("units", ""),
                from_stock=item.get("from_stock"),
                to_stock=item.get("to_stock"),
                non_negative=item.get("non_negative", True),
                x=item.get("x"),
                y=item.get("y"),
                graphical_function=build_graphical_function(item.get("graphical_function")),
            )
            added["flows"] += 1
        except (KeyError, ValueError) as exc:
            raise fail("flows", index, item, exc) from exc

    for index, item in enumerate(arguments.get("connectors") or []):
        try:
            model.add_connector(item["from_var"], item["to_var"])
            added["connectors"] += 1
        except (KeyError, ValueError) as exc:
            raise fail("connectors", index, item, exc) from exc

    for index, item in enumerate(arguments.get("modules") or []):
        try:
            model.create_module(item["name"], members=item.get("members"))
            view = item.get("view")
            if view is not None:
                model.set_module_view(
                    item["name"],
                    x=view["x"],
                    y=view["y"],
                    width=view["width"],
                    height=view["height"],
                )
            style = item.get("style")
            if style is not None:
                model.set_module_style(
                    item["name"],
                    border_color=style.get("border_color"),
                    background=style.get("background"),
                    font_color=style.get("font_color"),
                    font_size=style.get("font_size"),
                    label_side=style.get("label_side"),
                )
            added["modules"] += 1
        except (KeyError, ValueError) as exc:
            raise fail("modules", index, item, exc) from exc

    return added


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

    def _finalize_batch(model: StellaModel, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run connector sync and validation on the still-unregistered model.

        Must run BEFORE registration/swap: anything that can raise has to
        fire while the model is still outside the session, or a failed
        batch would leave partial state behind.
        """
        extras: dict[str, Any] = {}
        if arguments.get("sync_connectors", True):
            extras["connector_sync"] = model.sync_connectors_from_equations()
        if arguments.get("validate", True):
            issues = validate_model(model)
            extras["validation"] = {
                "passed": not any(issue.severity == "error" for issue in issues),
                "issues": [validation_issue_to_dict(issue) for issue in issues],
            }
        return extras

    def _batch_response(
        action: str,
        model_id: str,
        model: StellaModel,
        added: dict[str, int],
        extras: dict[str, Any],
    ) -> ToolResponse:
        payload: dict[str, Any] = {
            "model_id": model_id,
            "added": added,
            "model": model_to_summary(model_id, model),
            **extras,
        }
        counts_text = ", ".join(f"{count} {kind}" for kind, count in added.items() if count)
        return success_result(
            f"{action} model_id={model_id}: added {counts_text or 'nothing'}",
            payload,
        )

    @register("build_model")
    def _handle_build_model(arguments: dict[str, Any]) -> ToolResponse:
        requested_id = arguments.get("model_id")
        if requested_id and requested_id in get_session_models().models:
            raise ValueError(f"model_id '{requested_id}' already exists in this session")

        sim = arguments.get("sim_specs") or {}
        start = float(sim.get("start", 0))
        stop = float(sim.get("stop", 100))
        dt = float(sim.get("dt", 0.25))
        if dt <= 0:
            raise ValueError("dt must be > 0")
        if stop <= start:
            raise ValueError("stop must be greater than start")

        model = StellaModel(name=arguments["name"])
        model.sim_specs.start = start
        model.sim_specs.stop = stop
        model.sim_specs.dt = dt
        model.sim_specs.method = sim.get("method", "Euler")
        model.sim_specs.time_units = sim.get("time_units", "Years")

        # Items, connector sync, and validation all run on the unregistered
        # model: any failure raises before registration, so a failed batch
        # leaves no session trace.
        added = _apply_batch_items(model, arguments, build_graphical_function)
        extras = _finalize_batch(model, arguments)
        model_id = set_current_model(model, model_id=requested_id)
        return _batch_response("Built", model_id, model, added, extras)

    @register("add_variables")
    def _handle_add_variables(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        # Atomic: apply items, sync, and validate on a scratch copy; swap
        # into the session only after everything that can raise has run.
        scratch = copy.deepcopy(model)
        added = _apply_batch_items(scratch, arguments, build_graphical_function)
        extras = _finalize_batch(scratch, arguments)
        get_session_models().models[model_id] = scratch
        return _batch_response("Updated", model_id, scratch, added, extras)

    @register("set_sim_specs")
    def _handle_set_sim_specs(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        specs = model.set_sim_specs(
            start=arguments.get("start"),
            stop=arguments.get("stop"),
            dt=arguments.get("dt"),
            method=arguments.get("method"),
            time_units=arguments.get("time_units"),
        )
        return success_result(
            f"Updated simulation specs for model_id={model_id}",
            {
                "model_id": model_id,
                "sim_specs": {
                    "start": specs.start,
                    "stop": specs.stop,
                    "dt": specs.dt,
                    "method": specs.method,
                    "time_units": specs.time_units,
                },
            },
        )

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

    @register("update_stock")
    def _handle_update_stock(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        stock = model.update_stock(
            name=arguments["name"],
            initial_value=arguments.get("initial_value"),
            units=arguments.get("units"),
            non_negative=arguments.get("non_negative"),
            x=arguments.get("x"),
            y=arguments.get("y"),
        )
        key = model._normalize_name(arguments["name"])
        return success_result(
            f"Updated stock '{stock.name}' in model_id={model_id}",
            {"model_id": model_id, "stock": stock_to_dict(key, stock)},
        )

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

    @register("update_flow")
    def _handle_update_flow(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        flow = model.update_flow(
            name=arguments["name"],
            equation=arguments.get("equation"),
            units=arguments.get("units"),
            non_negative=arguments.get("non_negative"),
            x=arguments.get("x"),
            y=arguments.get("y"),
            graphical_function=build_graphical_function(arguments.get("graphical_function")),
        )
        key = model._normalize_name(arguments["name"])
        return success_result(
            f"Updated flow '{flow.name}' in model_id={model_id}",
            {"model_id": model_id, "flow": flow_to_dict(model, key, flow)},
        )

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

    @register("update_aux")
    def _handle_update_aux(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        aux = model.update_aux(
            name=arguments["name"],
            equation=arguments.get("equation"),
            units=arguments.get("units"),
            x=arguments.get("x"),
            y=arguments.get("y"),
            graphical_function=build_graphical_function(arguments.get("graphical_function")),
        )
        key = model._normalize_name(arguments["name"])
        return success_result(
            f"Updated auxiliary '{aux.name}' in model_id={model_id}",
            {"model_id": model_id, "auxiliary": aux_to_dict(key, aux)},
        )

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

    @register("sync_connectors_from_equations")
    def _handle_sync_connectors_from_equations(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        summary = model.sync_connectors_from_equations()
        return success_result(
            (
                f"Synced connectors for model_id={model_id}: "
                f"added={summary['added']}, existing={summary['existing']}"
            ),
            {"model_id": model_id, **summary},
        )

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

    @register("render_diagram")
    def _handle_render_diagram(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        if arguments.get("auto_layout", True):
            # Mirror save_model's layout prep so a freshly built model
            # renders sensibly.
            model._auto_layout()
            if model.modules:
                model.auto_place_module_boxes(only_missing=True)
        else:
            model._recalculate_flow_points()
            model._calculate_connector_angles()
            model._position_orphan_flows()
        svg = render_model_svg(model)
        result: dict[str, Any] = {"model_id": model_id, "svg": svg, "filepath": None}
        filepath = arguments.get("filepath")
        if filepath:
            path = Path(filepath)
            if not path.suffix:
                path = path.with_suffix(".svg")
            path.write_text(svg, encoding="utf-8")
            result["filepath"] = str(path)
        suffix = f" -> {result['filepath']}" if result["filepath"] else ""
        return success_result(
            f"Rendered model_id={model_id} to SVG ({len(svg)} bytes){suffix}",
            result,
        )

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
            return success_result("No templates available.", {"templates": []})
        lines = ["Available templates:"]
        for info in templates:
            counts = f"{info.stocks}S/{info.flows}F/{info.auxiliaries}A"
            tags = ", ".join(info.tags) if info.tags else "-"
            lines.append(
                f"  - {info.name} [{info.source}] | title={info.title} | vars={counts} | tags={tags}"
            )
            if info.description:
                lines.append(f"    {info.description}")
        return success_result(
            "\n".join(lines),
            {"templates": [template_info_to_dict(info) for info in templates]},
        )

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
        return success_result("\n".join(lines), {"template": template_info_to_dict(info)})

    @register("load_template")
    def _handle_load_template(arguments: dict[str, Any]) -> ToolResponse:
        info, model = load_template_model(arguments["template_name"])
        model_id = set_current_model(model, model_id=arguments.get("model_id"))
        n_stocks = len(model.stocks)
        n_flows = len(model.flows)
        n_aux = len(model.auxs)
        return success_result(
            (
                f"Loaded template '{info.name}' [{info.source}] as model_id={model_id} "
                f"with {n_stocks} stocks, {n_flows} flows, {n_aux} auxiliaries"
            ),
            {
                "model_id": model_id,
                "template": template_info_to_dict(info),
                "model": model_to_summary(model_id, model),
            },
        )

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
        return success_result(
            f"Saved model_id={model_id} as template '{info.name}' at {info.path}",
            {"template": template_info_to_dict(info)},
        )

    @register("list_models")
    def _handle_list_models(arguments: dict[str, Any]) -> ToolResponse:
        session_models = get_session_models()
        if not session_models.models:
            return success_result("No models created in this session.", {"models": []})

        lines = ["Session models:"]
        for mid, model in sorted(session_models.models.items()):
            current = " (current)" if mid == session_models.current_model_id else ""
            lines.append(f"  - {mid}: {model.name}{current}")
        models_payload = [
            {
                "model_id": mid,
                "name": model.name,
                "current": mid == session_models.current_model_id,
            }
            for mid, model in sorted(session_models.models.items())
        ]
        return success_result("\n".join(lines), {"models": models_payload})

    @register("simulate")
    def _handle_simulate(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        result = run_simulation(
            model,
            overrides=arguments.get("overrides"),
            max_points=arguments.get("max_points", 101),
            include=arguments.get("include"),
            save_results_csv=arguments.get("save_results_csv"),
        )
        finals = ", ".join(
            f"{s['name']}={s['summary']['final']}" for s in result["series"]
        )
        warn_text = f" ({len(result['warnings'])} warnings)" if result["warnings"] else ""
        return success_result(
            f"Simulated model_id={model_id} from {result['sim_specs']['start']} to "
            f"{result['sim_specs']['stop']}{warn_text}. Final values: {finals}",
            {"model_id": model_id, **result},
        )

    @register("compare_scenarios")
    def _handle_compare_scenarios(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        result = compare_scenarios(
            model,
            scenarios=arguments.get("scenarios"),
            baseline=arguments.get("baseline"),
            include=arguments.get("include"),
            max_points=arguments.get("max_points", 101),
            save_comparison_csv=arguments.get("save_comparison_csv"),
        )
        lines = []
        for scenario in result["scenarios"]:
            deltas = ", ".join(
                f"{var} {d['final_abs']:+.4g}" if d["final_abs"] is not None else f"{var} n/a"
                for var, d in scenario["delta_vs_baseline"].items()
            )
            lines.append(f"{scenario['name']}: {deltas}" if deltas else scenario["name"])
        summary = "; ".join(lines) if lines else "none"
        return success_result(
            f"Compared {len(result['scenarios'])} scenario(s) for model_id={model_id} "
            f"vs baseline. Final deltas: {summary}",
            {"model_id": model_id, **result},
        )

    @register("sensitivity_analysis")
    def _handle_sensitivity_analysis(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        result = sensitivity_analysis(
            model,
            parameters=arguments.get("parameters"),
            output=arguments.get("output"),
            mode=arguments.get("mode", "oat"),
            max_runs=arguments.get("max_runs", 200),
            include_series=arguments.get("include_series", False),
            save_sweep_csv=arguments.get("save_sweep_csv"),
        )
        ranked = sorted(
            result["parameters"],
            key=lambda p: abs(p["elasticity"]) if p["elasticity"] is not None else -1.0,
            reverse=True,
        )
        ranking = ", ".join(
            f"{p['name']} (elasticity {p['elasticity']:+.3g})"
            if p["elasticity"] is not None
            else f"{p['name']} (n/a)"
            for p in ranked
        )
        return success_result(
            f"Swept {len(result['parameters'])} parameter(s) over {result['total_runs']} "
            f"run(s) for {result['output']['variable']} ({result['output']['metric']}) on "
            f"model_id={model_id}. Ranked by |elasticity|: {ranking}",
            {"model_id": model_id, **result},
        )

    @register("delete_model")
    def _handle_delete_model(arguments: dict[str, Any]) -> ToolResponse:
        session_models = get_session_models()
        model_id = arguments["model_id"]
        if model_id not in session_models.models:
            raise ValueError(f"Unknown model_id '{model_id}' for this session")
        del session_models.models[model_id]
        if session_models.current_model_id == model_id:
            session_models.current_model_id = None
        remaining = sorted(session_models.models)
        return success_result(
            f"Deleted model_id={model_id} from session ({len(remaining)} remaining). "
            "Saved .stmx files are not affected.",
            {
                "deleted": model_id,
                "remaining": remaining,
                "current_model_id": session_models.current_model_id,
            },
        )

    @register("inspect_model")
    def _handle_inspect_model(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        summary = model_to_summary(model_id, model)
        payload: dict[str, Any] = {"model": summary}
        include_validation = arguments.get("include_validation", True)
        if include_validation:
            issues = validate_model(model)
            payload["validation"] = {
                "passed": not any(issue.severity == "error" for issue in issues),
                "issues": [validation_issue_to_dict(issue) for issue in issues],
            }
        return success_result(
            (
                f"Model {model_id}: {model.name} "
                f"({len(model.stocks)} stocks, {len(model.flows)} flows, "
                f"{len(model.auxs)} auxiliaries)"
            ),
            payload,
        )

    @register("list_modules")
    def _handle_list_modules(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        if not model.modules:
            return success_result(
                f"No modules in model_id={model_id}.",
                {"model_id": model_id, "modules": []},
            )

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
        return success_result(
            "\n".join(lines),
            {
                "model_id": model_id,
                "modules": [
                    module_to_dict(model, key, model.modules[key])
                    for key in sorted(model.modules)
                ],
            },
        )

    @register("list_connectors")
    def _handle_list_connectors(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        if not model.connectors:
            return success_result(
                f"No connectors in model_id={model_id}.",
                {"model_id": model_id, "connectors": []},
            )

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
        return success_result(
            "\n".join(lines),
            {
                "model_id": model_id,
                "connectors": [
                    connector_to_dict(model, conn)
                    for conn in sorted(model.connectors, key=lambda c: c.uid)
                ],
            },
        )

    @register("validate_model")
    def _handle_validate_model(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        errors = validate_model(model)
        if not errors:
            return success_result(
                "Model validation passed with no errors or warnings.",
                {"model_id": model_id, "passed": True, "issues": []},
            )

        result_lines = ["Model validation results:"]
        for err in errors:
            prefix = "ERROR" if err.severity == "error" else "WARNING"
            result_lines.append(f"  [{prefix}] {err.category}: {err.message}")
        return success_result(
            "\n".join(result_lines),
            {
                "model_id": model_id,
                "passed": not any(err.severity == "error" for err in errors),
                "issues": [validation_issue_to_dict(err) for err in errors],
            },
        )

    @register("list_variables")
    def _handle_list_variables(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        lines = [f"Model: {model.name}", ""]
        lines.insert(1, f"model_id: {model_id}")
        lines.insert(2, "")

        if model.stocks:
            lines.append("Stocks:")
            for stock in model.stocks.values():
                lines.append(f"  - {stock.name} = {stock.initial_value} [{stock.units}]")
            lines.append("")

        if model.flows:
            lines.append("Flows:")
            for flow in model.flows.values():
                from_str = flow.from_stock or "external"
                to_str = flow.to_stock or "external"
                lines.append(f"  - {flow.name}: {from_str} -> {to_str} = {flow.equation}")
            lines.append("")

        if model.auxs:
            lines.append("Auxiliaries:")
            for aux in model.auxs.values():
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

        return success_result(
            "\n".join(lines),
            {
                "model_id": model_id,
                "variables": model_to_summary(model_id, model)["variables"],
            },
        )

    @register("get_model_xml")
    def _handle_get_model_xml(arguments: dict[str, Any]) -> ToolResponse:
        _, model = get_model(arguments.get("model_id"))
        # Export mutates layout state (auto-layout, flow points, connector
        # angles), so previewing XML works on a copy — the tool is read-only.
        preview = copy.deepcopy(model)
        xml = preview.to_xml(
            auto_layout=arguments.get("auto_layout", True),
            resolve_layout_violations=arguments.get("resolve_layout_violations", False),
            compat_mode=arguments.get("compat_mode", "permissive"),
        )
        # Truncate if too long
        if len(xml) > 10000:
            xml = xml[:10000] + "\n... (truncated)"
        output = [TextContent(type="text", text=xml)]
        if preview.last_export_warnings:
            output.append(
                TextContent(
                    type="text",
                    text=(
                        f"Compatibility warnings ({len(preview.last_export_warnings)}):\n"
                        + "\n".join(f"- {msg}" for msg in preview.last_export_warnings[:5])
                    ),
                )
            )
        return output
