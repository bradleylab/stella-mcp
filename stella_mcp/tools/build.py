"""Schemas for model construction, variables, and connector tools.

This file is schema-heavy because construction owns fifteen related public
tools. Keeping their literal contracts together makes changes reviewable as one
domain even though the file is above the project's approximate line guideline.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from mcp.types import TextContent, Tool

from ..model_snapshot import (
    aux_to_dict,
    flow_to_dict,
    model_to_summary,
    stock_to_dict,
    validation_issue_to_dict,
)
from ..tool_results import success_result
from ..validator import validate_model
from ..xmile import StellaModel
from .shared import (
    HandlerContext,
    RegisterTool,
    SharedSchemas,
    ToolResponse,
    apply_batch_items,
    build_shared_schemas,
)


def build_tools(shared: SharedSchemas | None = None) -> list[Tool]:
    """Build construction-domain tool descriptors in public catalog order."""
    schemas = shared or build_shared_schemas()
    model_id_property = schemas.model_id_property
    graphical_function_schema = schemas.graphical_function
    batch_item_properties = schemas.batch_item_properties
    return [
        Tool(
            name="create_model",
            description="Create a new Stella model with specified time settings",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Model name"},
                    "model_id": {
                        "type": "string",
                        "description": "Optional model ID to assign in this session",
                    },
                    "start": {
                        "type": "number",
                        "description": "Simulation start time",
                        "default": 0,
                    },
                    "stop": {
                        "type": "number",
                        "description": "Simulation stop time",
                        "default": 100,
                    },
                    "dt": {"type": "number", "description": "Time step", "default": 0.25},
                    "method": {
                        "type": "string",
                        "description": "Integration method (Euler or RK4)",
                        "default": "Euler",
                    },
                    "time_units": {
                        "type": "string",
                        "description": "Time units",
                        "default": "Years",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="build_model",
            description=(
                "Create and populate a model in one call: sim specs, stocks, "
                "auxiliaries, flows, connectors, and modules. All-or-nothing — "
                "on any item error nothing is registered and the error names the "
                "failing item (stage + index). Connector sync and validation run "
                "by default, so the response doubles as an inspection."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Model name"},
                    "model_id": {
                        "type": "string",
                        "description": "Optional model ID to assign in this session",
                    },
                    "sim_specs": {
                        "type": "object",
                        "description": "Simulation time settings",
                        "properties": {
                            "start": {
                                "type": "number",
                                "description": "Simulation start time",
                                "default": 0,
                            },
                            "stop": {
                                "type": "number",
                                "description": "Simulation stop time",
                                "default": 100,
                            },
                            "dt": {
                                "type": "number",
                                "description": "Time step",
                                "default": 0.25,
                            },
                            "method": {
                                "type": "string",
                                "description": "Integration method (Euler or RK4)",
                                "default": "Euler",
                            },
                            "time_units": {
                                "type": "string",
                                "description": "Time units",
                                "default": "Years",
                            },
                        },
                    },
                    **batch_item_properties,
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="add_variables",
            description=(
                "Add multiple stocks, auxiliaries, flows, connectors, and/or "
                "modules to an existing model in one call. All-or-nothing — on "
                "any item error the model is left unchanged and the error names "
                "the failing item (stage + index)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    **batch_item_properties,
                },
            },
        ),
        Tool(
            name="set_sim_specs",
            description="Update simulation time settings on an existing model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "start": {"type": "number", "description": "Simulation start time"},
                    "stop": {"type": "number", "description": "Simulation stop time"},
                    "dt": {"type": "number", "description": "Time step"},
                    "method": {
                        "type": "string",
                        "description": "Integration method (Euler or RK4)",
                    },
                    "time_units": {"type": "string", "description": "Time units"},
                },
            },
        ),
        Tool(
            name="add_stock",
            description="Add a stock (reservoir) to the current model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "name": {"type": "string", "description": "Stock name"},
                    "initial_value": {
                        "type": "string",
                        "description": "Initial value (number or equation)",
                    },
                    "units": {"type": "string", "description": "Units", "default": ""},
                    "non_negative": {
                        "type": "boolean",
                        "description": "Prevent negative values",
                        "default": True,
                    },
                    "x": {
                        "type": "number",
                        "description": (
                            "X position (optional, auto-positioned if not specified)"
                        ),
                    },
                    "y": {
                        "type": "number",
                        "description": (
                            "Y position (optional, auto-positioned if not specified)"
                        ),
                    },
                },
                "required": ["name", "initial_value"],
            },
        ),
        Tool(
            name="update_stock",
            description="Update stock fields while preserving relationships",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "name": {"type": "string", "description": "Stock name"},
                    "initial_value": {"type": "string", "description": "Initial value"},
                    "units": {"type": "string", "description": "Units"},
                    "non_negative": {
                        "type": "boolean",
                        "description": "Prevent negative values",
                    },
                    "x": {"type": "number", "description": "X position"},
                    "y": {"type": "number", "description": "Y position"},
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="add_flow",
            description="Add a flow between stocks in the current model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "name": {"type": "string", "description": "Flow name"},
                    "equation": {"type": "string", "description": "Flow rate equation"},
                    "units": {"type": "string", "description": "Units", "default": ""},
                    "from_stock": {
                        "type": "string",
                        "description": "Source stock (null for external source)",
                    },
                    "to_stock": {
                        "type": "string",
                        "description": "Destination stock (null for external sink)",
                    },
                    "non_negative": {
                        "type": "boolean",
                        "description": "Prevent negative values",
                        "default": True,
                    },
                    "x": {
                        "type": "number",
                        "description": (
                            "X position (optional, auto-positioned if not specified)"
                        ),
                    },
                    "y": {
                        "type": "number",
                        "description": (
                            "Y position (optional, auto-positioned if not specified)"
                        ),
                    },
                    "graphical_function": graphical_function_schema,
                },
                "required": ["name", "equation"],
            },
        ),
        Tool(
            name="update_flow",
            description="Update flow fields while preserving structural stock links",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "name": {"type": "string", "description": "Flow name"},
                    "equation": {"type": "string", "description": "Flow rate equation"},
                    "units": {"type": "string", "description": "Units"},
                    "non_negative": {
                        "type": "boolean",
                        "description": "Prevent negative values",
                    },
                    "x": {"type": "number", "description": "X position"},
                    "y": {"type": "number", "description": "Y position"},
                    "graphical_function": graphical_function_schema,
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="add_aux",
            description=(
                "Add an auxiliary variable (parameter or intermediate calculation) "
                "to the current model"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "name": {"type": "string", "description": "Variable name"},
                    "equation": {
                        "type": "string",
                        "description": "Equation or constant value",
                    },
                    "units": {"type": "string", "description": "Units", "default": ""},
                    "x": {
                        "type": "number",
                        "description": (
                            "X position (optional, auto-positioned if not specified)"
                        ),
                    },
                    "y": {
                        "type": "number",
                        "description": (
                            "Y position (optional, auto-positioned if not specified)"
                        ),
                    },
                    "graphical_function": graphical_function_schema,
                },
                "required": ["name", "equation"],
            },
        ),
        Tool(
            name="update_aux",
            description="Update auxiliary variable fields",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "name": {"type": "string", "description": "Variable name"},
                    "equation": {
                        "type": "string",
                        "description": "Equation or constant value",
                    },
                    "units": {"type": "string", "description": "Units"},
                    "x": {"type": "number", "description": "X position"},
                    "y": {"type": "number", "description": "Y position"},
                    "graphical_function": graphical_function_schema,
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="add_connector",
            description="Add a connector (dependency arrow) between variables",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "from_var": {"type": "string", "description": "Source variable name"},
                    "to_var": {
                        "type": "string",
                        "description": "Target variable name (the one using from_var)",
                    },
                },
                "required": ["from_var", "to_var"],
            },
        ),
        Tool(
            name="sync_connectors_from_equations",
            description=(
                "Add missing dependency connectors inferred from flow and auxiliary equations"
            ),
            inputSchema={
                "type": "object",
                "properties": {"model_id": model_id_property},
            },
        ),
        Tool(
            name="set_connector_routing",
            description="Set connector angle and/or explicit routing waypoints",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "connector_uid": {
                        "type": "integer",
                        "description": (
                            "Connector UID. Optional if from_var+to_var uniquely "
                            "identify a connector."
                        ),
                    },
                    "from_var": {
                        "type": "string",
                        "description": (
                            "Connector source variable name (used for lookup when "
                            "connector_uid is omitted)"
                        ),
                    },
                    "to_var": {
                        "type": "string",
                        "description": (
                            "Connector target variable name (used for lookup when "
                            "connector_uid is omitted)"
                        ),
                    },
                    "angle": {"type": "number", "description": "Connector angle in degrees"},
                    "angle_locked": {
                        "type": "boolean",
                        "description": "Whether to preserve the explicit connector angle",
                    },
                    "points_locked": {
                        "type": "boolean",
                        "description": "Whether to preserve explicit connector waypoints",
                    },
                    "points": {
                        "type": "array",
                        "description": "Optional connector waypoint list",
                        "items": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                            },
                            "required": ["x", "y"],
                        },
                    },
                },
            },
        ),
        Tool(
            name="rename_variable",
            description="Rename a stock/flow/aux and update dependent references",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "old_name": {"type": "string", "description": "Existing variable name"},
                    "new_name": {"type": "string", "description": "New variable name"},
                },
                "required": ["old_name", "new_name"],
            },
        ),
        Tool(
            name="delete_variable",
            description="Delete a stock/flow/aux and clean connectors/module membership",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "name": {"type": "string", "description": "Variable name to delete"},
                    "force": {
                        "type": "boolean",
                        "description": (
                            "Allow deleting stocks that still have connected flows "
                            "(flows are detached)"
                        ),
                        "default": False,
                    },
                },
                "required": ["name"],
            },
        ),
    ]


def register_handlers(register: RegisterTool, context: HandlerContext) -> None:
    """Register construction-domain handlers."""
    get_model = context.get_model
    set_current_model = context.set_current_model
    contains_session_model = context.contains_session_model
    replace_session_model = context.replace_session_model
    build_graphical_function = context.build_graphical_function

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
        """Finish work on an unregistered model so batch failure stays atomic."""
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
        if requested_id and contains_session_model(requested_id):
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

        added = apply_batch_items(model, arguments, build_graphical_function)
        extras = _finalize_batch(model, arguments)
        model_id = set_current_model(model, model_id=requested_id)
        return _batch_response("Built", model_id, model, added, extras)

    @register("add_variables")
    def _handle_add_variables(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        scratch = copy.deepcopy(model)
        added = apply_batch_items(scratch, arguments, build_graphical_function)
        extras = _finalize_batch(scratch, arguments)
        replace_session_model(model_id, scratch)
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
