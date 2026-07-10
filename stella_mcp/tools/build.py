"""Schemas for model construction, variables, and connector tools.

This file is schema-heavy because construction owns fifteen related public
tools. Keeping their literal contracts together makes changes reviewable as one
domain even though the file is above the project's approximate line guideline.
"""

from __future__ import annotations

from mcp.types import Tool

from .shared import SharedSchemas, build_shared_schemas


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
