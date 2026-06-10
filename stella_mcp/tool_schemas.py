"""MCP tool schema definitions."""

from __future__ import annotations

from mcp.types import Tool


def build_tool_definitions() -> list[Tool]:
    """Build MCP tool descriptors."""
    model_id_property = {
        "type": "string",
        "description": "Session-scoped model ID. Optional; defaults to the current model for this session.",
    }
    graphical_function_schema = {
        "type": "object",
        "description": "Graphical function (lookup table) definition",
        "properties": {
            "ypts": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "description": "Y values for the lookup table",
            },
            "xscale": {
                "type": "object",
                "description": "X scale when x points are evenly spaced",
                "properties": {
                    "min": {"type": "number"},
                    "max": {"type": "number"},
                },
                "required": ["min", "max"],
            },
            "xpts": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "description": "Explicit X values (same length as ypts)",
            },
            "yscale": {
                "type": "object",
                "description": "Optional Y scale for display",
                "properties": {
                    "min": {"type": "number"},
                    "max": {"type": "number"},
                },
                "required": ["min", "max"],
            },
            "type": {
                "type": "string",
                "enum": ["continuous", "discrete"],
                "description": "Graphical function type (e.g., continuous or discrete)",
            },
        },
        "required": ["ypts"],
        "oneOf": [
            {"required": ["xscale"]},
            {"required": ["xpts"]},
        ],
    }
    return [
        Tool(
            name="create_model",
            description="Create a new Stella model with specified time settings",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Model name"},
                    "model_id": {"type": "string", "description": "Optional model ID to assign in this session"},
                    "start": {"type": "number", "description": "Simulation start time", "default": 0},
                    "stop": {"type": "number", "description": "Simulation stop time", "default": 100},
                    "dt": {"type": "number", "description": "Time step", "default": 0.25},
                    "method": {"type": "string", "description": "Integration method (Euler or RK4)", "default": "Euler"},
                    "time_units": {"type": "string", "description": "Time units", "default": "Years"},
                },
                "required": ["name"],
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
                    "method": {"type": "string", "description": "Integration method (Euler or RK4)"},
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
                    "initial_value": {"type": "string", "description": "Initial value (number or equation)"},
                    "units": {"type": "string", "description": "Units", "default": ""},
                    "non_negative": {"type": "boolean", "description": "Prevent negative values", "default": True},
                    "x": {"type": "number", "description": "X position (optional, auto-positioned if not specified)"},
                    "y": {"type": "number", "description": "Y position (optional, auto-positioned if not specified)"},
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
                    "non_negative": {"type": "boolean", "description": "Prevent negative values"},
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
                    "from_stock": {"type": "string", "description": "Source stock (null for external source)"},
                    "to_stock": {"type": "string", "description": "Destination stock (null for external sink)"},
                    "non_negative": {"type": "boolean", "description": "Prevent negative values", "default": True},
                    "x": {"type": "number", "description": "X position (optional, auto-positioned if not specified)"},
                    "y": {"type": "number", "description": "Y position (optional, auto-positioned if not specified)"},
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
                    "non_negative": {"type": "boolean", "description": "Prevent negative values"},
                    "x": {"type": "number", "description": "X position"},
                    "y": {"type": "number", "description": "Y position"},
                    "graphical_function": graphical_function_schema,
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="add_aux",
            description="Add an auxiliary variable (parameter or intermediate calculation) to the current model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "name": {"type": "string", "description": "Variable name"},
                    "equation": {"type": "string", "description": "Equation or constant value"},
                    "units": {"type": "string", "description": "Units", "default": ""},
                    "x": {"type": "number", "description": "X position (optional, auto-positioned if not specified)"},
                    "y": {"type": "number", "description": "Y position (optional, auto-positioned if not specified)"},
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
                    "equation": {"type": "string", "description": "Equation or constant value"},
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
                    "to_var": {"type": "string", "description": "Target variable name (the one using from_var)"},
                },
                "required": ["from_var", "to_var"],
            },
        ),
        Tool(
            name="sync_connectors_from_equations",
            description="Add missing dependency connectors inferred from flow and auxiliary equations",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                },
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
                        "description": "Connector UID. Optional if from_var+to_var uniquely identify a connector.",
                    },
                    "from_var": {
                        "type": "string",
                        "description": "Connector source variable name (used for lookup when connector_uid is omitted)",
                    },
                    "to_var": {
                        "type": "string",
                        "description": "Connector target variable name (used for lookup when connector_uid is omitted)",
                    },
                    "angle": {
                        "type": "number",
                        "description": "Connector angle in degrees",
                    },
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
                        "description": "Allow deleting stocks that still have connected flows (flows are detached)",
                        "default": False,
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="create_module",
            description="Create a logical module/group for organizing variables",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "name": {"type": "string", "description": "Module name"},
                    "members": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional initial member variable names",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="add_to_module",
            description="Add variables to an existing module",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "module_name": {"type": "string", "description": "Existing module name"},
                    "members": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variable names to add",
                    },
                },
                "required": ["module_name", "members"],
            },
        ),
        Tool(
            name="remove_from_module",
            description="Remove variables from an existing module",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "module_name": {"type": "string", "description": "Existing module name"},
                    "members": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variable names to remove",
                    },
                },
                "required": ["module_name", "members"],
            },
        ),
        Tool(
            name="rename_module",
            description="Rename an existing module",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "module_name": {"type": "string", "description": "Existing module name"},
                    "new_name": {"type": "string", "description": "New module name"},
                },
                "required": ["module_name", "new_name"],
            },
        ),
        Tool(
            name="delete_module",
            description="Delete a module",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "module_name": {"type": "string", "description": "Existing module name"},
                },
                "required": ["module_name"],
            },
        ),
        Tool(
            name="set_module_view",
            description="Set explicit view box geometry for a module",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "module_name": {"type": "string", "description": "Module name"},
                    "x": {"type": "number", "description": "Center X"},
                    "y": {"type": "number", "description": "Center Y"},
                    "width": {"type": "number", "description": "Box width"},
                    "height": {"type": "number", "description": "Box height"},
                },
                "required": ["module_name", "x", "y", "width", "height"],
            },
        ),
        Tool(
            name="set_module_style",
            description="Set module box visual style in the diagram view",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "module_name": {"type": "string", "description": "Module name"},
                    "border_color": {"type": "string", "description": "Module border/line color"},
                    "background": {"type": "string", "description": "Module fill/background color"},
                    "font_color": {"type": "string", "description": "Module label font color"},
                    "font_size": {"type": "string", "description": "Module label font size (e.g., 9pt)"},
                    "label_side": {
                        "type": "string",
                        "description": "Module label position: top, bottom, left, or right",
                    },
                },
                "required": ["module_name"],
            },
        ),
        Tool(
            name="auto_place_module_boxes",
            description="Auto-place module view boxes around their member variables",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "padding": {
                        "type": "number",
                        "description": "Padding around module members in pixels",
                        "default": 40.0,
                    },
                    "min_width": {
                        "type": "number",
                        "description": "Minimum module box width in pixels",
                        "default": 180.0,
                    },
                    "min_height": {
                        "type": "number",
                        "description": "Minimum module box height in pixels",
                        "default": 120.0,
                    },
                    "only_missing": {
                        "type": "boolean",
                        "description": "Only place boxes for modules without explicit view geometry",
                        "default": False,
                    },
                },
            },
        ),
        Tool(
            name="save_model",
            description="Save the current model to a .stmx file",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "filepath": {"type": "string", "description": "Output file path (.stmx)"},
                    "auto_layout": {
                        "type": "boolean",
                        "description": "Whether to auto-layout before export",
                        "default": True,
                    },
                    "resolve_layout_violations": {
                        "type": "boolean",
                        "description": "Whether to run layout crossing/collision post-processing before export",
                        "default": False,
                    },
                    "compat_mode": {
                        "type": "string",
                        "enum": ["permissive", "strict"],
                        "description": "Compatibility mode for export checks",
                        "default": "permissive",
                    },
                },
                "required": ["filepath"],
            },
        ),
        Tool(
            name="read_model",
            description="Read an existing .stmx file and load it as the current model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string", "description": "Optional model ID to load into/assign in this session"},
                    "filepath": {"type": "string", "description": "Path to .stmx file"},
                    "compat_mode": {
                        "type": "string",
                        "enum": ["permissive", "strict"],
                        "description": "Compatibility mode for import parsing",
                        "default": "permissive",
                    },
                },
                "required": ["filepath"],
            },
        ),
        Tool(
            name="list_templates",
            description="List built-in and user-defined templates",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["builtin", "user"],
                        "description": "Optional source filter",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional case-insensitive search against template name/title/description",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional required tags (all must match)",
                    },
                },
            },
        ),
        Tool(
            name="get_template_info",
            description="Get detailed metadata for one template",
            inputSchema={
                "type": "object",
                "properties": {
                    "template_name": {"type": "string", "description": "Template name"},
                },
                "required": ["template_name"],
            },
        ),
        Tool(
            name="load_template",
            description="Load a template into the current session as a model",
            inputSchema={
                "type": "object",
                "properties": {
                    "template_name": {"type": "string", "description": "Template name"},
                    "model_id": {"type": "string", "description": "Optional model ID for the loaded template"},
                },
                "required": ["template_name"],
            },
        ),
        Tool(
            name="save_as_template",
            description="Save the current model as a user-defined template",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "template_name": {"type": "string", "description": "Template name to save"},
                    "description": {
                        "type": "string",
                        "description": "Optional template description for discovery",
                        "default": "",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags for discovery/filtering",
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Whether to overwrite an existing user template with the same name",
                        "default": False,
                    },
                },
                "required": ["template_name"],
            },
        ),
        Tool(
            name="list_models",
            description="List all model IDs available in the current session",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="delete_model",
            description=(
                "Remove a model from the current session. Saved .stmx files are "
                "not touched. model_id is required — there is deliberately no "
                "implicit 'delete current model'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Session model ID to remove",
                    },
                },
                "required": ["model_id"],
            },
        ),
        Tool(
            name="inspect_model",
            description="Return a structured summary of the current model for agent inspection",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "include_validation": {
                        "type": "boolean",
                        "description": "Include validation issues in structured output",
                        "default": True,
                    },
                },
            },
        ),
        Tool(
            name="list_modules",
            description="List modules/groups in the current model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                },
            },
        ),
        Tool(
            name="list_connectors",
            description="List connector metadata (uid, endpoints, angle, routing lock/points)",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                },
            },
        ),
        Tool(
            name="validate_model",
            description="Validate the current model for errors and warnings",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                },
            },
        ),
        Tool(
            name="list_variables",
            description="List all variables (stocks, flows, auxiliaries) in the current model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                },
            },
        ),
        Tool(
            name="get_model_xml",
            description="Get the XMILE XML representation of the current model (for preview)",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "auto_layout": {
                        "type": "boolean",
                        "description": "Whether to auto-layout before export",
                        "default": True,
                    },
                    "resolve_layout_violations": {
                        "type": "boolean",
                        "description": "Whether to run layout crossing/collision post-processing before export",
                        "default": False,
                    },
                    "compat_mode": {
                        "type": "string",
                        "enum": ["permissive", "strict"],
                        "description": "Compatibility mode for export checks",
                        "default": "permissive",
                    },
                },
            },
        ),
    ]
