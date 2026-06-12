"""MCP tool schema definitions."""

from __future__ import annotations

from mcp.types import Tool, ToolAnnotations

# Annotation policy. Every tool must appear in exactly one set — the test in
# tests/test_mcp_surface.py asserts these partition the full tool list, so a
# new tool added without an annotation decision fails CI.
_READ_ONLY_TOOLS = {
    "get_model_xml", "get_template_info", "inspect_model", "list_connectors",
    "list_models", "list_modules", "list_templates", "list_variables",
    "validate_model",
}
_DESTRUCTIVE_TOOLS = {"delete_model", "delete_module", "delete_variable"}
# Optional file writes only; safe to repeat with the same arguments.
_IDEMPOTENT_TOOLS = {
    "compare_scenarios", "render_diagram", "sensitivity_analysis", "simulate",
}
_MUTATING_TOOLS = {
    "add_aux", "add_connector", "add_flow", "add_stock", "add_to_module",
    "add_variables", "auto_place_module_boxes", "build_model", "create_model",
    "create_module", "load_template", "read_model", "remove_from_module",
    "rename_module", "rename_variable", "save_as_template", "save_model",
    "set_connector_routing", "set_module_style", "set_module_view",
    "set_sim_specs", "sync_connectors_from_equations", "update_aux",
    "update_flow", "update_stock",
}


def _annotation_for(name: str) -> ToolAnnotations | None:
    if name in _READ_ONLY_TOOLS:
        return ToolAnnotations(readOnlyHint=True)
    if name in _DESTRUCTIVE_TOOLS:
        return ToolAnnotations(readOnlyHint=False, destructiveHint=True)
    if name in _IDEMPOTENT_TOOLS:
        return ToolAnnotations(readOnlyHint=False, idempotentHint=True)
    if name in _MUTATING_TOOLS:
        return ToolAnnotations(readOnlyHint=False, destructiveHint=False)
    return None


def _apply_tool_annotations(tools: list[Tool]) -> list[Tool]:
    for tool in tools:
        tool.annotations = _annotation_for(tool.name)
    return tools


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
    stock_item_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Stock name"},
            "initial_value": {"type": "string", "description": "Initial value (number or equation)"},
            "units": {"type": "string", "description": "Units", "default": ""},
            "non_negative": {"type": "boolean", "description": "Prevent negative values", "default": True},
            "x": {"type": "number", "description": "X position (optional, auto-positioned if not specified)"},
            "y": {"type": "number", "description": "Y position (optional, auto-positioned if not specified)"},
        },
        "required": ["name", "initial_value"],
    }
    flow_item_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Flow name"},
            "equation": {"type": "string", "description": "Flow rate equation"},
            "units": {"type": "string", "description": "Units", "default": ""},
            "from_stock": {"type": "string", "description": "Source stock (omit for external source)"},
            "to_stock": {"type": "string", "description": "Destination stock (omit for external sink)"},
            "non_negative": {"type": "boolean", "description": "Prevent negative values", "default": True},
            "x": {"type": "number", "description": "X position (optional)"},
            "y": {"type": "number", "description": "Y position (optional)"},
            "graphical_function": graphical_function_schema,
        },
        "required": ["name", "equation"],
    }
    aux_item_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Variable name"},
            "equation": {"type": "string", "description": "Equation or constant value"},
            "units": {"type": "string", "description": "Units", "default": ""},
            "x": {"type": "number", "description": "X position (optional)"},
            "y": {"type": "number", "description": "Y position (optional)"},
            "graphical_function": graphical_function_schema,
        },
        "required": ["name", "equation"],
    }
    connector_item_schema = {
        "type": "object",
        "properties": {
            "from_var": {"type": "string", "description": "Source variable name"},
            "to_var": {"type": "string", "description": "Target variable name (the one using from_var)"},
        },
        "required": ["from_var", "to_var"],
    }
    module_item_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Module name"},
            "members": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Member variable names",
            },
            "view": {
                "type": "object",
                "description": "Optional explicit module box geometry",
                "properties": {
                    "x": {"type": "number", "description": "Center X"},
                    "y": {"type": "number", "description": "Center Y"},
                    "width": {"type": "number", "description": "Box width"},
                    "height": {"type": "number", "description": "Box height"},
                },
                "required": ["x", "y", "width", "height"],
            },
            "style": {
                "type": "object",
                "description": "Optional module box style",
                "properties": {
                    "border_color": {"type": "string", "description": "Module border/line color"},
                    "background": {"type": "string", "description": "Module fill/background color"},
                    "font_color": {"type": "string", "description": "Module label font color"},
                    "font_size": {"type": "string", "description": "Module label font size (e.g., 9pt)"},
                    "label_side": {
                        "type": "string",
                        "description": "Module label position: top, bottom, left, or right",
                    },
                },
            },
        },
        "required": ["name"],
    }
    batch_item_properties = {
        "stocks": {
            "type": "array",
            "items": stock_item_schema,
            "description": "Stocks to add (applied first)",
        },
        "auxs": {
            "type": "array",
            "items": aux_item_schema,
            "description": "Auxiliary variables to add (applied after stocks)",
        },
        "flows": {
            "type": "array",
            "items": flow_item_schema,
            "description": "Flows to add (applied after stocks and auxs)",
        },
        "connectors": {
            "type": "array",
            "items": connector_item_schema,
            "description": "Explicit connectors to add (applied after variables)",
        },
        "modules": {
            "type": "array",
            "items": module_item_schema,
            "description": "Modules to create (applied last)",
        },
        "sync_connectors": {
            "type": "boolean",
            "description": "Run sync_connectors_from_equations after applying items",
            "default": True,
        },
        "validate": {
            "type": "boolean",
            "description": "Include validation results in the response",
            "default": True,
        },
    }
    tools = [
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
                    "model_id": {"type": "string", "description": "Optional model ID to assign in this session"},
                    "sim_specs": {
                        "type": "object",
                        "description": "Simulation time settings",
                        "properties": {
                            "start": {"type": "number", "description": "Simulation start time", "default": 0},
                            "stop": {"type": "number", "description": "Simulation stop time", "default": 100},
                            "dt": {"type": "number", "description": "Time step", "default": 0.25},
                            "method": {"type": "string", "description": "Integration method (Euler or RK4)", "default": "Euler"},
                            "time_units": {"type": "string", "description": "Time units", "default": "Years"},
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
            name="render_diagram",
            description=(
                "Render the current model as an SVG stock-and-flow diagram. "
                "The SVG is returned inline (for clients without file access) "
                "and optionally written to a file. Defaults to running "
                "auto-layout first so freshly built models render sensibly."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "filepath": {
                        "type": "string",
                        "description": "Optional output path (.svg); parent directory must exist",
                    },
                    "auto_layout": {
                        "type": "boolean",
                        "description": "Run auto-layout before rendering (same semantics as save_model)",
                        "default": True,
                    },
                },
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
            name="simulate",
            description=(
                "Run the model and return downsampled time series with per-"
                "variable summaries (initial/final/min/max). Requires the "
                "optional pysd dependency (pip install 'stella-mcp[sim]'). "
                "Integration is Euler regardless of the model's method setting."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "overrides": {
                        "type": "object",
                        "description": (
                            "Constant parameter overrides keyed by variable name "
                            "(display or underscore form)"
                        ),
                        "additionalProperties": {"type": "number"},
                    },
                    "include": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables to report (default: all stocks)",
                    },
                    "max_points": {
                        "type": "integer",
                        "description": "Maximum points per returned series",
                        "default": 101,
                        "minimum": 2,
                    },
                    "save_results_csv": {
                        "type": "string",
                        "description": "Optional path to write the full results table as CSV",
                    },
                },
            },
        ),
        Tool(
            name="compare_scenarios",
            description=(
                "Run several named what-if scenarios (each a set of constant "
                "parameter overrides) against a baseline and report how each "
                "diverges: per-variable final/max absolute deltas and final "
                "percent change. Requires the optional pysd dependency "
                "(pip install 'stella-mcp[sim]')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "scenarios": {
                        "type": "array",
                        "minItems": 1,
                        "description": "Named override sets to compare (names must be unique)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Unique scenario label",
                                },
                                "overrides": {
                                    "type": "object",
                                    "additionalProperties": {"type": "number"},
                                    "description": "Constant parameter overrides for this scenario",
                                },
                            },
                            "required": ["name", "overrides"],
                        },
                    },
                    "baseline": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": (
                            "Override set to measure deltas against "
                            "(default: the unmodified model)"
                        ),
                    },
                    "include": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables to report and compare (default: all stocks)",
                    },
                    "max_points": {
                        "type": "integer",
                        "description": "Maximum points per returned series",
                        "default": 101,
                        "minimum": 2,
                    },
                    "save_comparison_csv": {
                        "type": "string",
                        "description": "Optional path to write a wide variable-by-scenario CSV",
                    },
                },
                "required": ["scenarios"],
            },
        ),
        Tool(
            name="sensitivity_analysis",
            description=(
                "One-at-a-time sensitivity: sweep each parameter across a range "
                "(holding the others at their baseline) and report how one chosen "
                "output metric responds, with a range slope and a "
                "baseline-normalized elasticity for ranking. Requires the "
                "optional pysd dependency (pip install 'stella-mcp[sim]')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "parameters": {
                        "type": "array",
                        "minItems": 1,
                        "description": "Parameters to sweep, each one at a time",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Parameter variable name",
                                },
                                "start": {
                                    "type": "number",
                                    "description": "Sweep start (use with stop + steps)",
                                },
                                "stop": {
                                    "type": "number",
                                    "description": "Sweep stop (use with start + steps)",
                                },
                                "steps": {
                                    "type": "integer",
                                    "minimum": 2,
                                    "description": "Number of evenly spaced sweep points",
                                },
                                "values": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 2,
                                    "description": (
                                        "Explicit sweep values "
                                        "(alternative to start/stop/steps)"
                                    ),
                                },
                            },
                            "required": ["name"],
                        },
                    },
                    "output": {
                        "type": "object",
                        "description": "The single output metric to track across the sweep",
                        "properties": {
                            "variable": {
                                "type": "string",
                                "description": "Output variable to reduce to a metric",
                            },
                            "metric": {
                                "type": "string",
                                "enum": ["final", "max", "min", "mean", "time_to_threshold"],
                                "default": "final",
                            },
                            "threshold": {
                                "type": "number",
                                "description": "Required when metric is time_to_threshold",
                            },
                        },
                        "required": ["variable"],
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["oat"],
                        "default": "oat",
                        "description": (
                            "Sweep design; only one-at-a-time is available "
                            "(grid/montecarlo reserved)"
                        ),
                    },
                    "max_runs": {
                        "type": "integer",
                        "default": 200,
                        "minimum": 1,
                        "description": (
                            "Hard cap on total swept runs; the call errors "
                            "rather than truncating a larger sweep"
                        ),
                    },
                    "include_series": {
                        "type": "boolean",
                        "default": False,
                        "description": "Also return each run's downsampled output series",
                    },
                    "save_sweep_csv": {
                        "type": "string",
                        "description": (
                            "Optional path to write the long "
                            "(parameter, value, metric) CSV"
                        ),
                    },
                },
                "required": ["parameters", "output"],
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
    return _apply_tool_annotations(tools)
