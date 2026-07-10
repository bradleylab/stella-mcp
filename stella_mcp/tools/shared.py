"""Shared schema fragments and handler contracts for MCP tool domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SharedSchemas:
    """Schema fragments reused by more than one tool domain."""

    model_id_property: dict[str, Any]
    graphical_function: dict[str, Any]
    batch_item_properties: dict[str, Any]


def build_shared_schemas() -> SharedSchemas:
    """Build fresh shared fragments for one complete tool catalog."""
    model_id_property = {
        "type": "string",
        "description": (
            "Session-scoped model ID. Optional; defaults to the current model "
            "for this session."
        ),
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
                "description": "X position (optional, auto-positioned if not specified)",
            },
            "y": {
                "type": "number",
                "description": "Y position (optional, auto-positioned if not specified)",
            },
        },
        "required": ["name", "initial_value"],
    }
    flow_item_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Flow name"},
            "equation": {"type": "string", "description": "Flow rate equation"},
            "units": {"type": "string", "description": "Units", "default": ""},
            "from_stock": {
                "type": "string",
                "description": "Source stock (omit for external source)",
            },
            "to_stock": {
                "type": "string",
                "description": "Destination stock (omit for external sink)",
            },
            "non_negative": {
                "type": "boolean",
                "description": "Prevent negative values",
                "default": True,
            },
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
            "equation": {
                "type": "string",
                "description": "Equation or constant value",
            },
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
            "to_var": {
                "type": "string",
                "description": "Target variable name (the one using from_var)",
            },
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
                    "border_color": {
                        "type": "string",
                        "description": "Module border/line color",
                    },
                    "background": {
                        "type": "string",
                        "description": "Module fill/background color",
                    },
                    "font_color": {
                        "type": "string",
                        "description": "Module label font color",
                    },
                    "font_size": {
                        "type": "string",
                        "description": "Module label font size (e.g., 9pt)",
                    },
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
    return SharedSchemas(
        model_id_property=model_id_property,
        graphical_function=graphical_function_schema,
        batch_item_properties=batch_item_properties,
    )
