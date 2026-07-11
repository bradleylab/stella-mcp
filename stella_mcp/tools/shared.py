"""Shared schema fragments and handler contracts for MCP tool domains.

This module intentionally groups declarative schema fragments with the small
set of cross-domain handler protocols and the atomic batch primitive. These are
the only concepts shared by multiple tool domains, so keeping them here avoids
both circular imports and a second generic-utilities layer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mcp.types import CallToolResult, TextContent

from ..session_store import SessionDeleteResult, SessionModelEntry
from ..tool_results import BatchItemError
from ..xmile import GraphicalFunction, StellaModel

ToolResponse = list[TextContent] | CallToolResult
ToolHandler = Callable[[dict[str, Any]], ToolResponse]
RegisterTool = Callable[[str], Callable[[ToolHandler], ToolHandler]]


@dataclass(frozen=True)
class HandlerContext:
    """Server-owned operations made available to domain handler registrars."""

    get_model: Callable[[str | None], tuple[str, StellaModel]]
    set_current_model: Callable[[StellaModel, str | None], str]
    list_session_models: Callable[[], tuple[SessionModelEntry, ...]]
    delete_session_model: Callable[[str], SessionDeleteResult]
    contains_session_model: Callable[[str], bool]
    replace_session_model: Callable[[str, StellaModel], None]
    build_graphical_function: Callable[
        [dict[str, Any] | None], GraphicalFunction | None
    ]
    compat_warning_suffix: Callable[[list[str]], str]


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


def apply_batch_items(
    model: StellaModel,
    arguments: dict[str, Any],
    build_graphical_function: Callable[
        [dict[str, Any] | None], GraphicalFunction | None
    ],
) -> dict[str, int]:
    """Apply ordered batch items and identify any failing stage atomically."""
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
        # Inputs are not schema-enforced at the server, and numbers are common
        # for constant equations. Anything else must fail before model mutation.
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
