"""Schemas for module lifecycle and layout-box tools."""

from __future__ import annotations

from mcp.types import Tool

from .shared import SharedSchemas, build_shared_schemas


def build_tools(shared: SharedSchemas | None = None) -> list[Tool]:
    """Build module-domain tool descriptors in public catalog order."""
    model_id_property = (shared or build_shared_schemas()).model_id_property
    return [
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
                        "description": (
                            "Only place boxes for modules without explicit view geometry"
                        ),
                        "default": False,
                    },
                },
            },
        ),
    ]
