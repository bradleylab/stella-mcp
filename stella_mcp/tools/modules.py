"""Schemas and handlers for module lifecycle and layout-box tools.

This module is marginally above the approximate line guideline because eight
small tool contracts stay adjacent to their handler registrations; splitting
them again would make one domain require cross-file navigation.
"""

from __future__ import annotations

from typing import Any

from mcp.types import Tool

from ..model_snapshot import module_to_dict
from ..tool_results import success_result
from .shared import (
    HandlerContext,
    RegisterTool,
    SharedSchemas,
    ToolResponse,
    build_shared_schemas,
)


def build_tools(shared: SharedSchemas | None = None) -> list[Tool]:
    """Build module-domain tool descriptors in public catalog order."""
    model_id_property = (shared or build_shared_schemas()).model_id_property
    return [
        Tool(
            name="create_module",
            description="Create a logical module/group for organizing variables",
            input_schema={
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
            input_schema={
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
            input_schema={
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
            input_schema={
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
            input_schema={
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
            input_schema={
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
            input_schema={
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
            input_schema={
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


def register_handlers(register: RegisterTool, context: HandlerContext) -> None:
    """Register module-domain handlers."""
    get_model = context.get_model

    @register("create_module")
    def _handle_create_module(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        module = model.create_module(
            name=arguments["name"],
            members=arguments.get("members"),
        )
        key = model._normalize_name(module.name)
        return success_result(
            f"Created module '{module.name}' in model_id={model_id} with {len(module.members)} members",
            {"model_id": model_id, "module": module_to_dict(model, key, module)},
        )

    @register("add_to_module")
    def _handle_add_to_module(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        module = model.add_to_module(
            module_name=arguments["module_name"],
            members=arguments["members"],
        )
        key = model._normalize_name(module.name)
        return success_result(
            (
                f"Added {len(arguments['members'])} members to module '{module.name}' "
                f"in model_id={model_id} (total members: {len(module.members)})"
            ),
            {"model_id": model_id, "module": module_to_dict(model, key, module)},
        )

    @register("remove_from_module")
    def _handle_remove_from_module(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        module = model.remove_from_module(
            module_name=arguments["module_name"],
            members=arguments["members"],
        )
        key = model._normalize_name(module.name)
        return success_result(
            (
                f"Removed up to {len(arguments['members'])} members from module '{module.name}' "
                f"in model_id={model_id} (total members: {len(module.members)})"
            ),
            {"model_id": model_id, "module": module_to_dict(model, key, module)},
        )

    @register("rename_module")
    def _handle_rename_module(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        module = model.rename_module(
            module_name=arguments["module_name"],
            new_name=arguments["new_name"],
        )
        key = model._normalize_name(module.name)
        return success_result(
            f"Renamed module '{arguments['module_name']}' to '{module.name}' in model_id={model_id}",
            {"model_id": model_id, "module": module_to_dict(model, key, module)},
        )

    @register("delete_module")
    def _handle_delete_module(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        module = model.delete_module(arguments["module_name"])
        return success_result(
            f"Deleted module '{module.name}' from model_id={model_id}",
            {"model_id": model_id, "module_name": module.name, "deleted": module.name},
        )

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
        key = model._normalize_name(module.name)
        return success_result(
            (
                f"Set module view for '{module.name}' in model_id={model_id} "
                f"to center=({module.x}, {module.y}), size=({module.width}, {module.height})"
            ),
            {"model_id": model_id, "module": module_to_dict(model, key, module)},
        )

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
        key = model._normalize_name(module.name)
        return success_result(
            (
                f"Set module style for '{module.name}' in model_id={model_id}: "
                + ", ".join(style_parts)
            ),
            {"model_id": model_id, "module": module_to_dict(model, key, module)},
        )

    @register("auto_place_module_boxes")
    def _handle_auto_place_module_boxes(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        model.auto_place_module_boxes(
            padding=arguments.get("padding", 40.0),
            min_width=arguments.get("min_width", 180.0),
            min_height=arguments.get("min_height", 120.0),
            only_missing=arguments.get("only_missing", False),
        )
        return success_result(
            f"Auto-placed module boxes in model_id={model_id} for {len(model.modules)} modules",
            {
                "model_id": model_id,
                "modules": [
                    module_to_dict(model, key, model.modules[key])
                    for key in sorted(model.modules)
                ],
            },
        )
