"""Schemas and handlers for model inspection, validation, and deletion tools.

The module exceeds the approximate line guideline because eight declarative
schemas and their read-oriented handlers form one public inspection surface;
splitting that surface would add indirection without isolating more behavior.
"""

from __future__ import annotations

import copy
from typing import Any

from mcp.types import TextContent, Tool

from ..layout_quality import layout_report_to_dict, layout_warning_suffix
from ..model_snapshot import (
    connector_to_dict,
    model_to_summary,
    module_to_dict,
    validation_issue_to_dict,
)
from ..tool_results import success_result
from ..validator import validate_model
from .shared import (
    HandlerContext,
    RegisterTool,
    SharedSchemas,
    ToolResponse,
    build_shared_schemas,
)


def build_tools(shared: SharedSchemas | None = None) -> list[Tool]:
    """Build inspection-domain tool descriptors in public catalog order."""
    model_id_property = (shared or build_shared_schemas()).model_id_property
    return [
        Tool(
            name="list_models",
            description="List all model IDs available in the current session",
            inputSchema={"type": "object", "properties": {}},
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
            description=(
                "Return a structured summary of the current model for agent inspection"
            ),
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
                "properties": {"model_id": model_id_property},
            },
        ),
        Tool(
            name="list_connectors",
            description=(
                "List connector metadata (uid, endpoints, angle, routing lock/points)"
            ),
            inputSchema={
                "type": "object",
                "properties": {"model_id": model_id_property},
            },
        ),
        Tool(
            name="validate_model",
            description="Validate the current model for errors and warnings",
            inputSchema={
                "type": "object",
                "properties": {"model_id": model_id_property},
            },
        ),
        Tool(
            name="list_variables",
            description="List all variables (stocks, flows, auxiliaries) in the current model",
            inputSchema={
                "type": "object",
                "properties": {"model_id": model_id_property},
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
                        "description": (
                            "Whether to run layout crossing/collision post-processing "
                            "before export"
                        ),
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


def register_handlers(register: RegisterTool, context: HandlerContext) -> None:
    """Register inspection-domain handlers."""
    get_model = context.get_model
    list_session_models = context.list_session_models
    delete_session_model = context.delete_session_model

    @register("list_models")
    def _handle_list_models(arguments: dict[str, Any]) -> ToolResponse:
        session_models = list_session_models()
        if not session_models:
            return success_result("No models created in this session.", {"models": []})

        lines = ["Session models:"]
        for entry in session_models:
            current = " (current)" if entry.current else ""
            lines.append(f"  - {entry.model_id}: {entry.model.name}{current}")
        models_payload = [
            {
                "model_id": entry.model_id,
                "name": entry.model.name,
                "current": entry.current,
            }
            for entry in session_models
        ]
        return success_result("\n".join(lines), {"models": models_payload})

    @register("delete_model")
    def _handle_delete_model(arguments: dict[str, Any]) -> ToolResponse:
        model_id = arguments["model_id"]
        result = delete_session_model(model_id)
        return success_result(
            f"Deleted model_id={model_id} from session ({len(result.remaining)} remaining). "
            "Saved .stmx files are not affected.",
            {
                "deleted": result.deleted,
                "remaining": list(result.remaining),
                "current_model_id": result.current_model_id,
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
                ", ".join(model._display_name(member) for member in sorted(module.members))
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
        for connector in sorted(model.connectors, key=lambda item: item.uid):
            from_display = model._display_name(connector.from_var)
            to_display = model._display_name(connector.to_var)
            line = (
                f"  - uid={connector.uid}: {from_display} -> {to_display} | "
                f"angle={connector.angle} (locked={connector.angle_locked}) | "
                f"points={len(connector.points)} (locked={connector.points_locked})"
            )
            if connector.points:
                preview = ", ".join(f"({x:g},{y:g})" for x, y in connector.points[:3])
                if len(connector.points) > 3:
                    preview += ", ..."
                line += f" | pts={preview}"
            lines.append(line)
        return success_result(
            "\n".join(lines),
            {
                "model_id": model_id,
                "connectors": [
                    connector_to_dict(model, connector)
                    for connector in sorted(model.connectors, key=lambda item: item.uid)
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
        for error in errors:
            prefix = "ERROR" if error.severity == "error" else "WARNING"
            result_lines.append(f"  [{prefix}] {error.category}: {error.message}")
        return success_result(
            "\n".join(result_lines),
            {
                "model_id": model_id,
                "passed": not any(error.severity == "error" for error in errors),
                "issues": [validation_issue_to_dict(error) for error in errors],
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
                    ", ".join(
                        model._display_name(member) for member in sorted(module.members)
                    )
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
        model_id, model = get_model(arguments.get("model_id"))
        preview = copy.deepcopy(model)
        xml = preview.to_xml(
            auto_layout=arguments.get("auto_layout", True),
            resolve_layout_violations=arguments.get("resolve_layout_violations", False),
            compat_mode=arguments.get("compat_mode", "permissive"),
        )
        truncated = len(xml) > 10000
        if truncated:
            xml = xml[:10000] + "\n... (truncated)"
        result = success_result(
            xml,
            {
                "model_id": model_id,
                "xml": xml,
                "truncated": truncated,
                "compatibility_warnings": list(preview.last_export_warnings),
                "layout": layout_report_to_dict(preview.last_layout_result),
            },
        )
        if preview.last_export_warnings:
            result.content.append(
                TextContent(
                    type="text",
                    text=(
                        f"Compatibility warnings ({len(preview.last_export_warnings)}):\n"
                        + "\n".join(f"- {message}" for message in preview.last_export_warnings[:5])
                    ),
                )
            )
        layout_suffix = layout_warning_suffix(preview.last_layout_result)
        if layout_suffix:
            result.content.append(
                TextContent(type="text", text=f"Layout report{layout_suffix}")
            )
        return result
