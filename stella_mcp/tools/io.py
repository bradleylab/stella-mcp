"""Schemas and handlers for model I/O, rendering, and template tools.

The module exceeds the approximate line guideline because seven declarative
schemas are kept with their corresponding handlers. Its executable paths remain
independent and focused on the shared file/template boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.types import Tool

from ..layout_quality import layout_report_to_dict, layout_warning_suffix
from ..model_snapshot import model_to_summary, template_info_to_dict
from ..render_svg import render_model_svg
from ..templates import get_template_info, load_template_model, save_user_template
from ..templates import list_templates as list_available_templates
from ..tool_results import success_result
from ..xmile import parse_stmx
from .shared import (
    HandlerContext,
    RegisterTool,
    SharedSchemas,
    ToolResponse,
    build_shared_schemas,
)


def build_tools(shared: SharedSchemas | None = None) -> list[Tool]:
    """Build I/O-domain tool descriptors in public catalog order."""
    model_id_property = (shared or build_shared_schemas()).model_id_property
    return [
        Tool(
            name="save_model",
            description="Save the current model to a .stmx file",
            input_schema={
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
            input_schema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "filepath": {
                        "type": "string",
                        "description": (
                            "Optional output path (.svg); parent directory must exist"
                        ),
                    },
                    "auto_layout": {
                        "type": "boolean",
                        "description": (
                            "Run auto-layout before rendering (same semantics as save_model)"
                        ),
                        "default": True,
                    },
                },
            },
        ),
        Tool(
            name="read_model",
            description="Read an existing .stmx file and load it as the current model",
            input_schema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": (
                            "Optional model ID to load into/assign in this session"
                        ),
                    },
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
            input_schema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["builtin", "user"],
                        "description": "Optional source filter",
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "Optional case-insensitive search against template "
                            "name/title/description"
                        ),
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
            input_schema={
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
            input_schema={
                "type": "object",
                "properties": {
                    "template_name": {"type": "string", "description": "Template name"},
                    "model_id": {
                        "type": "string",
                        "description": "Optional model ID for the loaded template",
                    },
                },
                "required": ["template_name"],
            },
        ),
        Tool(
            name="save_as_template",
            description="Save the current model as a user-defined template",
            input_schema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "template_name": {
                        "type": "string",
                        "description": "Template name to save",
                    },
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
                        "description": (
                            "Whether to overwrite an existing user template with the same name"
                        ),
                        "default": False,
                    },
                },
                "required": ["template_name"],
            },
        ),
    ]


def register_handlers(register: RegisterTool, context: HandlerContext) -> None:
    """Register I/O-domain handlers."""
    get_model = context.get_model
    set_current_model = context.set_current_model
    compat_warning_suffix = context.compat_warning_suffix

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
        layout_suffix = layout_warning_suffix(model.last_layout_result)
        return success_result(
            f"Saved model_id={model_id} to {filepath}{warning_suffix}{layout_suffix}",
            {
                "model_id": model_id,
                "filepath": str(filepath),
                "compatibility_warnings": list(model.last_export_warnings),
                "layout": layout_report_to_dict(model.last_layout_result),
            },
        )

    @register("render_diagram")
    def _handle_render_diagram(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        if arguments.get("auto_layout", True):
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
            (
                f"Rendered model_id={model_id} to SVG ({len(svg)} bytes){suffix}"
                f"{layout_warning_suffix(model.last_layout_result)}"
            ),
            {
                **result,
                "layout": layout_report_to_dict(model.last_layout_result),
            },
        )

    @register("read_model")
    def _handle_read_model(arguments: dict[str, Any]) -> ToolResponse:
        filepath = Path(arguments["filepath"])
        model = parse_stmx(
            str(filepath),
            compat_mode=arguments.get("compat_mode", "permissive"),
        )
        model_id = set_current_model(model, model_id=arguments.get("model_id"))
        n_stocks = len(model.stocks)
        n_flows = len(model.flows)
        n_aux = len(model.auxs)
        warning_suffix = compat_warning_suffix(model.compatibility_warnings)
        return success_result(
            (
                f"Loaded model '{model.name}' as model_id={model_id} "
                f"with {n_stocks} stocks, {n_flows} flows, {n_aux} auxiliaries"
                f"{warning_suffix}"
            ),
            {
                "model_id": model_id,
                "filepath": str(filepath),
                "model": model_to_summary(model_id, model),
                "compatibility_warnings": list(model.compatibility_warnings),
            },
        )

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
