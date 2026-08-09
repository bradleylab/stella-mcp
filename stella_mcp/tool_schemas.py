"""Compatibility facade for domain-owned MCP tool schemas."""

from __future__ import annotations

from typing import Any

from mcp.types import Tool, ToolAnnotations

from .tools import build, inspect, io, modules, simulation
from .tools.shared import build_shared_schemas

# Annotation policy remains centralized. Every tool must appear in exactly one set.
_READ_ONLY_TOOLS = {
    "get_model_xml", "get_template_info", "inspect_model", "list_connectors",
    "list_models", "list_modules", "list_templates", "list_variables",
    "validate_model",
}
_DESTRUCTIVE_TOOLS = {"delete_model", "delete_module", "delete_variable"}
# Optional file writes only; safe to repeat with the same arguments.
# calibrate qualifies because differential_evolution is always seeded
# (seed=None is rejected), so a call is deterministic given its inputs; exposing
# an unseeded/random mode later would require moving it out of this set.
_IDEMPOTENT_TOOLS = {
    "calibrate", "compare_scenarios", "render_diagram", "sensitivity_analysis",
    "simulate",
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
_DESTRUCTIVE_TOOLS.add("revoke_workspace")
_MUTATING_TOOLS.add("create_workspace")

_WORKSPACE_FREE_TOOLS = {
    "create_workspace",
    "revoke_workspace",
    "list_templates",
    "get_template_info",
}

_WORKSPACE_PROPERTY = {
    "type": "string",
    "description": (
        "Opaque application workspace handle. Required by MCP 2026-07-28 clients; "
        "supported legacy stdio clients may omit it to use the process-local "
        "compatibility workspace."
    ),
}

# Top-level success contracts. Nested domain records intentionally remain open
# in 0.14.0 so additive snapshot fields do not become accidental breaking
# changes; required top-level fields still make omissions client-detectable.
_OUTPUT_REQUIRED: dict[str, tuple[str, ...]] = {
    "create_model": ("model_id", "model"),
    "build_model": ("model_id", "added", "model"),
    "add_variables": ("model_id", "added", "model"),
    "set_sim_specs": ("model_id", "sim_specs"),
    "add_stock": ("model_id", "stock"),
    "update_stock": ("model_id", "stock"),
    "add_flow": ("model_id", "flow"),
    "update_flow": ("model_id", "flow"),
    "add_aux": ("model_id", "auxiliary"),
    "update_aux": ("model_id", "auxiliary"),
    "add_connector": ("model_id", "connector"),
    "sync_connectors_from_equations": ("model_id", "added", "existing"),
    "set_connector_routing": ("model_id", "connector"),
    "rename_variable": ("model_id", "kind", "old_name", "new_name", "new_key"),
    "delete_variable": ("model_id", "name", "kind"),
    "create_module": ("model_id", "module"),
    "add_to_module": ("model_id", "module"),
    "remove_from_module": ("model_id", "module"),
    "rename_module": ("model_id", "module"),
    "delete_module": ("model_id", "module_name", "deleted"),
    "set_module_view": ("model_id", "module"),
    "set_module_style": ("model_id", "module"),
    "auto_place_module_boxes": ("model_id", "modules"),
    "save_model": ("model_id", "filepath", "compatibility_warnings", "layout"),
    "render_diagram": ("model_id", "svg", "filepath", "layout"),
    "read_model": ("model_id", "filepath", "model", "compatibility_warnings"),
    "list_templates": ("templates",),
    "get_template_info": ("template",),
    "load_template": ("model_id", "template", "model"),
    "save_as_template": ("template",),
    "simulate": ("model_id",),
    "compare_scenarios": ("model_id",),
    "sensitivity_analysis": ("model_id",),
    "calibrate": ("model_id",),
    "list_models": ("models",),
    "delete_model": ("deleted", "remaining", "current_model_id"),
    "inspect_model": ("model",),
    "list_modules": ("model_id", "modules"),
    "list_connectors": ("model_id", "connectors"),
    "validate_model": ("model_id", "passed", "issues"),
    "list_variables": ("model_id", "variables"),
    "get_model_xml": ("model_id", "xml", "truncated", "compatibility_warnings", "layout"),
    "create_workspace": ("workspace_id",),
    "revoke_workspace": ("workspace_id", "revoked"),
}

_STRING_OUTPUT_FIELDS = {
    "workspace_id", "model_id", "filepath", "svg", "xml", "kind", "old_name",
    "new_name", "new_key", "name", "module_name", "deleted",
}
_BOOLEAN_OUTPUT_FIELDS = {"passed", "truncated", "revoked"}
_ARRAY_OUTPUT_FIELDS = {
    "issues", "models", "modules", "connectors", "templates", "remaining",
    "compatibility_warnings",
}


def _output_property(tool_name: str, field: str) -> dict[str, Any]:
    if tool_name == "sync_connectors_from_equations" and field in {"added", "existing"}:
        return {"type": "integer"}
    if field == "filepath" or field == "current_model_id":
        return {"type": ["string", "null"]}
    if field == "layout":
        return {"type": ["object", "null"]}
    if field in _STRING_OUTPUT_FIELDS:
        return {"type": "string"}
    if field in _BOOLEAN_OUTPUT_FIELDS:
        return {"type": "boolean"}
    if field in _ARRAY_OUTPUT_FIELDS:
        return {"type": "array"}
    return {"type": "object"}


def _apply_output_contracts(tools: list[Tool]) -> list[Tool]:
    for tool in tools:
        required = _OUTPUT_REQUIRED[tool.name]
        tool.output_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                field: _output_property(tool.name, field) for field in required
            },
            "required": list(required),
        }
    return tools


def _annotation_for(name: str) -> ToolAnnotations | None:
    if name in _READ_ONLY_TOOLS:
        return ToolAnnotations(read_only_hint=True)
    if name in _DESTRUCTIVE_TOOLS:
        return ToolAnnotations(read_only_hint=False, destructive_hint=True)
    if name in _IDEMPOTENT_TOOLS:
        return ToolAnnotations(read_only_hint=False, idempotent_hint=True)
    if name in _MUTATING_TOOLS:
        return ToolAnnotations(read_only_hint=False, destructive_hint=False)
    return None


def _apply_tool_annotations(tools: list[Tool]) -> list[Tool]:
    for tool in tools:
        tool.annotations = _annotation_for(tool.name)
    return tools


def _apply_workspace_routing(
    tools: list[Tool], *, require_workspace_id: bool
) -> list[Tool]:
    """Add protocol-era-specific workspace routing to stateful tools."""
    for tool in tools:
        if tool.name not in _WORKSPACE_FREE_TOOLS:
            tool.input_schema.setdefault("properties", {})["workspace_id"] = dict(
                _WORKSPACE_PROPERTY
            )
            if require_workspace_id:
                required = tool.input_schema.setdefault("required", [])
                if "workspace_id" not in required:
                    required.append("workspace_id")
    return tools


def _workspace_tools() -> list[Tool]:
    """Return lifecycle tools appended after the established catalog."""
    return [
        Tool(
            name="create_workspace",
            description=(
                "Create an isolated application workspace for stateful Stella tool calls. "
                "The returned ID routes state and is not an authorization credential."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "ttl_seconds": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description": (
                            "Optional caller-selected lifetime. Omit for process-lifetime state."
                        ),
                    }
                },
            },
        ),
        Tool(
            name="revoke_workspace",
            description="Revoke a workspace and discard all models stored in it.",
            input_schema={
                "type": "object",
                "properties": {"workspace_id": dict(_WORKSPACE_PROPERTY)},
                "required": ["workspace_id"],
            },
        ),
    ]


def build_tool_definitions(*, require_workspace_id: bool = False) -> list[Tool]:
    """Build the public tool catalog in its stable 0.10.0 order."""
    shared = build_shared_schemas()
    tools = [
        *build.build_tools(shared),
        *modules.build_tools(shared),
        *io.build_tools(shared),
        *simulation.build_tools(shared),
        *inspect.build_tools(shared),
    ]
    tools.extend(_workspace_tools())
    return _apply_tool_annotations(
        _apply_output_contracts(
            _apply_workspace_routing(
                tools, require_workspace_id=require_workspace_id
            )
        )
    )
