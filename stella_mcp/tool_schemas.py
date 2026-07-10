"""Compatibility facade for domain-owned MCP tool schemas."""

from __future__ import annotations

from mcp.types import Tool, ToolAnnotations

from .tools import build, inspect, io, modules, simulation
from .tools.shared import build_shared_schemas

# Annotation policy remains centralized until the declared MCP minimum supports
# outputSchema. Every tool must appear in exactly one set.
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
    """Build the public tool catalog in its stable 0.10.0 order."""
    shared = build_shared_schemas()
    tools = [
        *build.build_tools(shared),
        *modules.build_tools(shared),
        *io.build_tools(shared),
        *simulation.build_tools(shared),
        *inspect.build_tools(shared),
    ]
    return _apply_tool_annotations(tools)
