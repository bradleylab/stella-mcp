"""Schemas for model I/O, rendering, and template tools."""

from __future__ import annotations

from mcp.types import Tool

from .shared import SharedSchemas, build_shared_schemas


def build_tools(shared: SharedSchemas | None = None) -> list[Tool]:
    """Build I/O-domain tool descriptors in public catalog order."""
    model_id_property = (shared or build_shared_schemas()).model_id_property
    return [
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
            inputSchema={
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
            inputSchema={
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
            inputSchema={
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
