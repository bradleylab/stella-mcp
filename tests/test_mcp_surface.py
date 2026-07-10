"""Tests for MCP tool annotations, resources, and prompts."""

import asyncio
import hashlib
import json
import xml.etree.ElementTree as ET

import pytest

from stella_mcp import server as server_mod
from stella_mcp.tool_schemas import (
    _DESTRUCTIVE_TOOLS,
    _IDEMPOTENT_TOOLS,
    _MUTATING_TOOLS,
    _READ_ONLY_TOOLS,
    build_tool_definitions,
)

_TOOL_NAMES_0_10 = (
    "create_model",
    "build_model",
    "add_variables",
    "set_sim_specs",
    "add_stock",
    "update_stock",
    "add_flow",
    "update_flow",
    "add_aux",
    "update_aux",
    "add_connector",
    "sync_connectors_from_equations",
    "set_connector_routing",
    "rename_variable",
    "delete_variable",
    "create_module",
    "add_to_module",
    "remove_from_module",
    "rename_module",
    "delete_module",
    "set_module_view",
    "set_module_style",
    "auto_place_module_boxes",
    "save_model",
    "render_diagram",
    "read_model",
    "list_templates",
    "get_template_info",
    "load_template",
    "save_as_template",
    "simulate",
    "compare_scenarios",
    "sensitivity_analysis",
    "calibrate",
    "list_models",
    "delete_model",
    "inspect_model",
    "list_modules",
    "list_connectors",
    "validate_model",
    "list_variables",
    "get_model_xml",
)
_ANNOTATION_FIELDS = (
    "title",
    "readOnlyHint",
    "destructiveHint",
    "idempotentHint",
    "openWorldHint",
)
_TOOL_CATALOG_SHA256_0_10 = "10b28141403d3fee5f36816efccc5ee9115f08384d3eaab8c6d7ff25b7360b83"


def test_tool_catalog_matches_0_10_snapshot():
    tools = build_tool_definitions()
    payload = [
        {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.inputSchema,
            "annotations": None
            if tool.annotations is None
            else {
                field: getattr(tool.annotations, field, None)
                for field in _ANNOTATION_FIELDS
            },
        }
        for tool in tools
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    assert tuple(tool.name for tool in tools) == _TOOL_NAMES_0_10
    assert hashlib.sha256(canonical.encode()).hexdigest() == _TOOL_CATALOG_SHA256_0_10


def test_annotation_sets_partition_all_tools():
    """Every tool must be categorized — a new tool with no annotation
    decision fails this test."""
    all_names = {t.name for t in build_tool_definitions()}
    categorized = _READ_ONLY_TOOLS | _DESTRUCTIVE_TOOLS | _IDEMPOTENT_TOOLS | _MUTATING_TOOLS
    assert categorized == all_names, (
        f"uncategorized: {all_names - categorized}; "
        f"unknown in policy: {categorized - all_names}"
    )
    # Sets must be mutually exclusive.
    sets = [_READ_ONLY_TOOLS, _DESTRUCTIVE_TOOLS, _IDEMPOTENT_TOOLS, _MUTATING_TOOLS]
    for i, a in enumerate(sets):
        for b in sets[i + 1:]:
            assert not (a & b), f"overlap: {a & b}"


def test_tool_schemas_match_registered_handlers():
    schema_names = {tool.name for tool in build_tool_definitions()}
    handler_names = set(server_mod._TOOL_HANDLERS)

    assert schema_names == handler_names, (
        f"schemas without handlers: {schema_names - handler_names}; "
        f"handlers without schemas: {handler_names - schema_names}"
    )


def test_calibrate_schema_matches_optimizer_defaults():
    calibrate_tool = next(tool for tool in build_tool_definitions() if tool.name == "calibrate")
    properties = calibrate_tool.inputSchema["properties"]

    assert properties["max_nfev"]["default"] == 1000
    assert properties["maxiter"]["type"] == ["integer", "null"]
    assert properties["maxiter"]["default"] == 100
    assert properties["popsize"]["default"] == 15
    assert properties["seed"]["default"] == 0


def test_read_only_tools_are_annotated_read_only():
    tools = {t.name: t for t in build_tool_definitions()}
    for name in _READ_ONLY_TOOLS:
        assert tools[name].annotations.readOnlyHint is True


def test_destructive_and_idempotent_hints():
    tools = {t.name: t for t in build_tool_definitions()}
    for name in _DESTRUCTIVE_TOOLS:
        assert tools[name].annotations.destructiveHint is True
        assert tools[name].annotations.readOnlyHint is False
    for name in _IDEMPOTENT_TOOLS:
        assert tools[name].annotations.idempotentHint is True


def test_get_model_xml_is_read_only_in_practice(monkeypatch):
    """Annotated read-only -> must not mutate the session model."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 5000)
    asyncio.run(server_mod.call_tool("create_model", {"name": "X", "model_id": "x"}))
    asyncio.run(server_mod.call_tool(
        "add_stock", {"model_id": "x", "name": "S", "initial_value": "1"}
    ))
    model = server_mod._get_session_models().models["x"]
    assert model.stocks["S"].x is None  # unpositioned before preview

    asyncio.run(server_mod.call_tool("get_model_xml", {"model_id": "x"}))

    # get_model_xml ran auto_layout on a copy; session model stays unpositioned.
    assert model.stocks["S"].x is None


def _fresh_session(monkeypatch, key):
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: key)


def test_list_resources_includes_templates_and_models(monkeypatch):
    _fresh_session(monkeypatch, 5001)
    asyncio.run(server_mod.call_tool("create_model", {"name": "Mine", "model_id": "mine"}))

    resources = asyncio.run(server_mod.list_resources())
    uris = {str(r.uri) for r in resources}

    assert any(u.startswith("stella://templates/") for u in uris)
    assert "stella://templates/sir" in uris
    assert "stella://models/mine" in uris
    # All five builtin templates present.
    template_uris = {u for u in uris if u.startswith("stella://templates/")}
    assert len(template_uris) >= 5


def test_read_template_resource_parses_as_xml(monkeypatch):
    _fresh_session(monkeypatch, 5002)
    from pydantic import AnyUrl

    contents = asyncio.run(server_mod.read_resource(AnyUrl("stella://templates/sir")))
    assert len(contents) == 1
    ET.fromstring(contents[0].content)  # well-formed XMILE
    assert contents[0].mime_type == "application/xml"


def test_read_model_resource_does_not_mutate_session(monkeypatch):
    _fresh_session(monkeypatch, 5003)
    from pydantic import AnyUrl

    asyncio.run(server_mod.call_tool("create_model", {"name": "M", "model_id": "m"}))
    asyncio.run(server_mod.call_tool(
        "add_stock", {"model_id": "m", "name": "S", "initial_value": "1"}
    ))
    model = server_mod._get_session_models().models["m"]

    contents = asyncio.run(server_mod.read_resource(AnyUrl("stella://models/m")))

    ET.fromstring(contents[0].content)
    assert model.stocks["S"].x is None  # export ran on a copy


def test_model_resource_uri_round_trips_with_special_chars(monkeypatch):
    """A model_id with a space must round-trip: the advertised URI must be
    readable (regression — AnyUrl percent-encodes the space)."""
    _fresh_session(monkeypatch, 5005)
    from pydantic import AnyUrl

    asyncio.run(server_mod.call_tool("create_model", {"name": "M", "model_id": "my model"}))
    asyncio.run(server_mod.call_tool(
        "add_stock", {"model_id": "my model", "name": "S", "initial_value": "1"}
    ))

    resources = asyncio.run(server_mod.list_resources())
    model_uri = next(str(r.uri) for r in resources if r.name == "my model")

    # Read back using the exact URI that list_resources advertised.
    contents = asyncio.run(server_mod.read_resource(AnyUrl(model_uri)))
    ET.fromstring(contents[0].content)


def test_read_unknown_resource_raises(monkeypatch):
    _fresh_session(monkeypatch, 5004)
    from pydantic import AnyUrl

    with pytest.raises(ValueError):
        asyncio.run(server_mod.read_resource(AnyUrl("stella://templates/does_not_exist")))


def test_list_prompts_and_get_prompt():
    prompts = asyncio.run(server_mod.list_prompts())
    assert [p.name for p in prompts] == ["build-stella-model"]
    assert prompts[0].arguments[0].name == "description"

    result = asyncio.run(server_mod.get_prompt(
        "build-stella-model", {"description": "a predator-prey system"}
    ))
    text = result.messages[0].content.text
    assert "predator-prey system" in text
    assert "build_model" in text


def test_get_unknown_prompt_raises():
    with pytest.raises(ValueError):
        asyncio.run(server_mod.get_prompt("nope", None))
