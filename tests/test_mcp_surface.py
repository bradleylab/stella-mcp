"""Tests for MCP tool annotations, resources, and prompts."""

import asyncio
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
