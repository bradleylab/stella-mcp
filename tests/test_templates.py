"""Tests for template storage and loading."""

import asyncio

import stella_mcp.server as server_mod
from stella_mcp.templates import (
    get_template_info,
    list_templates,
    load_template_model,
    resolve_template,
    save_user_template,
)
from stella_mcp.xmile import StellaModel


def _tool_text(result):
    """Return first text content from either legacy list or CallToolResult responses."""
    if hasattr(result, "content"):
        return result.content[0].text
    return result[0].text


def test_builtin_templates_are_available():
    """Built-in templates should be discoverable."""
    names = {info.name for info in list_templates() if info.source == "builtin"}
    assert "exponential_growth" in names
    assert "sir" in names
    assert "lotka_volterra" in names
    assert "carbon_cycle_2box" in names
    assert "nutrient_box_2box" in names


def test_builtin_templates_include_metadata_and_counts():
    """Built-in templates should expose metadata for discovery."""
    by_name = {info.name: info for info in list_templates(source="builtin")}
    sir = by_name["sir"]
    assert sir.title == "SIR"
    assert "epidemiology" in sir.tags
    assert sir.description
    assert sir.stocks > 0
    assert sir.flows > 0
    assert sir.auxiliaries > 0


def test_save_and_load_user_template(monkeypatch, tmp_path):
    """User template save/load should round-trip a model."""
    monkeypatch.setenv("STELLA_MCP_TEMPLATE_DIR", str(tmp_path))

    model = StellaModel("User Template Source")
    model.add_stock("Population", "100")
    saved = save_user_template(
        "My Template",
        model,
        description="Simple reusable starter",
        tags=["Demo", "Population"],
    )
    assert saved.name == "my_template"
    assert saved.source == "user"
    assert saved.path.exists()
    assert saved.title == "User Template Source"
    assert saved.description == "Simple reusable starter"
    assert saved.tags == ("demo", "population")

    info, loaded = load_template_model("my template")
    assert info.name == "my_template"
    assert info.source == "user"
    assert "Population" in loaded.stocks
    detailed = get_template_info("my template")
    assert detailed.description == "Simple reusable starter"
    assert detailed.tags == ("demo", "population")


def test_list_templates_filtering(monkeypatch, tmp_path):
    """Template discovery should support source, query, and tag filters."""
    monkeypatch.setenv("STELLA_MCP_TEMPLATE_DIR", str(tmp_path))
    alpha = StellaModel("Alpha Model")
    alpha.add_stock("A", "1")
    beta = StellaModel("Beta Model")
    beta.add_stock("B", "1")

    save_user_template(
        "Alpha Template",
        alpha,
        description="Alpha pattern for testing",
        tags=["demo", "alpha"],
    )
    save_user_template(
        "Beta Template",
        beta,
        description="Beta pattern for testing",
        tags=["demo", "beta"],
    )

    filtered = list_templates(source="user", query="alpha", tags=["demo", "alpha"])
    assert [info.name for info in filtered] == ["alpha_template"]


def test_user_template_overrides_builtin_name(monkeypatch, tmp_path):
    """User templates should override built-ins with the same canonical name."""
    monkeypatch.setenv("STELLA_MCP_TEMPLATE_DIR", str(tmp_path))

    model = StellaModel("Custom SIR")
    model.add_stock("OnlyStock", "1")
    save_user_template("sir", model, overwrite=True)

    resolved = resolve_template("sir")
    assert resolved.source == "user"
    _, loaded = load_template_model("sir")
    assert loaded.name == "Custom SIR"


def test_server_template_discovery_tools(monkeypatch, tmp_path):
    """Server template tools should expose metadata and discovery filters."""
    monkeypatch.setenv("STELLA_MCP_TEMPLATE_DIR", str(tmp_path))
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 1515)

    asyncio.run(server_mod.call_tool("create_model", {"name": "TemplateServer", "model_id": "m1"}))
    asyncio.run(server_mod.call_tool("add_stock", {"model_id": "m1", "name": "S", "initial_value": "100"}))
    asyncio.run(
        server_mod.call_tool(
            "save_as_template",
            {
                "model_id": "m1",
                "template_name": "Alpha Template",
                "description": "Alpha template for discovery tests.",
                "tags": ["demo", "smoke"],
                "overwrite": True,
            },
        )
    )

    listed = asyncio.run(
        server_mod.call_tool(
            "list_templates",
            {"source": "user", "query": "alpha", "tags": ["demo"]},
        )
    )
    assert "alpha_template [user]" in _tool_text(listed)
    assert "vars=1S/0F/0A" in _tool_text(listed)

    details = asyncio.run(
        server_mod.call_tool("get_template_info", {"template_name": "alpha_template"})
    )
    assert "description: Alpha template for discovery tests." in _tool_text(details)
    assert "tags: demo, smoke" in _tool_text(details)


def test_list_templates_tool_returns_structured_templates(monkeypatch, tmp_path):
    """Template discovery should expose structured template metadata."""
    monkeypatch.setenv("STELLA_MCP_TEMPLATE_DIR", str(tmp_path))
    result = asyncio.run(server_mod.call_tool("list_templates", {"source": "builtin"}))

    assert result.structuredContent["templates"]
    first = result.structuredContent["templates"][0]
    assert {"name", "source", "title", "stocks", "flows", "auxiliaries"}.issubset(first)
