"""Tests for template storage and loading."""

import asyncio
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import stella_mcp.server as server_mod
from stella_mcp.render_svg import render_model_svg
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


def test_sir_template_layout_fits_one_page_with_compact_connectors():
    """The built-in SIR model should open as a usable first-page diagram."""
    info, model = load_template_model("sir")
    root = ET.parse(info.path).getroot()
    view = root.find(".//{*}view")
    assert view is not None
    page_width = float(view.attrib["page_width"])
    page_height = float(view.attrib["page_height"])
    assert view.attrib["{http://iseesystems.com/XMILE}page_cols"] == "1"
    assert view.attrib["{http://iseesystems.com/XMILE}page_rows"] == "1"

    elements = {**model.stocks, **model.flows, **model.auxs}
    positions = {name: (element.x, element.y) for name, element in elements.items()}
    assert all(x is not None and 0 <= x <= page_width for x, _ in positions.values())
    assert all(y is not None and 0 <= y <= page_height for _, y in positions.values())

    susceptible_x, susceptible_y = positions["Susceptible"]
    infected_x, infected_y = positions["Infected"]
    recovered_x, recovered_y = positions["Recovered"]
    infection_x, _ = positions["infection"]
    recovery_x, _ = positions["recovery"]
    assert susceptible_x < infection_x < infected_x < recovery_x < recovered_x
    assert susceptible_y == infected_y == recovered_y

    for flow in model.flows.values():
        source = model.stocks[flow.from_stock]
        target = model.stocks[flow.to_stock]
        assert flow.points[0][0] == source.x + source.width / 2
        assert flow.points[-1][0] == target.x - target.width / 2
        assert all(point[1] == flow.y for point in flow.points)

    max_connector_length = min(page_width, page_height) / 2
    for connector in model.connectors:
        source = positions[connector.from_var]
        target = positions[connector.to_var]
        assert math.dist(source, target) <= max_connector_length


def test_readme_sir_diagram_matches_current_builtin_template():
    """The README diagram should not lag behind template identifier migrations."""
    _, model = load_template_model("sir")
    model._auto_layout()

    diagram = (Path(__file__).parents[1] / "docs/images/sir.svg").read_text(encoding="utf-8")

    assert diagram == render_model_svg(model)
    assert "transmission rate" in diagram
    assert "recovery rate" in diagram


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
    server_mod._clear_session_store()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 1515)

    asyncio.run(server_mod.call_tool("create_model", {"name": "TemplateServer", "model_id": "m1"}))
    asyncio.run(
        server_mod.call_tool("add_stock", {"model_id": "m1", "name": "S", "initial_value": "100"})
    )
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

    assert result.structured_content["templates"]
    first = result.structured_content["templates"][0]
    assert {"name", "source", "title", "stocks", "flows", "auxiliaries"}.issubset(first)
