"""Tests for variable lifecycle consistency (rename/delete operations)."""

import asyncio

import pytest
from mcp.types import CallToolResult

import stella_mcp.server as server_mod
from stella_mcp.xmile import StellaModel


def _tool_text(result):
    """Return first text content from either legacy list or CallToolResult responses."""
    if isinstance(result, CallToolResult):
        return result.content[0].text
    return result[0].text


def test_rename_stock_updates_equations_connectors_and_modules():
    """Renaming a stock should update all dependent references."""
    model = StellaModel("RenameStock")
    model.add_stock("Susceptible", "100")
    model.add_stock("Infected", "10")
    model.add_aux("beta", "0.3")
    model.add_flow(
        "infection",
        "beta * Susceptible",
        from_stock="Susceptible",
        to_stock="Infected",
    )
    model.add_aux("force", "infection + Susceptible")
    model.add_connector("Susceptible", "infection")
    model.create_module("Core", members=["Susceptible", "infection"])

    kind, new_key = model.rename_variable("Susceptible", "S")
    assert kind == "stock"
    assert new_key == "S"
    assert "S" in model.stocks
    assert "Susceptible" not in model.stocks
    assert model.flows["infection"].from_stock == "S"
    assert model.flows["infection"].equation == "beta * S"
    assert model.auxs["force"].equation == "infection + S"
    assert model.connectors[0].from_var == "S"
    assert "S" in model.modules["Core"].members
    assert "Susceptible" not in model.modules["Core"].members


def test_rename_flow_updates_stock_lists_and_equations():
    """Renaming a flow should update stock inflow/outflow lists and equation refs."""
    model = StellaModel("RenameFlow")
    model.add_stock("S1", "100")
    model.add_stock("S2", "10")
    model.add_flow("f", "1", from_stock="S1", to_stock="S2")
    model.add_aux("tracker", "f")
    model.add_connector("S1", "f")

    kind, new_key = model.rename_variable("f", "flow_renamed")
    assert kind == "flow"
    assert new_key == "flow_renamed"
    assert model.stocks["S1"].outflows == ["flow_renamed"]
    assert model.stocks["S2"].inflows == ["flow_renamed"]
    assert model.auxs["tracker"].equation == "flow_renamed"
    assert model.connectors[0].to_var == "flow_renamed"


def test_delete_flow_cleans_references():
    """Deleting a flow should remove stock links, connectors, and module membership."""
    model = StellaModel("DeleteFlow")
    model.add_stock("S1", "100")
    model.add_stock("S2", "10")
    model.add_aux("rate", "0.1")
    model.add_flow("f", "rate", from_stock="S1", to_stock="S2")
    model.add_connector("rate", "f")
    model.create_module("Core", members=["f", "rate"])

    summary = model.delete_variable("f")
    assert summary["kind"] == "flow"
    assert summary["removed_connectors"] == 1
    assert summary["removed_module_memberships"] == 1
    assert summary["detached_flows"] == 0
    assert "f" not in model.flows
    assert model.stocks["S1"].outflows == []
    assert model.stocks["S2"].inflows == []
    assert model.modules["Core"].members == ["rate"]


def test_delete_stock_requires_force_when_connected():
    """Deleting a connected stock should require force=True."""
    model = StellaModel("DeleteStock")
    model.add_stock("S1", "100")
    model.add_stock("S2", "10")
    model.add_flow("f", "1", from_stock="S1", to_stock="S2")

    with pytest.raises(ValueError, match="use force=true"):
        model.delete_variable("S1")

    summary = model.delete_variable("S1", force=True)
    assert summary["kind"] == "stock"
    assert summary["detached_flows"] == 1
    assert "S1" not in model.stocks
    assert model.flows["f"].from_stock is None


def test_delete_variable_blocks_equation_references():
    """Deleting a variable used by equations should be rejected."""
    model = StellaModel("DeleteWithRefs")
    model.add_aux("k", "0.1")
    model.add_flow("f", "k")

    with pytest.raises(ValueError, match="referenced in equations"):
        model.delete_variable("k")


def test_duplicate_variable_names_are_rejected():
    """Variable names should be unique across stock/flow/aux types."""
    model = StellaModel("UniqueNames")
    model.add_stock("S", "1")
    with pytest.raises(ValueError, match="already exists"):
        model.add_aux("S", "2")
    with pytest.raises(ValueError, match="already exists"):
        model.add_flow("S", "3")


def test_server_rename_and_delete_variable_tools(monkeypatch):
    """Server variable lifecycle tools should update and report consistent state."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 1616)

    asyncio.run(server_mod.call_tool("create_model", {"name": "LifecycleServer", "model_id": "m1"}))
    asyncio.run(server_mod.call_tool("add_stock", {"model_id": "m1", "name": "S1", "initial_value": "100"}))
    asyncio.run(server_mod.call_tool("add_stock", {"model_id": "m1", "name": "S2", "initial_value": "0"}))
    asyncio.run(
        server_mod.call_tool(
            "add_flow",
            {"model_id": "m1", "name": "f", "equation": "1", "from_stock": "S1", "to_stock": "S2"},
        )
    )

    renamed = asyncio.run(
        server_mod.call_tool(
            "rename_variable",
            {"model_id": "m1", "old_name": "S1", "new_name": "Source"},
        )
    )
    assert "Renamed stock 'S1' to 'Source'" in _tool_text(renamed)

    listed = asyncio.run(server_mod.call_tool("list_variables", {"model_id": "m1"}))
    assert "Source = 100" in _tool_text(listed)
    assert "f: Source -> S2" in _tool_text(listed)

    deleted = asyncio.run(
        server_mod.call_tool(
            "delete_variable",
            {"model_id": "m1", "name": "f"},
        )
    )
    assert "Deleted flow 'f'" in _tool_text(deleted)


def test_server_delete_variable_requires_force(monkeypatch):
    """delete_variable should return invalid_input when force is required."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 1717)

    asyncio.run(server_mod.call_tool("create_model", {"name": "LifecycleForce", "model_id": "m1"}))
    asyncio.run(server_mod.call_tool("add_stock", {"model_id": "m1", "name": "S1", "initial_value": "100"}))
    asyncio.run(server_mod.call_tool("add_stock", {"model_id": "m1", "name": "S2", "initial_value": "0"}))
    asyncio.run(
        server_mod.call_tool(
            "add_flow",
            {"model_id": "m1", "name": "f", "equation": "1", "from_stock": "S1", "to_stock": "S2"},
        )
    )

    result = asyncio.run(
        server_mod.call_tool("delete_variable", {"model_id": "m1", "name": "S1"})
    )
    assert isinstance(result, CallToolResult)
    assert result.isError is True
    assert result.structuredContent["error"]["code"] == "invalid_input"


def test_update_stock_flow_aux_and_sim_specs():
    """Model update methods should change only provided fields."""
    model = StellaModel("Update")
    model.add_stock("S", "100", units="people")
    model.add_aux("k", "0.1")
    model.add_flow("loss", "S * k", from_stock="S")

    model.set_sim_specs(start=1, stop=50, dt=0.5, method="RK4", time_units="Days")
    model.update_stock("S", initial_value="200", units="GtC", non_negative=False, x=10, y=20)
    model.update_aux("k", equation="0.2", units="1/day", x=30, y=40)
    model.update_flow("loss", equation="S * k * 2", units="GtC/day", non_negative=False, x=50, y=60)

    assert model.sim_specs.start == 1
    assert model.sim_specs.stop == 50
    assert model.sim_specs.dt == 0.5
    assert model.sim_specs.method == "RK4"
    assert model.sim_specs.time_units == "Days"
    assert model.stocks["S"].initial_value == "200"
    assert model.stocks["S"].non_negative is False
    assert model.auxs["k"].equation == "0.2"
    assert model.flows["loss"].equation == "S * k * 2"
