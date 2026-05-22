"""Tests for server session model state and graphical function validation."""

import asyncio
from pathlib import Path

import pytest
from mcp.types import CallToolResult

import stella_mcp.server as server_mod
from stella_mcp.xmile import StellaModel


def _tool_text(result):
    """Return first text content from either legacy list or CallToolResult responses."""
    if isinstance(result, CallToolResult):
        return result.content[0].text
    return result[0].text


def test_session_scoped_models_are_isolated(monkeypatch):
    """Models with the same model_id in different sessions should not collide."""
    server_mod._session_models.clear()

    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 101)
    server_mod._set_current_model(StellaModel("SessionOne"), model_id="m1")
    _, model_one = server_mod.get_model("m1")
    assert model_one.name == "SessionOne"

    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 202)
    with pytest.raises(ValueError, match="Unknown model_id"):
        server_mod.get_model("m1")
    server_mod._set_current_model(StellaModel("SessionTwo"), model_id="m1")
    _, model_two = server_mod.get_model("m1")
    assert model_two.name == "SessionTwo"

    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 101)
    _, model_one_again = server_mod.get_model("m1")
    assert model_one_again.name == "SessionOne"


def test_graphical_function_requires_exactly_one_x_domain():
    """Graphical functions must define exactly one x-domain strategy."""
    with pytest.raises(ValueError, match="exactly one of xscale or xpts"):
        server_mod.build_graphical_function({"ypts": [0, 1]})

    with pytest.raises(ValueError, match="exactly one of xscale or xpts"):
        server_mod.build_graphical_function(
            {"ypts": [0, 1], "xscale": {"min": 0, "max": 1}, "xpts": [0, 1]}
        )


def test_graphical_function_validates_lengths_and_scales():
    """Graphical function table lengths and scale bounds should be validated."""
    with pytest.raises(ValueError, match="same length"):
        server_mod.build_graphical_function({"ypts": [0, 1, 2], "xpts": [0, 1]})

    with pytest.raises(ValueError, match="min < max"):
        server_mod.build_graphical_function(
            {"ypts": [0, 1], "xscale": {"min": 2, "max": 1}}
        )

    with pytest.raises(ValueError, match="type must be one of"):
        server_mod.build_graphical_function(
            {"ypts": [0, 1], "xpts": [0, 1], "type": "invalid"}
        )


def test_graphical_function_valid_payload():
    """A valid graphical function payload should parse successfully."""
    gf = server_mod.build_graphical_function(
        {
            "ypts": [0, 1, 4],
            "xpts": [0, 1, 2],
            "yscale": {"min": 0, "max": 5},
            "type": "continuous",
        }
    )
    assert gf is not None
    assert gf.gf_type == "continuous"
    assert gf.xpts == [0.0, 1.0, 2.0]
    assert gf.ypts == [0.0, 1.0, 4.0]


def test_list_models_reports_current(monkeypatch):
    """list_models should return all session model IDs and mark current."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 303)

    server_mod._set_current_model(StellaModel("First"), model_id="m1")
    server_mod._set_current_model(StellaModel("Second"), model_id="m2")

    result = asyncio.run(server_mod.call_tool("list_models", {}))
    text = _tool_text(result)
    assert "m1: First" in text
    assert "m2: Second (current)" in text


def test_list_models_empty_session(monkeypatch):
    """list_models should return a clear message when session has no models."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 404)

    result = asyncio.run(server_mod.call_tool("list_models", {}))
    assert _tool_text(result) == "No models created in this session."


def test_get_model_xml_respects_auto_layout_flag(monkeypatch):
    """get_model_xml(auto_layout=False) should skip auto-positioning."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 505)
    server_mod._set_current_model(StellaModel("NoLayout"), model_id="m1")
    model_id, model = server_mod.get_model("m1")
    model.add_stock("S", "100")

    result = asyncio.run(
        server_mod.call_tool(
            "get_model_xml",
            {"model_id": model_id, "auto_layout": False},
        )
    )
    text = _tool_text(result)
    assert '<stock x="0" y="0"' in text


def test_save_model_respects_auto_layout_flag(monkeypatch, tmp_path: Path):
    """save_model(auto_layout=False) should write XML without auto-positioning."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 606)
    server_mod._set_current_model(StellaModel("NoLayoutSave"), model_id="m1")
    model_id, model = server_mod.get_model("m1")
    model.add_stock("S", "100")
    outpath = tmp_path / "nolayout_export.stmx"

    asyncio.run(
        server_mod.call_tool(
            "save_model",
            {"model_id": model_id, "filepath": str(outpath), "auto_layout": False},
        )
    )
    xml = outpath.read_text(encoding="utf-8")
    assert '<stock x="0" y="0"' in xml


def test_get_model_xml_resolve_layout_violations_flag(monkeypatch):
    """get_model_xml(resolve_layout_violations=True) should invoke resolver."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 707)
    server_mod._set_current_model(StellaModel("Resolve"), model_id="m1")
    model_id, model = server_mod.get_model("m1")
    model.add_stock("S", "100")

    called = {"value": False}

    def fake_resolver(max_iterations: int = 10):
        called["value"] = True

    model._resolve_layout_violations = fake_resolver  # type: ignore[attr-defined]

    asyncio.run(
        server_mod.call_tool(
            "get_model_xml",
            {"model_id": model_id, "resolve_layout_violations": True},
        )
    )
    assert called["value"] is True


def test_unknown_tool_returns_structured_error():
    """Unknown tool should return isError with stable error code."""
    result = asyncio.run(server_mod.call_tool("does_not_exist", {}))
    assert isinstance(result, CallToolResult)
    assert result.isError is True
    assert result.structuredContent["error"]["code"] == "unknown_tool"


def test_model_not_found_returns_structured_error(monkeypatch):
    """Model lookup errors should return model_not_found code."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 808)

    result = asyncio.run(server_mod.call_tool("list_variables", {"model_id": "missing"}))
    assert isinstance(result, CallToolResult)
    assert result.isError is True
    assert result.structuredContent["error"]["code"] == "model_not_found"
    assert result.structuredContent["error"]["category"] == "user_input"


def test_internal_error_returns_structured_error(monkeypatch):
    """Unexpected exceptions should return internal_error."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 909)
    server_mod._set_current_model(StellaModel("Boom"), model_id="m1")
    _, model = server_mod.get_model("m1")

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    model.to_xml = boom  # type: ignore[method-assign]
    result = asyncio.run(server_mod.call_tool("get_model_xml", {"model_id": "m1"}))
    assert isinstance(result, CallToolResult)
    assert result.isError is True
    assert result.structuredContent["error"]["code"] == "internal_error"
    assert result.structuredContent["error"]["category"] == "internal"


def test_call_tool_session_isolation_end_to_end(monkeypatch):
    """End-to-end tool calls should isolate model registries per session key."""
    server_mod._session_models.clear()
    state = {"session_key": 1}
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: state["session_key"])

    # Session 1
    result1 = asyncio.run(
        server_mod.call_tool("create_model", {"name": "SessionOneModel", "model_id": "shared"})
    )
    assert "model_id=shared" in _tool_text(result1)
    list1 = asyncio.run(server_mod.call_tool("list_models", {}))
    assert "shared: SessionOneModel (current)" in _tool_text(list1)

    # Session 2 with same model_id should be independent
    state["session_key"] = 2
    result2 = asyncio.run(
        server_mod.call_tool("create_model", {"name": "SessionTwoModel", "model_id": "shared"})
    )
    assert "model_id=shared" in _tool_text(result2)
    list2 = asyncio.run(server_mod.call_tool("list_models", {}))
    assert "shared: SessionTwoModel (current)" in _tool_text(list2)

    # Back to Session 1, original model should still be present
    state["session_key"] = 1
    list1_again = asyncio.run(server_mod.call_tool("list_models", {}))
    assert "shared: SessionOneModel (current)" in _tool_text(list1_again)


def test_list_templates_tool_includes_builtin():
    """list_templates tool should expose built-in templates."""
    result = asyncio.run(server_mod.call_tool("list_templates", {}))
    text = _tool_text(result)
    assert "exponential_growth [builtin]" in text
    assert "sir [builtin]" in text


def test_set_connector_routing_tool_updates_connector_points(monkeypatch):
    """set_connector_routing should update connector angle/points metadata."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 1101)

    asyncio.run(server_mod.call_tool("create_model", {"name": "Routing", "model_id": "m1"}))
    asyncio.run(
        server_mod.call_tool(
            "add_stock",
            {"model_id": "m1", "name": "S", "initial_value": "100", "x": 400, "y": 300},
        )
    )
    asyncio.run(
        server_mod.call_tool(
            "add_aux",
            {"model_id": "m1", "name": "k", "equation": "1", "x": 200, "y": 150},
        )
    )
    asyncio.run(
        server_mod.call_tool(
            "add_connector",
            {"model_id": "m1", "from_var": "k", "to_var": "S"},
        )
    )

    updated = asyncio.run(
        server_mod.call_tool(
            "set_connector_routing",
            {
                "model_id": "m1",
                "from_var": "k",
                "to_var": "S",
                "angle": -20,
                "points": [
                    {"x": 240, "y": 170},
                    {"x": 320, "y": 210},
                ],
                "points_locked": True,
            },
        )
    )
    assert "Updated connector uid=" in _tool_text(updated)
    _, model = server_mod.get_model("m1")
    assert len(model.connectors) == 1
    conn = model.connectors[0]
    assert conn.angle == -20.0
    assert conn.angle_locked is True
    assert conn.points == [(240.0, 170.0), (320.0, 210.0)]
    assert conn.points_locked is True

    xml_result = asyncio.run(
        server_mod.call_tool(
            "get_model_xml",
            {"model_id": "m1", "auto_layout": False},
        )
    )
    xml_text = _tool_text(xml_result)
    assert 'angle="-20.0"' in xml_text


def test_set_connector_routing_requires_lookup_fields(monkeypatch):
    """set_connector_routing should fail if neither uid nor endpoint pair is provided."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 1102)
    asyncio.run(server_mod.call_tool("create_model", {"name": "RoutingFail", "model_id": "m1"}))

    result = asyncio.run(
        server_mod.call_tool(
            "set_connector_routing",
            {"model_id": "m1", "angle": 10},
        )
    )
    assert isinstance(result, CallToolResult)
    assert result.isError is True
    assert result.structuredContent["error"]["code"] == "invalid_input"


def test_list_connectors_empty(monkeypatch):
    """list_connectors should return clear message when no connectors exist."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 1103)
    asyncio.run(server_mod.call_tool("create_model", {"name": "NoConn", "model_id": "m1"}))

    result = asyncio.run(server_mod.call_tool("list_connectors", {"model_id": "m1"}))
    assert _tool_text(result) == "No connectors in model_id=m1."


def test_list_connectors_includes_routing_metadata(monkeypatch):
    """list_connectors should expose uid, endpoints, angle, and point lock metadata."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 1104)

    asyncio.run(server_mod.call_tool("create_model", {"name": "ConnList", "model_id": "m1"}))
    asyncio.run(
        server_mod.call_tool(
            "add_stock",
            {"model_id": "m1", "name": "S", "initial_value": "100", "x": 400, "y": 300},
        )
    )
    asyncio.run(
        server_mod.call_tool(
            "add_aux",
            {"model_id": "m1", "name": "k", "equation": "1", "x": 220, "y": 150},
        )
    )
    asyncio.run(
        server_mod.call_tool("add_connector", {"model_id": "m1", "from_var": "k", "to_var": "S"})
    )
    asyncio.run(
        server_mod.call_tool(
            "set_connector_routing",
            {
                "model_id": "m1",
                "from_var": "k",
                "to_var": "S",
                "angle": -12,
                "points": [{"x": 260, "y": 175}, {"x": 320, "y": 225}],
                "points_locked": True,
            },
        )
    )

    listed = asyncio.run(server_mod.call_tool("list_connectors", {"model_id": "m1"}))
    text = _tool_text(listed)
    assert "uid=1" in text
    assert "k -> S" in text
    assert "angle=-12.0 (locked=True)" in text
    assert "points=2 (locked=True)" in text


def test_save_and_load_template_tools(monkeypatch, tmp_path):
    """save_as_template and load_template should work through tool API."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 1001)
    monkeypatch.setenv("STELLA_MCP_TEMPLATE_DIR", str(tmp_path))

    create = asyncio.run(
        server_mod.call_tool("create_model", {"name": "Template Source", "model_id": "source"})
    )
    assert "model_id=source" in _tool_text(create)

    asyncio.run(
        server_mod.call_tool(
            "add_stock",
            {"model_id": "source", "name": "Population", "initial_value": "100"},
        )
    )

    saved = asyncio.run(
        server_mod.call_tool(
            "save_as_template",
            {"model_id": "source", "template_name": "custom_pop"},
        )
    )
    assert "template 'custom_pop'" in _tool_text(saved)

    listed = asyncio.run(server_mod.call_tool("list_templates", {}))
    assert "custom_pop [user]" in _tool_text(listed)

    loaded = asyncio.run(
        server_mod.call_tool(
            "load_template",
            {"template_name": "custom_pop", "model_id": "loaded"},
        )
    )
    assert "as model_id=loaded" in _tool_text(loaded)


def test_read_model_reports_compatibility_warnings(monkeypatch, tmp_path):
    """read_model in permissive mode should report import compatibility warnings."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 2001)
    path = tmp_path / "compat_warn.stmx"
    path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<xmile version="1.0" xmlns="http://docs.oasis-open.org/xmile/ns/XMILE/v1.0">
  <header><name>CompatWarn</name></header>
  <sim_specs method="Euler" time_units="Years"><start>0</start><stop>10</stop><dt reciprocal="true">0</dt></sim_specs>
  <model><variables/></model>
</xmile>
""",
        encoding="utf-8",
    )

    result = asyncio.run(
        server_mod.call_tool("read_model", {"filepath": str(path), "model_id": "m1"})
    )
    assert "compatibility warnings: 1" in _tool_text(result)


def test_read_model_strict_returns_invalid_input(monkeypatch, tmp_path):
    """read_model strict mode should fail on compatibility issues."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 2002)
    path = tmp_path / "compat_strict.stmx"
    path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<xmile version="1.0" xmlns="http://docs.oasis-open.org/xmile/ns/XMILE/v1.0">
  <header><name>CompatStrict</name></header>
  <sim_specs method="Euler" time_units="Years"><start>0</start><stop>10</stop><dt reciprocal="true">0</dt></sim_specs>
  <model><variables/></model>
</xmile>
""",
        encoding="utf-8",
    )

    result = asyncio.run(
        server_mod.call_tool(
            "read_model",
            {"filepath": str(path), "model_id": "m1", "compat_mode": "strict"},
        )
    )
    assert isinstance(result, CallToolResult)
    assert result.isError is True
    assert result.structuredContent["error"]["code"] == "invalid_input"


def test_get_model_xml_strict_mode_returns_invalid_input(monkeypatch):
    """get_model_xml strict mode should fail fast on export compatibility issues."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 2003)
    server_mod._set_current_model(StellaModel("CompatExportStrict"), model_id="m1")
    _, model = server_mod.get_model("m1")
    model.add_stock("S", "100")
    model.sim_specs.dt = -1

    result = asyncio.run(
        server_mod.call_tool("get_model_xml", {"model_id": "m1", "compat_mode": "strict"})
    )
    assert isinstance(result, CallToolResult)
    assert result.isError is True
    assert result.structuredContent["error"]["code"] == "invalid_input"


def test_list_models_returns_structured_content(monkeypatch):
    """list_models should return a machine-readable model list."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 2101)
    asyncio.run(server_mod.call_tool("create_model", {"name": "M1", "model_id": "m1"}))
    result = asyncio.run(server_mod.call_tool("list_models", {}))

    assert isinstance(result, CallToolResult)
    assert result.structuredContent["models"] == [
        {"model_id": "m1", "name": "M1", "current": True}
    ]


def test_validate_model_returns_structured_issues(monkeypatch):
    """validate_model should expose validation issues as dictionaries."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 2102)
    asyncio.run(server_mod.call_tool("create_model", {"name": "Broken", "model_id": "m1"}))
    asyncio.run(
        server_mod.call_tool(
            "add_stock",
            {"model_id": "m1", "name": "S", "initial_value": "100"},
        )
    )
    result = asyncio.run(server_mod.call_tool("validate_model", {"model_id": "m1"}))

    assert isinstance(result, CallToolResult)
    assert result.structuredContent["model_id"] == "m1"
    assert result.structuredContent["issues"][0]["category"] == "mass_balance"


def test_inspect_model_returns_complete_structured_summary(monkeypatch):
    """inspect_model should be the primary structured model introspection tool."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 2103)
    asyncio.run(server_mod.call_tool("create_model", {"name": "Inspect", "model_id": "m1"}))
    asyncio.run(
        server_mod.call_tool("add_stock", {"model_id": "m1", "name": "S", "initial_value": "100"})
    )
    asyncio.run(server_mod.call_tool("add_aux", {"model_id": "m1", "name": "k", "equation": "0.1"}))
    asyncio.run(
        server_mod.call_tool(
            "add_flow",
            {"model_id": "m1", "name": "loss", "equation": "S * k", "from_stock": "S"},
        )
    )
    asyncio.run(server_mod.call_tool("add_connector", {"model_id": "m1", "from_var": "S", "to_var": "loss"}))
    asyncio.run(server_mod.call_tool("add_connector", {"model_id": "m1", "from_var": "k", "to_var": "loss"}))

    result = asyncio.run(
        server_mod.call_tool("inspect_model", {"model_id": "m1", "include_validation": True})
    )

    assert isinstance(result, CallToolResult)
    assert result.structuredContent["model"]["model_id"] == "m1"
    assert result.structuredContent["model"]["counts"]["stocks"] == 1
    assert result.structuredContent["validation"]["passed"] is True


def test_update_tools_return_structured_content(monkeypatch):
    """Update tools should mutate model fields and return structured payloads."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 2104)
    asyncio.run(server_mod.call_tool("create_model", {"name": "Update", "model_id": "m1"}))
    asyncio.run(
        server_mod.call_tool("add_stock", {"model_id": "m1", "name": "S", "initial_value": "100"})
    )
    asyncio.run(server_mod.call_tool("add_aux", {"model_id": "m1", "name": "k", "equation": "0.1"}))
    asyncio.run(
        server_mod.call_tool(
            "add_flow",
            {"model_id": "m1", "name": "loss", "equation": "S * k", "from_stock": "S"},
        )
    )

    specs = asyncio.run(
        server_mod.call_tool("set_sim_specs", {"model_id": "m1", "stop": 20, "dt": 0.5})
    )
    stock = asyncio.run(
        server_mod.call_tool("update_stock", {"model_id": "m1", "name": "S", "initial_value": "200"})
    )
    aux = asyncio.run(
        server_mod.call_tool("update_aux", {"model_id": "m1", "name": "k", "equation": "0.2"})
    )
    flow = asyncio.run(
        server_mod.call_tool(
            "update_flow",
            {"model_id": "m1", "name": "loss", "equation": "S * k * 2"},
        )
    )

    assert specs.structuredContent["sim_specs"]["stop"] == 20
    assert stock.structuredContent["stock"]["initial_value"] == "200"
    assert aux.structuredContent["auxiliary"]["equation"] == "0.2"
    assert flow.structuredContent["flow"]["equation"] == "S * k * 2"


def test_sync_connectors_from_equations_tool(monkeypatch):
    """Tool should add missing equation connectors and report counts."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 2105)
    asyncio.run(server_mod.call_tool("create_model", {"name": "Sync", "model_id": "m1"}))
    asyncio.run(
        server_mod.call_tool("add_stock", {"model_id": "m1", "name": "S", "initial_value": "100"})
    )
    asyncio.run(server_mod.call_tool("add_aux", {"model_id": "m1", "name": "k", "equation": "0.1"}))
    asyncio.run(
        server_mod.call_tool(
            "add_flow",
            {"model_id": "m1", "name": "loss", "equation": "S * k", "from_stock": "S"},
        )
    )

    result = asyncio.run(
        server_mod.call_tool("sync_connectors_from_equations", {"model_id": "m1"})
    )

    assert result.structuredContent["added"] == 2
    listed = asyncio.run(server_mod.call_tool("list_connectors", {"model_id": "m1"}))
    assert len(listed.structuredContent["connectors"]) == 2
