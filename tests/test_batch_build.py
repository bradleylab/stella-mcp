"""Tests for the batch construction tools (build_model, add_variables)."""

import asyncio

from stella_mcp import server as server_mod

SIR_BATCH = {
    "name": "SIR",
    "model_id": "sir",
    "sim_specs": {"start": 0, "stop": 100, "dt": 0.125, "time_units": "Days"},
    "stocks": [
        {"name": "Susceptible", "initial_value": "9999", "units": "people"},
        {"name": "Infected", "initial_value": "1", "units": "people"},
        {"name": "Recovered", "initial_value": "0", "units": "people"},
    ],
    "auxs": [
        {"name": "contact_rate", "equation": "6"},
        {"name": "infectivity", "equation": "0.25"},
        {"name": "recovery_time", "equation": "2", "units": "days"},
        {"name": "total_population", "equation": "Susceptible + Infected + Recovered"},
    ],
    "flows": [
        {
            "name": "infection",
            "equation": "Susceptible * contact_rate * infectivity * Infected / total_population",
            "from_stock": "Susceptible",
            "to_stock": "Infected",
        },
        {
            "name": "recovery",
            "equation": "Infected / recovery_time",
            "from_stock": "Infected",
            "to_stock": "Recovered",
        },
    ],
    "modules": [
        {"name": "Disease Dynamics", "members": ["Susceptible", "Infected", "Recovered"]}
    ],
}


def _call(name, arguments):
    return asyncio.run(server_mod.call_tool(name, arguments))


def _fresh_session(monkeypatch, key):
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: key)


def test_build_model_creates_full_model_in_one_call(monkeypatch):
    _fresh_session(monkeypatch, 3001)

    result = _call("build_model", SIR_BATCH)

    assert not result.isError
    sc = result.structuredContent
    assert sc["model_id"] == "sir"
    assert sc["added"] == {
        "stocks": 3, "flows": 2, "auxiliaries": 4, "connectors": 0, "modules": 1,
    }
    # Connector sync ran by default and wired the equation dependencies.
    assert sc["connector_sync"]["added"] > 0
    assert sc["validation"]["passed"] is True
    assert sc["model"]["counts"]["stocks"] == 3
    # The built model is registered and current.
    listed = _call("list_models", {})
    assert [m["model_id"] for m in listed.structuredContent["models"]] == ["sir"]
    assert listed.structuredContent["models"][0]["current"] is True


def test_build_model_failure_is_atomic(monkeypatch):
    """An invalid item must leave the session without the model."""
    _fresh_session(monkeypatch, 3002)
    bad = {
        "name": "Broken",
        "model_id": "broken",
        "stocks": [{"name": "S", "initial_value": "1"}],
        "flows": [
            {"name": "ok_flow", "equation": "1", "from_stock": "S"},
            {"name": "bad_flow", "equation": "1", "from_stock": "NoSuchStock"},
        ],
    }

    result = _call("build_model", bad)

    assert result.isError
    err = result.structuredContent["error"]
    assert err["code"] == "invalid_input"
    assert err["stage"] == "flows"
    assert err["index"] == 1
    assert err["item_name"] == "bad_flow"
    listed = _call("list_models", {})
    assert listed.structuredContent["models"] == []


def test_add_variables_extends_existing_model(monkeypatch):
    _fresh_session(monkeypatch, 3003)
    _call("create_model", {"name": "Pop", "model_id": "pop"})
    _call("add_stock", {"model_id": "pop", "name": "Population", "initial_value": "100"})

    result = _call("add_variables", {
        "model_id": "pop",
        "auxs": [{"name": "growth_rate", "equation": "0.1"}],
        "flows": [
            {
                "name": "growth",
                "equation": "Population * growth_rate",
                "to_stock": "Population",
            }
        ],
    })

    assert not result.isError
    sc = result.structuredContent
    assert sc["added"]["flows"] == 1
    assert sc["added"]["auxiliaries"] == 1
    assert sc["connector_sync"]["added"] == 2
    assert sc["validation"]["passed"] is True


def test_add_variables_failure_leaves_model_unchanged(monkeypatch):
    _fresh_session(monkeypatch, 3004)
    _call("create_model", {"name": "Pop", "model_id": "pop"})
    _call("add_stock", {"model_id": "pop", "name": "Population", "initial_value": "100"})

    result = _call("add_variables", {
        "model_id": "pop",
        "auxs": [
            {"name": "good_aux", "equation": "1"},
            {"name": "Population", "equation": "2"},  # duplicate name -> error
        ],
    })

    assert result.isError
    err = result.structuredContent["error"]
    assert err["stage"] == "auxs"
    assert err["index"] == 1
    inspected = _call("inspect_model", {"model_id": "pop", "include_validation": False})
    counts = inspected.structuredContent["model"]["counts"]
    # Neither aux landed: the partial batch was rolled back wholesale.
    assert counts["auxiliaries"] == 0
    assert counts["stocks"] == 1


def test_build_model_sync_connectors_false_respected(monkeypatch):
    _fresh_session(monkeypatch, 3005)
    args = {
        "name": "NoSync",
        "model_id": "nosync",
        "sync_connectors": False,
        "stocks": [{"name": "S", "initial_value": "1"}],
        "auxs": [{"name": "k", "equation": "0.5"}],
        "flows": [{"name": "out", "equation": "S * k", "from_stock": "S"}],
    }

    result = _call("build_model", args)

    sc = result.structuredContent
    assert "connector_sync" not in sc
    assert sc["model"]["counts"]["connectors"] == 0


def test_build_model_accepts_graphical_function_items(monkeypatch):
    _fresh_session(monkeypatch, 3006)
    args = {
        "name": "GF",
        "model_id": "gf",
        "auxs": [
            {
                "name": "lookup_rate",
                "equation": "GRAPH(TIME)",
                "graphical_function": {
                    "xscale": {"min": 0, "max": 100},
                    "ypts": [0.1, 0.2, 0.4, 0.6],
                    "type": "continuous",
                },
            }
        ],
    }

    result = _call("build_model", args)

    assert not result.isError
    aux = result.structuredContent["model"]["variables"]["auxiliaries"][0]
    assert aux["graphical_function"]["ypts"] == [0.1, 0.2, 0.4, 0.6]


def test_build_model_missing_required_field_names_item(monkeypatch):
    _fresh_session(monkeypatch, 3007)
    args = {
        "name": "Missing",
        "stocks": [{"name": "S"}],  # missing initial_value
    }

    result = _call("build_model", args)

    assert result.isError
    err = result.structuredContent["error"]
    assert err["stage"] == "stocks"
    assert err["index"] == 0
    assert "initial_value" in err["message"]


def test_build_model_module_view_and_style_applied(monkeypatch):
    _fresh_session(monkeypatch, 3008)
    args = {
        "name": "Mods",
        "model_id": "mods",
        "stocks": [{"name": "S", "initial_value": "1"}],
        "modules": [
            {
                "name": "Core",
                "members": ["S"],
                "view": {"x": 100, "y": 100, "width": 300, "height": 200},
                "style": {"background": "#FFF7E6", "label_side": "top"},
            }
        ],
    }

    result = _call("build_model", args)

    assert not result.isError
    module = result.structuredContent["model"]["modules"][0]
    assert module["box"] == {"x": 100, "y": 100, "width": 300, "height": 200}
    assert module["style"]["background"] == "#FFF7E6"
    assert module["style"]["label_side"] == "top"
