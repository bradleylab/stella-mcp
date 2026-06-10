"""Tests for the simulate tool (pysd-backed, optional extra).

The whole module is skipped when pysd is not installed; the
dependency-missing error path is covered in test_server_state_and_gf.py,
which runs without the sim extra.
"""

import asyncio

import pytest

pysd = pytest.importorskip("pysd")

from stella_mcp import server as server_mod  # noqa: E402


def _call(name, arguments):
    return asyncio.run(server_mod.call_tool(name, arguments))


def _fresh_session(monkeypatch, key):
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: key)


def _build_growth_model(monkeypatch, key):
    _fresh_session(monkeypatch, key)
    _call("build_model", {
        "name": "Growth",
        "model_id": "g",
        "sim_specs": {"start": 0, "stop": 10, "dt": 0.25},
        "stocks": [{"name": "Population", "initial_value": "100"}],
        "auxs": [{"name": "growth rate", "equation": "0.1"}],
        "flows": [
            {
                "name": "growth",
                "equation": 'Population * "growth rate"',
                "to_stock": "Population",
            }
        ],
    })


def test_simulate_growth_model(monkeypatch):
    _build_growth_model(monkeypatch, 4001)

    result = _call("simulate", {"model_id": "g", "max_points": 20})

    assert not result.isError
    sc = result.structuredContent
    assert sc["model_id"] == "g"
    [series] = sc["series"]
    assert series["name"] == "Population"
    assert len(series["points"]) <= 20
    assert series["points"][0]["t"] == 0
    assert series["points"][-1]["t"] == 10
    assert series["summary"]["initial"] == 100.0
    assert series["summary"]["final"] > 100.0


def test_simulate_override_flattens_growth(monkeypatch):
    _build_growth_model(monkeypatch, 4002)

    result = _call("simulate", {
        "model_id": "g",
        "overrides": {"growth_rate": 0},
    })

    [series] = result.structuredContent["series"]
    assert series["summary"]["final"] == 100.0


def test_simulate_unknown_override_lists_candidates(monkeypatch):
    _build_growth_model(monkeypatch, 4003)

    result = _call("simulate", {"model_id": "g", "overrides": {"nope": 1}})

    assert result.isError
    err = result.structuredContent["error"]
    assert err["code"] == "invalid_input"
    assert "growth rate" in err["message"]


def test_simulate_rk4_model_warns_euler_only(monkeypatch):
    _fresh_session(monkeypatch, 4004)
    _call("build_model", {
        "name": "RK",
        "model_id": "rk",
        "sim_specs": {"start": 0, "stop": 5, "dt": 0.25, "method": "RK4"},
        "stocks": [{"name": "S", "initial_value": "1"}],
        "flows": [{"name": "f", "equation": "S * 0.1", "to_stock": "S"}],
    })

    result = _call("simulate", {"model_id": "rk"})

    assert not result.isError
    assert any("Euler" in w for w in result.structuredContent["warnings"])


def test_simulate_does_not_mutate_session_model(monkeypatch):
    """The session model's layout state and equations must be untouched."""
    _fresh_session(monkeypatch, 4005)
    _call("build_model", {
        "name": "GF",
        "model_id": "gf",
        "sim_specs": {"start": 0, "stop": 10, "dt": 0.25},
        "stocks": [{"name": "Population", "initial_value": "100"}],
        "auxs": [
            {"name": "rate", "equation": "0.05"},
            {
                "name": "seasonal",
                "equation": "GRAPH(TIME)",
                "graphical_function": {
                    "xscale": {"min": 0, "max": 10},
                    "ypts": [1.0, 1.2, 0.8, 1.0],
                },
            },
        ],
        "flows": [
            {
                "name": "growth",
                "equation": "Population * rate * seasonal",
                "to_stock": "Population",
            }
        ],
    })
    session = server_mod._get_session_models()
    model = session.models["gf"]
    export_warnings_before = list(model.last_export_warnings)

    result = _call("simulate", {"model_id": "gf"})

    assert not result.isError
    # GRAPH shim applied only to the simulation copy.
    assert model.auxs["seasonal"].equation == "GRAPH(TIME)"
    # to_xml() ran on a deep copy: no layout positions appeared on the
    # session model and export warnings did not change.
    assert model.stocks["Population"].x is None
    assert model.last_export_warnings == export_warnings_before


def test_simulate_saves_csv_with_time_column(monkeypatch, tmp_path):
    _build_growth_model(monkeypatch, 4006)
    csv_path = tmp_path / "results.csv"

    result = _call("simulate", {"model_id": "g", "save_results_csv": str(csv_path)})

    assert not result.isError
    assert result.structuredContent["csv_path"] == str(csv_path)
    header = csv_path.read_text().splitlines()[0]
    assert header.startswith("time,")
    assert "Population" in header


def test_simulate_include_selects_variables(monkeypatch):
    _build_growth_model(monkeypatch, 4007)

    result = _call("simulate", {"model_id": "g", "include": ["growth"]})

    names = [s["name"] for s in result.structuredContent["series"]]
    assert names == ["growth"]
