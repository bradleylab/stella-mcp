"""Tests for agent-facing model snapshot serialization."""

from stella_mcp.model_snapshot import (
    connector_to_dict,
    model_to_summary,
    module_to_dict,
    validation_issue_to_dict,
)
from stella_mcp.validator import ValidationError
from stella_mcp.xmile import StellaModel


def test_model_to_summary_includes_core_sections():
    model = StellaModel("Carbon")
    model.sim_specs.start = 0
    model.sim_specs.stop = 10
    model.sim_specs.dt = 0.5
    model.add_stock("Atmosphere", "100", units="GtC", x=100, y=200)
    model.add_aux("rate", "0.1", units="1/year")
    model.add_flow("sink", "Atmosphere * rate", from_stock="Atmosphere")
    model.add_connector("Atmosphere", "sink")
    model.add_connector("rate", "sink")
    model.create_module("Core", members=["Atmosphere", "sink", "rate"])

    summary = model_to_summary("carbon_v1", model)

    assert summary["model_id"] == "carbon_v1"
    assert summary["name"] == "Carbon"
    assert summary["sim_specs"] == {
        "start": 0,
        "stop": 10,
        "dt": 0.5,
        "method": "Euler",
        "time_units": "Years",
    }
    assert summary["counts"] == {
        "stocks": 1,
        "flows": 1,
        "auxiliaries": 1,
        "connectors": 2,
        "modules": 1,
    }
    assert summary["variables"]["stocks"][0]["name"] == "Atmosphere"
    assert summary["variables"]["flows"][0]["from_stock"] == "Atmosphere"
    assert summary["variables"]["auxiliaries"][0]["equation"] == "0.1"
    assert summary["modules"][0]["name"] == "Core"
    assert summary["connectors"][0]["uid"] == 1


def test_connector_and_module_dicts_preserve_routing_and_members():
    model = StellaModel("Routing")
    model.add_stock("S", "100")
    model.add_aux("k", "1")
    connector = model.add_connector("k", "S")
    connector.angle = 42
    connector.angle_locked = True
    connector.points = [(1.5, 2.5)]
    connector.points_locked = True
    model.create_module("M", members=["S", "k"])
    model.set_module_view("M", x=10, y=20, width=30, height=40)

    assert connector_to_dict(model, connector) == {
        "uid": 1,
        "from_var": "k",
        "from_display": "k",
        "to_var": "S",
        "to_display": "S",
        "angle": 42,
        "angle_locked": True,
        "points": [{"x": 1.5, "y": 2.5}],
        "points_locked": True,
    }
    assert module_to_dict(model, "M", model.modules["M"])["members"] == ["S", "k"]


def test_validation_issue_to_dict():
    issue = ValidationError(
        severity="error",
        category="undefined_variable",
        message="Flow references missing variable",
        variable="flow_x",
    )
    assert validation_issue_to_dict(issue) == {
        "severity": "error",
        "category": "undefined_variable",
        "message": "Flow references missing variable",
        "variable": "flow_x",
    }
