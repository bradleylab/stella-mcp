"""Focused contracts for XMILE serialization boundaries."""

from stella_mcp import xmile_export, xmile_io
from stella_mcp.model_types import GraphicalFunction
from stella_mcp.xmile import StellaModel
from stella_mcp.xmile_export import _stella_connector_points


def test_xmile_io_reexports_exporter_functions():
    assert xmile_io.gf_eqn_text is xmile_export.gf_eqn_text
    assert xmile_io.model_to_xml is xmile_export.model_to_xml


def test_graphical_function_export_uses_spec_equation_and_point_lists():
    model = StellaModel("Graphical Function")
    model.uuid = "export-contract"
    model.add_aux(
        "lookup",
        "GRAPH(Time)",
        graphical_function=GraphicalFunction(
            ypts=[0.0, 2.5, 5.0],
            xscale=(0.0, 10.0),
            yscale=(0.0, 5.0),
            gf_type="continuous",
        ),
    )

    direct_xml = xmile_export.model_to_xml(model, auto_layout=False)

    assert "<eqn>Time</eqn>" in direct_xml
    assert '<gf type="continuous">' in direct_xml
    assert '<xscale min="0" max="10"/>' in direct_xml
    assert '<yscale min="0" max="5"/>' in direct_xml
    assert "<ypts>0,2.5,5</ypts>" in direct_xml
    assert model.to_xml(auto_layout=False) == direct_xml


def test_model_xml_formatting_methods_delegate_to_export_module(monkeypatch):
    model = StellaModel("Delegate")
    calls: list[tuple[StellaModel, float | None]] = []

    def fake_dt_xml(delegate_model: StellaModel, dt: float | None = None) -> str:
        calls.append((delegate_model, dt))
        return "<dt>delegate</dt>"

    monkeypatch.setattr(xmile_export, "_dt_xml", fake_dt_xml)

    assert model._dt_xml(0.5) == "<dt>delegate</dt>"
    assert calls == [(model, 0.5)]


def test_generated_connector_routes_gain_stella_bezier_midpoint_anchors():
    model = StellaModel("Connector anchors")
    model.add_aux("source", "1", x=100, y=100)
    model.add_aux("target", "source", x=300, y=200)
    connector = model.add_connector("source", "target")
    connector.points = [(118.0, 100.0), (200.0, 100.0), (282.0, 200.0)]

    assert _stella_connector_points(connector) == (
        (118.0, 100.0),
        (159.0, 100.0),
        (200.0, 100.0),
        (241.0, 150.0),
        (282.0, 200.0),
    )

    connector.points_locked = True
    assert _stella_connector_points(connector) == tuple(connector.points)
