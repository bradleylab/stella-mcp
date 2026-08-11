"""Tests for round-trip preservation of detailed layout data."""

from pathlib import Path

from stella_mcp.xmile import StellaModel, parse_stmx
from tests.support.layout_fixtures import build_fanout


def test_round_trip_preserves_stock_size_and_flow_points(tmp_path: Path):
    """Imported stock sizes and flow waypoints should survive save/load cycles."""
    filepath = tmp_path / "layout_data.stmx"

    model1 = StellaModel("LayoutRoundTrip")
    model1.add_stock("A", "100", x=120, y=220)
    model1.add_stock("B", "100", x=360, y=220)
    flow = model1.add_flow("transfer", "10", from_stock="A", to_stock="B", x=240, y=220)

    # Simulate imported/manual layout data.
    model1.stocks["A"].width = 88
    model1.stocks["A"].height = 62
    model1.stocks["A"].size_locked = True
    flow.points = [(170.0, 190.0), (310.0, 190.0)]
    flow.points_locked = True

    filepath.write_text(model1.to_xml(), encoding="utf-8")

    model2 = parse_stmx(str(filepath))
    assert model2.stocks["A"].width == 88
    assert model2.stocks["A"].height == 62
    assert (model2.stocks["A"].x, model2.stocks["A"].y) == (120, 220)
    assert model2.stocks["A"].size_locked is True
    assert model2.flows["transfer"].points == [(170.0, 190.0), (310.0, 190.0)]
    assert model2.flows["transfer"].points_locked is True

    xml2 = model2.to_xml()
    assert 'stock x="76.0" y="189.0" width="88" height="62" name="A"' in xml2
    assert 'width="88"' in xml2
    assert 'height="62"' in xml2
    assert 'pt x="170.0" y="190.0"' in xml2
    assert 'pt x="310.0" y="190.0"' in xml2


def test_round_trip_preserves_connector_routing_points(tmp_path: Path):
    """Connector routing points should survive save/load cycles."""
    filepath = tmp_path / "connector_routing.stmx"

    model1 = StellaModel("ConnectorRoutingRoundTrip")
    model1.add_stock("A", "100", x=120, y=220)
    model1.add_aux("k", "1", x=320, y=120)
    connector = model1.add_connector("k", "A")
    connector.points = [(300.0, 140.0), (220.0, 170.0), (165.0, 200.0)]
    connector.points_locked = True

    filepath.write_text(model1.to_xml(auto_layout=False), encoding="utf-8")

    model2 = parse_stmx(str(filepath))
    assert len(model2.connectors) == 1
    assert model2.connectors[0].points == [(300.0, 140.0), (220.0, 170.0), (165.0, 200.0)]
    assert model2.connectors[0].points_locked is True

    xml2 = model2.to_xml(auto_layout=False)
    assert "<connector" in xml2
    assert "<pts>" in xml2
    assert 'pt x="300.0" y="140.0"' in xml2
    assert 'pt x="220.0" y="170.0"' in xml2
    assert 'pt x="165.0" y="200.0"' in xml2


def test_generated_layout_strict_round_trip_preserves_semantics_and_geometry(
    tmp_path: Path,
):
    filepath = tmp_path / "generated_fanout.stmx"
    model = build_fanout()
    model._auto_layout()
    expected_flow_routes = {
        name: list(flow.points) for name, flow in model.flows.items()
    }
    expected_connector_routes = {
        connector.uid: list(connector.points) for connector in model.connectors
    }

    filepath.write_text(
        model.to_xml(auto_layout=False, compat_mode="strict"),
        encoding="utf-8",
    )
    assert {name: flow.points for name, flow in model.flows.items()} == expected_flow_routes
    assert {
        connector.uid: connector.points for connector in model.connectors
    } == expected_connector_routes
    loaded = parse_stmx(str(filepath), compat_mode="strict")

    assert {
        name: (stock.initial_value, tuple(stock.inflows), tuple(stock.outflows))
        for name, stock in loaded.stocks.items()
    } == {
        name: (stock.initial_value, tuple(stock.inflows), tuple(stock.outflows))
        for name, stock in model.stocks.items()
    }
    assert {
        name: (flow.equation, flow.from_stock, flow.to_stock)
        for name, flow in loaded.flows.items()
    } == {
        name: (flow.equation, flow.from_stock, flow.to_stock)
        for name, flow in model.flows.items()
    }
    assert {name: aux.equation for name, aux in loaded.auxs.items()} == {
        name: aux.equation for name, aux in model.auxs.items()
    }
    assert (loaded.view_page_width, loaded.view_page_height) == (
        model.view_page_width,
        model.view_page_height,
    )
    assert (loaded.view_page_columns, loaded.view_page_rows) == (
        model.view_page_columns,
        model.view_page_rows,
    )
    assert {
        name: element.label_side
        for registry in (loaded.stocks, loaded.flows, loaded.auxs)
        for name, element in registry.items()
    } == {
        name: element.label_side
        for registry in (model.stocks, model.flows, model.auxs)
        for name, element in registry.items()
    }
    assert {name: flow.points for name, flow in loaded.flows.items()} == expected_flow_routes
    assert {
        connector.uid: connector.points[::2] for connector in loaded.connectors
    } == expected_connector_routes
