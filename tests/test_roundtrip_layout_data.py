"""Tests for round-trip preservation of detailed layout data."""

from pathlib import Path

from stella_mcp.xmile import StellaModel, parse_stmx


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
    assert model2.stocks["A"].size_locked is True
    assert model2.flows["transfer"].points == [(170.0, 190.0), (310.0, 190.0)]
    assert model2.flows["transfer"].points_locked is True

    xml2 = model2.to_xml()
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
