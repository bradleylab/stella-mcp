"""Tests for typed layout metadata and coordinate provenance."""

import hashlib
import json
from pathlib import Path

import pytest

from stella_mcp.xmile import StellaModel, parse_stmx

FIXTURES = Path(__file__).parent / "fixtures" / "layout"


def test_api_coordinates_are_user_authored_and_missing_coordinates_are_auto():
    model = StellaModel("Provenance")

    stock = model.add_stock("Pinned", "1", x=120.25, y=240.75)
    flow = model.add_flow("Loose flow", "1")
    aux = model.add_aux("Pinned aux", "1", x=300.5, y=150.25)

    assert stock.position_source == "user"
    assert aux.position_source == "user"
    assert flow.position_source == "auto"

    model._auto_layout()

    assert (stock.x, stock.y) == (120.25, 240.75)
    assert (aux.x, aux.y) == (300.5, 150.25)
    assert flow.x is not None and flow.x.is_integer()
    assert flow.y is not None and flow.y.is_integer()


def test_partial_authored_coordinates_fill_only_the_missing_axis():
    model = StellaModel("Partial provenance")
    source = model.add_stock("Source", "10", x=120.5)
    model.add_stock("Destination", "0")
    flow = model.add_flow(
        "Transfer",
        "Control",
        from_stock="Source",
        to_stock="Destination",
        x=250.75,
    )
    control = model.add_aux("Control", "1", y=80.25)
    model.sync_connectors_from_equations()

    result = model._auto_layout()

    assert source.x == 120.5
    assert source.y is not None
    assert flow.x == 250.75
    assert flow.y is not None
    assert control.x is not None
    assert control.y == 80.25
    assert {source.position_source, flow.position_source, control.position_source} == {"user"}
    assert result.metrics.missing_positions == ()


def test_updated_coordinates_become_user_authored_and_survive_layout():
    model = StellaModel("Updated provenance")
    stock = model.add_stock("Stock", "10")
    flow = model.add_flow("Flow", "Control", from_stock="Stock")
    aux = model.add_aux("Control", "1")
    model.sync_connectors_from_equations()

    model.update_stock("Stock", x=120.25, y=240.75)
    model.update_flow("Flow", x=300.5, y=240.75)
    model.update_aux("Control", x=300.5, y=120.25)

    assert {stock.position_source, flow.position_source, aux.position_source} == {"user"}

    model._auto_layout()

    assert (stock.x, stock.y) == (120.25, 240.75)
    assert (flow.x, flow.y) == (300.5, 240.75)
    assert (aux.x, aux.y) == (300.5, 120.25)


def test_auto_coordinates_recompute_after_model_extension():
    model = StellaModel("Incremental")
    model.add_stock("First", "10")
    model.add_stock("Second", "0")
    model.add_aux("Initial rate", "1")
    model.add_flow("Initial transfer", "Initial rate", from_stock="First", to_stock="Second")
    model.sync_connectors_from_equations()
    model._auto_layout()

    before = {
        name: (element.x, element.y)
        for name, element in {
            **model.stocks,
            **model.flows,
            **model.auxs,
        }.items()
    }
    assert all(element.position_source == "auto" for element in model.stocks.values())
    assert all(element.position_source == "auto" for element in model.flows.values())
    assert all(element.position_source == "auto" for element in model.auxs.values())

    model.add_stock("Third", "0")
    model.add_aux("Added rate", "2")
    model.add_flow("Added transfer", "Added rate", from_stock="Second", to_stock="Third")
    model.sync_connectors_from_equations()
    model._auto_layout()

    after = {
        name: (element.x, element.y)
        for name, element in {
            **model.stocks,
            **model.flows,
            **model.auxs,
        }.items()
        if name in before
    }
    assert any(after[name] != before[name] for name in before)


def test_auto_geometry_snaps_to_whole_pixels():
    model = StellaModel("PixelGrid")
    model.add_stock("A", "1")
    model.add_stock("B", "0")
    model.add_flow("transfer", "1", from_stock="A", to_stock="B")
    model.add_aux("rate", "1")
    model.add_connector("rate", "transfer")

    model._auto_layout()

    elements = [*model.stocks.values(), *model.flows.values(), *model.auxs.values()]
    assert all(element.x is not None and element.x.is_integer() for element in elements)
    assert all(element.y is not None and element.y.is_integer() for element in elements)
    assert all(
        coordinate.is_integer()
        for flow in model.flows.values()
        for point in flow.points
        for coordinate in point
    )


def test_viewport_label_sides_and_user_geometry_round_trip(tmp_path: Path):
    source = tmp_path / "typed-layout.stmx"
    source.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<xmile version="1.0" xmlns="http://docs.oasis-open.org/xmile/ns/XMILE/v1.0" xmlns:isee="http://iseesystems.com/XMILE">
  <header><name>Typed Layout</name></header>
  <sim_specs method="Euler" time_units="Years"><start>0</start><stop>10</stop><dt>1</dt></sim_specs>
  <model>
    <variables>
      <stock name="S"><eqn>1</eqn><outflow>f</outflow></stock>
      <flow name="f"><eqn>a</eqn></flow>
      <aux name="a"><eqn>1</eqn></aux>
    </variables>
    <views>
      <view type="stock_flow" page_width="640.5" page_height="480.25" isee:page_cols="3" isee:page_rows="4">
        <style font_size="11pt"><stock font_size="15pt"/><flow font_size="12pt"/><aux font_size="10.5pt"/></style>
        <stock name="S" x="100.25" y="200.75" width="45" height="35" label_side="top"/>
        <flow name="f" x="180.5" y="200.75" label_side="left"><pts><pt x="122.75" y="200.75"/><pt x="240.25" y="200.75"/></pts></flow>
        <aux name="a" x="180.5" y="100.125" label_side="right"/>
      </view>
    </views>
  </model>
</xmile>
""",
        encoding="utf-8",
    )

    model = parse_stmx(str(source), compat_mode="strict")

    assert model.view_page_width == 640.5
    assert model.view_page_height == 480.25
    assert model.view_page_columns == 3
    assert model.view_page_rows == 4
    assert model.view_stock_font_points == 15.0
    assert model.view_flow_font_points == 12.0
    assert model.view_aux_font_points == 10.5
    assert model.stocks["S"].label_side == "top"
    assert model.flows["f"].label_side == "left"
    assert model.auxs["a"].label_side == "right"
    assert model.stocks["S"].position_source == "user"
    assert model.flows["f"].position_source == "user"
    assert model.auxs["a"].position_source == "user"

    model._auto_layout()
    assert (model.stocks["S"].x, model.stocks["S"].y) == (122.75, 218.25)
    assert model.flows["f"].points == [(122.75, 200.75), (240.25, 200.75)]

    exported = model.to_xml(auto_layout=False, compat_mode="strict")
    assert 'page_width="640.5"' in exported
    assert 'page_height="480.25"' in exported
    assert 'isee:page_cols="3"' in exported
    assert 'isee:page_rows="4"' in exported
    assert 'x="100.25" y="200.75"' in exported
    assert 'label_side="top"' in exported
    assert 'label_side="left"' in exported
    assert 'label_side="right"' in exported
    assert 'stock font_size="15pt"' in exported
    assert "position_source" not in exported

    round_trip = tmp_path / "typed-layout-roundtrip.stmx"
    round_trip.write_text(exported, encoding="utf-8")
    loaded = parse_stmx(str(round_trip), compat_mode="strict")
    assert loaded.view_page_width == pytest.approx(640.5)
    assert loaded.view_stock_font_points == pytest.approx(15.0)
    assert loaded.view_flow_font_points == pytest.approx(12.0)
    assert loaded.view_aux_font_points == pytest.approx(10.5)
    assert loaded.stocks["S"].x == pytest.approx(122.75)
    assert loaded.flows["f"].points == [(122.75, 200.75), (240.25, 200.75)]


def test_format_spike_fixture_exercises_stella_layout_fields():
    model = parse_stmx(str(FIXTURES / "format_spike_source.stmx"), compat_mode="strict")

    assert (model.view_page_width, model.view_page_height) == (640.0, 480.0)
    assert (model.view_page_columns, model.view_page_rows) == (3, 2)
    assert (model.stocks["Reservoir"].x, model.stocks["Reservoir"].y) == (422.5, 317.5)
    assert model.stocks["Reservoir"].label_side == "top"
    assert model.flows["routed_input"].label_side == "bottom"
    assert model.auxs["left_control"].label_side == "left"
    assert model.auxs["right_note"].label_side == "right"
    assert model.flows["routed_input"].points[2:4] == [(200.0, 220.0), (320.0, 220.0)]
    assert (model.flows["routed_input"].x, model.flows["routed_input"].y) == (260.0, 220.0)
    assert model.connectors[0].points == [
        (100.0, 118.0),
        (100.0, 160.0),
        (180.0, 160.0),
        (180.0, 220.0),
        (260.0, 220.0),
    ]


def test_stella_saved_format_spike_matches_manifest_and_round_trips():
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))["fixtures"][0]
    for key in ("source", "saved"):
        artifact = FIXTURES / manifest[f"{key}_file"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == manifest[f"{key}_sha256"]

    model = parse_stmx(str(FIXTURES / manifest["saved_file"]), compat_mode="strict")

    assert (model.view_page_width, model.view_page_height) == (776.0, 588.0)
    assert (model.view_page_columns, model.view_page_rows) == (3, 2)
    assert model.flows["routed_input"].points == [
        (80.0, 300.0),
        (200.0, 300.0),
        (200.0, 220.0),
        (320.0, 220.0),
        (320.0, 300.0),
        (400.0, 300.0),
    ]
    assert model.connectors[0].points == [
        (100.0, 109.0),
        (100.0, 160.0),
        (180.0, 160.0),
        (180.0, 220.0),
        (251.031, 220.747),
    ]
    assert model.connectors[0].angle == 270.0


def test_stella_omitted_page_counts_default_to_one():
    one_page = parse_stmx(
        str(FIXTURES / "0.12_sir_stella_4_1_1_release.stmx"),
        compat_mode="strict",
    )
    two_columns = parse_stmx(
        str(FIXTURES / "0.12_chain_stella_4_1_1_release.stmx"),
        compat_mode="strict",
    )

    assert (one_page.view_page_columns, one_page.view_page_rows) == (1, 1)
    assert (two_columns.view_page_columns, two_columns.view_page_rows) == (2, 1)
