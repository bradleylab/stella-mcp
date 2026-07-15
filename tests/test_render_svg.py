"""Tests for SVG diagram rendering."""

import xml.etree.ElementTree as ET

import pytest

from stella_mcp.render_svg import render_model_svg
from stella_mcp.xmile import StellaModel


def _positioned_model() -> StellaModel:
    """A small model with explicit positions (2 stocks, 1 flow, 1 aux, 1 module)."""
    model = StellaModel("Diagram")
    model.add_stock("Source Stock", "100", x=100, y=100)
    model.add_stock("Sink Stock", "0", x=300, y=100)
    model.add_aux("rate", "0.1", x=200, y=220)
    model.add_flow("transfer", "Source_Stock * rate", from_stock="Source Stock", to_stock="Sink Stock")
    model.flows["transfer"].x = 200
    model.flows["transfer"].y = 100
    model.flows["transfer"].points = [(122, 100), (278, 100)]
    model.add_connector("Source Stock", "transfer")
    model.add_connector("rate", "transfer")
    model.create_module("Core", members=["Source Stock", "Sink Stock"])
    model.set_module_view("Core", x=200, y=100, width=320, height=160)
    return model


def _count_by_class(svg: str) -> dict[str, int]:
    root = ET.fromstring(svg)
    counts: dict[str, int] = {}
    for element in root.iter():
        cls = element.get("class")
        if cls:
            counts[cls] = counts.get(cls, 0) + 1
    return counts


def test_render_returns_well_formed_xml():
    svg = render_model_svg(_positioned_model())
    root = ET.fromstring(svg)  # raises on malformed XML
    assert root.tag.endswith("svg")


def test_render_element_counts_by_class():
    svg = render_model_svg(_positioned_model())
    counts = _count_by_class(svg)
    assert counts["stock"] == 2
    assert counts["aux"] == 1
    assert counts["flow-pipe"] == 1
    assert counts["flow-valve"] == 1
    assert counts["connector"] == 2
    assert counts["module"] == 1


def test_render_includes_display_name_labels():
    svg = render_model_svg(_positioned_model())
    # Underscores in internal names render as spaces in labels.
    assert ">Source Stock<" in svg
    assert ">rate<" in svg
    assert ">transfer<" in svg


def test_render_uses_imported_view_font_size_for_label_geometry():
    model = StellaModel("Custom font")
    model.add_aux("large label", "1", x=100, y=100)
    model.view_aux_font_points = 18.0

    svg = render_model_svg(model)

    assert 'style="font-size:24px"' in svg


def test_render_is_deterministic():
    model = _positioned_model()
    assert render_model_svg(model) == render_model_svg(model)


def test_render_element_coords_within_viewbox():
    model = _positioned_model()
    svg = render_model_svg(model, margin=40.0)
    root = ET.fromstring(svg)
    min_x, min_y, width, height = (float(v) for v in root.get("viewBox").split())
    max_x, max_y = min_x + width, min_y + height
    # Every stock rect must fall inside the viewBox.
    for element in root.iter():
        if element.get("class") == "stock":
            x, y = float(element.get("x")), float(element.get("y"))
            w, h = float(element.get("width")), float(element.get("height"))
            assert min_x <= x and x + w <= max_x
            assert min_y <= y and y + h <= max_y


def test_render_escapes_special_characters():
    model = StellaModel("Escaping")
    model.add_aux("a & b <c>", "1", x=50, y=50)
    svg = render_model_svg(model)
    ET.fromstring(svg)  # would raise if the raw & / < leaked in unescaped
    assert "a &amp; b &lt;c&gt;" in svg


def test_render_missing_positions_raises_naming_variables():
    model = StellaModel("Unpositioned")
    model.add_stock("Population", "100")  # no x/y
    model.add_aux("growth", "0.1", x=10, y=10)
    with pytest.raises(ValueError) as exc:
        render_model_svg(model)
    assert "Population" in str(exc.value)
    assert "auto_layout=true" in str(exc.value)


def test_render_after_auto_layout_succeeds():
    model = StellaModel("Auto")
    model.add_stock("Population", "100")
    model.add_aux("growth_rate", "0.1")
    model.add_flow("growth", "Population * growth_rate", to_stock="Population")
    model.add_connector("Population", "growth")
    model.add_connector("growth_rate", "growth")
    model._auto_layout()
    svg = render_model_svg(model)
    ET.fromstring(svg)
    assert _count_by_class(svg)["stock"] == 1


def test_render_overlapping_stocks_does_not_crash():
    """Imported models can place elements at identical coords; that is data,
    not a renderer error."""
    model = StellaModel("Overlap")
    model.add_stock("A", "1", x=100, y=100)
    model.add_stock("B", "2", x=100, y=100)
    svg = render_model_svg(model)
    assert _count_by_class(svg)["stock"] == 2


def test_render_orphan_flow_after_auto_layout_succeeds():
    """A flow with no source or destination stock is positioned by layout and
    renders with source/sink clouds, rather than blocking the render
    (regression: orphan flows kept x/y == None and render_model_svg raised)."""
    model = StellaModel("Orphan")
    model.add_stock("S", "10")
    model.add_flow("leak", "1")  # no from_stock / to_stock
    model._auto_layout()
    leak = model.flows[model._normalize_name("leak")]
    assert leak.x is not None and leak.y is not None
    svg = render_model_svg(model)
    ET.fromstring(svg)  # would raise if render failed
    counts = _count_by_class(svg)
    assert counts["flow-pipe"] == 1
    # Both ends unattached -> a source cloud and a sink cloud.
    assert counts.get("cloud", 0) == 2


def test_render_locked_connector_uses_stored_waypoints():
    model = StellaModel("Locked")
    model.add_aux("k", "1", x=50, y=50)
    model.add_stock("S", "100", x=200, y=50)
    connector = model.add_connector("k", "S")
    connector.points = [(60, 60), (130, 90), (190, 55)]
    connector.points_locked = True
    svg = render_model_svg(model)
    root = ET.fromstring(svg)
    locked = [e for e in root.iter() if e.get("class") == "connector"]
    assert len(locked) == 1
    assert locked[0].tag.endswith("polyline")  # waypoints, not a computed arc
    assert "130,90" in locked[0].get("points")


def test_render_auto_routed_connector_uses_stored_waypoints():
    model = StellaModel("Routed")
    model.add_aux("k", "1", x=50, y=50)
    model.add_stock("S", "100", x=200, y=50)
    connector = model.add_connector("k", "S")
    connector.points = [(68, 50), (100, 50), (100, 32), (178, 32)]

    root = ET.fromstring(render_model_svg(model))
    [rendered] = [element for element in root.iter() if element.get("class") == "connector"]

    assert rendered.tag.endswith("polyline")
    assert rendered.get("points") == "68,50 100,50 100,32 178,32"


@pytest.mark.parametrize(
    ("side", "axis", "comparison"),
    [
        ("top", "y", "less"),
        ("bottom", "y", "greater"),
        ("left", "x", "less"),
        ("right", "x", "greater"),
    ],
)
def test_render_respects_element_label_side(side, axis, comparison):
    model = StellaModel("Labels")
    aux = model.add_aux("control", "1", x=100, y=100)
    aux.label_side = side

    root = ET.fromstring(render_model_svg(model))
    [label] = [element for element in root.iter() if element.get("class") == "label"]
    coordinate = float(label.get(axis))

    if comparison == "less":
        assert coordinate < 100
    else:
        assert coordinate > 100
