"""Tests for the directed stock-flow layout graph."""

from stella_mcp.layout_graph import build_stock_graph, place_stock_backbone
from stella_mcp.xmile import StellaModel
from tests.support.layout_fixtures import build_chain, build_feedback, build_mixed_pins


def test_chain_uses_increasing_ranks_and_one_aligned_pipe():
    model = build_chain()

    graph, warnings = place_stock_backbone(model)

    positions = [(model.stocks[f"stock_{index}"].x, model.stocks[f"stock_{index}"].y) for index in range(10)]
    assert all(positions[index][0] < positions[index + 1][0] for index in range(9))
    assert len({position[1] for position in positions}) == 1
    assert set(dict(graph.ranks).values()) == set(range(10))
    assert warnings == ()


def test_feedback_cycle_collapses_to_one_ring_component():
    model = build_feedback()

    graph, warnings = place_stock_backbone(model)

    assert len(graph.components) == 1
    assert len(graph.components[0]) == 6
    assert len({(stock.x, stock.y) for stock in model.stocks.values()}) == 6
    assert min(model.stocks, key=lambda name: (model.stocks[name].y, name)) == "stock_0"
    assert warnings == ()


def test_mixed_pins_remain_exact_and_free_stock_is_between_them():
    model = build_mixed_pins()

    _, warnings = place_stock_backbone(model)

    source = model.stocks["pinned_source"]
    middle = model.stocks["free_middle"]
    destination = model.stocks["pinned_destination"]
    assert (source.x, source.y) == (120.5, 280.25)
    assert (destination.x, destination.y) == (620.5, 280.25)
    assert source.x < middle.x < destination.x
    assert middle.y == 280.0
    assert warnings == ()


def test_disconnected_stock_components_pack_into_rows():
    model = StellaModel("Packing")
    for index in range(8):
        model.add_stock(f"stock_{index}", "0")

    graph, _ = place_stock_backbone(model)

    assert len(graph.weak_components) == 8
    assert max(stock.x for stock in model.stocks.values()) < model.view_page_width
    assert len({stock.y for stock in model.stocks.values()}) > 1


def test_cross_component_controls_align_single_component_rows():
    model = StellaModel("Cross-component controls")
    model.add_stock("Predator", "1")
    model.add_stock("Prey", "1")
    model.add_flow("predator_births", "1", to_stock="Predator")
    model.add_flow("predator_deaths", "1", from_stock="Predator")
    model.add_flow("prey_births", "1", to_stock="Prey")
    model.add_flow("predation", "1", from_stock="Prey")
    model.add_connector("Prey", "predator_births")
    model.add_connector("Predator", "predation")

    place_stock_backbone(model)

    assert model.stocks["Prey"].x < model.stocks["Predator"].x
    assert model.stocks["Prey"].y > model.stocks["Predator"].y


def test_backward_pins_emit_conflict_without_moving_them():
    model = StellaModel("Pinned conflict")
    model.add_stock("source", "1", x=500, y=200)
    model.add_stock("target", "0", x=100, y=200)
    model.add_flow("transfer", "1", from_stock="source", to_stock="target")

    graph = build_stock_graph(model)
    _, warnings = place_stock_backbone(model)

    assert dict(graph.ranks)[dict(graph.component_by_node)["source"]] == 0
    assert (model.stocks["source"].x, model.stocks["target"].x) == (500, 100)
    assert [warning.code for warning in warnings] == ["layout.pinned_conflict"]
