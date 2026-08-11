"""Reproducible model builders for layout regression tests."""

from __future__ import annotations

from collections.abc import Callable

from stella_mcp.templates import list_templates, load_template_model
from stella_mcp.xmile import StellaModel

FixtureBuilder = Callable[[], StellaModel]


def reset_layout(model: StellaModel) -> StellaModel:
    """Clear authored geometry so a template exercises automatic layout."""
    for stock in model.stocks.values():
        stock.x = None
        stock.y = None
        stock.size_locked = False
        stock.position_source = "auto"
    for aux in model.auxs.values():
        aux.x = None
        aux.y = None
        aux.position_source = "auto"
    for flow in model.flows.values():
        flow.x = None
        flow.y = None
        flow.position_source = "auto"
        flow.points = []
        flow.points_locked = False
    for connector in model.connectors:
        connector.angle_locked = False
        connector.points = []
        connector.points_locked = False
    return model


def build_chain() -> StellaModel:
    model = StellaModel("Ten-stock chain")
    for index in range(10):
        model.add_stock(f"stock_{index}", "100")
    for index in range(9):
        rate = f"rate_{index}"
        flow = f"flow_{index}"
        model.add_aux(rate, "1")
        model.add_flow(
            flow,
            rate,
            from_stock=f"stock_{index}",
            to_stock=f"stock_{index + 1}",
        )
    model.sync_connectors_from_equations()
    return model


def build_fanout() -> StellaModel:
    model = StellaModel("Eight-way fanout")
    model.add_stock("hub", "1000")
    for index in range(8):
        destination = f"destination_{index}"
        rate = f"rate_{index}"
        model.add_stock(destination, "0")
        model.add_aux(rate, "0.001")
        model.add_flow(
            f"flow_{index}",
            f"hub * {rate}",
            from_stock="hub",
            to_stock=destination,
        )
    model.sync_connectors_from_equations()
    return model


def build_feedback() -> StellaModel:
    model = StellaModel("Six-stock feedback")
    for index in range(6):
        model.add_stock(f"stock_{index}", "100")
        model.add_aux(f"factor_{index}", "0.01")
        model.add_aux(
            f"control_{index}",
            f"stock_{(index + 2) % 6} * factor_{index}",
        )
    for index in range(6):
        model.add_flow(
            f"flow_{index}",
            f"control_{index}",
            from_stock=f"stock_{index}",
            to_stock=f"stock_{(index + 1) % 6}",
        )
    model.sync_connectors_from_equations()
    return model


def build_disconnected() -> StellaModel:
    model = StellaModel("Disconnected components")
    model.add_stock("source", "100")
    model.add_stock("destination", "0")
    model.add_aux("transfer rate", "1")
    model.add_flow(
        "transfer",
        '"transfer rate"',
        from_stock="source",
        to_stock="destination",
    )
    model.add_aux("observation", "10")
    model.add_aux("scaled observation", 'observation * 2')
    model.add_flow("external exchange", "0")
    model.sync_connectors_from_equations()
    return model


def build_mixed_pins() -> StellaModel:
    model = StellaModel("Mixed pinned and free")
    model.add_stock("pinned source", "100", x=120.5, y=280.25)
    model.add_stock("free middle", "0")
    model.add_stock("pinned destination", "0", x=620.5, y=280.25)
    model.add_flow(
        "first transfer",
        "1",
        from_stock="pinned source",
        to_stock="free middle",
    )
    model.add_flow(
        "second transfer",
        "1",
        from_stock="free middle",
        to_stock="pinned destination",
    )
    return model


def build_special_flows() -> StellaModel:
    model = StellaModel("Special flow forms")
    model.add_stock("reservoir", "100")
    model.add_flow("self loop", "0", from_stock="reservoir", to_stock="reservoir")
    model.add_flow("external input", "1", to_stock="reservoir")
    model.add_flow("external output", "1", from_stock="reservoir")
    model.add_flow("orphan exchange", "0")
    return model


def build_long_labels() -> StellaModel:
    model = StellaModel("Long labels")
    model.add_stock("upstream dissolved nutrient reservoir", "100")
    model.add_stock("downstream biologically available nutrient reservoir", "0")
    model.add_aux("temperature adjusted transformation coefficient", "0.01")
    model.add_flow(
        "temperature dependent nutrient transformation flux",
        '"temperature adjusted transformation coefficient"',
        from_stock="upstream dissolved nutrient reservoir",
        to_stock="downstream biologically available nutrient reservoir",
    )
    model.sync_connectors_from_equations()
    return model


def build_dense_planar() -> StellaModel:
    model = StellaModel("Dense planar dependency chain")
    model.add_stock("source", "100")
    model.add_stock("destination", "0")
    previous = "base factor"
    model.add_aux(previous, "1")
    for index in range(8):
        name = f"derived factor {index}"
        model.add_aux(name, f'"{previous}" + 1')
        previous = name
    model.add_flow(
        "transfer",
        f'"{previous}"',
        from_stock="source",
        to_stock="destination",
    )
    model.sync_connectors_from_equations()
    return model


def build_nonplanar() -> StellaModel:
    model = StellaModel("Non-planar dependency graph")
    left = [f"left_{index}" for index in range(3)]
    right = [f"right_{index}" for index in range(3)]
    for name in left:
        model.add_aux(name, "1")
    expression = " + ".join(left)
    for name in right:
        model.add_aux(name, expression)
    model.sync_connectors_from_equations()
    return model


def build_incremental_base() -> StellaModel:
    model = StellaModel("Incremental layout")
    model.add_stock("first", "100", x=120, y=240)
    model.add_stock("second", "0")
    model.add_aux("initial rate", "1")
    model.add_flow(
        "initial transfer",
        '"initial rate"',
        from_stock="first",
        to_stock="second",
    )
    model.sync_connectors_from_equations()
    return model


def extend_incremental(model: StellaModel) -> StellaModel:
    model.add_stock("third", "0")
    model.add_aux("added rate", "1")
    model.add_flow(
        "added transfer",
        '"added rate"',
        from_stock="first",
        to_stock="third",
    )
    model.sync_connectors_from_equations()
    return model


def fixture_builders() -> dict[str, FixtureBuilder]:
    """Return every non-template benchmark builder in deterministic order."""
    return {
        "chain": build_chain,
        "dense_planar": build_dense_planar,
        "disconnected": build_disconnected,
        "fanout": build_fanout,
        "feedback": build_feedback,
        "long_labels": build_long_labels,
        "mixed_pins": build_mixed_pins,
        "nonplanar": build_nonplanar,
        "special_flows": build_special_flows,
    }


def template_models() -> dict[str, StellaModel]:
    """Load every built-in template with authored geometry cleared."""
    models: dict[str, StellaModel] = {}
    for template in list_templates(source="builtin"):
        _, model = load_template_model(template.name)
        models[f"template_{template.name}"] = reset_layout(model)
    return models
