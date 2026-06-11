"""Tests for the unused-variable validation check."""

from stella_mcp.templates import load_template_model
from stella_mcp.validator import validate_model
from stella_mcp.xmile import StellaModel


def _unused(model: StellaModel) -> set[str]:
    return {
        e.variable for e in validate_model(model) if e.category == "unused_variable"
    }


def test_unreferenced_aux_is_flagged():
    model = StellaModel("Unused")
    model.add_stock("Population", "100")
    model.add_aux("growth_rate", "0.1")
    model.add_aux("orphan", "42")  # referenced by nothing
    model.add_flow("growth", "Population * growth_rate", to_stock="Population")
    assert _unused(model) == {"orphan"}


def test_aux_used_in_equation_not_flagged():
    model = StellaModel("Used")
    model.add_stock("Population", "100")
    model.add_aux("growth_rate", "0.1")
    model.add_flow("growth", "Population * growth_rate", to_stock="Population")
    assert "growth_rate" not in _unused(model)


def test_aux_used_only_via_connector_not_flagged():
    model = StellaModel("Connector")
    model.add_stock("S", "1", x=10, y=10)
    model.add_aux("lookup_input", "1", x=50, y=50)
    # No equation references it, but a connector does (e.g. a gf input).
    model.add_connector("lookup_input", "S")
    assert "lookup_input" not in _unused(model)


def test_aux_used_only_in_quoted_identifier_not_flagged():
    model = StellaModel("Quoted")
    model.add_stock("Population", "100")
    model.add_aux("net growth rate", "0.1")
    model.add_flow("growth", '"net growth rate" * Population', to_stock="Population")
    assert "net_growth_rate" not in _unused(model)


def test_aux_used_only_in_stock_initial_value_not_flagged():
    model = StellaModel("Initial")
    model.add_aux("seed", "50")
    model.add_stock("Population", "seed * 2")  # aux used in stock init only
    assert "seed" not in _unused(model)


def test_stocks_and_flows_are_never_flagged_unused():
    model = StellaModel("NoFlag")
    model.add_stock("Isolated", "100")  # no flows, but stocks are never flagged
    model.add_flow("drift", "1")  # orphan flow, but flows are never flagged
    assert _unused(model) == set()


def test_builtin_templates_have_no_unused_auxiliaries():
    for template in ("exponential_growth", "sir", "lotka_volterra",
                     "carbon_cycle_2box", "nutrient_box_2box"):
        _info, model = load_template_model(template)
        assert _unused(model) == set(), f"{template} has unexpected unused auxiliaries"
