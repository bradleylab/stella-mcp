"""Tests for units consistency validation warnings."""

from stella_mcp.validator import validate_model
from stella_mcp.xmile import StellaModel


def _categories(model: StellaModel) -> list[str]:
    return [e.category for e in validate_model(model)]


def test_inconsistent_flow_units_fire():
    model = StellaModel("Carbon")
    model.sim_specs.time_units = "Years"
    model.add_stock("Atmosphere", "100", units="GtC")
    model.add_aux("rate", "0.1")
    # Flow units 'GtC' (no per-time) against a GtC stock -> should warn.
    model.add_flow("respiration", "Atmosphere * rate", units="GtC", from_stock="Atmosphere")
    issues = validate_model(model)
    bad = [e for e in issues if e.category == "units_inconsistent"]
    assert len(bad) == 1
    assert "GtC/Years" in bad[0].message
    assert bad[0].severity == "warning"


def test_consistent_flow_units_exact_match_silent():
    model = StellaModel("Carbon")
    model.sim_specs.time_units = "Years"
    model.add_stock("Atmosphere", "100", units="GtC")
    model.add_flow("respiration", "1", units="GtC/Years", from_stock="Atmosphere")
    assert "units_inconsistent" not in _categories(model)


def test_consistent_flow_units_plural_variant_silent():
    model = StellaModel("Carbon")
    model.sim_specs.time_units = "Years"
    model.add_stock("Atmosphere", "100", units="GtC")
    # 'GtC/year' (singular) vs time unit 'Years' (plural) must still be silent.
    model.add_flow("respiration", "1", units="GtC/year", from_stock="Atmosphere")
    assert "units_inconsistent" not in _categories(model)


def test_unitless_model_is_silent():
    model = StellaModel("Teaching")
    model.add_stock("Population", "100")
    model.add_aux("rate", "0.1")
    model.add_flow("growth", "Population * rate", to_stock="Population")
    cats = _categories(model)
    assert "units_missing" not in cats
    assert "units_inconsistent" not in cats


def test_one_united_stock_fires_units_missing_on_blank_flow():
    model = StellaModel("Mixed")
    model.add_stock("Atmosphere", "100", units="GtC")
    model.add_flow("respiration", "1", from_stock="Atmosphere")  # no units
    missing = [e for e in validate_model(model) if e.category == "units_missing"]
    assert {e.variable for e in missing} == {"respiration"}


def test_aux_without_units_never_warns():
    model = StellaModel("Mixed")
    model.add_stock("Atmosphere", "100", units="GtC")
    model.add_flow("respiration", "1", units="GtC/Years", from_stock="Atmosphere")
    model.add_aux("rate", "0.1")  # dimensionless parameter, no units
    missing = [e for e in validate_model(model) if e.category == "units_missing"]
    assert all(e.variable != "rate" for e in missing)
    assert missing == []  # stock + flow both have units; aux is exempt


def test_flow_with_units_but_stock_without_is_silent_for_inconsistency():
    model = StellaModel("Partial")
    model.sim_specs.time_units = "Years"
    model.add_stock("Atmosphere", "100")  # no units
    model.add_flow("respiration", "1", units="GtC/Years", from_stock="Atmosphere")
    # Insufficient information (stock has no units) != inconsistency.
    assert "units_inconsistent" not in _categories(model)


def test_conversion_flow_between_differing_stock_units_silent():
    model = StellaModel("Conversion")
    model.sim_specs.time_units = "Years"
    model.add_stock("Carbon", "100", units="GtC")
    model.add_stock("Mass", "0", units="kg")
    # A flow connecting differently-united stocks is a legitimate conversion.
    model.add_flow("convert", "1", units="GtC/Years", from_stock="Carbon", to_stock="Mass")
    assert "units_inconsistent" not in _categories(model)


def test_complex_flow_units_with_multiple_slashes_silent():
    model = StellaModel("Complex")
    model.sim_specs.time_units = "Years"
    model.add_stock("Atmosphere", "100", units="GtC")
    # Two slashes: too complex to judge confidently -> stay silent.
    model.add_flow("respiration", "1", units="GtC/Years/m2", from_stock="Atmosphere")
    assert "units_inconsistent" not in _categories(model)
