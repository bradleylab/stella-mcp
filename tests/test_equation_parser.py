"""Tests for shared equation parsing logic."""

from stella_mcp.equation_parser import (
    extract_quoted_references,
    extract_variable_references,
)
from stella_mcp.validator import ModelValidator, validate_model
from stella_mcp.xmile import StellaModel


def test_extract_variable_references_filters_reserved_tokens():
    refs = extract_variable_references("IF Population > 100 THEN rate ELSE 0")
    assert refs == {"Population", "rate"}


def test_extended_builtins_are_not_variable_refs():
    eq = "SINWAVE(amplitude, period) + ARCTAN(x) + EXPRND(rate) + CLOCKTIME"
    refs = extract_variable_references(eq)
    assert refs == {"amplitude", "period", "x", "rate"}


def test_quoted_spans_returned_as_candidate_refs():
    # Quoted spans are candidate variable references (XMILE quoted
    # identifiers); callers filter against actual model variables.
    refs = extract_variable_references('LOOKUP("some label", Population)')
    assert refs == {"some label", "Population"}


def test_quoted_identifiers_extracted_as_refs():
    refs = extract_variable_references('"net growth rate" * Population')
    assert refs == {"net growth rate", "Population"}


def test_quoted_identifier_with_function_call():
    refs = extract_variable_references('MAX("carrying capacity" - Population, 0)')
    assert "carrying capacity" in refs


def test_extract_quoted_references_skips_reserved_and_numeric():
    assert extract_quoted_references('"TIME" + "100" + "real var"') == {"real var"}


def test_unresolved_quoted_ref_is_warning_not_error():
    model = StellaModel("Quoted")
    model.add_stock("Population", "100")
    model.add_aux("k", '"no such variable" * 2')
    issues = validate_model(model)
    quoted_issues = [e for e in issues if e.category == "unresolved_quoted_reference"]
    assert len(quoted_issues) == 1
    assert quoted_issues[0].severity == "warning"
    assert not any(e.category == "undefined_variable" for e in issues)


def test_resolved_quoted_ref_participates_in_validation():
    model = StellaModel("Quoted")
    model.add_stock("Population", "100")
    model.add_aux("net growth rate", "0.1")
    model.add_flow("growth", '"net growth rate" * Population', to_stock="Population")
    issues = validate_model(model)
    # The quoted ref resolves to a real variable: no undefined/unresolved
    # issues, but the missing-connection warnings fire for it like any ref.
    assert not any(
        e.category in ("undefined_variable", "unresolved_quoted_reference")
        for e in issues
    )
    missing = [e for e in issues if e.category == "missing_connection"]
    assert any("net growth rate" in e.message for e in missing)


def test_circular_dependency_check_ignores_unresolved_quoted_refs():
    model = StellaModel("Quoted")
    model.add_aux("a", '"not a variable" + 1')
    issues = validate_model(model)
    assert not any(e.category == "circular_dependency" for e in issues)


def test_sync_connectors_handles_quoted_refs():
    model = StellaModel("QuotedSync")
    model.add_stock("Population", "100")
    model.add_aux("net growth rate", "0.1")
    model.add_flow(
        "growth", '"net growth rate" * Population + "ignore me"',
        to_stock="Population",
    )
    summary = model.sync_connectors_from_equations()
    endpoints = {(c.from_var, c.to_var) for c in model.connectors}
    assert ("net_growth_rate", "growth") in endpoints
    assert ("Population", "growth") in endpoints
    assert not any("ignore" in f for f, _ in endpoints)
    assert summary["added"] == 2


def test_xmile_and_validator_use_same_reference_extraction():
    model = StellaModel("ParserConsistency")
    equation = 'MAX(Stock_A, LOOKUP("label", input_var)) + aux_1'

    xmile_refs = model._extract_variable_refs(equation)
    validator_refs = ModelValidator(model)._extract_variable_references(equation)

    # Both layers see identical candidates, including the quoted span
    # ("label" — unresolved quoted candidates are filtered downstream).
    # xmile normalizes names; these refs are already underscore-friendly.
    assert xmile_refs == validator_refs == {"Stock_A", "input_var", "aux_1", "label"}


def test_sync_connectors_from_equations_adds_missing_and_preserves_existing():
    """Connector sync should add missing equation dependencies without duplicating."""
    model = StellaModel("Sync")
    model.add_stock("S", "100")
    model.add_aux("k", "0.1")
    model.add_aux("modifier", "2")
    model.add_flow("loss", "S * k * modifier", from_stock="S")
    existing = model.add_connector("S", "loss")

    summary = model.sync_connectors_from_equations()

    endpoints = {(connector.from_var, connector.to_var) for connector in model.connectors}
    assert endpoints == {("S", "loss"), ("k", "loss"), ("modifier", "loss")}
    assert existing.uid == 1
    assert summary == {"added": 2, "existing": 1}
