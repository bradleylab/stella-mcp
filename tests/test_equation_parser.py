"""Tests for shared equation parsing logic."""

from stella_mcp.equation_parser import extract_variable_references
from stella_mcp.validator import ModelValidator
from stella_mcp.xmile import StellaModel


def test_extract_variable_references_filters_reserved_tokens():
    refs = extract_variable_references("IF Population > 100 THEN rate ELSE 0")
    assert refs == {"Population", "rate"}


def test_extract_variable_references_ignores_string_literals():
    refs = extract_variable_references('LOOKUP("Population", Population)')
    assert refs == {"Population"}


def test_xmile_and_validator_use_same_reference_extraction():
    model = StellaModel("ParserConsistency")
    equation = 'MAX(Stock_A, LOOKUP("label", input_var)) + aux_1'

    xmile_refs = model._extract_variable_refs(equation)
    validator_refs = ModelValidator(model)._extract_variable_references(equation)

    # xmile normalizes names, parser already emits underscore-friendly names here.
    assert xmile_refs == validator_refs == {"Stock_A", "input_var", "aux_1"}


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
