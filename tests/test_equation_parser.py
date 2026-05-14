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
