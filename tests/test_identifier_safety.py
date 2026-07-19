"""Stella reserved-identifier validation and compatibility behavior."""

from __future__ import annotations

import asyncio

import pytest

from stella_mcp import server as server_mod
from stella_mcp.equation_parser import (
    StellaReservedIdentifierError,
    is_stella_reserved_identifier,
)
from stella_mcp.templates import load_template_model
from stella_mcp.validator import validate_model
from stella_mcp.xmile import StellaModel, parse_stmx


def test_identifier_predicate_excludes_legacy_parser_only_aliases():
    assert is_stella_reserved_identifier("beta") is True
    assert is_stella_reserved_identifier("GAMMA") is True
    assert is_stella_reserved_identifier("time") is True
    assert is_stella_reserved_identifier("GRAPH") is False
    assert is_stella_reserved_identifier("beta_1") is False
    assert is_stella_reserved_identifier("transmission rate") is False


def test_validator_warns_for_reserved_identifiers():
    model = StellaModel("Reserved")
    model.add_aux("beta", "0.3")
    model.add_aux("safe rate", "0.1")

    issues = validate_model(model)

    reserved = [issue for issue in issues if issue.category == "reserved_identifier"]
    assert [(issue.severity, issue.variable) for issue in reserved] == [("warning", "beta")]


def test_permissive_export_warns_but_strict_export_rejects(tmp_path):
    model = StellaModel("Reserved")
    model.add_aux("beta", "0.3")

    xml = model.to_xml(auto_layout=False, compat_mode="permissive")
    assert 'aux name="beta"' in xml
    assert any("may be renamed by Stella" in warning for warning in model.last_export_warnings)

    with pytest.raises(StellaReservedIdentifierError) as caught:
        model.to_xml(auto_layout=False, compat_mode="strict")
    assert caught.value.details == {"identifiers": ["beta"]}

    path = tmp_path / "reserved.stmx"
    path.write_text(xml, encoding="utf-8")
    permissive = parse_stmx(str(path), compat_mode="permissive")
    assert any("may be renamed by Stella" in warning for warning in permissive.compatibility_warnings)
    with pytest.raises(StellaReservedIdentifierError):
        parse_stmx(str(path), compat_mode="strict")


def test_mcp_strict_export_returns_structured_reserved_identifier_error(tmp_path):
    server_mod._clear_session_store()
    model = StellaModel("Reserved")
    model.add_aux("gamma", "0.1")
    server_mod._set_current_model(model, "reserved")

    result = asyncio.run(
        server_mod.call_tool(
            "save_model",
            {
                "model_id": "reserved",
                "filepath": str(tmp_path / "reserved.stmx"),
                "auto_layout": False,
                "compat_mode": "strict",
            },
        )
    )

    assert result.isError is True
    assert result.structuredContent["error"] == {
        "code": "reserved_identifier",
        "message": "Stella-reserved variable identifiers: gamma",
        "category": "compatibility",
        "identifiers": ["gamma"],
    }


def test_builtin_sir_uses_stella_safe_parameter_names():
    _, model = load_template_model("sir")

    assert set(model.auxs) == {"transmission_rate", "recovery_rate", "population_total"}
    assert model.flows["infection"].equation == (
        "transmission_rate * Susceptible * Infected / population_total"
    )
    assert model.flows["recovery"].equation == "recovery_rate * Infected"
    assert not [
        issue for issue in validate_model(model) if issue.category == "reserved_identifier"
    ]
    model.to_xml(auto_layout=False, compat_mode="strict")
