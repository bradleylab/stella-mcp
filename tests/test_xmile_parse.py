"""Focused contracts for XMILE parser boundaries."""

from pathlib import Path

import pytest

from stella_mcp import xmile_io, xmile_parse
from stella_mcp.model_snapshot import model_to_summary
from stella_mcp.templates import builtin_template_dir
from stella_mcp.xmile import parse_stmx

CORPUS_DIR = Path(__file__).resolve().parent / "fixtures" / "compat_corpus"
BUILTIN_TEMPLATES = sorted(builtin_template_dir().glob("*.stmx"))


def test_xmile_io_reexports_parser_function():
    assert xmile_io.parse_stmx_file is xmile_parse.parse_stmx_file


def test_permissive_parser_preserves_warning_order_and_text():
    model = xmile_parse.parse_stmx_file(
        str(CORPUS_DIR / "malformed_permissive.stmx"),
        compat_mode="permissive",
    )

    assert model.compatibility_warnings == [
        "sim_specs.dt reciprocal value must be > 0, got 0.0",
        "Connector uid=2 missing from/to endpoint; skipped",
    ]


def test_strict_parser_retains_first_compatibility_failure():
    with pytest.raises(
        ValueError,
        match=r"sim_specs\.dt reciprocal value must be > 0, got 0\.0",
    ):
        xmile_parse.parse_stmx_file(
            str(CORPUS_DIR / "malformed_permissive.stmx"),
            compat_mode="strict",
        )


def test_xmile_facade_still_routes_to_parser(monkeypatch):
    sentinel = object()
    calls: list[tuple[str, str]] = []

    def fake_parse(filepath: str, compat_mode: str = "permissive"):
        calls.append((filepath, compat_mode))
        return sentinel

    monkeypatch.setattr(xmile_io, "parse_stmx_file", fake_parse)

    assert parse_stmx("model.stmx", compat_mode="strict") is sentinel
    assert calls == [("model.stmx", "strict")]


@pytest.mark.parametrize("template_path", BUILTIN_TEMPLATES, ids=lambda path: path.stem)
def test_builtin_template_parse_export_parse_is_equivalent(template_path, tmp_path):
    first = xmile_parse.parse_stmx_file(str(template_path), compat_mode="strict")
    exported = first.to_xml(auto_layout=False, compat_mode="strict")
    roundtrip_path = tmp_path / template_path.name
    roundtrip_path.write_text(exported, encoding="utf-8")
    second = xmile_parse.parse_stmx_file(str(roundtrip_path), compat_mode="strict")

    first_summary = model_to_summary(template_path.stem, first)
    second_summary = model_to_summary(template_path.stem, second)
    first_summary.pop("uuid")
    second_summary.pop("uuid")

    assert second_summary == first_summary
