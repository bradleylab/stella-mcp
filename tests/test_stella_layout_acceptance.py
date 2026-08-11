"""Release gates for Stella Professional 4.1.1 layout evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pytest

from stella_mcp.model_types import AUX_RADIUS
from stella_mcp.xmile import parse_stmx

FIXTURES = Path(__file__).parent / "fixtures" / "layout"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
DESKTOP_CASES = [case for case in MANIFEST["fixtures"] if case["case"] != "format_spike"]
CONNECTOR_ENDPOINT_TOLERANCE = AUX_RADIUS + 1


def _rename(value: str | None, names: dict[str, str]) -> str | None:
    return names.get(value, value) if value is not None else None


def _rename_expression(expression: str, names: dict[str, str]) -> str:
    for source, target in sorted(names.items(), key=lambda item: -len(item[0])):
        expression = re.sub(rf"\b{re.escape(source)}\b", target, expression)
    return expression


def _semantic_signature(model, names: dict[str, str]):
    stocks = {
        _rename(name, names): (
            _rename_expression(stock.initial_value, names),
            tuple(_rename(flow, names) for flow in stock.inflows),
            tuple(_rename(flow, names) for flow in stock.outflows),
            stock.width,
            stock.height,
        )
        for name, stock in model.stocks.items()
    }
    flows = {
        _rename(name, names): (
            _rename_expression(flow.equation, names),
            _rename(flow.from_stock, names),
            _rename(flow.to_stock, names),
        )
        for name, flow in model.flows.items()
    }
    auxiliaries = {
        _rename(name, names): _rename_expression(aux.equation, names)
        for name, aux in model.auxs.items()
    }
    connectors = tuple(
        sorted(
            (
                connector.uid,
                _rename(connector.from_var, names),
                _rename(connector.to_var, names),
            )
            for connector in model.connectors
        )
    )
    simulation = (
        model.sim_specs.start,
        model.sim_specs.stop,
        model.sim_specs.dt,
        model.sim_specs.method,
        model.sim_specs.time_units,
    )
    return stocks, flows, auxiliaries, connectors, simulation


def _layout_elements(model, names: dict[str, str]):
    return {
        _rename(name, names): (element.x, element.y, element.label_side)
        for registry in (model.stocks, model.flows, model.auxs)
        for name, element in registry.items()
    }


@pytest.mark.parametrize("case", DESKTOP_CASES, ids=lambda case: case["case"])
def test_stella_4_1_1_release_fixture_semantics_and_layout(case):
    for artifact_key in ("source", "saved"):
        artifact = FIXTURES / case[f"{artifact_key}_file"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == case[
            f"{artifact_key}_sha256"
        ]

    names = case["identifier_renames"]
    # The historical SIR pair records the beta/gamma renames that now cause
    # intentional 0.13 strict-mode rejection. Preserve that 0.12 evidence in
    # permissive mode; all cases without known renames remain strict fixtures.
    compat_mode = "permissive" if names else "strict"
    source = parse_stmx(str(FIXTURES / case["source_file"]), compat_mode=compat_mode)
    saved = parse_stmx(str(FIXTURES / case["saved_file"]), compat_mode=compat_mode)

    assert _semantic_signature(source, names) == _semantic_signature(saved, {})
    assert _layout_elements(source, names) == _layout_elements(saved, {})
    assert (
        source.view_page_width,
        source.view_page_height,
        source.view_page_columns,
        source.view_page_rows,
    ) == (
        saved.view_page_width,
        saved.view_page_height,
        saved.view_page_columns,
        saved.view_page_rows,
    )

    for source_name, source_flow in source.flows.items():
        saved_flow = saved.flows[_rename(source_name, names)]
        assert source_flow.points == saved_flow.points

    saved_connectors = {connector.uid: connector for connector in saved.connectors}
    for source_connector in source.connectors:
        saved_connector = saved_connectors[source_connector.uid]
        assert len(source_connector.points) == len(saved_connector.points)
        assert source_connector.points[1:-1] == saved_connector.points[1:-1]
        assert math.dist(source_connector.points[0], saved_connector.points[0]) <= (
            CONNECTOR_ENDPOINT_TOLERANCE
        )
        assert math.dist(source_connector.points[-1], saved_connector.points[-1]) <= (
            CONNECTOR_ENDPOINT_TOLERANCE
        )
