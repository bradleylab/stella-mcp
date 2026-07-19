"""Generate reproducible layout-quality records and visual artifacts."""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluation.layout_fixtures import (
    build_incremental_base,
    extend_incremental,
    fixture_builders,
    template_models,
)
from stella_mcp.layout_quality import (
    ROUTE_BEND_CAP,
    ROUTE_LENGTH_MANHATTAN_MULTIPLIER,
    LayoutMetrics,
    LayoutResult,
    analyze_layout,
    layout_report_to_dict,
)
from stella_mcp.render_svg import render_model_svg
from stella_mcp.xmile import StellaModel

SCHEMA_VERSION = 1
EVALUATION_UUID_NAMESPACE = "https://github.com/bradleylab/stella-mcp/layout-evaluation/"
REPORT_FLOAT_DECIMAL_PLACES = 9


def _canonicalize_report_value(value: Any) -> Any:
    """Normalize report containers and insignificant float representation."""
    if isinstance(value, float):
        return round(value, REPORT_FLOAT_DECIMAL_PLACES)
    if isinstance(value, dict):
        return {key: _canonicalize_report_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize_report_value(item) for item in value]
    return value


def _positions(model: StellaModel) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for registry in (model.stocks, model.flows, model.auxs):
        for name, element in sorted(registry.items()):
            if element.x is not None and element.y is not None:
                result[name] = [element.x, element.y]
    return result


def _position_sources(model: StellaModel) -> dict[str, str]:
    return {
        name: element.position_source
        for registry in (model.stocks, model.flows, model.auxs)
        for name, element in sorted(registry.items())
    }


def _routes(model: StellaModel) -> dict[str, list[list[float]]]:
    result: dict[str, list[list[float]]] = {}
    for name, flow in sorted(model.flows.items()):
        result[f"flow:{name}"] = [[x, y] for x, y in flow.points]
    for connector in sorted(model.connectors, key=lambda item: item.uid):
        result[f"connector:{connector.uid}"] = [[x, y] for x, y in connector.points]
    return result


def _metric_counts(metrics: LayoutMetrics) -> dict[str, int | float]:
    record = metrics.to_dict()
    return {
        key: len(value) if isinstance(value, tuple) else value
        for key, value in record.items()
        if key not in {"bounds", "page_overflow"}
    }


def _case_record(
    model: StellaModel,
    metrics: LayoutMetrics,
    result: LayoutResult | None = None,
) -> dict[str, Any]:
    return {
        "elements": len(model.stocks) + len(model.flows) + len(model.auxs),
        "stocks": len(model.stocks),
        "flows": len(model.flows),
        "auxiliaries": len(model.auxs),
        "connectors": len(model.connectors),
        "positions": _positions(model),
        "position_sources": _position_sources(model),
        "routes": _routes(model),
        "metrics": asdict(metrics),
        "metric_counts": _metric_counts(metrics),
        "layout": layout_report_to_dict(result),
    }


def _write_artifacts(output_dir: Path, case_name: str, model: StellaModel) -> None:
    model.uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, EVALUATION_UUID_NAMESPACE + case_name))
    (output_dir / f"{case_name}.svg").write_text(render_model_svg(model), encoding="utf-8")
    (output_dir / f"{case_name}.stmx").write_text(
        model.to_xml(auto_layout=False),
        encoding="utf-8",
    )


def _incremental_record(output_dir: Path) -> dict[str, Any]:
    model = build_incremental_base()
    first_result = model._auto_layout()
    first_positions = _positions(model)
    first_metrics = analyze_layout(model)
    first_record = _case_record(model, first_metrics, first_result)
    _write_artifacts(output_dir, "incremental_before", model)

    extend_incremental(model)
    second_result = model._auto_layout()
    second_positions = _positions(model)
    second_metrics = analyze_layout(model)
    _write_artifacts(output_dir, "incremental_after", model)

    shared = sorted(set(first_positions).intersection(second_positions))
    displacement = {
        name: ((second_positions[name][0] - first_positions[name][0]) ** 2
               + (second_positions[name][1] - first_positions[name][1]) ** 2) ** 0.5
        for name in shared
    }
    return {
        "before": first_record,
        "after": _case_record(model, second_metrics, second_result),
        "moved_elements": sum(distance > 0 for distance in displacement.values()),
        "total_displacement": sum(displacement.values()),
        "displacement": displacement,
    }


def run_layout_evaluation(output_dir: Path) -> dict[str, Any]:
    """Run every layout fixture and write the artifacts to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    models = template_models()
    models.update({name: builder() for name, builder in fixture_builders().items()})
    cases: dict[str, Any] = {}
    for case_name, model in sorted(models.items()):
        result = model._auto_layout()
        metrics = analyze_layout(model)
        cases[case_name] = _case_record(model, metrics, result)
        _write_artifacts(output_dir, case_name, model)

    raw_record = {
        "schema_version": SCHEMA_VERSION,
        "acceptance": {
            "route_bend_cap": ROUTE_BEND_CAP,
            "route_length_manhattan_multiplier": ROUTE_LENGTH_MANHATTAN_MULTIPLIER,
        },
        "cases": cases,
        "incremental": _incremental_record(output_dir),
    }
    record = _canonicalize_report_value(raw_record)
    (output_dir / "layout-report.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the JSON record and generated SVG/STMX artifacts",
    )
    arguments = parser.parse_args()
    run_layout_evaluation(arguments.output_dir.resolve())


if __name__ == "__main__":
    main()
