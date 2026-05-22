"""Agent-facing structured snapshots for Stella models."""

from __future__ import annotations

from typing import Any

from .templates import TemplateInfo
from .validator import ValidationError
from .xmile import Aux, Connector, Flow, GraphicalFunction, Module, StellaModel, Stock


def _point_to_dict(point: tuple[float, float]) -> dict[str, float]:
    return {"x": point[0], "y": point[1]}


def graphical_function_to_dict(gf: GraphicalFunction | None) -> dict[str, Any] | None:
    if gf is None:
        return None
    return {
        "ypts": gf.ypts,
        "xscale": {"min": gf.xscale[0], "max": gf.xscale[1]} if gf.xscale else None,
        "xpts": gf.xpts,
        "yscale": {"min": gf.yscale[0], "max": gf.yscale[1]} if gf.yscale else None,
        "type": gf.gf_type,
    }


def stock_to_dict(key: str, stock: Stock) -> dict[str, Any]:
    return {
        "key": key,
        "name": stock.name,
        "initial_value": stock.initial_value,
        "units": stock.units,
        "inflows": stock.inflows,
        "outflows": stock.outflows,
        "non_negative": stock.non_negative,
        "position": {"x": stock.x, "y": stock.y},
        "size": {"width": stock.width, "height": stock.height, "locked": stock.size_locked},
    }


def flow_to_dict(model: StellaModel, key: str, flow: Flow) -> dict[str, Any]:
    return {
        "key": key,
        "name": flow.name,
        "equation": flow.equation,
        "units": flow.units,
        "from_stock": model._display_name(flow.from_stock) if flow.from_stock else None,
        "to_stock": model._display_name(flow.to_stock) if flow.to_stock else None,
        "non_negative": flow.non_negative,
        "position": {"x": flow.x, "y": flow.y},
        "points": [_point_to_dict(point) for point in flow.points],
        "points_locked": flow.points_locked,
        "graphical_function": graphical_function_to_dict(flow.graphical_function),
    }


def aux_to_dict(key: str, aux: Aux) -> dict[str, Any]:
    return {
        "key": key,
        "name": aux.name,
        "equation": aux.equation,
        "units": aux.units,
        "position": {"x": aux.x, "y": aux.y},
        "graphical_function": graphical_function_to_dict(aux.graphical_function),
    }


def connector_to_dict(model: StellaModel, connector: Connector) -> dict[str, Any]:
    return {
        "uid": connector.uid,
        "from_var": connector.from_var,
        "from_display": model._display_name(connector.from_var),
        "to_var": connector.to_var,
        "to_display": model._display_name(connector.to_var),
        "angle": connector.angle,
        "angle_locked": connector.angle_locked,
        "points": [_point_to_dict(point) for point in connector.points],
        "points_locked": connector.points_locked,
    }


def module_to_dict(model: StellaModel, key: str, module: Module) -> dict[str, Any]:
    return {
        "key": key,
        "name": module.name,
        "members": [model._display_name(member) for member in module.members],
        "box": {
            "x": module.x,
            "y": module.y,
            "width": module.width,
            "height": module.height,
        },
        "style": {
            "border_color": module.border_color,
            "background": module.background,
            "font_color": module.font_color,
            "font_size": module.font_size,
            "label_side": module.label_side,
        },
    }


def validation_issue_to_dict(issue: ValidationError) -> dict[str, Any]:
    return {
        "severity": issue.severity,
        "category": issue.category,
        "message": issue.message,
        "variable": issue.variable,
    }


def template_info_to_dict(info: TemplateInfo) -> dict[str, Any]:
    return {
        "name": info.name,
        "source": info.source,
        "path": str(info.path),
        "title": info.title,
        "description": info.description,
        "tags": list(info.tags),
        "stocks": info.stocks,
        "flows": info.flows,
        "auxiliaries": info.auxiliaries,
        "modules": info.modules,
        "updated_at": info.updated_at,
    }


def model_to_summary(model_id: str, model: StellaModel) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "name": model.name,
        "uuid": model.uuid,
        "sim_specs": {
            "start": model.sim_specs.start,
            "stop": model.sim_specs.stop,
            "dt": model.sim_specs.dt,
            "method": model.sim_specs.method,
            "time_units": model.sim_specs.time_units,
        },
        "counts": {
            "stocks": len(model.stocks),
            "flows": len(model.flows),
            "auxiliaries": len(model.auxs),
            "connectors": len(model.connectors),
            "modules": len(model.modules),
        },
        "variables": {
            "stocks": [stock_to_dict(key, model.stocks[key]) for key in sorted(model.stocks)],
            "flows": [flow_to_dict(model, key, model.flows[key]) for key in sorted(model.flows)],
            "auxiliaries": [aux_to_dict(key, model.auxs[key]) for key in sorted(model.auxs)],
        },
        "connectors": [
            connector_to_dict(model, connector)
            for connector in sorted(model.connectors, key=lambda item: item.uid)
        ],
        "modules": [module_to_dict(model, key, model.modules[key]) for key in sorted(model.modules)],
        "compatibility_warnings": model.compatibility_warnings,
        "last_export_warnings": model.last_export_warnings,
    }
