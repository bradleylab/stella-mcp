"""Deterministic semantic signatures and differences for Stella models."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from stella_mcp.model_snapshot import model_to_summary
from stella_mcp.xmile import StellaModel


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _element_signature(element: ET.Element) -> tuple[Any, ...]:
    """Normalize XML syntax while retaining structure, order, names, and equations."""
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or "").strip(),
        tuple(_element_signature(child) for child in element),
    )


def unsupported_xml_signature(root: ET.Element) -> dict[str, Any]:
    """Return preserved-only structures at their required XMILE levels."""
    models = [child for child in root if _local_name(child.tag) == "model"]
    if not models:
        return {
            "root_extras": (),
            "variable_dimensions": (),
            "variables_extras": (),
            "additional_models": (),
        }
    primary_model = models[0]
    variables = next(
        (child for child in primary_model if _local_name(child.tag) == "variables"),
        None,
    )
    variable_dimensions = []
    variables_extras = []
    if variables is not None:
        for variable in variables:
            if _local_name(variable.tag) not in {"stock", "flow", "aux"}:
                continue
            dimensions = [
                _element_signature(child)
                for child in variable
                if _local_name(child.tag) == "dimensions"
            ]
            if dimensions:
                variable_dimensions.append(
                    (_local_name(variable.tag), variable.get("name"), tuple(dimensions))
                )
        variables_extras = [
            _element_signature(child)
            for child in variables
            if _local_name(child.tag) not in {"stock", "flow", "aux", "group"}
        ]

    return {
        "root_extras": tuple(
            _element_signature(child)
            for child in root
            if _local_name(child.tag) not in {"header", "sim_specs", "prefs", "model"}
        ),
        "variable_dimensions": tuple(variable_dimensions),
        "variables_extras": tuple(variables_extras),
        "additional_models": tuple(_element_signature(model) for model in models[1:]),
    }


def model_semantic_signature(model: StellaModel) -> dict[str, Any]:
    """Return all supported model semantics in stable JSON-safe form."""
    summary = model_to_summary("fidelity", model)
    variables = summary["variables"]
    for entry in variables["stocks"]:
        entry["label_side"] = model.stocks[entry["key"]].label_side
    for entry in variables["flows"]:
        entry["label_side"] = model.flows[entry["key"]].label_side
    for entry in variables["auxiliaries"]:
        entry["label_side"] = model.auxs[entry["key"]].label_side
    modules = summary["modules"]
    for module in modules:
        module["members"] = sorted(module["members"])
    return {
        "name": summary["name"],
        "sim_specs": summary["sim_specs"],
        "view": {
            "page_width": model.view_page_width,
            "page_height": model.view_page_height,
            "page_columns": model.view_page_columns,
            "page_rows": model.view_page_rows,
            "stock_font_points": model.view_stock_font_points,
            "flow_font_points": model.view_flow_font_points,
            "aux_font_points": model.view_aux_font_points,
        },
        "variables": variables,
        "connectors": summary["connectors"],
        "modules": modules,
    }


def model_metadata_signature(model: StellaModel) -> dict[str, Any]:
    """Return non-semantic model metadata that may vary across round-trips."""
    return {"uuid": model.uuid}


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def structured_diff(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
    """Return stable JSON Pointer changes between two JSON-safe values."""
    if isinstance(before, dict) and isinstance(after, dict):
        changes: list[dict[str, Any]] = []
        for key in sorted(before.keys() | after.keys()):
            child_path = f"{path}/{_pointer_token(str(key))}"
            if key not in before:
                changes.append(
                    {"path": child_path, "kind": "added", "before": None, "after": after[key]}
                )
            elif key not in after:
                changes.append(
                    {
                        "path": child_path,
                        "kind": "removed",
                        "before": before[key],
                        "after": None,
                    }
                )
            else:
                changes.extend(structured_diff(before[key], after[key], child_path))
        return changes

    if isinstance(before, list) and isinstance(after, list):
        changes = []
        common = min(len(before), len(after))
        for index in range(common):
            changes.extend(structured_diff(before[index], after[index], f"{path}/{index}"))
        for index in range(common, len(before)):
            changes.append(
                {
                    "path": f"{path}/{index}",
                    "kind": "removed",
                    "before": before[index],
                    "after": None,
                }
            )
        for index in range(common, len(after)):
            changes.append(
                {
                    "path": f"{path}/{index}",
                    "kind": "added",
                    "before": None,
                    "after": after[index],
                }
            )
        return changes

    if before != after:
        return [{"path": path or "/", "kind": "changed", "before": before, "after": after}]
    return []


def compare_model_fidelity(before: StellaModel, after: StellaModel) -> dict[str, Any]:
    """Compare supported semantics and separately report metadata drift."""
    before_semantics = model_semantic_signature(before)
    after_semantics = model_semantic_signature(after)
    before_metadata = model_metadata_signature(before)
    after_metadata = model_metadata_signature(after)
    semantic_changes = structured_diff(before_semantics, after_semantics)
    metadata_changes = structured_diff(before_metadata, after_metadata)
    return {
        "semantic_equal": not semantic_changes,
        "semantic_changes": semantic_changes,
        "metadata_changes": metadata_changes,
    }
