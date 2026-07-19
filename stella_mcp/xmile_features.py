"""Detect XMILE constructs that require explicit compatibility handling."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _direct_children(parent: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(parent) if _local_name(child.tag) == name]


def _truthy_flag(value: str | None) -> bool:
    return bool(value and value.strip().lower() not in {"0", "false", "no"})


@dataclass(frozen=True)
class XmileFeatureFinding:
    """One deterministic compatibility finding from an XMILE document."""

    code: str
    status: str
    message: str
    locations: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.locations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "status": self.status,
            "message": self.message,
            "count": self.count,
            "locations": list(self.locations),
        }


@dataclass(frozen=True)
class XmileFeatureReport:
    """Feature support report retained with an imported model."""

    findings: tuple[XmileFeatureFinding, ...] = ()

    @property
    def preserved_only(self) -> tuple[XmileFeatureFinding, ...]:
        return tuple(finding for finding in self.findings if finding.status == "preserved_only")

    @property
    def preserved_only_codes(self) -> tuple[str, ...]:
        return tuple(finding.code for finding in self.preserved_only)

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": not self.preserved_only,
            "findings": [finding.to_dict() for finding in self.findings],
        }


class UnsupportedModelFeatureError(ValueError):
    """Raised when strict or simulation behavior encounters preserved-only XMILE."""

    def __init__(self, findings: tuple[XmileFeatureFinding, ...]):
        self.findings = findings
        codes = ", ".join(finding.code for finding in findings)
        super().__init__(f"Unsupported XMILE model features: {codes}")

    @property
    def details(self) -> dict[str, Any]:
        return {
            "feature_codes": [finding.code for finding in self.findings],
            "findings": [finding.to_dict() for finding in self.findings],
        }


def detect_xmile_features(root: ET.Element) -> XmileFeatureReport:
    """Return preserved-only feature findings before model conversion."""
    findings: list[XmileFeatureFinding] = []
    models = _direct_children(root, "model")

    array_locations: list[str] = []
    if _direct_children(root, "dimensions"):
        array_locations.append("/xmile/dimensions")
    headers = _direct_children(root, "header")
    if headers:
        smiles = _direct_children(headers[0], "smile")
        if smiles and _truthy_flag(smiles[0].get("uses_arrays")):
            array_locations.append("/xmile/header/smile/@uses_arrays")
    for model_index, model in enumerate(models):
        variables = _direct_children(model, "variables")
        if not variables:
            continue
        for variable in list(variables[0]):
            if _direct_children(variable, "dimensions"):
                variable_name = variable.get("name", "")
                array_locations.append(
                    f"/xmile/model[{model_index}]/variables/{_local_name(variable.tag)}"
                    f"[@name={variable_name!r}]/dimensions"
                )
    if array_locations:
        findings.append(
            XmileFeatureFinding(
                code="xmile.arrays",
                status="preserved_only",
                message=(
                    "XMILE arrays and dimensions are preserved for permissive "
                    "round-trips but are not modeled or simulated"
                ),
                locations=tuple(dict.fromkeys(array_locations)),
            )
        )

    module_locations: list[str] = []
    for model_index, model in enumerate(models):
        variables = _direct_children(model, "variables")
        if not variables:
            continue
        for module in _direct_children(variables[0], "module"):
            module_locations.append(
                f"/xmile/model[{model_index}]/variables/module"
                f"[@name={module.get('name', '')!r}]"
            )
    if module_locations:
        findings.append(
            XmileFeatureFinding(
                code="xmile.module_instances",
                status="preserved_only",
                message=(
                    "XMILE module instances are preserved for permissive round-trips "
                    "but are not modeled or simulated"
                ),
                locations=tuple(module_locations),
            )
        )

    if len(models) > 1:
        findings.append(
            XmileFeatureFinding(
                code="xmile.nested_models",
                status="preserved_only",
                message=(
                    "Additional top-level XMILE models are preserved for permissive "
                    "round-trips but are not modeled or simulated"
                ),
                locations=tuple(
                    f"/xmile/model[{index}][@name={model.get('name', '')!r}]"
                    for index, model in enumerate(models[1:], start=1)
                ),
            )
        )

    return XmileFeatureReport(findings=tuple(findings))


def ensure_supported_for_simulation(report: XmileFeatureReport) -> None:
    """Reject retained features that the PySD bridge cannot represent safely."""
    if report.preserved_only:
        raise UnsupportedModelFeatureError(report.preserved_only)
