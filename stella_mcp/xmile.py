"""Compatibility facade for Stella model types and XMILE parsing."""

from __future__ import annotations

from .model import StellaModel
from .model_types import (
    AUX_RADIUS,
    ISEE_NS,
    XMILE_NS,
    Aux,
    Connector,
    Flow,
    GraphicalFunction,
    Module,
    SimSpecs,
    Stock,
)

__all__ = [
    "AUX_RADIUS",
    "ISEE_NS",
    "XMILE_NS",
    "Aux",
    "Connector",
    "Flow",
    "GraphicalFunction",
    "Module",
    "SimSpecs",
    "StellaModel",
    "Stock",
    "parse_stmx",
]


def parse_stmx(filepath: str, compat_mode: str = "permissive") -> StellaModel:
    """Parse an existing .stmx file into a StellaModel."""
    from .xmile_io import parse_stmx_file

    return parse_stmx_file(filepath, compat_mode=compat_mode)
