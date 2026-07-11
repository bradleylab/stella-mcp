"""Core Stella model data types and shared namespace constants."""

from __future__ import annotations

from dataclasses import dataclass, field

XMILE_NS = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"
ISEE_NS = "http://iseesystems.com/XMILE"
AUX_RADIUS = 18


@dataclass
class Stock:
    """Represents a stock (reservoir) in the model."""

    name: str
    initial_value: str
    units: str = ""
    inflows: list[str] = field(default_factory=list)
    outflows: list[str] = field(default_factory=list)
    non_negative: bool = True
    x: float | None = None
    y: float | None = None
    width: int = 45
    height: int = 35
    size_locked: bool = False
    extra_attrs: dict[str, str] = field(default_factory=dict)
    extra_children_xml: list[str] = field(default_factory=list)
    view_extra_attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class GraphicalFunction:
    """Represents a graphical function (lookup table) definition."""

    ypts: list[float]
    xscale: tuple[float, float] | None = None
    xpts: list[float] | None = None
    yscale: tuple[float, float] | None = None
    gf_type: str | None = None


@dataclass
class Flow:
    """Represents a flow between stocks."""

    name: str
    equation: str
    units: str = ""
    from_stock: str | None = None
    to_stock: str | None = None
    non_negative: bool = True
    x: float | None = None
    y: float | None = None
    points: list[tuple[float, float]] = field(default_factory=list)
    points_locked: bool = False
    graphical_function: GraphicalFunction | None = None
    extra_attrs: dict[str, str] = field(default_factory=dict)
    extra_children_xml: list[str] = field(default_factory=list)
    view_extra_attrs: dict[str, str] = field(default_factory=dict)
    view_extra_children_xml: list[str] = field(default_factory=list)


@dataclass
class Aux:
    """Represents an auxiliary variable."""

    name: str
    equation: str
    units: str = ""
    x: float | None = None
    y: float | None = None
    graphical_function: GraphicalFunction | None = None
    extra_attrs: dict[str, str] = field(default_factory=dict)
    extra_children_xml: list[str] = field(default_factory=list)
    view_extra_attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class Connector:
    """Represents a dependency connector between variables."""

    uid: int
    from_var: str
    to_var: str
    angle: float = 0
    angle_locked: bool = False
    points: list[tuple[float, float]] = field(default_factory=list)
    points_locked: bool = False
    extra_attrs: dict[str, str] = field(default_factory=dict)
    extra_children_xml: list[str] = field(default_factory=list)


@dataclass
class Module:
    """Represents a logical module/group of model variables."""

    name: str
    members: list[str] = field(default_factory=list)
    x: float | None = None
    y: float | None = None
    width: float | None = None
    height: float | None = None
    border_color: str | None = None
    background: str | None = None
    font_color: str | None = None
    font_size: str | None = None
    label_side: str | None = None
    extra_attrs: dict[str, str] = field(default_factory=dict)
    extra_children_xml: list[str] = field(default_factory=list)
    view_extra_attrs: dict[str, str] = field(default_factory=dict)
    view_extra_children_xml: list[str] = field(default_factory=list)


@dataclass
class SimSpecs:
    """Simulation specifications."""

    start: float = 0
    stop: float = 100
    dt: float = 0.25
    method: str = "Euler"
    time_units: str = "Years"
    extra_attrs: dict[str, str] = field(default_factory=dict)
    extra_children_xml: list[str] = field(default_factory=list)
