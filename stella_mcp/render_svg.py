"""Render a Stella model to SVG for visual verification.

Pure stdlib — no rasterization dependency. The output mirrors Stella's
visual vocabulary (stocks as rectangles, auxiliaries as circles, flows as
valved pipes, connectors as routed polylines) closely enough to read at a glance and
to spot layout problems without opening Stella.

Coordinate system: Stella stores element ``.x/.y`` as centers and uses a
y-down screen frame, which matches SVG, so positions pass through directly.
This renderer is pure — it does not run layout. Callers that want a
freshly built model positioned should run ``model._auto_layout()`` first;
rendering a model with unpositioned variables raises ``ValueError``.
"""

from __future__ import annotations

from xml.sax.saxutils import escape, quoteattr

from .layout_quality import (
    CSS_PIXELS_PER_POINT,
    DEFAULT_FONT_POINTS,
    estimate_label_box,
    label_font_points,
)
from .layout_router import element_box
from .xmile import AUX_RADIUS, StellaModel

_VALVE_HALF = 7.0  # Half-size of the flow valve bowtie glyph
_CLOUD_R = 11.0  # Radius of source/sink cloud glyph
_CONNECTOR_BOW = 0.18  # Perpendicular bow as a fraction of connector length


def _fmt(value: float) -> str:
    """Format a coordinate compactly and deterministically."""
    return f"{value:.2f}".rstrip("0").rstrip(".") if value % 1 else str(int(value))


def _label(
    cx: float,
    baseline_y: float,
    text: str,
    font_pixels: float = DEFAULT_FONT_POINTS * CSS_PIXELS_PER_POINT,
) -> str:
    default_pixels = DEFAULT_FONT_POINTS * CSS_PIXELS_PER_POINT
    font_size = "" if font_pixels == default_pixels else f' style="font-size:{_fmt(font_pixels)}px"'
    return (
        f'<text class="label" x="{_fmt(cx)}" y="{_fmt(baseline_y)}" '
        f'text-anchor="middle"{font_size}>{escape(text)}</text>'
    )


def _element_label_svg(model: StellaModel, key: str) -> tuple[str, list[tuple[float, float]]]:
    element = next(
        registry[key] for registry in (model.stocks, model.flows, model.auxs) if key in registry
    )
    glyph = element_box(model, key)
    assert glyph is not None
    font_points = label_font_points(model, key)
    label = estimate_label_box(
        glyph,
        model._display_name(element.name),
        element.label_side or "bottom",
        font_points,
    )
    baseline_y = label.y + label.height * 0.35
    return _label(
        label.x,
        baseline_y,
        model._display_name(element.name),
        font_points * CSS_PIXELS_PER_POINT,
    ), [
        (label.left, label.top),
        (label.right, label.bottom),
    ]


def _unpositioned(model: StellaModel) -> list[str]:
    """Names of variables missing an x or y coordinate (display form)."""
    missing: list[str] = []
    for registry in (model.stocks, model.auxs, model.flows):
        for key in sorted(registry):
            element = registry[key]
            if element.x is None or element.y is None:
                missing.append(model._display_name(element.name))
    return missing


def _stock_svg(model: StellaModel) -> tuple[list[str], list[tuple[float, float]]]:
    parts: list[str] = []
    extents: list[tuple[float, float]] = []
    for key in sorted(model.stocks):
        stock = model.stocks[key]
        half_w, half_h = stock.width / 2, stock.height / 2
        left, top = stock.x - half_w, stock.y - half_h
        parts.append(
            f'<rect class="stock" x="{_fmt(left)}" y="{_fmt(top)}" '
            f'width="{_fmt(stock.width)}" height="{_fmt(stock.height)}"/>'
        )
        label, label_extents = _element_label_svg(model, key)
        parts.append(label)
        extents += [(left, top), (stock.x + half_w, stock.y + half_h)]
        extents += label_extents
    return parts, extents


def _aux_svg(model: StellaModel) -> tuple[list[str], list[tuple[float, float]]]:
    parts: list[str] = []
    extents: list[tuple[float, float]] = []
    for key in sorted(model.auxs):
        aux = model.auxs[key]
        parts.append(
            f'<circle class="aux" cx="{_fmt(aux.x)}" cy="{_fmt(aux.y)}" r="{_fmt(AUX_RADIUS)}"/>'
        )
        label, label_extents = _element_label_svg(model, key)
        parts.append(label)
        extents += [
            (aux.x - AUX_RADIUS, aux.y - AUX_RADIUS),
            (aux.x + AUX_RADIUS, aux.y + AUX_RADIUS),
        ]
        extents += label_extents
    return parts, extents


def _cloud_svg(cx: float, cy: float) -> str:
    """A small lobed cloud marking an unattached flow end (source/sink)."""
    r = _CLOUD_R
    return (
        f'<path class="cloud" d="M{_fmt(cx - r)},{_fmt(cy)} '
        f"a{_fmt(r * 0.5)},{_fmt(r * 0.5)} 0 0 1 {_fmt(r * 0.5)},{_fmt(-r * 0.5)} "
        f"a{_fmt(r * 0.5)},{_fmt(r * 0.5)} 0 0 1 {_fmt(r)},0 "
        f"a{_fmt(r * 0.5)},{_fmt(r * 0.5)} 0 0 1 {_fmt(r * 0.5)},{_fmt(r * 0.5)} "
        f'a{_fmt(r * 0.5)},{_fmt(r * 0.5)} 0 0 1 {_fmt(-r * 2)},0 z"/>'
    )


def _flow_svg(model: StellaModel) -> tuple[list[str], list[tuple[float, float]]]:
    parts: list[str] = []
    extents: list[tuple[float, float]] = []
    for key in sorted(model.flows):
        flow = model.flows[key]
        points = flow.points or [(flow.x, flow.y)]
        pts_attr = " ".join(f"{_fmt(px)},{_fmt(py)}" for px, py in points)
        parts.append(f'<polyline class="flow-pipe" points="{pts_attr}"/>')
        parts.append(
            f'<polyline class="flow-pipe-inner" points="{pts_attr}" marker-end="url(#flow-arrow)"/>'
        )
        # Valve bowtie centered on the flow position.
        vx, vy, h = flow.x, flow.y, _VALVE_HALF
        parts.append(
            f'<path class="flow-valve" d="M{_fmt(vx - h)},{_fmt(vy - h)} '
            f"L{_fmt(vx + h)},{_fmt(vy + h)} L{_fmt(vx + h)},{_fmt(vy - h)} "
            f'L{_fmt(vx - h)},{_fmt(vy + h)} z"/>'
        )
        # Clouds mark ends not attached to a stock.
        if flow.from_stock is None and points:
            parts.append(_cloud_svg(*points[0]))
        if flow.to_stock is None and points:
            parts.append(_cloud_svg(*points[-1]))
        label, label_extents = _element_label_svg(model, key)
        parts.append(label)
        extents += [(vx - h, vy - h), (vx + h, vy + h)]
        extents += [(px, py) for px, py in points]
        extents += label_extents
    return parts, extents


def _element_center(model: StellaModel, key: str) -> tuple[float, float] | None:
    for registry in (model.stocks, model.auxs, model.flows):
        element = registry.get(key)
        if element is not None and element.x is not None and element.y is not None:
            return (element.x, element.y)
    return None


def _connector_svg(model: StellaModel) -> tuple[list[str], list[tuple[float, float]]]:
    parts: list[str] = []
    extents: list[tuple[float, float]] = []
    # Sort by uid for deterministic output.
    for connector in sorted(model.connectors, key=lambda c: c.uid):
        if connector.points:
            pts = " ".join(f"{_fmt(px)},{_fmt(py)}" for px, py in connector.points)
            parts.append(
                f'<polyline class="connector" points="{pts}" marker-end="url(#connector-arrow)"/>'
            )
            extents += [(px, py) for px, py in connector.points]
            continue
        src = _element_center(model, connector.from_var)
        dst = _element_center(model, connector.to_var)
        if src is None or dst is None:
            continue
        (sx, sy), (dx, dy) = src, dst
        # Imported connectors without point data retain the legacy arc preview.
        mx, my = (sx + dx) / 2, (sy + dy) / 2
        vx, vy = dx - sx, dy - sy
        cx, cy = mx - vy * _CONNECTOR_BOW, my + vx * _CONNECTOR_BOW
        parts.append(
            f'<path class="connector" d="M{_fmt(sx)},{_fmt(sy)} '
            f'Q{_fmt(cx)},{_fmt(cy)} {_fmt(dx)},{_fmt(dy)}" '
            f'marker-end="url(#connector-arrow)"/>'
        )
        extents += [(sx, sy), (dx, dy), (cx, cy)]
    return parts, extents


def _module_svg(model: StellaModel) -> tuple[list[str], list[tuple[float, float]]]:
    parts: list[str] = []
    extents: list[tuple[float, float]] = []
    for key in sorted(model.modules):
        module = model.modules[key]
        if None in (module.x, module.y, module.width, module.height):
            continue
        left, top = module.x - module.width / 2, module.y - module.height / 2
        style = f" fill={quoteattr(module.background)}" if module.background else ""
        style += f" stroke={quoteattr(module.border_color)}" if module.border_color else ""
        parts.append(
            f'<rect class="module" x="{_fmt(left)}" y="{_fmt(top)}" '
            f'width="{_fmt(module.width)}" height="{_fmt(module.height)}" rx="6"{style}/>'
        )
        label_y = top - 4 if module.label_side == "top" else top + 14
        parts.append(_label(module.x, label_y, model._display_name(module.name)))
        extents += [(left, top), (module.x + module.width / 2, module.y + module.height / 2)]
    return parts, extents


def render_model_svg(model: StellaModel, *, margin: float = 40.0) -> str:
    """Render the model to an SVG document string.

    Pure function: requires that every stock, auxiliary, and flow already
    has a position. Raises ``ValueError`` naming the unpositioned variables
    otherwise (the caller runs layout first when ``auto_layout`` is set).
    """
    missing = _unpositioned(model)
    if missing:
        raise ValueError(
            "Cannot render: these variables have no position: "
            f"{', '.join(missing)}. Render with auto_layout=true to position them."
        )

    # Modules render behind elements; connectors and flows beneath stocks/auxs.
    module_parts, module_ext = _module_svg(model)
    connector_parts, connector_ext = _connector_svg(model)
    flow_parts, flow_ext = _flow_svg(model)
    stock_parts, stock_ext = _stock_svg(model)
    aux_parts, aux_ext = _aux_svg(model)

    extents = module_ext + connector_ext + flow_ext + stock_ext + aux_ext
    if extents:
        min_x = min(x for x, _ in extents) - margin
        min_y = min(y for _, y in extents) - margin
        max_x = max(x for x, _ in extents) + margin
        max_y = max(y for _, y in extents) + margin
    else:
        min_x = min_y = 0.0
        max_x = max_y = margin * 2
    width, height = max_x - min_x, max_y - min_y

    body = module_parts + connector_parts + flow_parts + stock_parts + aux_parts
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{_fmt(min_x)} {_fmt(min_y)} {_fmt(width)} {_fmt(height)}" '
        f'width="{_fmt(width)}" height="{_fmt(height)}">',
        '<defs><marker id="flow-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" fill="#000"/></marker>'
        '<marker id="connector-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" fill="#FF007F"/></marker></defs>',
        "<style>"
        ".canvas{fill:#fff}"
        ".stock{fill:#fff;stroke:#000;stroke-width:1.5}"
        ".aux{fill:#fff;stroke:#000;stroke-width:1.5}"
        ".flow-pipe{fill:none;stroke:#000;stroke-width:5;stroke-linejoin:round}"
        ".flow-pipe-inner{fill:none;stroke:#fff;stroke-width:2;stroke-linejoin:round}"
        ".flow-valve{fill:#fff;stroke:#000;stroke-width:1.5}"
        ".cloud{fill:#fff;stroke:#000;stroke-width:1}"
        ".connector{fill:none;stroke:#FF007F;stroke-width:1}"
        ".module{fill:none;stroke:#666;stroke-width:1}"
        ".label{font-family:Arial,sans-serif;font-size:12px;fill:#000}"
        "</style>",
        f'<rect class="canvas" x="{_fmt(min_x)}" y="{_fmt(min_y)}" '
        f'width="{_fmt(width)}" height="{_fmt(height)}" fill="#fff"/>',
        *body,
        "</svg>",
    ]
    return "\n".join(lines)
