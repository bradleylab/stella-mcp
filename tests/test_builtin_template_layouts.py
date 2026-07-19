"""Release gates for the authored layouts shipped with built-in templates."""

from __future__ import annotations

import pytest

from stella_mcp.layout_quality import analyze_layout
from stella_mcp.templates import list_templates, load_template_model

_HARD_VIOLATION_FIELDS = (
    "missing_positions",
    "glyph_overlaps",
    "label_glyph_overlaps",
    "label_label_overlaps",
    "flow_glyph_crossings",
    "connector_glyph_crossings",
    "flow_label_crossings",
    "connector_label_crossings",
    "flow_flow_crossings",
    "connector_flow_crossings",
    "connector_connector_crossings",
    "flow_shared_segments",
    "route_self_intersections",
    "repeated_route_points",
    "avoidable_route_detours",
)


@pytest.mark.parametrize(
    "template_name",
    [template.name for template in list_templates(source="builtin")],
)
def test_builtin_template_authored_layout_is_clean_and_single_page(template_name: str) -> None:
    _, model = load_template_model(template_name)
    metrics = analyze_layout(model)

    for field in _HARD_VIOLATION_FIELDS:
        assert getattr(metrics, field) == (), f"{template_name}: {field}"
    left, top, right, bottom = metrics.bounds
    assert 0 <= left <= right <= model.view_page_width
    assert 0 <= top <= bottom <= model.view_page_height
