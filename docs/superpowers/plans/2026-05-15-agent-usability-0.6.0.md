# Stella MCP Agent Usability 0.6.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Stella MCP easier and safer for agents to inspect, edit, validate, and automate by adding structured tool outputs, model inspection, explicit update tools, lint enforcement, and better docs.

**Architecture:** Keep the core model in `stella_mcp/xmile.py`, but move agent-facing serialization into a new focused module so MCP handlers can return stable `structuredContent` without duplicating formatting logic. Add update/edit methods on `StellaModel` only where they preserve existing invariants, then expose them through small handlers and schemas.

**Tech Stack:** Python 3.10+, MCP Python SDK, pytest, ruff, setuptools packaging.

---

## File Structure

- Create `stella_mcp/model_snapshot.py`: pure serialization helpers for models, variables, connectors, modules, validation issues, and templates.
- Create `stella_mcp/tool_results.py`: small helpers for successful `CallToolResult` responses with text plus `structuredContent`.
- Modify `stella_mcp/tool_handlers.py`: use `CallToolResult` for success responses where structured data is useful; add handlers for `inspect_model`, `set_sim_specs`, `update_stock`, `update_flow`, `update_aux`, and `sync_connectors_from_equations`.
- Modify `stella_mcp/tool_schemas.py`: add schemas for the new tools and document structured payload expectations.
- Modify `stella_mcp/xmile.py`: add minimal update methods and connector-sync logic on `StellaModel`.
- Modify `README.md`: add a concise "Recommended Agent Workflow" section and document the new tools.
- Modify `pyproject.toml`: add dev/test optional dependencies and ruff config.
- Modify `.github/workflows/ci.yml`: install dev extras and run ruff before pytest.
- Modify tests:
  - `tests/test_model_snapshot.py`
  - `tests/test_server_state_and_gf.py`
  - `tests/test_variable_lifecycle.py`
  - `tests/test_equation_parser.py`
  - `tests/test_positioning.py`
  - `tests/test_force_directed.py`

---

### Task 1: Add Dev Tooling and Make Full Lint Pass

**Files:**
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`
- Modify: `tests/test_positioning.py`
- Modify: `tests/test_force_directed.py`

- [ ] **Step 1: Add failing CI/tooling expectation**

Run:

```bash
uv run --with ruff ruff check .
```

Expected before changes: FAIL with unused imports/redefinitions in `tests/test_positioning.py` and `tests/test_force_directed.py`.

- [ ] **Step 2: Add dev extras and ruff config**

In `pyproject.toml`, add after `dependencies`:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8",
    "ruff>=0.8",
]

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
ignore = ["E501"]
```

- [ ] **Step 3: Clean `tests/test_force_directed.py` imports**

Replace:

```python
from stella_mcp.layout import BoundingBox, force_directed_layout
```

with:

```python
from stella_mcp.layout import force_directed_layout
```

- [ ] **Step 4: Clean `tests/test_positioning.py` top imports**

Replace:

```python
import pytest

from stella_mcp.layout import BoundingBox, segments_intersect, segment_intersects_box
from stella_mcp.xmile import StellaModel, parse_stmx
```

with:

```python
from stella_mcp.xmile import StellaModel, parse_stmx
```

Keep the local imports inside the geometry-specific tests at lines around 743-872.

- [ ] **Step 5: Fix unused local variable in connector crossing test**

In `tests/test_positioning.py`, replace:

```python
        # After layout, connector should not cross stock B
        if rate_pos[0] is not None and rate_pos[1] is not None:
            crosses = segment_intersects_box(rate_pos, flow_pos, stock_b_box)
            # Note: may still cross in some cases - this tests detection works
            # The important thing is the detection method exists and works
```

with:

```python
        # The assertion is on the helper behavior, not on the current best-effort layout.
        if rate_pos[0] is not None and rate_pos[1] is not None:
            assert isinstance(segment_intersects_box(rate_pos, flow_pos, stock_b_box), bool)
```

- [ ] **Step 6: Update CI to run lint and use dev extras**

In `.github/workflows/ci.yml`, replace the install/test steps with:

```yaml
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      - name: Run lint
        run: ruff check .
      - name: Run compatibility corpus tests
        run: python -m pytest tests/test_compatibility_corpus.py
      - name: Run full test suite
        run: python -m pytest
```

- [ ] **Step 7: Verify tooling**

Run:

```bash
uv run --extra dev ruff check .
uv run --extra dev python -m pytest
```

Expected: ruff passes; pytest reports `134 passed` or more.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .github/workflows/ci.yml tests/test_positioning.py tests/test_force_directed.py
git commit -m "test: add lint enforcement"
```

---

### Task 2: Add Model Snapshot Serialization Helpers

**Files:**
- Create: `stella_mcp/model_snapshot.py`
- Create: `tests/test_model_snapshot.py`

- [ ] **Step 1: Write failing snapshot tests**

Create `tests/test_model_snapshot.py`:

```python
"""Tests for agent-facing model snapshot serialization."""

from stella_mcp.model_snapshot import (
    connector_to_dict,
    model_to_summary,
    module_to_dict,
    validation_issue_to_dict,
)
from stella_mcp.validator import ValidationError
from stella_mcp.xmile import StellaModel


def test_model_to_summary_includes_core_sections():
    model = StellaModel("Carbon")
    model.sim_specs.start = 0
    model.sim_specs.stop = 10
    model.sim_specs.dt = 0.5
    model.add_stock("Atmosphere", "100", units="GtC", x=100, y=200)
    model.add_aux("rate", "0.1", units="1/year")
    model.add_flow("sink", "Atmosphere * rate", from_stock="Atmosphere")
    model.add_connector("Atmosphere", "sink")
    model.add_connector("rate", "sink")
    model.create_module("Core", members=["Atmosphere", "sink", "rate"])

    summary = model_to_summary("carbon_v1", model)

    assert summary["model_id"] == "carbon_v1"
    assert summary["name"] == "Carbon"
    assert summary["sim_specs"] == {
        "start": 0,
        "stop": 10,
        "dt": 0.5,
        "method": "Euler",
        "time_units": "Years",
    }
    assert summary["counts"] == {
        "stocks": 1,
        "flows": 1,
        "auxiliaries": 1,
        "connectors": 2,
        "modules": 1,
    }
    assert summary["variables"]["stocks"][0]["name"] == "Atmosphere"
    assert summary["variables"]["flows"][0]["from_stock"] == "Atmosphere"
    assert summary["variables"]["auxiliaries"][0]["equation"] == "0.1"
    assert summary["modules"][0]["name"] == "Core"
    assert summary["connectors"][0]["uid"] == 1


def test_connector_and_module_dicts_preserve_routing_and_members():
    model = StellaModel("Routing")
    model.add_stock("S", "100")
    model.add_aux("k", "1")
    connector = model.add_connector("k", "S")
    connector.angle = 42
    connector.angle_locked = True
    connector.points = [(1.5, 2.5)]
    connector.points_locked = True
    model.create_module("M", members=["S", "k"])
    model.set_module_view("M", x=10, y=20, width=30, height=40)

    assert connector_to_dict(model, connector) == {
        "uid": 1,
        "from_var": "k",
        "from_display": "k",
        "to_var": "S",
        "to_display": "S",
        "angle": 42,
        "angle_locked": True,
        "points": [{"x": 1.5, "y": 2.5}],
        "points_locked": True,
    }
    assert module_to_dict(model, "M", model.modules["M"])["members"] == ["S", "k"]


def test_validation_issue_to_dict():
    issue = ValidationError(
        severity="error",
        category="undefined_variable",
        message="Flow references missing variable",
        variable="flow_x",
    )
    assert validation_issue_to_dict(issue) == {
        "severity": "error",
        "category": "undefined_variable",
        "message": "Flow references missing variable",
        "variable": "flow_x",
    }
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run --extra dev python -m pytest tests/test_model_snapshot.py -v
```

Expected: FAIL because `stella_mcp.model_snapshot` does not exist.

- [ ] **Step 3: Create snapshot implementation**

Create `stella_mcp/model_snapshot.py`:

```python
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
        "modules": [
            module_to_dict(model, key, model.modules[key])
            for key in sorted(model.modules)
        ],
        "compatibility_warnings": model.compatibility_warnings,
        "last_export_warnings": model.last_export_warnings,
    }
```

- [ ] **Step 4: Run snapshot tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_model_snapshot.py -v
```

Expected: PASS.

- [ ] **Step 5: Run full tests**

Run:

```bash
uv run --extra dev python -m pytest
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add stella_mcp/model_snapshot.py tests/test_model_snapshot.py
git commit -m "feat: add structured model snapshots"
```

---

### Task 3: Return Structured Content From Existing Inspection Tools

**Files:**
- Create: `stella_mcp/tool_results.py`
- Modify: `stella_mcp/tool_handlers.py`
- Modify: `tests/test_server_state_and_gf.py`
- Modify: `tests/test_templates.py`

- [ ] **Step 1: Write failing tests for structured tool responses**

Append to `tests/test_server_state_and_gf.py`:

```python
def test_list_models_returns_structured_content(monkeypatch):
    """list_models should return a machine-readable model list."""
    import asyncio

    from stella_mcp import server as server_mod

    server_mod._session_models.clear()
    asyncio.run(server_mod.call_tool("create_model", {"name": "M1", "model_id": "m1"}))
    result = asyncio.run(server_mod.call_tool("list_models", {}))

    assert result.structuredContent["models"] == [
        {"model_id": "m1", "name": "M1", "current": True}
    ]


def test_validate_model_returns_structured_issues(monkeypatch):
    """validate_model should expose validation issues as dictionaries."""
    import asyncio

    from stella_mcp import server as server_mod

    server_mod._session_models.clear()
    asyncio.run(server_mod.call_tool("create_model", {"name": "Broken", "model_id": "m1"}))
    asyncio.run(server_mod.call_tool("add_stock", {"model_id": "m1", "name": "S", "initial_value": "100"}))
    result = asyncio.run(server_mod.call_tool("validate_model", {"model_id": "m1"}))

    assert result.structuredContent["model_id"] == "m1"
    assert result.structuredContent["issues"][0]["category"] == "mass_balance"
```

Append to `tests/test_templates.py`:

```python
def test_list_templates_tool_returns_structured_templates(monkeypatch, tmp_path):
    """Template discovery should expose structured template metadata."""
    import asyncio

    from stella_mcp import server as server_mod

    monkeypatch.setenv("STELLA_MCP_TEMPLATE_DIR", str(tmp_path))
    result = asyncio.run(server_mod.call_tool("list_templates", {"source": "builtin"}))

    assert result.structuredContent["templates"]
    first = result.structuredContent["templates"][0]
    assert {"name", "source", "title", "stocks", "flows", "auxiliaries"}.issubset(first)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run --extra dev python -m pytest tests/test_server_state_and_gf.py::test_list_models_returns_structured_content tests/test_server_state_and_gf.py::test_validate_model_returns_structured_issues tests/test_templates.py::test_list_templates_tool_returns_structured_templates -v
```

Expected: FAIL because handlers return `list[TextContent]`.

- [ ] **Step 3: Add success result helper**

Create `stella_mcp/tool_results.py`:

```python
"""Helpers for MCP tool result construction."""

from __future__ import annotations

from typing import Any

from mcp.types import CallToolResult, TextContent


def success_result(text: str, structured: dict[str, Any] | None = None) -> CallToolResult:
    """Return a successful MCP tool result with optional structured content."""
    return CallToolResult(
        isError=False,
        content=[TextContent(type="text", text=text)],
        structuredContent=structured or {},
    )
```

- [ ] **Step 4: Update selected handlers to use structured content**

In `stella_mcp/tool_handlers.py`, add imports:

```python
from .model_snapshot import (
    connector_to_dict,
    model_to_summary,
    module_to_dict,
    template_info_to_dict,
    validation_issue_to_dict,
)
from .tool_results import success_result
```

Then update handlers:

- `list_templates`: structured key `templates`.
- `get_template_info`: structured key `template`.
- `load_template`: structured keys `model_id`, `template`, `model`.
- `save_as_template`: structured key `template`.
- `list_models`: structured key `models`.
- `list_modules`: structured keys `model_id`, `modules`.
- `list_connectors`: structured keys `model_id`, `connectors`.
- `validate_model`: structured keys `model_id`, `issues`, `passed`.
- `list_variables`: structured keys `model_id`, `variables`.

For `list_models`, replace the final return with:

```python
        models_payload = [
            {
                "model_id": mid,
                "name": model.name,
                "current": mid == session_models.current_model_id,
            }
            for mid, model in sorted(session_models.models.items())
        ]
        return success_result("\n".join(lines), {"models": models_payload})
```

For `validate_model`, make sure the handler captures `model_id`:

```python
        model_id, model = get_model(arguments.get("model_id"))
        errors = validate_model(model)
        if not errors:
            return success_result(
                "Model validation passed with no errors or warnings.",
                {"model_id": model_id, "passed": True, "issues": []},
            )
```

Then return issues with:

```python
        return success_result(
            "\n".join(result_lines),
            {
                "model_id": model_id,
                "passed": not any(err.severity == "error" for err in errors),
                "issues": [validation_issue_to_dict(err) for err in errors],
            },
        )
```

- [ ] **Step 5: Run targeted tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_server_state_and_gf.py tests/test_templates.py -v
```

Expected: PASS.

- [ ] **Step 6: Run full tests and lint**

Run:

```bash
uv run --extra dev ruff check .
uv run --extra dev python -m pytest
```

Expected: both pass.

- [ ] **Step 7: Commit**

```bash
git add stella_mcp/tool_results.py stella_mcp/tool_handlers.py tests/test_server_state_and_gf.py tests/test_templates.py
git commit -m "feat: return structured inspection results"
```

---

### Task 4: Add `inspect_model`

**Files:**
- Modify: `stella_mcp/tool_schemas.py`
- Modify: `stella_mcp/tool_handlers.py`
- Modify: `tests/test_server_state_and_gf.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing test**

Append to `tests/test_server_state_and_gf.py`:

```python
def test_inspect_model_returns_complete_structured_summary(monkeypatch):
    """inspect_model should be the primary structured model introspection tool."""
    import asyncio

    from stella_mcp import server as server_mod

    server_mod._session_models.clear()
    asyncio.run(server_mod.call_tool("create_model", {"name": "Inspect", "model_id": "m1"}))
    asyncio.run(server_mod.call_tool("add_stock", {"model_id": "m1", "name": "S", "initial_value": "100"}))
    asyncio.run(server_mod.call_tool("add_aux", {"model_id": "m1", "name": "k", "equation": "0.1"}))
    asyncio.run(server_mod.call_tool("add_flow", {"model_id": "m1", "name": "loss", "equation": "S * k", "from_stock": "S"}))
    asyncio.run(server_mod.call_tool("add_connector", {"model_id": "m1", "from_var": "S", "to_var": "loss"}))
    asyncio.run(server_mod.call_tool("add_connector", {"model_id": "m1", "from_var": "k", "to_var": "loss"}))

    result = asyncio.run(server_mod.call_tool("inspect_model", {"model_id": "m1", "include_validation": True}))

    assert result.structuredContent["model"]["model_id"] == "m1"
    assert result.structuredContent["model"]["counts"]["stocks"] == 1
    assert result.structuredContent["validation"]["passed"] is True
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run --extra dev python -m pytest tests/test_server_state_and_gf.py::test_inspect_model_returns_complete_structured_summary -v
```

Expected: FAIL with unknown tool.

- [ ] **Step 3: Add tool schema**

In `stella_mcp/tool_schemas.py`, add a `Tool(...)` near model inspection tools:

```python
        Tool(
            name="inspect_model",
            description="Return a structured summary of the current model for agent inspection",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "include_validation": {
                        "type": "boolean",
                        "description": "Include validation issues in structured output",
                        "default": True,
                    },
                },
            },
        ),
```

- [ ] **Step 4: Add handler**

In `stella_mcp/tool_handlers.py`, register:

```python
    @register("inspect_model")
    def _handle_inspect_model(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        summary = model_to_summary(model_id, model)
        payload: dict[str, Any] = {"model": summary}
        include_validation = arguments.get("include_validation", True)
        if include_validation:
            issues = validate_model(model)
            payload["validation"] = {
                "passed": not any(issue.severity == "error" for issue in issues),
                "issues": [validation_issue_to_dict(issue) for issue in issues],
            }
        return success_result(
            (
                f"Model {model_id}: {model.name} "
                f"({len(model.stocks)} stocks, {len(model.flows)} flows, "
                f"{len(model.auxs)} auxiliaries)"
            ),
            payload,
        )
```

- [ ] **Step 5: Document `inspect_model`**

In `README.md`, add `inspect_model` to the Model Inspection table:

```markdown
| `inspect_model` | Return a structured model summary for agent inspection |
```

Add an example after `list_models`:

```json
{"name":"inspect_model","arguments":{"model_id":"sir_baseline","include_validation":true}}
```

- [ ] **Step 6: Run tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_server_state_and_gf.py::test_inspect_model_returns_complete_structured_summary -v
uv run --extra dev python -m pytest
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add stella_mcp/tool_schemas.py stella_mcp/tool_handlers.py tests/test_server_state_and_gf.py README.md
git commit -m "feat: add structured model inspection"
```

---

### Task 5: Add Explicit Update Tools

**Files:**
- Modify: `stella_mcp/xmile.py`
- Modify: `stella_mcp/tool_schemas.py`
- Modify: `stella_mcp/tool_handlers.py`
- Modify: `tests/test_variable_lifecycle.py`
- Modify: `tests/test_server_state_and_gf.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing model tests**

Append to `tests/test_variable_lifecycle.py`:

```python
def test_update_stock_flow_aux_and_sim_specs():
    """Model update methods should change only provided fields."""
    model = StellaModel("Update")
    model.add_stock("S", "100", units="people")
    model.add_aux("k", "0.1")
    model.add_flow("loss", "S * k", from_stock="S")

    model.set_sim_specs(start=1, stop=50, dt=0.5, method="RK4", time_units="Days")
    model.update_stock("S", initial_value="200", units="GtC", non_negative=False, x=10, y=20)
    model.update_aux("k", equation="0.2", units="1/day", x=30, y=40)
    model.update_flow("loss", equation="S * k * 2", units="GtC/day", non_negative=False, x=50, y=60)

    assert model.sim_specs.start == 1
    assert model.sim_specs.stop == 50
    assert model.sim_specs.dt == 0.5
    assert model.sim_specs.method == "RK4"
    assert model.sim_specs.time_units == "Days"
    assert model.stocks["S"].initial_value == "200"
    assert model.stocks["S"].non_negative is False
    assert model.auxs["k"].equation == "0.2"
    assert model.flows["loss"].equation == "S * k * 2"
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run --extra dev python -m pytest tests/test_variable_lifecycle.py::test_update_stock_flow_aux_and_sim_specs -v
```

Expected: FAIL because update methods do not exist.

- [ ] **Step 3: Add model update methods**

In `stella_mcp/xmile.py`, add methods after `delete_variable`:

```python
    def set_sim_specs(
        self,
        start: Optional[float] = None,
        stop: Optional[float] = None,
        dt: Optional[float] = None,
        method: Optional[str] = None,
        time_units: Optional[str] = None,
    ) -> SimSpecs:
        """Update simulation specs while preserving omitted fields."""
        new_start = self.sim_specs.start if start is None else float(start)
        new_stop = self.sim_specs.stop if stop is None else float(stop)
        new_dt = self.sim_specs.dt if dt is None else float(dt)
        if new_dt <= 0:
            raise ValueError("dt must be > 0")
        if new_stop <= new_start:
            raise ValueError("stop must be greater than start")
        self.sim_specs.start = new_start
        self.sim_specs.stop = new_stop
        self.sim_specs.dt = new_dt
        if method is not None:
            self.sim_specs.method = str(method)
        if time_units is not None:
            self.sim_specs.time_units = str(time_units)
        return self.sim_specs

    def update_stock(
        self,
        name: str,
        initial_value: Optional[str] = None,
        units: Optional[str] = None,
        non_negative: Optional[bool] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ) -> Stock:
        """Update stock fields while preserving relationships."""
        norm_name = self._normalize_name(name)
        stock = self.stocks.get(norm_name)
        if stock is None:
            raise ValueError(f"Stock '{name}' does not exist")
        if initial_value is not None:
            stock.initial_value = str(initial_value)
        if units is not None:
            stock.units = str(units)
        if non_negative is not None:
            stock.non_negative = bool(non_negative)
        if x is not None:
            stock.x = float(x)
        if y is not None:
            stock.y = float(y)
        return stock

    def update_flow(
        self,
        name: str,
        equation: Optional[str] = None,
        units: Optional[str] = None,
        non_negative: Optional[bool] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        graphical_function: Optional[GraphicalFunction] = None,
    ) -> Flow:
        """Update flow fields while preserving structural stock links."""
        norm_name = self._normalize_name(name)
        flow = self.flows.get(norm_name)
        if flow is None:
            raise ValueError(f"Flow '{name}' does not exist")
        if equation is not None:
            flow.equation = str(equation)
        if units is not None:
            flow.units = str(units)
        if non_negative is not None:
            flow.non_negative = bool(non_negative)
        if x is not None:
            flow.x = float(x)
        if y is not None:
            flow.y = float(y)
        if graphical_function is not None:
            flow.graphical_function = graphical_function
        return flow

    def update_aux(
        self,
        name: str,
        equation: Optional[str] = None,
        units: Optional[str] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        graphical_function: Optional[GraphicalFunction] = None,
    ) -> Aux:
        """Update auxiliary fields."""
        norm_name = self._normalize_name(name)
        aux = self.auxs.get(norm_name)
        if aux is None:
            raise ValueError(f"Auxiliary '{name}' does not exist")
        if equation is not None:
            aux.equation = str(equation)
        if units is not None:
            aux.units = str(units)
        if x is not None:
            aux.x = float(x)
        if y is not None:
            aux.y = float(y)
        if graphical_function is not None:
            aux.graphical_function = graphical_function
        return aux
```

- [ ] **Step 4: Add tool schemas**

In `stella_mcp/tool_schemas.py`, add tools:

- `set_sim_specs`: optional `model_id`, `start`, `stop`, `dt`, `method`, `time_units`.
- `update_stock`: required `name`; optional `model_id`, `initial_value`, `units`, `non_negative`, `x`, `y`.
- `update_flow`: required `name`; optional `model_id`, `equation`, `units`, `non_negative`, `x`, `y`, `graphical_function`.
- `update_aux`: required `name`; optional `model_id`, `equation`, `units`, `x`, `y`, `graphical_function`.

Use the existing `model_id_property` and `graphical_function_schema`.

- [ ] **Step 5: Add tool handlers**

In `stella_mcp/tool_handlers.py`, register handlers that call the new model methods and return `success_result` with structured snapshots of the updated object.

For `set_sim_specs`:

```python
    @register("set_sim_specs")
    def _handle_set_sim_specs(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        specs = model.set_sim_specs(
            start=arguments.get("start"),
            stop=arguments.get("stop"),
            dt=arguments.get("dt"),
            method=arguments.get("method"),
            time_units=arguments.get("time_units"),
        )
        return success_result(
            f"Updated simulation specs for model_id={model_id}",
            {
                "model_id": model_id,
                "sim_specs": {
                    "start": specs.start,
                    "stop": specs.stop,
                    "dt": specs.dt,
                    "method": specs.method,
                    "time_units": specs.time_units,
                },
            },
        )
```

- [ ] **Step 6: Add server tests**

Append to `tests/test_server_state_and_gf.py`:

```python
def test_update_tools_return_structured_content(monkeypatch):
    """Update tools should mutate model fields and return structured payloads."""
    import asyncio

    from stella_mcp import server as server_mod

    server_mod._session_models.clear()
    asyncio.run(server_mod.call_tool("create_model", {"name": "Update", "model_id": "m1"}))
    asyncio.run(server_mod.call_tool("add_stock", {"model_id": "m1", "name": "S", "initial_value": "100"}))
    asyncio.run(server_mod.call_tool("add_aux", {"model_id": "m1", "name": "k", "equation": "0.1"}))
    asyncio.run(server_mod.call_tool("add_flow", {"model_id": "m1", "name": "loss", "equation": "S * k", "from_stock": "S"}))

    specs = asyncio.run(server_mod.call_tool("set_sim_specs", {"model_id": "m1", "stop": 20, "dt": 0.5}))
    stock = asyncio.run(server_mod.call_tool("update_stock", {"model_id": "m1", "name": "S", "initial_value": "200"}))
    aux = asyncio.run(server_mod.call_tool("update_aux", {"model_id": "m1", "name": "k", "equation": "0.2"}))
    flow = asyncio.run(server_mod.call_tool("update_flow", {"model_id": "m1", "name": "loss", "equation": "S * k * 2"}))

    assert specs.structuredContent["sim_specs"]["stop"] == 20
    assert stock.structuredContent["stock"]["initial_value"] == "200"
    assert aux.structuredContent["auxiliary"]["equation"] == "0.2"
    assert flow.structuredContent["flow"]["equation"] == "S * k * 2"
```

- [ ] **Step 7: Document update tools**

In `README.md`, add rows for `set_sim_specs`, `update_stock`, `update_flow`, and `update_aux`, plus one JSON example:

```json
{"name":"update_flow","arguments":{"model_id":"pop_v1","name":"growth","equation":"Population * growth_rate * stress_modifier"}}
```

- [ ] **Step 8: Verify**

Run:

```bash
uv run --extra dev ruff check .
uv run --extra dev python -m pytest
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add stella_mcp/xmile.py stella_mcp/tool_schemas.py stella_mcp/tool_handlers.py tests/test_variable_lifecycle.py tests/test_server_state_and_gf.py README.md
git commit -m "feat: add explicit model update tools"
```

---

### Task 6: Add Connector Sync From Equations

**Files:**
- Modify: `stella_mcp/xmile.py`
- Modify: `stella_mcp/tool_schemas.py`
- Modify: `stella_mcp/tool_handlers.py`
- Modify: `tests/test_equation_parser.py`
- Modify: `tests/test_server_state_and_gf.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing model test**

Append to `tests/test_equation_parser.py`:

```python
def test_sync_connectors_from_equations_adds_missing_and_preserves_existing():
    """Connector sync should add missing equation dependencies without duplicating."""
    from stella_mcp.xmile import StellaModel

    model = StellaModel("Sync")
    model.add_stock("S", "100")
    model.add_aux("k", "0.1")
    model.add_aux("modifier", "2")
    model.add_flow("loss", "S * k * modifier", from_stock="S")
    existing = model.add_connector("S", "loss")

    summary = model.sync_connectors_from_equations()

    endpoints = {(connector.from_var, connector.to_var) for connector in model.connectors}
    assert endpoints == {("S", "loss"), ("k", "loss"), ("modifier", "loss")}
    assert existing.uid == 1
    assert summary == {"added": 2, "existing": 1}
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
uv run --extra dev python -m pytest tests/test_equation_parser.py::test_sync_connectors_from_equations_adds_missing_and_preserves_existing -v
```

Expected: FAIL because method does not exist.

- [ ] **Step 3: Add model method**

In `stella_mcp/xmile.py`, add after `add_connector`:

```python
    def sync_connectors_from_equations(self) -> dict[str, int]:
        """Add missing connectors for equation references on flows and auxiliaries."""
        existing = {(conn.from_var, conn.to_var) for conn in self.connectors}
        added = 0
        already_present = 0

        targets: list[tuple[str, str]] = []
        for name, flow in self.flows.items():
            for ref in sorted(self._extract_variable_refs(flow.equation)):
                if ref != name and self._has_variable(ref):
                    targets.append((ref, name))
        for name, aux in self.auxs.items():
            for ref in sorted(self._extract_variable_refs(aux.equation)):
                if ref != name and self._has_variable(ref):
                    targets.append((ref, name))

        for from_var, to_var in targets:
            if (from_var, to_var) in existing:
                already_present += 1
                continue
            self.add_connector(from_var, to_var)
            existing.add((from_var, to_var))
            added += 1

        return {"added": added, "existing": already_present}
```

- [ ] **Step 4: Add schema and handler**

Add `sync_connectors_from_equations` to `tool_schemas.py`:

```python
        Tool(
            name="sync_connectors_from_equations",
            description="Add missing dependency connectors inferred from flow and auxiliary equations",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                },
            },
        ),
```

Add handler:

```python
    @register("sync_connectors_from_equations")
    def _handle_sync_connectors_from_equations(arguments: dict[str, Any]) -> ToolResponse:
        model_id, model = get_model(arguments.get("model_id"))
        summary = model.sync_connectors_from_equations()
        return success_result(
            (
                f"Synced connectors for model_id={model_id}: "
                f"added={summary['added']}, existing={summary['existing']}"
            ),
            {"model_id": model_id, **summary},
        )
```

- [ ] **Step 5: Add server test**

Append to `tests/test_server_state_and_gf.py`:

```python
def test_sync_connectors_from_equations_tool(monkeypatch):
    """Tool should add missing equation connectors and report counts."""
    import asyncio

    from stella_mcp import server as server_mod

    server_mod._session_models.clear()
    asyncio.run(server_mod.call_tool("create_model", {"name": "Sync", "model_id": "m1"}))
    asyncio.run(server_mod.call_tool("add_stock", {"model_id": "m1", "name": "S", "initial_value": "100"}))
    asyncio.run(server_mod.call_tool("add_aux", {"model_id": "m1", "name": "k", "equation": "0.1"}))
    asyncio.run(server_mod.call_tool("add_flow", {"model_id": "m1", "name": "loss", "equation": "S * k", "from_stock": "S"}))

    result = asyncio.run(server_mod.call_tool("sync_connectors_from_equations", {"model_id": "m1"}))

    assert result.structuredContent["added"] == 2
    listed = asyncio.run(server_mod.call_tool("list_connectors", {"model_id": "m1"}))
    assert len(listed.structuredContent["connectors"]) == 2
```

- [ ] **Step 6: Document connector sync**

In `README.md`, add a row:

```markdown
| `sync_connectors_from_equations` | Add missing dependency connectors inferred from equations |
```

Add example:

```json
{"name":"sync_connectors_from_equations","arguments":{"model_id":"pop_v1"}}
```

- [ ] **Step 7: Verify and commit**

Run:

```bash
uv run --extra dev ruff check .
uv run --extra dev python -m pytest
```

Expected: PASS.

Commit:

```bash
git add stella_mcp/xmile.py stella_mcp/tool_schemas.py stella_mcp/tool_handlers.py tests/test_equation_parser.py tests/test_server_state_and_gf.py README.md
git commit -m "feat: infer connectors from equations"
```

---

### Task 7: Add Agent Workflow Documentation and Release Prep

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml`
- Modify: `stella_mcp/__init__.py`

- [ ] **Step 1: Add README workflow section**

After the Configuration section in `README.md`, add:

```markdown
## Recommended Agent Workflow

For a new model:

1. `create_model` with a stable `model_id`.
2. Add stocks, flows, and auxiliaries.
3. Run `sync_connectors_from_equations`.
4. Run `inspect_model` with `include_validation=true`.
5. Fix validation errors with `update_*`, `rename_variable`, or `delete_variable`.
6. Save with `save_model`.

For imported models:

1. `read_model` with `compat_mode="permissive"` to inspect warnings.
2. Run `inspect_model` to understand model structure.
3. Use `compat_mode="strict"` before final save when round-trip fidelity matters.
```

- [ ] **Step 2: Bump version to 0.6.0**

In `pyproject.toml`:

```toml
version = "0.6.0"
```

In `stella_mcp/__init__.py`:

```python
__version__ = "0.6.0"
```

- [ ] **Step 3: Verify package build includes metadata**

Run:

```bash
uv run --extra dev python -m pytest
uv run --with build python -m build --outdir /private/tmp/stella-mcp-0.6.0-dist
python -m zipfile -l /private/tmp/stella-mcp-0.6.0-dist/stella_mcp-0.6.0-py3-none-any.whl | rg 'builtin_templates|METADATA|LICENSE'
```

Expected:
- pytest passes.
- build creates `stella_mcp-0.6.0.tar.gz` and `stella_mcp-0.6.0-py3-none-any.whl`.
- wheel listing includes built-in templates, metadata, and license.

- [ ] **Step 4: Commit**

```bash
git add README.md pyproject.toml stella_mcp/__init__.py
git commit -m "docs: document agent workflow for 0.6.0"
```

---

## Self-Review

- Spec coverage: This plan covers the recommended next release scope: structured outputs, inspection, explicit update tools, connector sync, lint enforcement, docs, and version bump.
- Placeholder scan: No `TBD`, `TODO`, or "implement later" placeholders are present. Code snippets name concrete files, functions, methods, and commands.
- Type consistency: New helpers consistently use `dict[str, Any]`, existing `StellaModel`/dataclass names, and existing MCP `CallToolResult`/`TextContent` return types.
- Scope check: This is one coherent `0.6.0` stabilization release. It does not include broader Stella compatibility corpus expansion or simulation/runtime execution; those should remain separate follow-up releases.
