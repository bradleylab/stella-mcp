---
title: Better Element Positioning in Stella MCP
type: feat
date: 2026-02-02
---

# Better Element Positioning in Stella MCP

## Overview

Improve how the MCP positions stocks, flows, and auxiliaries in Stella diagrams. Currently all elements are placed in simple horizontal rows regardless of their relationships, making complex models hard to read. This plan adds user-specified positioning and smarter automatic layout based on variable dependencies.

## Problem Statement

**Current behavior:**
- `_auto_layout()` in `xmile.py:171-227` runs unconditionally before every XML generation
- All stocks placed in a single horizontal row at y=300
- All auxiliaries placed in a single horizontal row at y=150
- No consideration for which auxs connect to which flows/stocks
- User cannot specify positions via MCP tools
- `parse_stmx()` does not preserve positions from loaded models (they get reset on save)

**Result:** Models with many auxiliaries become a visual mess where connectors cross each other and related variables are far apart.

## Proposed Solution

A three-phase approach, each deliverable independently:

### Phase 1: Respect User-Specified Positions

Allow users to set positions when creating elements, and make auto-layout skip elements that already have positions.

### Phase 2: Smarter Default Layout

Group auxiliaries near the flows/stocks they affect instead of a flat horizontal row.

### Phase 3: Post-Creation Repositioning (Optional)

Add a `set_position` tool to move elements after creation.

---

## Critical Design Decisions

These questions must be answered before implementation:

### 1. Position Sentinel Value

**Problem:** Using `x == 0 and y == 0` to detect "unpositioned" makes (0,0) an unusable coordinate.

**Decision:** Change dataclasses to use `Optional[float]` with `None` meaning unspecified:

```python
@dataclass
class Stock:
    name: str
    initial_value: str
    units: str = ""
    inflows: list[str] = field(default_factory=list)
    outflows: list[str] = field(default_factory=list)
    non_negative: bool = True
    x: Optional[float] = None  # Changed from 0
    y: Optional[float] = None  # Changed from 0
```

Auto-layout check becomes:
```python
if stock.x is None or stock.y is None:
    stock.x = x
    stock.y = stock_y
```

### 2. Position Preservation on Model Load

**Problem:** `parse_stmx()` doesn't read position data from `<view>` section. Loading and saving a model resets all positions.

**Decision:** Update `parse_stmx()` to extract positions from view elements. This is required for Phase 1 to be useful with existing models.

### 3. Flow Pipe Recalculation

**Problem:** When stocks are user-positioned, `flow.points` must be recalculated to connect visually.

**Decision:** Always recalculate `flow.points` in `_auto_layout()` based on current stock positions, even if `flow.x`/`flow.y` are user-specified.

### 4. Coordinate Bounds

**Decision:** Accept any positive coordinates. Warn (via `validate_model`) if coordinates exceed typical canvas bounds (0-1500 for x, 0-1200 for y based on default page dimensions).

---

## Technical Approach

### Phase 1: User-Specified Positions

**Changes to `stella_mcp/xmile.py` (dataclasses):**

```python
# Change defaults from 0 to None for all position fields
@dataclass
class Stock:
    # ... existing fields ...
    x: Optional[float] = None
    y: Optional[float] = None

@dataclass
class Flow:
    # ... existing fields ...
    x: Optional[float] = None
    y: Optional[float] = None
    points: list[tuple[float, float]] = field(default_factory=list)

@dataclass
class Aux:
    # ... existing fields ...
    x: Optional[float] = None
    y: Optional[float] = None
```

**Changes to `StellaModel` methods:**

```python
def add_stock(
    self,
    name: str,
    initial_value: str,
    units: str = "",
    inflows: Optional[list[str]] = None,
    outflows: Optional[list[str]] = None,
    non_negative: bool = True,
    x: Optional[float] = None,  # NEW
    y: Optional[float] = None   # NEW
) -> Stock:
    stock = Stock(
        name=name,
        initial_value=initial_value,
        units=units,
        inflows=[self._normalize_name(f) for f in (inflows or [])],
        outflows=[self._normalize_name(f) for f in (outflows or [])],
        non_negative=non_negative,
        x=x,  # NEW
        y=y   # NEW
    )
    self.stocks[self._normalize_name(name)] = stock
    return stock
```

Similar changes to `add_flow()` and `add_aux()`.

**Changes to `_auto_layout()`:**

```python
def _auto_layout(self):
    stock_spacing = 200
    aux_spacing = 80
    stock_y = 300
    aux_y = 150
    start_x = 200

    # Position stocks - only if not already positioned
    x = start_x
    for name, stock in self.stocks.items():
        if stock.x is None or stock.y is None:
            stock.x = x
            stock.y = stock_y
            x += stock_spacing

    # ALWAYS recalculate flow positions and points based on stock positions
    for name, flow in self.flows.items():
        from_stock = self.stocks.get(flow.from_stock)
        to_stock = self.stocks.get(flow.to_stock)

        if from_stock and to_stock:
            if flow.x is None or flow.y is None:
                flow.x = (from_stock.x + to_stock.x) / 2
                flow.y = (from_stock.y + to_stock.y) / 2
            # Always recalculate points to connect stocks
            flow.points = [
                (from_stock.x + 22.5, from_stock.y),
                (to_stock.x - 22.5, to_stock.y)
            ]
        elif from_stock:
            if flow.x is None or flow.y is None:
                flow.x = from_stock.x + 90
                flow.y = from_stock.y
            flow.points = [
                (from_stock.x + 22.5, from_stock.y),
                (from_stock.x + 160, from_stock.y)
            ]
        elif to_stock:
            if flow.x is None or flow.y is None:
                flow.x = to_stock.x - 90
                flow.y = to_stock.y
            flow.points = [
                (to_stock.x - 160, to_stock.y),
                (to_stock.x - 22.5, to_stock.y)
            ]
        else:
            if flow.x is None or flow.y is None:
                flow.x = start_x
                flow.y = stock_y

    # Position auxiliaries - only if not already positioned
    x = start_x
    for name, aux in self.auxs.items():
        if aux.x is None or aux.y is None:
            aux.x = x
            aux.y = aux_y
            x += aux_spacing
```

**Changes to `parse_stmx()` - Add position extraction:**

```python
# After parsing variables, extract positions from view section
views = find_child(model_elem, "views") if model_elem is not None else None
view = find_child(views, "view") if views is not None else None

if view is not None:
    # Extract stock positions
    for stock_elem in findall_children(view, "stock"):
        name = stock_elem.get("name")
        x = stock_elem.get("x")
        y = stock_elem.get("y")
        if name:
            norm_name = model._normalize_name(name)
            if norm_name in model.stocks:
                if x is not None:
                    model.stocks[norm_name].x = float(x)
                if y is not None:
                    model.stocks[norm_name].y = float(y)

    # Extract flow positions
    for flow_elem in findall_children(view, "flow"):
        name = flow_elem.get("name")
        x = flow_elem.get("x")
        y = flow_elem.get("y")
        if name:
            norm_name = model._normalize_name(name)
            if norm_name in model.flows:
                if x is not None:
                    model.flows[norm_name].x = float(x)
                if y is not None:
                    model.flows[norm_name].y = float(y)

    # Extract aux positions
    for aux_elem in findall_children(view, "aux"):
        name = aux_elem.get("name")
        x = aux_elem.get("x")
        y = aux_elem.get("y")
        if name:
            norm_name = model._normalize_name(name)
            if norm_name in model.auxs:
                if x is not None:
                    model.auxs[norm_name].x = float(x)
                if y is not None:
                    model.auxs[norm_name].y = float(y)

    # Existing connector parsing...
```

**Changes to `stella_mcp/server.py`:**

Add `x` and `y` to tool schemas:

```python
Tool(
    name="add_stock",
    inputSchema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Stock name"},
            "initial_value": {"type": "string", "description": "Initial value"},
            "units": {"type": "string", "default": ""},
            "non_negative": {"type": "boolean", "default": True},
            "x": {"type": "number", "description": "X position (optional, auto-positioned if not specified)"},
            "y": {"type": "number", "description": "Y position (optional, auto-positioned if not specified)"},
        },
        "required": ["name", "initial_value"],
    },
),
```

Update handlers:

```python
elif name == "add_stock":
    model = get_model()
    model.add_stock(
        name=arguments["name"],
        initial_value=arguments["initial_value"],
        units=arguments.get("units", ""),
        non_negative=arguments.get("non_negative", True),
        x=arguments.get("x"),  # NEW
        y=arguments.get("y"),  # NEW
    )
    pos_info = ""
    if arguments.get("x") is not None and arguments.get("y") is not None:
        pos_info = f" at position ({arguments['x']}, {arguments['y']})"
    return [TextContent(
        type="text",
        text=f"Added stock '{arguments['name']}' with initial value {arguments['initial_value']}{pos_info}"
    )]
```

**Files:**
- `stella_mcp/xmile.py:16-26` (Stock dataclass)
- `stella_mcp/xmile.py:29-40` (Flow dataclass)
- `stella_mcp/xmile.py:43-49` (Aux dataclass)
- `stella_mcp/xmile.py:97-116` (StellaModel.add_stock)
- `stella_mcp/xmile.py:118-153` (StellaModel.add_flow)
- `stella_mcp/xmile.py:155-159` (StellaModel.add_aux)
- `stella_mcp/xmile.py:171-227` (_auto_layout)
- `stella_mcp/xmile.py:366-527` (parse_stmx - add position extraction)
- `stella_mcp/server.py:49-62` (add_stock schema)
- `stella_mcp/server.py:63-78` (add_flow schema)
- `stella_mcp/server.py:79-91` (add_aux schema)
- `stella_mcp/server.py:171-182` (add_stock handler)
- `stella_mcp/server.py:184-203` (add_flow handler)
- `stella_mcp/server.py:205-215` (add_aux handler)

### Phase 2: Smarter Default Layout

**Goal:** Group auxiliaries logically instead of a flat row.

**Heuristic approach:**
1. Parse each aux's equation to find variable references
2. Categorize auxs:
   - **Constants/Parameters**: No variable references (e.g., `"0.1"`, `"100"`) - place top-left
   - **Flow modifiers**: Referenced by flows - place near the flow they affect
   - **Stock-dependent**: Reference stocks - place below the relevant stock
   - **Calculated intermediates**: Reference other auxs - place in middle tier

**Helper method (unify with validator.py):**

```python
# Shared constant - consider moving to a common module
STELLA_FUNCTIONS = {
    'IF', 'THEN', 'ELSE', 'AND', 'OR', 'NOT',
    'MIN', 'MAX', 'ABS', 'SIN', 'COS', 'TAN',
    'EXP', 'LN', 'LOG', 'LOG10', 'SQRT', 'INT',
    'ROUND', 'MOD', 'TIME', 'DT', 'STARTTIME', 'STOPTIME',
    'DELAY', 'DELAY1', 'DELAY3', 'DELAYN',
    'SMOOTH', 'SMOOTH3', 'SMOOTHN', 'SMTH1', 'SMTH3', 'SMTHN',
    'TREND', 'FORCST', 'PULSE', 'STEP', 'RAMP',
    'RANDOM', 'NORMAL', 'POISSON', 'EXPRND',
    'PREVIOUS', 'INIT', 'SELF',
}

def _extract_variable_refs(self, equation: str) -> set[str]:
    """Extract variable names referenced in an equation."""
    import re
    tokens = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', equation)
    return {self._normalize_name(t) for t in tokens if t.upper() not in STELLA_FUNCTIONS}
```

**Algorithm for smart layout:**

```python
def _auto_layout(self):
    stock_spacing = 200
    aux_spacing = 80
    stock_y = 300
    aux_y = 150
    start_x = 200

    # Position stocks first (unchanged from Phase 1)
    # ...

    # Categorize auxiliaries (only those needing positioning)
    unpositioned_auxs = {n: a for n, a in self.auxs.items() if a.x is None or a.y is None}

    constants = []
    flow_modifiers = {}  # flow_name -> [aux_names]
    stock_deps = {}      # stock_name -> [aux_names]
    intermediates = []

    for name, aux in unpositioned_auxs.items():
        refs = self._extract_variable_refs(aux.equation)

        if not refs:
            constants.append(name)
            continue

        # Check if this aux is used by any flow
        placed = False
        for flow_name, flow in self.flows.items():
            flow_refs = self._extract_variable_refs(flow.equation)
            if name in flow_refs:
                flow_modifiers.setdefault(flow_name, []).append(name)
                placed = True
                break

        if placed:
            continue

        # Check if aux references a stock
        for ref in refs:
            if ref in self.stocks:
                stock_deps.setdefault(ref, []).append(name)
                placed = True
                break

        if not placed:
            intermediates.append(name)

    # Position constants in top-left cluster
    x = start_x
    for name in constants:
        self.auxs[name].x = x
        self.auxs[name].y = aux_y - 60
        x += aux_spacing

    # Position flow modifiers near their flows
    for flow_name, aux_names in flow_modifiers.items():
        flow = self.flows[flow_name]
        for i, aux_name in enumerate(aux_names):
            self.auxs[aux_name].x = flow.x + ((i % 3 - 1) * aux_spacing)
            self.auxs[aux_name].y = aux_y + ((i // 3) * 50)

    # Position stock-dependent auxs below their stocks
    for stock_name, aux_names in stock_deps.items():
        stock = self.stocks[stock_name]
        for i, aux_name in enumerate(aux_names):
            self.auxs[aux_name].x = stock.x + ((i - len(aux_names)//2) * aux_spacing)
            self.auxs[aux_name].y = stock_y + 80

    # Position intermediates in remaining row
    x = start_x
    for name in intermediates:
        self.auxs[name].x = x
        self.auxs[name].y = aux_y
        x += aux_spacing
```

**Files:**
- `stella_mcp/xmile.py:171-227` (_auto_layout - rewrite)
- `stella_mcp/xmile.py` (add `_extract_variable_refs` helper)
- Consider: `stella_mcp/constants.py` for shared `STELLA_FUNCTIONS`

### Phase 3: Post-Creation Repositioning (Optional)

**Add new tool to `server.py`:**

```python
Tool(
    name="set_position",
    description="Set the visual position of an element in the model diagram",
    inputSchema={
        "type": "object",
        "properties": {
            "element_name": {"type": "string", "description": "Name of stock, flow, or auxiliary"},
            "x": {"type": "number", "description": "X coordinate"},
            "y": {"type": "number", "description": "Y coordinate"},
        },
        "required": ["element_name", "x", "y"],
    },
),
```

**Handler:**

```python
elif name == "set_position":
    model = get_model()
    elem_name = model._normalize_name(arguments["element_name"])
    x = arguments["x"]
    y = arguments["y"]

    if elem_name in model.stocks:
        model.stocks[elem_name].x = x
        model.stocks[elem_name].y = y
        elem_type = "stock"
    elif elem_name in model.flows:
        model.flows[elem_name].x = x
        model.flows[elem_name].y = y
        elem_type = "flow"
    elif elem_name in model.auxs:
        model.auxs[elem_name].x = x
        model.auxs[elem_name].y = y
        elem_type = "auxiliary"
    else:
        return [TextContent(type="text", text=f"Element '{arguments['element_name']}' not found")]

    return [TextContent(
        type="text",
        text=f"Set {elem_type} '{arguments['element_name']}' position to ({x}, {y})"
    )]
```

**Files:**
- `stella_mcp/server.py:29-150` (add to list_tools)
- `stella_mcp/server.py:153-300` (add handler)

---

## Acceptance Criteria

### Phase 1
- [x] `add_stock`, `add_flow`, `add_aux` accept optional `x` and `y` parameters
- [x] Elements with user-specified positions keep those positions after `to_xml()`
- [x] Elements without specified positions still get auto-positioned
- [x] Loaded models (via `read_model`) preserve their original positions
- [x] Round-trip test: load model, save without changes, positions unchanged
- [x] Flow points always recalculated to connect to current stock positions

### Phase 2
- [x] Constants/parameters cluster in top-left area (y = aux_y - 60)
- [x] Auxs that affect flows appear near those flows
- [x] Auxs that depend on stocks appear below those stocks
- [x] Connector crossings reduced compared to current flat layout
- [x] Minimum spacing (aux_spacing = 80) enforced between elements

### Phase 3
- [ ] `set_position` tool moves any element type
- [ ] Position persists through subsequent `save_model` calls
- [ ] Clear error message if element not found

---

## Testing Strategy

**Unit tests for Phase 1:**

```python
# tests/test_positioning.py

def test_user_specified_position_preserved():
    """User-specified positions should not be overwritten."""
    model = StellaModel("Test")
    model.add_stock("Population", "100", x=400, y=500)
    xml = model.to_xml()
    assert 'x="400"' in xml
    assert 'y="500"' in xml

def test_unspecified_position_auto_laid_out():
    """Elements without positions should be auto-positioned."""
    model = StellaModel("Test")
    model.add_stock("Population", "100")  # No x, y
    xml = model.to_xml()
    assert 'x="200"' in xml  # start_x
    assert 'y="300"' in xml  # stock_y

def test_mixed_positioning():
    """Mix of positioned and unpositioned elements."""
    model = StellaModel("Test")
    model.add_stock("A", "100", x=400, y=300)
    model.add_stock("B", "100")  # Should get x=200 (start_x)
    xml = model.to_xml()
    # A keeps its position
    assert 'name="A"' in xml and 'x="400"' in xml
    # B gets auto-positioned
    assert 'name="B"' in xml and 'x="200"' in xml

def test_position_zero_is_valid():
    """User should be able to position at (0, 0)."""
    model = StellaModel("Test")
    model.add_stock("Origin", "100", x=0, y=0)
    xml = model.to_xml()
    assert 'name="Origin"' in xml
    assert 'x="0"' in xml
    assert 'y="0"' in xml

def test_flow_points_recalculated():
    """Flow points should connect to stocks at their actual positions."""
    model = StellaModel("Test")
    model.add_stock("A", "100", x=100, y=300)
    model.add_stock("B", "100", x=500, y=300)
    model.add_flow("transfer", "10", from_stock="A", to_stock="B")
    xml = model.to_xml()
    # Flow points should span from A to B
    assert '<pt x="122.5"' in xml  # A.x + 22.5
    assert '<pt x="477.5"' in xml  # B.x - 22.5

def test_round_trip_preserves_positions(tmp_path):
    """Loading and saving should preserve positions."""
    # Create and save a model
    model1 = StellaModel("Test")
    model1.add_stock("Pop", "100", x=400, y=350)
    filepath = tmp_path / "test.stmx"
    filepath.write_text(model1.to_xml())

    # Load and save again
    model2 = parse_stmx(str(filepath))
    assert model2.stocks["Pop"].x == 400
    assert model2.stocks["Pop"].y == 350
```

**Integration test for Phase 2:**

```python
def test_constants_grouped_top_left():
    """Constants should cluster in top-left area."""
    model = StellaModel("Test")
    model.add_stock("Population", "100")
    model.add_aux("growth_rate", "0.02")  # Constant - no refs
    model.add_aux("capacity", "1000")      # Constant - no refs

    xml = model.to_xml()
    # Constants should be at aux_y - 60 = 90
    # Parse and verify y positions

def test_flow_modifier_near_flow():
    """Auxs used by flows should be positioned near those flows."""
    model = StellaModel("Test")
    model.add_stock("Population", "100")
    model.add_aux("birth_rate", "0.02")
    model.add_flow("births", "Population * birth_rate", to_stock="Population")
    model.add_connector("birth_rate", "births")

    xml = model.to_xml()
    # birth_rate should be near births flow position
```

---

## Implementation Order

1. **Phase 1** - Foundation for all positioning work
   - Change dataclasses to Optional[float]
   - Add x/y to tool schemas and handlers
   - Update _auto_layout() with None checks
   - Update parse_stmx() to extract positions
   - Add tests

2. **Phase 2** - Main improvement
   - Add _extract_variable_refs() helper
   - Rewrite _auto_layout() with categorization
   - Add smart layout tests

3. **Phase 3** - Nice to have
   - Add set_position tool
   - Add handler
   - Add tests

Each phase can be merged independently.

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Equation parsing misses variable references | Medium | Low | Use comprehensive STELLA_FUNCTIONS list from validator.py; fall back to intermediates category |
| User-specified positions create element overlap | Medium | Low | Add validation warning for overlapping elements in validator.py |
| Smart layout causes overlapping labels | Medium | Low | Enforce minimum spacing (aux_spacing); document limitation |
| Flow pipes don't visually connect after stock repositioning | Low | High | Always recalculate flow.points based on stock positions |
| Aux referenced by multiple flows - ambiguous placement | Medium | Low | Use first match; document this behavior |

---

## Open Questions (Lower Priority)

1. **Label collision detection** - Should we check if labels overlap? (Defer to future)
2. **Reset to auto-layout** - Should there be a tool to reset positions? (Defer to future)
3. **Connector angle calculation** - Should we compute angles based on positions? (Defer to future)

---

## References

### Internal References
- `stella_mcp/xmile.py:171-227` - Current `_auto_layout()` implementation
- `stella_mcp/xmile.py:15-59` - Stock, Flow, Aux dataclasses with x, y fields
- `stella_mcp/xmile.py:366-527` - `parse_stmx()` function
- `stella_mcp/server.py:29-150` - Tool definitions
- `stella_mcp/server.py:153-300` - Tool handlers
- `stella_mcp/validator.py:60-70` - STELLA_FUNCTIONS reference

### External References
- [XMILE 1.0 Specification](https://docs.oasis-open.org/xmile/xmile/v1.0/xmile-v1.0.html) - View element positioning
- Stella Professional documentation on diagram layout
