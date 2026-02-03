---
title: "fix: Layout Algorithm Issues (Angles, Ordering, Overlap, Subsystems, Aux Placement)"
type: fix
date: 2026-02-02
---

# Fix Layout Algorithm Issues

## Overview

The graph-based auto-layout implementation has 5 issues affecting diagram readability when opened in Stella:

1. **Connector angles broken** - All connectors have `angle="0"` creating sweeping loops
2. **Stock ordering wrong** - Stocks don't follow flow topology (carbon flow direction)
3. **Auxs overlap** - Multiple auxs connecting to same target stack on top of each other
4. **Subsystems not separated** - Main model and error tracking should be in distinct regions
5. **Auxs scattered** - Parameters should be directly above their target flows

## Problem Statement

After implementing graph-based layout (commit a191e59), diagrams render but have visual issues:

- Connector loops obscure the diagram because angles aren't calculated
- Stock ordering is alphabetical, not by flow direction (Atmosphere → Vegetation → SOM)
- When multiple auxs (e.g., GPP_base, Atm_ref, Km_CO2, resp_fraction) all connect to the same flow, they stack at identical positions
- Independent subsystems (main model vs. calibration/error tracking) aren't visually separated
- Auxs like `tau_som` should appear directly above `Heterotrophic_Respiration`, not scattered

## Technical Approach

### Order of Operations

The layout algorithm must execute in this sequence:

```
1. Build dependency graph (connectors + implicit flow-stock edges)
2. Detect subsystems (connected components via undirected DFS)
3. Within each subsystem:
   a. Find stock chains following flow topology
   b. Position stocks in chains (left-to-right by flow direction)
   c. Position flows at midpoints between stocks
   d. Group auxs by target (flow or stock they connect to)
   e. Position aux groups with horizontal spread above targets
4. Calculate subsystem bounding boxes
5. Arrange subsystems with gaps (largest left, others right)
6. Calculate connector angles from final positions
```

### Fix 1: Connector Angle Calculation

**File:** `stella_mcp/xmile.py`

**Problem:** `Connector.angle` defaults to `0` and is never calculated.

**Solution:** Add `_calculate_connector_angles()` method called after all positions are finalized.

```python
def _calculate_connector_angles(self) -> None:
    """Calculate connector angles based on source and target positions."""
    import math

    # Build position lookup
    positions: dict[str, tuple[float, float]] = {}
    for stock in self.stocks.values():
        if stock.x is not None and stock.y is not None:
            positions[stock.name] = (stock.x, stock.y)
    for flow in self.flows.values():
        if flow.x is not None and flow.y is not None:
            positions[flow.name] = (flow.x, flow.y)
    for aux in self.auxs.values():
        if aux.x is not None and aux.y is not None:
            positions[aux.name] = (aux.x, aux.y)

    for conn in self.connectors:
        from_pos = positions.get(conn.from_var)
        to_pos = positions.get(conn.to_var)

        if from_pos and to_pos:
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]

            # Handle zero distance (same position)
            if abs(dx) < 0.001 and abs(dy) < 0.001:
                conn.angle = 0
            else:
                # Stella uses degrees, 0 = right, counter-clockwise positive
                conn.angle = math.degrees(math.atan2(-dy, dx))  # -dy because y increases downward
```

**Location:** Call at end of `_auto_layout()` (after line 593).

### Fix 2: Stock Ordering by Flow Topology

**File:** `stella_mcp/xmile.py`, method `_find_stock_chains()` (lines 223-274)

**Problem:** Start stocks are selected alphabetically when multiple chains exist. Chains should follow flow direction.

**Solution:** Improve chain detection to:
1. Identify true "source" stocks (stocks that receive from external or have only outflows)
2. Order chains by flow direction within the subsystem
3. Handle circular flows by breaking at the stock with the most connections

```python
def _find_stock_chains(self, subsystem: set[str]) -> list[list[str]]:
    """Find linear chains of stocks connected by flows, ordered by flow direction."""
    stocks_in_subsystem = [s for s in subsystem if s in self.stocks]
    if not stocks_in_subsystem:
        return []

    # Build directed stock adjacency from flows
    stock_adj: dict[str, list[str]] = {s: [] for s in stocks_in_subsystem}
    incoming_count: dict[str, int] = {s: 0 for s in stocks_in_subsystem}

    for flow in self.flows.values():
        if flow.from_stock in stocks_in_subsystem and flow.to_stock in stocks_in_subsystem:
            stock_adj[flow.from_stock].append(flow.to_stock)
            incoming_count[flow.to_stock] += 1

    # Find true sources (no incoming flows from other stocks in subsystem)
    # Sort by: incoming_count ASC, then alphabetically for determinism
    start_candidates = [(incoming_count[s], s) for s in stocks_in_subsystem]
    start_candidates.sort()

    chains = []
    visited = set()

    for _, start in start_candidates:
        if start in visited:
            continue

        chain = []
        current = start

        while current and current not in visited:
            visited.add(current)
            chain.append(current)

            # Follow flow direction to next stock
            next_stocks = [s for s in stock_adj.get(current, []) if s not in visited]
            current = next_stocks[0] if next_stocks else None

        if chain:
            chains.append(chain)

    return chains
```

### Fix 3: Aux Spread for Same-Target Overlap

**File:** `stella_mcp/xmile.py`, method `_position_subsystem()` (lines 310-430)

**Problem:** All auxs targeting the same element get `aux.x = target_pos[0]`.

**Solution:** Track target positions and apply horizontal spread.

```python
def _position_subsystem(self, subsystem: set[str], start_x: float,
                        stock_y: float, aux_y: float,
                        outgoing: dict, incoming: dict) -> float:
    # ... existing stock and flow positioning ...

    # Group auxs by their primary target
    target_to_auxs: dict[str, list[str]] = {}

    for aux_name in auxs_in_subsystem:
        aux = self.auxs[aux_name]
        if aux.x is not None and aux.y is not None:
            continue  # User-specified position

        targets = outgoing.get(aux_name, set())
        if targets:
            # Use first target (sorted for determinism)
            primary_target = sorted(targets)[0]
            if primary_target not in target_to_auxs:
                target_to_auxs[primary_target] = []
            target_to_auxs[primary_target].append(aux_name)

    # Position each group with horizontal spread
    AUX_SPREAD_SPACING = 70  # pixels between auxs in a group

    for target, aux_names in target_to_auxs.items():
        # Get target position
        target_x = None
        if target in self.stocks and self.stocks[target].x is not None:
            target_x = self.stocks[target].x
        elif target in self.flows and self.flows[target].x is not None:
            target_x = self.flows[target].x
        elif target in self.auxs and self.auxs[target].x is not None:
            target_x = self.auxs[target].x

        if target_x is None:
            continue

        # Sort aux names for determinism
        aux_names = sorted(aux_names)
        n = len(aux_names)

        # Center the group horizontally above target
        # Offsets: for n=3 -> [-70, 0, +70], for n=2 -> [-35, +35]
        start_offset = -((n - 1) * AUX_SPREAD_SPACING) / 2

        for i, aux_name in enumerate(aux_names):
            aux = self.auxs[aux_name]
            aux.x = target_x + start_offset + (i * AUX_SPREAD_SPACING)
            aux.y = aux_y
```

### Fix 4: Subsystem Separation

**File:** `stella_mcp/xmile.py`, method `_find_subsystems()` and `_arrange_subsystems()`

**Current behavior:** Works but needs verification. Subsystems detected via undirected DFS, arranged with `SUBSYSTEM_GAP = 250`.

**Verification needed:**
- Ensure error-tracking elements (Observed_CO2, Squared_Error, Cumulative_SE, RMSE) form a separate subsystem
- Confirm subsystems are ordered by relevance (main carbon cycle first, calibration/error tracking second)

**Potential fix:** If subsystem detection groups everything together, we may need to identify "auxiliary" subsystems by name patterns (e.g., containing "Error", "Squared", "RMSE", "Observed").

```python
def _classify_subsystems(self, subsystems: list[set[str]]) -> list[set[str]]:
    """Reorder subsystems: main model first, calibration/error tracking last."""
    ERROR_KEYWORDS = {'error', 'squared', 'rmse', 'observed', 'calibration', 'se'}

    main_subsystems = []
    aux_subsystems = []

    for subsystem in subsystems:
        # Check if subsystem looks like error tracking
        names_lower = {name.lower() for name in subsystem}
        is_error_subsystem = any(
            any(kw in name for kw in ERROR_KEYWORDS)
            for name in names_lower
        )

        if is_error_subsystem and len(subsystem) < len(subsystems[0]) / 2:
            aux_subsystems.append(subsystem)
        else:
            main_subsystems.append(subsystem)

    return main_subsystems + aux_subsystems
```

### Fix 5: Aux Positioning Above Target Flows

**Already addressed in Fix 3.** The aux spread algorithm positions auxs at `aux_y` (which is above `stock_y`), with x-coordinate centered on the target flow.

**Additional refinement:** For auxs that connect to flows (not stocks), ensure they're positioned above the flow's y-coordinate:

```python
# In the aux positioning loop
if target in self.flows:
    flow = self.flows[target]
    if flow.y is not None:
        # Position aux above the flow (smaller y = higher on screen)
        aux.y = flow.y - 80  # 80px above flow
    else:
        aux.y = aux_y
else:
    aux.y = aux_y
```

## Acceptance Criteria

- [x] **Connector angles:** All connectors have calculated angles pointing from source to target
- [x] **Stock ordering:** Stocks in a chain are ordered left-to-right following flow direction
- [x] **Aux overlap:** Multiple auxs targeting same element are spread horizontally with 70px spacing
- [x] **Subsystem separation:** Main model and error tracking subsystems are visually separated (≥200px gap)
- [x] **Aux positioning:** Auxs are positioned above their target flows, not scattered
- [x] **Regression:** Existing tests pass; user-specified positions still preserved

## Test Plan

### New Test Cases

```python
# test_positioning.py

def test_connector_angles_calculated():
    """Connectors should have angles pointing from source to target."""
    model = StellaModel("test")
    model.add_stock("A", "100", x=100, y=300)
    model.add_aux("rate", "0.1", x=100, y=150)
    model.add_connector("rate", "A")

    xml = model.to_xml()
    # Connector from (100, 150) to (100, 300) points downward
    # Angle should be -90 (or 270) degrees
    assert 'angle="-90' in xml or 'angle="270' in xml

def test_stock_chain_follows_flow_direction():
    """Stocks should be ordered by flow topology, not alphabetically."""
    model = StellaModel("test")
    # Add in wrong alphabetical order
    model.add_stock("Vegetation", "100")  # Should be middle
    model.add_stock("Atmosphere", "100")  # Should be first (source)
    model.add_stock("SOM", "100")         # Should be last (sink)

    model.add_flow("GPP", "10", from_stock="Atmosphere", to_stock="Vegetation")
    model.add_flow("Litter", "5", from_stock="Vegetation", to_stock="SOM")

    xml = model.to_xml()
    # Extract x positions - Atmosphere should have smallest x
    # (verify by parsing or checking relative positions)

def test_multiple_auxs_spread_horizontally():
    """Multiple auxs targeting same flow should spread out."""
    model = StellaModel("test")
    model.add_stock("A", "100")
    model.add_stock("B", "100")
    model.add_flow("growth", "rate1 * rate2", from_stock="A", to_stock="B")
    model.add_aux("rate1", "0.1")
    model.add_aux("rate2", "0.2")
    model.add_aux("rate3", "0.3")
    model.add_connector("rate1", "growth")
    model.add_connector("rate2", "growth")
    model.add_connector("rate3", "growth")

    xml = model.to_xml()
    # All three auxs should have different x positions
    # Parse and verify rate1.x != rate2.x != rate3.x

def test_subsystems_visually_separated():
    """Independent subsystems should have spatial gap."""
    model = StellaModel("test")
    # Main subsystem
    model.add_stock("A", "100")
    model.add_stock("B", "100")
    model.add_flow("f1", "1", from_stock="A", to_stock="B")

    # Error tracking subsystem (not connected to main)
    model.add_aux("Observed", "10")
    model.add_aux("Squared_Error", "(A - Observed)^2")
    model.add_connector("Observed", "Squared_Error")

    xml = model.to_xml()
    # Error subsystem should be ≥200px right of main subsystem
```

## Implementation Phases

### Phase 1: Connector Angles
- Add `_calculate_connector_angles()` method
- Call it at end of `_auto_layout()`
- Add test for angle calculation

### Phase 2: Stock Ordering
- Refactor `_find_stock_chains()` to prioritize by incoming_count
- Verify chains follow flow direction
- Add test for topology ordering

### Phase 3: Aux Spread
- Modify `_position_subsystem()` to group auxs by target
- Apply horizontal spread with centering
- Add test for aux spreading

### Phase 4: Verification & Polish
- Verify subsystem separation works correctly
- Test with real carbon cycle model
- Ensure all existing tests pass

## Files to Modify

| File | Changes |
|------|---------|
| `stella_mcp/xmile.py:223-274` | Refactor `_find_stock_chains()` for topology ordering |
| `stella_mcp/xmile.py:310-430` | Modify `_position_subsystem()` for aux spread |
| `stella_mcp/xmile.py:~600` | Add `_calculate_connector_angles()` method |
| `stella_mcp/xmile.py:593` | Call angle calculation after layout |
| `tests/test_positioning.py` | Add 4 new test cases |

## Open Questions

1. **Connector angle convention:** Stella documentation should confirm angle reference frame (assumed: degrees, 0=right, counter-clockwise positive, y-down coordinate system)

2. **Circular flows:** For A→B→C→A, which stock is "first"? (Assumed: lowest incoming_count, then alphabetical)

3. **Maximum spread:** If 20 auxs target one flow, should they wrap to multiple rows? (Assumed: single row, may extend beyond typical bounds)

## References

- Prior plan: `docs/plans/2026-02-02-feat-graph-based-auto-layout-plan.md`
- Brainstorm: `docs/brainstorms/2026-02-02-graph-based-auto-layout-brainstorm.md`
- Implementation: `stella_mcp/xmile.py` lines 140-550
- Tests: `tests/test_positioning.py` lines 150-280
