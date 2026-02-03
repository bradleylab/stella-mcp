---
title: Graph-Based Auto-Layout Algorithm
type: feat
date: 2026-02-02
brainstorm: docs/brainstorms/2026-02-02-graph-based-auto-layout-brainstorm.md
---

# Graph-Based Auto-Layout Algorithm

## Overview

Replace the current equation-parsing layout with a graph-based algorithm that uses `model.connectors` to position elements based on their actual relationships. This produces cleaner layouts where related elements cluster together and independent subsystems are visually separated.

## Problem Statement

The current `_auto_layout()` in `xmile.py:230-387`:
- Places all stocks in a horizontal row at y=300
- Places all auxs in rows at y=90, y=150, or y=380 based on equation parsing
- **Ignores connector relationships entirely**
- Results in unreadable diagrams for complex models

Example: In a carbon cycle model, `GPP_base`, `Atm_ref`, and `Km_CO2` should cluster near the `GPP` flow they feed into, not scattered in a generic row.

## Proposed Solution

A hierarchical graph-based layout algorithm:

1. **Build graph from connectors** (not equations)
2. **Detect subsystems** via connected components
3. **Arrange stock chains** horizontally following flow direction
4. **Layer auxs** by graph distance from their targets
5. **Separate subsystems** (largest centered, smaller offset)

## Technical Approach

### Phase 1: Graph Construction

**New method `_build_dependency_graph()`:**

```python
def _build_dependency_graph(self) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build bidirectional adjacency lists from connectors.

    Returns:
        (outgoing, incoming) where:
        - outgoing[node] = set of nodes this node connects TO
        - incoming[node] = set of nodes that connect TO this node
    """
    outgoing: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, set[str]] = defaultdict(set)

    # Add all elements as nodes (even if no connectors)
    for name in self.stocks:
        outgoing.setdefault(name, set())
        incoming.setdefault(name, set())
    for name in self.flows:
        outgoing.setdefault(name, set())
        incoming.setdefault(name, set())
    for name in self.auxs:
        outgoing.setdefault(name, set())
        incoming.setdefault(name, set())

    # Add connector edges
    for conn in self.connectors:
        outgoing[conn.from_var].add(conn.to_var)
        incoming[conn.to_var].add(conn.from_var)

    # Add implicit flow-stock edges
    for name, flow in self.flows.items():
        if flow.from_stock:
            outgoing[flow.from_stock].add(name)
            incoming[name].add(flow.from_stock)
        if flow.to_stock:
            outgoing[name].add(flow.to_stock)
            incoming[flow.to_stock].add(name)

    return outgoing, incoming
```

**Files:** `stella_mcp/xmile.py`

### Phase 2: Subsystem Detection

**New method `_find_subsystems()`:**

```python
def _find_subsystems(self, graph: dict[str, set[str]]) -> list[set[str]]:
    """Find connected components (subsystems) in the graph.

    Returns list of node sets, sorted by size (largest first).
    """
    all_nodes = set(self.stocks) | set(self.flows) | set(self.auxs)
    visited: set[str] = set()
    subsystems: list[set[str]] = []

    def dfs(node: str, component: set[str]):
        if node in visited or node not in all_nodes:
            return
        visited.add(node)
        component.add(node)
        for neighbor in graph.get(node, set()):
            dfs(neighbor, component)

    # Build undirected graph for component detection
    undirected: dict[str, set[str]] = defaultdict(set)
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            undirected[node].add(neighbor)
            undirected[neighbor].add(node)

    for node in sorted(all_nodes):  # Sorted for determinism
        if node not in visited:
            component: set[str] = set()
            dfs(node, component)
            if component:
                subsystems.append(component)

    return sorted(subsystems, key=len, reverse=True)
```

**Files:** `stella_mcp/xmile.py`

### Phase 3: Stock Chain Detection

**New method `_find_stock_chains()`:**

```python
def _find_stock_chains(self, subsystem: set[str]) -> list[list[str]]:
    """Find linear chains of stocks connected by flows within a subsystem.

    Returns list of chains, where each chain is ordered by flow direction.
    """
    stocks_in_subsystem = [s for s in subsystem if s in self.stocks]
    if not stocks_in_subsystem:
        return []

    # Build stock-to-stock adjacency via flows
    stock_adj: dict[str, list[str]] = defaultdict(list)
    for name, flow in self.flows.items():
        if flow.from_stock in subsystem and flow.to_stock in subsystem:
            stock_adj[flow.from_stock].append(flow.to_stock)

    # Find chain starting points (stocks with no incoming flow from within subsystem)
    has_incoming = set()
    for sources in stock_adj.values():
        has_incoming.update(sources)

    start_stocks = [s for s in stocks_in_subsystem if s not in has_incoming]
    if not start_stocks:
        # Cycle - pick alphabetically first for determinism
        start_stocks = [sorted(stocks_in_subsystem)[0]]

    # Trace chains
    chains: list[list[str]] = []
    visited: set[str] = set()

    for start in sorted(start_stocks):
        if start in visited:
            continue
        chain = []
        current = start
        while current and current not in visited:
            visited.add(current)
            chain.append(current)
            # Follow first outgoing flow (sorted for determinism)
            next_stocks = sorted(stock_adj.get(current, []))
            current = next_stocks[0] if next_stocks else None
        if chain:
            chains.append(chain)

    return chains
```

**Files:** `stella_mcp/xmile.py`

### Phase 4: Aux Layering

**New method `_layer_auxs()`:**

```python
def _layer_auxs(
    self,
    subsystem: set[str],
    incoming: dict[str, set[str]]
) -> dict[str, int]:
    """Assign layer numbers to auxs based on distance from stocks/flows.

    Layer 0: Elements directly connected to stocks or flows
    Layer 1: Elements connected to layer 0
    etc.

    Returns dict of aux_name -> layer_number
    """
    auxs_in_subsystem = [a for a in subsystem if a in self.auxs]
    stocks_and_flows = set(self.stocks) | set(self.flows)

    layers: dict[str, int] = {}
    current_layer = 0

    # Layer 0: auxs that connect directly to stocks or flows
    layer_0 = []
    for aux in auxs_in_subsystem:
        targets = incoming.get(aux, set())  # What does this aux feed into?
        # Check outgoing from this aux
        for conn in self.connectors:
            if conn.from_var == aux and conn.to_var in stocks_and_flows:
                layer_0.append(aux)
                break

    for aux in layer_0:
        layers[aux] = 0

    # BFS for remaining layers
    remaining = set(auxs_in_subsystem) - set(layer_0)
    while remaining:
        current_layer += 1
        next_layer = []
        for aux in remaining:
            # Check if this aux connects to something in previous layers
            for conn in self.connectors:
                if conn.from_var == aux and conn.to_var in layers:
                    next_layer.append(aux)
                    break

        if not next_layer:
            # Remaining auxs are orphans within subsystem - put in last layer
            for aux in remaining:
                layers[aux] = current_layer
            break

        for aux in next_layer:
            layers[aux] = current_layer
        remaining -= set(next_layer)

    return layers
```

**Files:** `stella_mcp/xmile.py`

### Phase 5: Positioning Algorithm

**Rewrite `_auto_layout()`:**

```python
def _auto_layout(self):
    """Auto-arrange visual positions using graph-based hierarchical layout.

    Algorithm:
    1. Build dependency graph from connectors
    2. Find subsystems (connected components)
    3. For each subsystem:
       a. Find stock chains
       b. Position stocks horizontally
       c. Position flows between stocks
       d. Layer and position auxs
    4. Arrange subsystems (largest centered)
    """
    # Skip if no elements need positioning
    needs_layout = any(
        s.x is None or s.y is None for s in self.stocks.values()
    ) or any(
        a.x is None or a.y is None for a in self.auxs.values()
    )

    if not needs_layout and not self.flows:
        return

    # Constants
    STOCK_SPACING = 200
    AUX_SPACING = 80
    LAYER_SPACING = 60
    SUBSYSTEM_GAP = 250
    STOCK_Y = 300
    AUX_BASE_Y = 150
    START_X = 200

    # Fallback: if no connectors, use equation-based layout
    if not self.connectors:
        self._equation_based_layout()  # Extract current logic to this method
        return

    # Build graph
    outgoing, incoming = self._build_dependency_graph()

    # Find subsystems
    subsystems = self._find_subsystems(outgoing)

    # Position each subsystem
    subsystem_bounds: list[tuple[float, float, float, float]] = []  # (min_x, min_y, max_x, max_y)

    for subsystem in subsystems:
        bounds = self._position_subsystem(
            subsystem, outgoing, incoming,
            STOCK_SPACING, AUX_SPACING, LAYER_SPACING, STOCK_Y, AUX_BASE_Y, START_X
        )
        subsystem_bounds.append(bounds)

    # Arrange subsystems relative to each other
    self._arrange_subsystems(subsystems, subsystem_bounds, SUBSYSTEM_GAP)

    # Always recalculate flow points
    self._recalculate_flow_points()
```

**Files:** `stella_mcp/xmile.py:230-387` (rewrite)

### Phase 6: Subsystem Positioning

**New method `_position_subsystem()`:**

```python
def _position_subsystem(
    self,
    subsystem: set[str],
    outgoing: dict[str, set[str]],
    incoming: dict[str, set[str]],
    stock_spacing: float,
    aux_spacing: float,
    layer_spacing: float,
    stock_y: float,
    aux_base_y: float,
    start_x: float
) -> tuple[float, float, float, float]:
    """Position all elements in a subsystem. Returns bounding box."""

    # Find stock chains
    chains = self._find_stock_chains(subsystem)

    # Position stocks in chains
    x = start_x
    stock_positions: dict[str, tuple[float, float]] = {}

    for chain in chains:
        for stock_name in chain:
            stock = self.stocks[stock_name]
            if stock.x is None or stock.y is None:
                stock.x = x
                stock.y = stock_y
            stock_positions[stock_name] = (stock.x, stock.y)
            x += stock_spacing

    # Position flows between their stocks
    for name, flow in self.flows.items():
        if name not in subsystem:
            continue
        if flow.x is not None and flow.y is not None:
            continue

        from_pos = stock_positions.get(flow.from_stock)
        to_pos = stock_positions.get(flow.to_stock)

        if from_pos and to_pos:
            flow.x = (from_pos[0] + to_pos[0]) / 2
            flow.y = (from_pos[1] + to_pos[1]) / 2
        elif from_pos:
            flow.x = from_pos[0] + 90
            flow.y = from_pos[1]
        elif to_pos:
            flow.x = to_pos[0] - 90
            flow.y = to_pos[1]

    # Layer auxs
    aux_layers = self._layer_auxs(subsystem, incoming)

    # Position auxs by layer, clustered near their targets
    for aux_name, layer in sorted(aux_layers.items(), key=lambda x: (x[1], x[0])):
        aux = self.auxs[aux_name]
        if aux.x is not None and aux.y is not None:
            continue

        # Find target position (what this aux connects to)
        target_x = self._find_aux_target_x(aux_name, outgoing)

        # Y position based on layer (above or below stocks)
        if layer == 0:
            aux.y = aux_base_y  # Near flows/stocks
        else:
            aux.y = aux_base_y - (layer * layer_spacing)  # Stack upward

        aux.x = target_x if target_x else start_x

    # Calculate bounding box
    all_x = [self.stocks[s].x for s in subsystem if s in self.stocks and self.stocks[s].x]
    all_x += [self.auxs[a].x for a in subsystem if a in self.auxs and self.auxs[a].x]
    all_y = [self.stocks[s].y for s in subsystem if s in self.stocks and self.stocks[s].y]
    all_y += [self.auxs[a].y for a in subsystem if a in self.auxs and self.auxs[a].y]

    if all_x and all_y:
        return (min(all_x), min(all_y), max(all_x), max(all_y))
    return (start_x, aux_base_y, start_x + stock_spacing, stock_y)
```

### Phase 7: Subsystem Arrangement

**New method `_arrange_subsystems()`:**

```python
def _arrange_subsystems(
    self,
    subsystems: list[set[str]],
    bounds: list[tuple[float, float, float, float]],
    gap: float
):
    """Arrange subsystems: largest centered, smaller offset to the right."""
    if len(subsystems) <= 1:
        return

    # First subsystem (largest) stays in place
    # Offset subsequent subsystems to the right
    current_x = bounds[0][2] + gap  # max_x of first + gap

    for i, subsystem in enumerate(subsystems[1:], start=1):
        min_x, min_y, max_x, max_y = bounds[i]
        offset_x = current_x - min_x

        # Shift all elements in this subsystem
        for name in subsystem:
            if name in self.stocks and self.stocks[name].x is not None:
                self.stocks[name].x += offset_x
            if name in self.flows and self.flows[name].x is not None:
                self.flows[name].x += offset_x
            if name in self.auxs and self.auxs[name].x is not None:
                self.auxs[name].x += offset_x

        current_x = current_x + (max_x - min_x) + gap
```

### Phase 8: Helper Methods

**New method `_find_aux_target_x()`:**

```python
def _find_aux_target_x(self, aux_name: str, outgoing: dict[str, set[str]]) -> Optional[float]:
    """Find the x-coordinate where an aux should be positioned (near its targets)."""
    targets = []

    for conn in self.connectors:
        if conn.from_var == aux_name:
            target = conn.to_var
            if target in self.stocks and self.stocks[target].x is not None:
                targets.append(self.stocks[target].x)
            elif target in self.flows and self.flows[target].x is not None:
                targets.append(self.flows[target].x)
            elif target in self.auxs and self.auxs[target].x is not None:
                targets.append(self.auxs[target].x)

    if targets:
        return sum(targets) / len(targets)  # Center between targets
    return None
```

**New method `_equation_based_layout()`:**

Extract current logic from `_auto_layout()` as fallback when no connectors exist.

**New method `_recalculate_flow_points()`:**

Extract flow point calculation from current `_auto_layout()`.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary data source | Connectors | Explicit relationships are more reliable than equation parsing |
| Fallback when no connectors | Equation-based layout | Backward compatibility |
| Cycle handling | All cycle members same layer | Feedback loops are peers |
| Orphan elements | Treat as constants (top-left) | Consistent with current behavior |
| Multi-target aux position | Geometric center of targets | Minimizes connector crossings |
| Subsystem arrangement | Horizontal (left-to-right by size) | Clear visual separation |
| Determinism | Sort all collections before processing | Same model = same layout |

---

## Acceptance Criteria

### Core Functionality
- [ ] Auxs positioned near the flows/stocks they connect to (via connectors)
- [ ] Independent subsystems visually separated
- [ ] Stock chains arranged horizontally following flow direction
- [ ] No element overlap
- [ ] Deterministic: same model always produces same layout

### Edge Cases
- [ ] Empty model: no error, no positioning needed
- [ ] No connectors: falls back to equation-based layout
- [ ] Cycles in connector graph: handled without infinite loop
- [ ] Orphan elements: positioned as constants
- [ ] Aux connected to multiple targets: centered between them
- [ ] Bidirectional flows: both positioned without overlap

### Backward Compatibility
- [ ] User-specified positions still preserved
- [ ] Models without connectors work as before
- [ ] Existing tests pass

---

## Testing Strategy

### Unit Tests

```python
# tests/test_graph_layout.py

class TestGraphConstruction:
    def test_build_graph_from_connectors(self):
        """Graph includes connector edges."""
        model = StellaModel("Test")
        model.add_stock("A", "100")
        model.add_aux("rate", "0.1")
        model.add_flow("flow1", "A * rate", from_stock="A")
        model.add_connector("rate", "flow1")

        outgoing, incoming = model._build_dependency_graph()
        assert "flow1" in outgoing["rate"]
        assert "rate" in incoming["flow1"]

    def test_graph_includes_flow_stock_edges(self):
        """Graph includes implicit flow-stock relationships."""
        model = StellaModel("Test")
        model.add_stock("A", "100")
        model.add_stock("B", "100")
        model.add_flow("transfer", "10", from_stock="A", to_stock="B")

        outgoing, incoming = model._build_dependency_graph()
        assert "transfer" in outgoing["A"]
        assert "B" in outgoing["transfer"]


class TestSubsystemDetection:
    def test_single_subsystem(self):
        """All connected elements form one subsystem."""
        model = StellaModel("Test")
        model.add_stock("A", "100")
        model.add_stock("B", "100")
        model.add_flow("f", "10", from_stock="A", to_stock="B")

        outgoing, _ = model._build_dependency_graph()
        subsystems = model._find_subsystems(outgoing)
        assert len(subsystems) == 1
        assert {"A", "B", "f"} <= subsystems[0]

    def test_multiple_subsystems(self):
        """Disconnected elements form separate subsystems."""
        model = StellaModel("Test")
        model.add_stock("A", "100")
        model.add_stock("B", "100")  # No connection to A
        model.add_aux("orphan", "5")  # No connectors

        outgoing, _ = model._build_dependency_graph()
        subsystems = model._find_subsystems(outgoing)
        assert len(subsystems) == 3  # Each isolated


class TestStockChains:
    def test_linear_chain(self):
        """Stocks connected by flows form a chain."""
        model = StellaModel("Test")
        model.add_stock("A", "100")
        model.add_stock("B", "100")
        model.add_stock("C", "100")
        model.add_flow("f1", "10", from_stock="A", to_stock="B")
        model.add_flow("f2", "10", from_stock="B", to_stock="C")

        chains = model._find_stock_chains({"A", "B", "C", "f1", "f2"})
        assert chains == [["A", "B", "C"]]


class TestAuxLayering:
    def test_direct_connection_is_layer_0(self):
        """Aux connecting to flow is layer 0."""
        model = StellaModel("Test")
        model.add_stock("A", "100")
        model.add_aux("rate", "0.1")
        model.add_flow("flow1", "A * rate", to_stock="A")
        model.add_connector("rate", "flow1")

        _, incoming = model._build_dependency_graph()
        layers = model._layer_auxs({"A", "rate", "flow1"}, incoming)
        assert layers["rate"] == 0


class TestFullLayout:
    def test_carbon_cycle_layout(self):
        """Complex model with subsystems lays out correctly."""
        model = StellaModel("Carbon")

        # Main cycle
        model.add_stock("Atmosphere", "750")
        model.add_stock("Vegetation", "550")
        model.add_aux("GPP_base", "120")
        model.add_flow("GPP", "GPP_base", from_stock="Atmosphere", to_stock="Vegetation")
        model.add_connector("GPP_base", "GPP")

        # Calibration (separate subsystem)
        model.add_aux("Observed", "400")
        model.add_aux("Error", "Atmosphere - Observed")
        model.add_connector("Observed", "Error")
        # Note: Error references Atmosphere in equation but no connector
        # This makes it a separate subsystem unless we add connector

        model._auto_layout()

        # GPP_base should be near GPP flow
        assert abs(model.auxs["GPP_base"].x - model.flows["GPP"].x) < 100

        # Calibration should be offset from main cycle
        main_max_x = max(model.stocks["Atmosphere"].x, model.stocks["Vegetation"].x)
        assert model.auxs["Observed"].x > main_max_x


class TestFallback:
    def test_no_connectors_uses_equation_layout(self):
        """Without connectors, falls back to equation-based layout."""
        model = StellaModel("Test")
        model.add_stock("A", "100")
        model.add_aux("rate", "0.1")  # No connector
        model.add_flow("f", "A * rate", to_stock="A")

        model._auto_layout()

        # Should still position elements (using equation-based fallback)
        assert model.stocks["A"].x is not None
        assert model.auxs["rate"].x is not None
```

---

## Implementation Order

1. **Phase 1-2:** Graph construction and subsystem detection
   - Add `_build_dependency_graph()` and `_find_subsystems()`
   - Add unit tests for graph building

2. **Phase 3-4:** Stock chains and aux layering
   - Add `_find_stock_chains()` and `_layer_auxs()`
   - Add unit tests

3. **Phase 5-7:** Main layout algorithm
   - Extract current logic to `_equation_based_layout()`
   - Rewrite `_auto_layout()` with graph-based algorithm
   - Add `_position_subsystem()` and `_arrange_subsystems()`

4. **Phase 8:** Helpers and integration
   - Add `_find_aux_target_x()` and `_recalculate_flow_points()`
   - Integration tests with complex models

5. **Testing:** Run against real models (carbon cycle, HW1)

---

## Files to Modify

| File | Changes |
|------|---------|
| `stella_mcp/xmile.py` | Rewrite `_auto_layout()`, add 8 new methods |
| `tests/test_positioning.py` | Update existing tests for new behavior |
| `tests/test_graph_layout.py` | New test file for graph-specific tests |

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Graph algorithm bugs cause infinite loops | Medium | High | Cycle detection, visited tracking, iteration limits |
| Layout worse than current for some models | Medium | Medium | Keep equation-based as fallback, test on multiple models |
| Performance issues on large models | Low | Medium | Lazy evaluation, early exit when all positioned |
| Non-deterministic output | Medium | Low | Sort all collections, use stable algorithms |

---

## References

### Internal
- Current `_auto_layout()`: `stella_mcp/xmile.py:230-387`
- DFS pattern for cycles: `stella_mcp/validator.py:232-271`
- Connector dataclass: `stella_mcp/xmile.py:69-75`
- Brainstorm: `docs/brainstorms/2026-02-02-graph-based-auto-layout-brainstorm.md`

### External
- [Sugiyama hierarchical layout](https://en.wikipedia.org/wiki/Layered_graph_drawing) - inspiration for layering approach
