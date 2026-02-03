---
title: Graph-Based Auto-Layout for Stella MCP
date: 2026-02-02
status: ready-for-planning
---

# Graph-Based Auto-Layout

## What We're Building

A graph-based auto-layout algorithm that positions model elements based on their actual connections (from `add_connector` calls) rather than simple equation parsing. The layout should:

1. **Use the connector graph** - Position auxs near the flows/stocks they connect to
2. **Detect subsystems** - Identify and separate independent subgraphs (e.g., main model vs. calibration infrastructure)
3. **Arrange stock chains horizontally** - Stocks connected by flows form left-to-right chains
4. **Position auxs by relationship** - Parameters above, outputs below, but prioritize clarity over rigid rules

## Why This Approach

The current layout puts everything in two horizontal rows (stocks at y=300, auxs at y=150), ignoring the connector information entirely. This makes complex models unreadable.

**Graph-based hierarchical layout** was chosen over:
- **Force-directed**: Non-deterministic, can produce messy results
- **Template-based**: Fails on novel structures, maintenance burden

Graph-based is deterministic, uses actual relationship data, and naturally handles subsystem separation.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Layout algorithm | Graph-based hierarchical | Uses connectors directly, deterministic, handles subsystems |
| Subsystem arrangement | By size (largest centered) | Natural visual hierarchy |
| Stock chain orientation | Horizontal (left-to-right) | Matches flow direction, standard SD convention |
| Aux placement | Mixed by type, clarity first | Parameters generally above, outputs below, but layout clarity trumps rigid rules |

## Algorithm Sketch

```
1. Build dependency graph from connectors
   - Nodes: all stocks, flows, auxs
   - Edges: connector relationships (from_var → to_var)

2. Detect subsystems (connected components)
   - Use DFS/BFS to find disconnected subgraphs
   - Sort by size (node count)

3. For each subsystem:
   a. Find stock chain(s)
      - Trace flows between stocks
      - Arrange horizontally, left-to-right

   b. Layer auxs by graph distance from stocks
      - Direct connections to stocks/flows: near those elements
      - Parameters (no incoming edges): above the chain
      - Outputs/indicators (no outgoing edges): below

   c. Position flows between their source/target stocks

4. Arrange subsystems
   - Largest subsystem centered
   - Smaller subsystems positioned around edges
   - Maintain spacing between subsystems
```

## Example: Carbon Cycle Model

**Input:** Atmosphere, Vegetation, SOM stocks with flows; GPP_base, Atm_ref, Km_CO2 auxs feeding GPP; calibration subsystem (Observed_CO2, Squared_Error, etc.)

**Expected output:**
```
Main carbon cycle (centered, largest):
    [GPP_base] [Atm_ref] [Km_CO2]
           \     |      /
            \    |     /
[Emissions] → [Atmosphere] ←→ [Vegetation] → [SOM]
                  ↑               ↑            ↑
            [emissions_base]  [resp_frac]  [tau params]

Calibration subsystem (offset to side):
    [Observed_CO2]
          ↓
    [Squared_Error] → [Cumulative_SE] → [RMSE]
```

## Open Questions

1. **Bidirectional flows** - How to handle Atmosphere ↔ Vegetation (two flows)? Suggest: position as single bidirectional arrow visually
2. **Aux connected to multiple targets** - Where to place an aux that feeds multiple flows? Suggest: center between targets
3. **Spacing parameters** - What default spacing values work well? Suggest: start with current values, tune empirically

## Success Criteria

- [ ] Auxs positioned near the flows/stocks they connect to (via connectors)
- [ ] Independent subsystems visually separated
- [ ] Stock chains arranged horizontally following flow direction
- [ ] No element overlap
- [ ] Deterministic: same model always produces same layout

## Next Steps

Run `/workflows:plan` to create implementation plan.
