# Layout Quality Milestone Specification

Status: approved, blocking Stella MCP 0.12.0 publication

## Objective

Replace the current best-effort force-directed auto-layout with a deterministic
hybrid pipeline that produces readable Stella diagrams for linear, branching,
cyclic, disconnected, and dense small models. The pipeline must preserve
explicit user geometry, prevent visible elements and routed lines from passing
through unrelated elements, minimize avoidable crossings and route length, fit
the declared Stella page grid, and provide reproducible quality evidence.

The feature is complete only after the generated fixtures pass automated layout
gates and visual validation in Stella Professional 4.1.1. This work is a release
gate for 0.12.0, not a follow-up after publication.

## Why This Is Needed

The existing physics layout has useful guarantees, including determinism,
size-aware separation, fixed-position support, and a retained 30-element runtime
budget. It does not organize the directed stock-flow structure or route the
finished diagram globally:

- `model_layout._position_subsystem()` places stocks and auxiliaries but excludes
  flows from the force simulation (`stella_mcp/model_layout.py:100-170`).
- `_auto_layout()` positions flows and calculates connector angles after node
  placement, but does not call the available crossing-repair pass
  (`stella_mcp/model_layout.py:361-403`).
- flow routing is local to each stock and does not optimize the complete set of
  routes (`stella_mcp/model_layout.py:504-623`).
- the repair pass only moves source auxiliaries for connector crossings and can
  introduce new collisions (`stella_mcp/model_layout.py:874-1082`).
- auto-layout uses a `1600` by `1000` canvas while new exports declare two
  `768` by `596` pages in each direction, so placement and page geometry do not
  share one source of truth (`stella_mcp/layout.py:168-169` and
  `stella_mcp/xmile_export.py:433-436`).
- labels are rendered after layout and are not represented in collision checks
  (`stella_mcp/render_svg.py:35-39`).

The original force-directed plan explicitly deferred crossing avoidance,
left-to-right ordering, and Stella tuning to a second phase
(`docs/plans/2026-02-08-feat-force-directed-layout-engine-plan.md`). This
milestone completes that deferred phase.

## Users And Observable Outcome

The affected users are people and agents that build models through Stella MCP,
then inspect or edit the result in Stella Professional or through
`render_diagram`.

After this milestone:

1. A directed stock-flow chain reads in flow direction without manual movement.
2. Branching flows use distinct, compact routes rather than long shared trunks or
   loops.
3. Feedback cycles remain visibly cyclic but do not collapse into tangles.
4. Connectors and flows avoid unrelated stocks, auxiliaries, valves, and labels.
5. Compact models use the smallest Stella page grid that contains the complete
   diagram, rather than appearing tiny in a fixed two-by-two grid.
6. If a graph cannot be drawn without line-to-line crossings, Stella MCP returns
   the best deterministic result and a structured warning describing the
   remaining violation.

## Scope

### Included

- A directed, layered layout for the stock-flow backbone.
- Strongly connected component handling for feedback loops.
- Deterministic packing of disconnected subsystems.
- Target-aware auxiliary placement.
- Port assignment and obstacle-aware flow and connector routing.
- Label-side selection and label collision checks.
- Dynamic Stella page rows and columns derived from final visual bounds.
- A shared layout analyzer, structured report, benchmark fixtures, and release
  gates.
- Stella Professional 4.1.1 open, run, save, and visual evidence for the release
  fixtures.
- Documentation and release-note updates for the changed behavior.

### Excluded

- A graphical editor or interactive drag-layout UI.
- A guarantee of zero line-to-line crossings for arbitrary non-planar graphs.
- Moving explicitly positioned stocks or auxiliaries, or rewriting explicitly
  locked flow and connector points.
- Automatic redesign of imported, fully positioned models when
  `auto_layout=False`.
- Changes to model equations, units, simulation methods, or numerical results.
- Module-level edge bundling or inter-module routing. Existing module boxes
  continue to enclose their members after element layout.
- A new mandatory runtime dependency. The core installation remains MCP plus the
  Python standard library.

## Required Invariants

The new pipeline must preserve these existing contracts:

- Same model and options produce identical positions, routes, label sides, page
  geometry, metrics, and warning order.
- User-specified stock and auxiliary coordinates remain byte-for-byte equivalent
  through export.
- Locked flow and connector point lists remain unchanged.
- Empty models, singleton models, orphan flows, and auxiliary-only subsystems
  still export and render.
- Layout does not change names, equations, units, connector endpoints, flow-stock
  relationships, simulation settings, or module membership.
- `auto_layout=False` preserves authored geometry and only derives missing,
  unlocked visual metadata.
- `resolve_layout_violations=True` remains accepted by the MCP and Python APIs.
  It invokes the new validator and safe router on existing geometry; it must not
  call the current mutation-only repair loop.
- The existing 30-element `to_xml()` runtime gate remains below two seconds on
  the test environment (`tests/test_force_directed.py`).

One current behavior intentionally changes. Today the first `to_xml()` writes
auto-layout positions onto elements, and every later run treats any element
with coordinates as pinned (`stella_mcp/model_layout.py:119-128`), so a
session model's layout freezes after its first export and elements added
later are squeezed around the frozen skeleton. With the position provenance
introduced below, every layout run recomputes all `auto` positions, so
incremental building through the MCP tools re-ranks the whole diagram on each
export, while `user` positions and locked routes never move. Unchanged models
still satisfy the determinism invariant because recomputation is
deterministic. Release notes must call out this behavior change.

## Architecture

### Data Structures

Add pure layout dataclasses, independent of `StellaModel`:

- `LayoutViewport`: page width, page height, rows, columns, and derived bounds.
- `LayoutBox`: named visual obstacle with kind, center, size, and lock state.
- `LayoutPort`: element boundary point and outward direction.
- `LayoutRoute`: endpoint names, ordered points, route kind, and lock state.
- `LayoutMetrics`: every quality count and route statistic defined below.
- `LayoutWarning`: stable code, message, and sorted affected element names.
- `LayoutResult`: positions, flow routes, connector routes, label sides, viewport,
  metrics, and warnings.

`StellaModel` gains typed view-page fields instead of discarding the known page
attributes during parsing. Stocks, flows, and auxiliaries gain an optional
`label_side` field with the Stella-supported values `top`, `bottom`, `left`, and
`right`. Unknown view attributes continue to round-trip through the existing
extra-attribute dictionaries.

Stocks, flows, and auxiliaries also gain a position-provenance marker
(`position_source`: `user` or `auto`). Coordinates parsed from an imported
document or supplied through the API are `user`; coordinates written by the
layout pipeline are `auto`. Only `user` coordinates are pinned. The marker is
internal session state and is not serialized to XMILE: a fully positioned
imported model is authored geometry, which the Scope section already excludes
from redesign.

All new spacing is derived from existing visual geometry rather than duplicated
magic numbers:

- rank-center spacing is the greater of `DEFAULT_IDEAL_EDGE_LENGTH` and the two
  facing half-widths plus `_SEPARATION_GAP`, widened when necessary so the
  valve glyph and the widest flow label box between the two ranks fit with
  `_SEPARATION_GAP` clearance on both sides;
- within-rank spacing is the two facing half-heights plus the label box height
  of the upper element plus `_SEPARATION_GAP` — labels render below their
  element, so a formula without the label height makes the zero
  label-glyph-overlap gate unsatisfiable at minimum spacing;
- component packing uses `DEFAULT_IDEAL_EDGE_LENGTH` between component boxes;
- port and route clearance uses `_SEPARATION_GAP` outside glyph and label boxes;
- the imported view font controls label size. The default `9pt` style becomes
  `12` CSS pixels using the standard `96` pixels-per-inch and `72`
  points-per-inch conversion. The dependency-free label box is one font pixel
  high and `0.6` font pixels wide per display-name code point as the initial
  estimate; Phase 1 must calibrate this factor against the recorded Stella
  Professional screenshots before any gate depends on it. A full font pixel
  per code point is roughly double the true average glyph width and would
  force exactly the sprawl and page-fit failures this milestone exists to
  remove. The existing renderer label offset separates that box from its
  glyph.

All auto-assigned coordinates and route points snap to whole pixels. Integer
geometry makes the orientation and collinearity predicates exact rather than
epsilon-tuned, makes every lexicographic tie-break on coordinates exact, and
gives the strict-improvement loops below a finite score lattice so their
termination is provable rather than assumed. User-supplied coordinates are
never snapped; the analyzer must accept mixed integer and float geometry.

### Pipeline

`_auto_layout()` becomes an orchestration wrapper around the following pure,
testable stages.

#### 1. Normalize Geometry

- Calculate stock sizes using the existing connectivity rule.
- Load page dimensions from imported view geometry or the export defaults of
  `768` by `596`.
- Record every supplied coordinate and every locked point list before making any
  change.
- Build display-label boxes from the configured view font and display names
  through one shared deterministic estimator used by layout and SVG rendering.

#### 2. Build The Layout Graph

- Create a directed stock graph from flows with both a source and destination
  stock.
- Preserve source-only, destination-only, self-loop, and orphan flows as typed
  edges rather than dropping them from the graph.
- Build the dependency graph for auxiliaries and connectors separately.
- Find weakly connected components and strongly connected components using
  deterministic name ordering.
- Collapse strongly connected components into a directed acyclic condensation
  graph for ranking.

#### 3. Place The Stock-Flow Backbone

- Assign acyclic components to left-to-right ranks following flow direction.
- Order nodes within adjacent ranks using deterministic barycentric sweeps until
  a forward-and-backward sweep pair no longer strictly improves the crossing
  count. A proposed order with an equal score is rejected; initial and local ties
  resolve by the normalized element name. Termination follows from the integer
  crossing count strictly decreasing.
- After ordering, assign within-rank coordinates with a deterministic alignment
  pass: stocks joined by a flow share the same cross-rank coordinate whenever no
  hard constraint prevents it, with priority to longer flow chains and ties
  resolved by normalized name, so a linear chain renders as one straight
  zero-bend pipe. Ordering alone does not produce straight pipes, and the
  router cannot repair misaligned ports after placement; this alignment step is
  the primary source of the "reads in flow direction" outcome.
- Place multi-stock strongly connected components as compact local rings, then
  treat each ring as one ranked block in the condensation graph. Ring radius is
  derived from the member boxes and separation gap. Evaluate every rotation and
  both directions of the normalized-name order, and select the lexicographically
  best crossing, route-length, then name-order result. Score candidates on the
  ring's local crossings and route length only, not a full-layout pass; the
  existing 30-element runtime budget bounds the candidate count.
- Keep pinned nodes fixed. Free nodes in a partially pinned component use the
  pinned positions as anchors; an unsatisfiable pin arrangement produces a
  warning instead of moving a pin.
- Use the current force-directed solver only for auxiliary-only components and
  local cyclic refinement. It is no longer responsible for global stock-flow
  ordering.
- Pack disconnected component boxes into rows within the viewport. Start a new
  row when the next component would exceed the current page width; do not append
  every subsystem indefinitely to the right.

Stage order matters: auxiliaries are placed before authoritative routing so
routes can avoid them, and free label sides are chosen after routing so labels
avoid routes. Each stage names the obstacle set it sees; no stage may consult
geometry a later stage produces.

#### 4. Assign Flow Ports And Provisional Routes

- Allocate distinct stock-edge ports for all incident flows before routing any
  flow. Port order follows the destination rank and position, with normalized
  names as the deterministic tie-breaker.
- Give each flow a provisional route: the straight port-to-port segment for
  aligned ports, otherwise the direct polyline between its ports. Provisional
  routes exist so stage 5 has valve estimates and obstacle context; the
  authoritative routes are produced in stage 6.
- Place the provisional valve at the half-length point of the provisional
  polyline.
- Give self-loop flows a compact route around their stock. A route may not leave
  and later re-enter its endpoint bounding region unless it represents a true
  self-loop.
- Source-only and destination-only flows use the nearest free outward port.
  Orphan flows retain deterministic fallback placement in a separate packed
  region.

#### 5. Place Auxiliaries

- Place free auxiliaries in a deterministic order: topological order of the
  auxiliary dependency graph, so an auxiliary is placed after any auxiliary it
  targets, with the normalized name as the tie-breaker. Each placed auxiliary
  joins the obstacle set for the auxiliaries that follow.
- For each free auxiliary, derive its preferred location from all connector
  targets. One target uses candidate positions around that target; multiple
  targets use candidates around their geometric median. Flow targets use the
  provisional valve position from stage 4.
- The first candidate ring contains the four cardinal and four diagonal
  positions whose clearance is the target half-extent, auxiliary radius, and
  separation gap. If every candidate has a hard conflict, expand by one
  separation gap and repeat. If expansion exhausts the working viewport, place
  the auxiliary at the deterministic least-bad candidate and emit
  `layout.placement_exhausted`.
- Score the finite candidate set using the same lexicographic hard constraints,
  then connector crossings, connector length, and deterministic coordinates.
  Obstacles at this stage are glyph boxes, provisional flow routes, and
  already-placed auxiliaries; free label boxes do not exist yet.

#### 6. Route Flows And Connectors

This stage produces the authoritative routes. Obstacles are stock, auxiliary,
and valve boxes, plus imported label boxes whose sides are locked by authored
geometry. Free label boxes are chosen in stage 7 and are not routing
obstacles; labels avoid routes, not the reverse.

- Use a direct route when the two ports have an unobstructed segment and no
  other route needs the same segment.
- Otherwise enumerate candidate routes in increasing complexity: the two
  single-bend L-routes, then the two-bend Z- and U-routes whose free
  coordinates come from expanded obstacle edges. Score candidates with the
  lexicographic criteria below and accept the best clean candidate.
- Only when no enumerated candidate is clean, fall back to a rectilinear
  visibility graph built from the obstacle boundaries expanded by the existing
  layout separation gap. Candidate axes are every port coordinate and every
  expanded obstacle edge. A graph vertex exists at each unobstructed axis
  intersection; an edge joins adjacent visible vertices on the same axis.
  Search state includes the incoming direction so bend count is part of path
  selection. The enumeration-first order exists because models within the
  30-element budget almost never need the full graph, and the fallback is the
  highest-complexity component in the milestone.
- Both the candidate enumeration and the visibility-graph search select the
  lexicographically best route by:
  1. hard-obstacle violations;
  2. crossings and shared segments;
  3. route length;
  4. bend count;
  5. ordered point coordinates.
- Attach connectors to element boundaries rather than center points. Preserve
  locked connector points exactly. Prefer a direct or single-bend connector
  route before accepting additional bends; flow segments are crossing
  penalties for connectors.
- Route locked paths first, then unlocked flows ordered by source rank,
  destination rank, and normalized name, then connectors by UID. After the
  first pass, reroute each unlocked path in the same order and accept a change
  only when the complete layout score strictly improves. Stop after a complete
  pass accepts no route, or after three complete passes; hitting the pass cap
  emits `layout.routing_fallback`. Integer geometry keeps the score lattice
  finite, so termination does not depend on the cap; the cap bounds runtime.
- Place each valve at the half-length point of its final polyline. Remove
  repeated points and redundant collinear points. When a reroute moves a
  valve, mark the auxiliaries targeting that valve for the stage 8 retry
  check.
- Serialize the selected points through the existing XMILE connector `<pts>`
  support. Connector angles remain derived compatibility metadata. Whether
  Stella honors polyline connector points is decided by the Phase 1 format
  spike before any router work; see Implementation Sequence.

#### 7. Choose Label Sides And Module Boxes

- Choose label sides after all routes exist. Candidate label boxes may not
  overlap glyphs, other labels, routed flows, or routed connectors, excepting
  the label's own element and its own attached route. If no side is clean,
  choose the deterministic least-bad side and emit `layout.label_conflict`.
- Existing imported label sides remain fixed when authored geometry is being
  preserved; their boxes were already routing obstacles in stage 6.
- Process elements in normalized-name order; each chosen label box joins the
  obstacle set for the labels that follow.
- Place module boxes after labels and routes, using the existing member bounds
  plus module padding.

#### 8. Validate And Select Page Geometry

- Analyze the complete diagram after all routes and labels exist.
- Retry only the affected placement or route when a hard violation remains,
  including auxiliary placements invalidated by valve movement in stage 6.
  Each retry must compare the complete layout score and may not accept a result
  that improves one category by worsening an earlier lexicographic category.
- Derive page columns and rows from complete glyph, label, and route bounds
  using the model's page width and page height.
- When no geometry is pinned, shift the whole diagram as one rigid body so the
  declared page area contains it. When any geometry is pinned, do not shift:
  moving free geometry relative to pinned geometry can re-create the exact
  violations this stage just validated away. Instead grow the page grid to
  cover the bounds, and emit `layout.page_overflow` only when pinned geometry
  lies outside any page grid reachable from the declared page size.
- Store `LayoutMetrics` and sorted `LayoutWarning` values on the model as the
  latest layout report.

## Layout Metrics

The analyzer must report at least:

- missing element positions;
- glyph-glyph overlaps;
- label-glyph and label-label overlaps;
- flow or connector segments through unrelated glyphs;
- flow or connector segments through unrelated labels;
- connector-flow crossings;
- connector-connector crossings excluding shared endpoints;
- flow-flow crossings and shared segments excluding shared stock ports;
- route self-intersections;
- repeated or redundant route points;
- backward acyclic stock-flow edges;
- total and maximum flow length;
- total and maximum connector length;
- total bend count;
- diagram bounds and page overflow on each edge;
- movement of pinned coordinates or locked points.

Intersection handling must correctly classify collinear overlap, endpoint touch,
and proper crossing. Endpoint touches that belong to the same declared port are
not violations; all exclusions must be explicit in the analyzer rather than
hidden in the geometry primitive.

## Warnings And API Behavior

Introduce stable warning codes:

- `layout.pinned_conflict`
- `layout.locked_route_conflict`
- `layout.unavoidable_crossing`
- `layout.page_overflow`
- `layout.label_conflict`
- `layout.placement_exhausted`
- `layout.routing_fallback`

`save_model`, `get_model_xml`, and `render_diagram` retain their current input
schemas. Their text response appends a concise warning summary when the latest
layout report is non-empty. Structured output includes the metrics and warnings
without removing or renaming existing fields.

No warning is emitted for a clean benchmark layout. Warnings are deterministic
and never suppress export; strict XMILE compatibility errors retain their current
behavior.

## Benchmark Corpus

Add reproducible builders for these layouts:

1. every built-in template after clearing authored positions and unlocked
   routes;
2. a directed ten-stock chain with one controller per flow;
3. a multi-destination stock fan-out with explicit equation dependency
   connectors;
4. a multi-stock feedback cycle with cross-cycle auxiliary dependencies;
5. disconnected stock-flow, auxiliary-only, and orphan-flow components;
6. mixed pinned and free elements;
7. a self-loop plus source-only and destination-only flows;
8. long display names that exercise label placement;
9. a dense planar dependency graph;
10. a known non-planar graph that must produce a deterministic unavoidable-
    crossing warning rather than claim a clean layout;
11. an incremental sequence that builds a model, exports it, adds a stock, a
    flow, and a controller, and exports again. The record captures both
    layouts and the auto-position churn between them (count of moved elements
    and total displacement). This fixture gates the position-provenance
    behavior: the second export must re-run the full pipeline rather than
    treating first-export positions as pinned, and any `user` coordinate must
    be identical in both exports.

Builders must create semantically valid models that run in Stella. Generated
artifacts belong under the existing evaluation output location, not in package
runtime data.

## Automated Acceptance

### Hard Gates For Planar Fixtures

- Zero missing positions.
- Zero movement of pinned coordinates or locked routes.
- Zero glyph, label-glyph, and label-label overlaps.
- Zero flow or connector segments through unrelated glyphs or labels.
- Zero route self-intersections, repeated points, and redundant collinear points.
- Zero connector-flow crossings.
- Zero connector-connector and flow-flow crossings except explicitly shared
  endpoint ports.
- Zero backward stock-flow edges outside strongly connected components or pinned
  conflicts.
- All complete visual bounds lie inside the declared page grid.
- Every route uses the direct port-to-port segment whenever that segment is
  unobstructed and unshared.
- No route is longer than a documented multiple of the Manhattan distance
  between its ports, and no route exceeds a documented bend cap. Both
  constants are recorded in the analyzer and derived from the Phase 1 baseline
  data before they gate a release. (An earlier draft gated on "the
  lexicographically best route returned by the visibility graph", which only
  tests the router against itself and pins implementation internals; these
  property gates replace it.)
- Repeated layout runs produce equal normalized `LayoutResult` values.

Graph planarity does not guarantee that a layered, port-constrained,
rectilinear drawing has zero crossings; barycentric ordering is a heuristic.
Phase 2 must therefore demonstrate that every zero-crossing gate above is
attainable on its named fixture before that gate becomes a release gate. If a
fixture proves unattainable, the fixture or the gate is amended through an
explicit spec change with operator approval, never by silently re-baselining.

### Fixture-Specific Page Gates

- Each freshly auto-laid-out built-in template fits a single `768` by `596`
  Stella page.
- The chain fixture fits within the existing two-page horizontal width and one
  page vertically. The margin is thin: nine rank gaps at minimum spacing plus
  the end stocks is roughly `1400` of the `1536` available pixels before flow
  labels widen any gap, so Phase 1 must verify this arithmetic with the
  calibrated label estimator before the gate is fixed.
- The fan-out and feedback fixtures fit one page each.
- The non-planar fixture may contain line-to-line crossings but may not violate
  any glyph, label, lock, or page hard constraint.

### Compatibility And Performance Gates

- Generated XMILE strict-imports and preserves supported semantics after export,
  Stella save, and re-import.
- Connector point lists, label sides, and page geometry round-trip.
- The SVG renderer uses the same routes and label sides as XMILE export.
- The native MCP tests cover layout warnings and structured metrics.
- The existing 30-element export completes in less than two seconds.
- Core tests run without simulation extras and without a new runtime dependency.
- Ruff, the core test suite, simulation suite, MCP-floor suite, evaluator,
  package build, and artifact checks all pass.

## Stella Desktop Acceptance

Generate SIR, Lotka-Volterra, chain, fan-out, and feedback fixtures from the new
pipeline. For each fixture in Stella Professional 4.1.1:

1. Open without a repair prompt or invalid-equation indicator.
2. Inspect at `100%` zoom and record a screenshot.
3. Confirm every label is readable and associated with one visible element.
4. Confirm no stock, auxiliary, valve, label, flow, or connector obscures an
   unrelated symbol or label.
5. Confirm flow direction is traceable without leaving and re-entering the page
   or following an avoidable crossing.
6. Run to the configured final time.
7. Save from Stella, strict-import the saved artifact, and compare supported
   semantics and layout geometry.
8. Record application version, date, source fixture, source artifact SHA-256,
   saved artifact SHA-256, page grid, automated metrics, and visual findings in
   the compatibility manifest or a linked layout-evaluation manifest.

Any failed item blocks 0.12.0. A manual visual exception requires a named fixture,
the unresolved warning code, a screenshot, and explicit release approval.

## Implementation Sequence

### Phase 1: Reproducible Baseline, Geometry, And Format Spike

- Add the benchmark builders and layout analyzer before changing placement.
- Record current normalized metrics and SVGs from the same runner.
- Replace the strict-CCW segment test with robust orientation and collinearity
  handling, and adopt whole-pixel snapping for auto-assigned geometry.
- Add typed viewport, position-provenance, and label-side round-tripping.
- Calibrate the label-width factor against the recorded Stella Professional
  screenshots, then verify the chain fixture's two-page arithmetic.
- Run the Stella format spike: hand-author one minimal `.stmx` exercising
  multi-bend connector `<pts>`, a valve at an arbitrary polyline midpoint,
  each `label_side` value, and non-default `isee:page_cols` and
  `isee:page_rows`; open and save it in Stella Professional 4.1.1 and diff the
  saved artifact. This front-loads the release-stopping risk named under
  Failure Handling instead of discovering it after the router is built. If
  Stella rewrites polyline connector points into arcs, the stage 6 connector
  deliverable changes from polyline routing to obstacle-aware selection of the
  connector arc angle, with polyline routes retained for `render_diagram`
  only; that decision is made and recorded here, before any router work.

### Phase 2: Directed Backbone And Component Packing

- Implement strongly connected components, condensation ranks, barycentric
  ordering, within-rank coordinate alignment, ring placement, pinned anchors,
  and row packing.
- Demonstrate that each zero-crossing hard gate is attainable on its fixture,
  per Automated Acceptance.
- Retain the force solver as the documented fallback only.

### Phase 3: Ports, Flow Routing, And Page Selection

- Implement boundary ports, provisional routes, candidate-enumeration routing
  with the visibility-graph fallback, route normalization, valve placement,
  self-loops, source/sink flows, and dynamic page rows/columns.

### Phase 4: Auxiliary, Label, And Connector Routing

- Implement target-aware auxiliary placement, label-side selection, label boxes,
  connector ports, and connector polylines.
- Replace `_resolve_layout_violations()` with validation-driven local repair or a
  compatibility wrapper around the new pipeline.

### Phase 5: Integration And Release Evidence

- Wire layout reports into Python and MCP responses.
- Update SVG rendering, snapshots, strict round-trip tests, README guidance,
  architecture documentation, changelog, and 0.12.0 release notes.
- Run every automated gate, then complete and record the Stella desktop protocol.

Each phase must leave the test suite green. Changes should be committed by phase
so a regression can be isolated or reverted without discarding the complete
milestone.

## Expected File Map

| Path | Responsibility |
|---|---|
| `stella_mcp/layout.py` | Robust geometry primitives and retained force fallback |
| `stella_mcp/layout_graph.py` | Directed graph, SCC, ranks, ordering, and component packing |
| `stella_mcp/layout_routing.py` | Ports, visibility graph, flow and connector routing |
| `stella_mcp/layout_quality.py` | Boxes, labels, metrics, validation, warnings, and result scoring |
| `stella_mcp/model_layout.py` | `StellaModel` orchestration and application of `LayoutResult` |
| `stella_mcp/model_types.py` | Typed label-side and view/layout metadata |
| `stella_mcp/xmile_parse.py` | Page geometry and label-side parsing |
| `stella_mcp/xmile_export.py` | Dynamic page geometry and routed-point export |
| `stella_mcp/render_svg.py` | Shared label sides, boxes, and routed polylines |
| `stella_mcp/tools/io.py` | Structured layout metrics and warning summaries |
| `evaluation/` | Reproducible fixture builders, runner, records, and report |
| `tests/` | Unit, integration, MCP, round-trip, benchmark, and performance gates |
| `docs/evaluation/` | Stella screenshots, provenance, and visual findings |

The exact module split may be adjusted to respect the repository's file-size and
ownership conventions, but graph ordering, routing, quality analysis, and model
orchestration must remain independently testable.

## Failure Handling And Rollback

- If directed placement regresses a fixture, retain the benchmark and select the
  old force layout only behind an internal fallback warning while the defect is
  fixed. A fallback warning does not satisfy the planar-fixture release gates.
- If routed connector points are rejected or rewritten by Stella, stop the
  release and update the connector representation based on the Stella-saved
  artifact. Do not guess at undocumented point semantics.
- If the two-second budget fails, profile the analyzer and router before reducing
  quality checks. Hard overlap and obstacle checks may not be disabled for
  performance.
- The entire milestone can be reverted phase-by-phase because equations and
  simulation behavior remain untouched. Version 0.12.0 must not be tagged until
  the new layout path or an explicitly approved scope reduction passes all
  release gates.

## Definition Of Done

- All included pipeline stages are implemented and documented.
- All automated hard, page, compatibility, and performance gates pass.
- The non-planar fixture emits only its expected deterministic crossing warning.
- Stella Professional 4.1.1 evidence passes for all five desktop fixtures.
- README and MCP descriptions no longer say only that layout is "reasonable";
  they describe the guarantees and warning behavior accurately.
- `CHANGELOG.md` and `docs/releases/0.12.0.md` summarize the completed layout
  milestone without overstating arbitrary-graph crossing guarantees.
- The 0.12.0 package build contains the expected runtime modules and no benchmark
  artifacts or new mandatory dependency.
