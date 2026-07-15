# Architecture

Stella MCP is organized around a behavior-preserving core model, an MCP
protocol layer, and optional simulation capabilities. Public compatibility is
maintained through narrow facade modules while implementation responsibilities
live in focused modules.

## Dependency Boundaries

The package has one unconditional runtime dependency: `mcp>=1.19.0,<2`.
MCP 1.19.0 is the first tested SDK release whose low-level stdio server
correctly transports this package's direct `CallToolResult` responses with
structured content. The upper bound keeps the package on the v1 API until an
MCP v2 migration is tested.

| Layer | Modules | Dependency rule |
| --- | --- | --- |
| Model core | `model_types`, `model`, `model_layout`, `layout_graph`, `layout_router`, `layout_quality`, `layout_pipeline`, `xmile_parse`, `xmile_export`, `validator`, `render_svg` | Uses the Python standard library and project modules. It must not require simulation dependencies. |
| MCP | `server`, `mcp_resources`, `tool_schemas`, `tool_handlers`, `tools/*`, `session_store` | Depends on `mcp` and the model core. Importing the server must work with only core dependencies installed. |
| Simulation | `simulate`, `analysis`, `calibrate` | Uses the optional `sim` extra: PySD, NumPy, pandas, and SciPy. Optional packages are imported only on simulation call paths. |

The `sim` extra is an additive capability. Model creation, XMILE import/export,
validation, templates, and SVG rendering remain available without it.

## MCP Request Flow

`server.py` owns protocol registration and converts exceptions into stable tool
errors. Tool definitions and implementations are composed through two public
facades:

1. `tool_schemas.build_tool_definitions()` concatenates each domain's
   `build_tools()` result in the established order and applies annotations.
2. `tool_handlers.register_tool_handlers()` creates a `HandlerContext`
   containing model and session operations, then delegates registration to each
   domain's `register_handlers()` function.
3. `server.call_tool()` dispatches through the registered handler.

The files under `stella_mcp/tools/` own both the schema and handler for their
domain:

| Domain | Responsibility |
| --- | --- |
| `build.py` | Model creation, batch construction, variables, flows, connectors, and simulation specifications. |
| `io.py` | Model import/export, templates, and diagram rendering. |
| `inspect.py` | Model listing, validation, XML preview, and deletion. |
| `modules.py` | Module lifecycle, membership, and module-box layout controls. |
| `simulation.py` | Simulation, scenario comparison, sensitivity analysis, and calibration. |
| `shared.py` | Shared schema fragments, handler protocols, response types, and batch helpers. |

New tools belong in the closest existing domain. A new domain should expose the
same `build_tools()` and `register_handlers()` pair and be composed through the
two facades without changing existing tool order.

## Model And XMILE Ownership

| Module | Responsibility |
| --- | --- |
| `model_types.py` | Stock, flow, auxiliary, connector, module, graphical-function, and simulation-specification records plus XMILE constants. |
| `model.py` | `StellaModel` construction, naming, variable lifecycle, connectors, modules, simulation specifications, and compatibility metadata. |
| `model_layout.py` | Stock sizing and retained compatibility delegates for older layout helper APIs. |
| `layout_graph.py` | Directed stock graph, strongly connected components, ranks, ordering, and component packing. |
| `layout_router.py` | Boundary-port allocation, route normalization, candidate routing, and visibility-graph fallback. |
| `layout_quality.py` | Shared glyph and label geometry, intersection analysis, metrics, warning types, and structured reports. |
| `layout_pipeline.py` | Staged stock, flow, auxiliary, connector, label, and page-grid orchestration. |
| `xmile_parse.py` | File parsing, namespace handling, numeric parsing, compatibility warnings, and retention of unknown supported content. |
| `xmile_export.py` | XML serialization, namespace declarations, preserved fragments, view styles, point lists, and graphical functions. |
| `xmile.py` | Public facade for `StellaModel`, model records, constants, and `parse_stmx`. |
| `xmile_io.py` | Compatibility facade for `model_to_xml`, `parse_stmx_file`, and `gf_eqn_text`. |

`StellaModel` keeps its established layout and XML methods as delegates to the
layout pipeline and `xmile_export`. Callers should continue importing public
model types from `stella_mcp.xmile`; implementation code may import from the
owner modules directly when that clarifies the dependency.

Parser changes must preserve strict and permissive warning behavior. Exporter
changes must preserve unknown attributes and elements in supported sections.
Internal stock geometry is center-based. XMILE parsing and export convert the
Stella convention in which stock `x`/`y` is upper-left when the corresponding
dimension is explicit and center-based when it is omitted. Generated connector
routes are logical polylines internally and in SVG; XMILE export inserts Bezier
anchors so Stella's rendered curve follows those analyzed segments.
Imported stock, flow, and auxiliary view-font sizes are retained as typed point
values and drive the same label-box estimator used by layout analysis and SVG.
The compatibility corpus, focused parser/exporter tests, and exact built-in
template exports guard these contracts.

## Session State

`SessionStore` owns every session's model registry and current-model pointer.
Handlers receive narrow operations through `HandlerContext`; they do not read
or mutate registry dictionaries directly.

The stdio transport uses `id(session)` as the live session key. Tests and calls
without an MCP session use the fallback key `-1`. Any future HTTP transport
must provide a stable transport identity or call `SessionStore.clear(key)` at
session teardown. Retaining an object-identity key after the session object is
released is unsafe because Python may reuse that identity.

Adding a transport therefore requires explicit lifecycle wiring and isolation
tests. It must not infer cleanup from model deletion because one session may
own multiple models.

## Compatibility Contracts

Version 0.12 retains these public surfaces from 0.11 except for the documented
layout-result additions and auto-position recomputation behavior:

- tool names, order, input schemas, annotations, text results, and structured
  results;
- imports from `stella_mcp.xmile`, `stella_mcp.xmile_io`,
  `stella_mcp.tool_schemas`, and `stella_mcp.tool_handlers`;
- strict and permissive XMILE parsing behavior;
- generated XMILE and model snapshots; authored layout geometry remains fixed,
  while automatic layout and SVG routing use the new deterministic pipeline;
- simulation, analysis, and calibration behavior when the `sim` extra is
  installed;
- stable structured error codes and categories.

Tests should be added at the owning module first. Facade-level contract tests
remain necessary where ordering, import compatibility, or exact output is part
of the public behavior.

## Release Contract

Each release synchronizes the version in `pyproject.toml`, `uv.lock`,
`stella_mcp/__init__.py`, `CITATION.cff`, and the dated `CHANGELOG.md` heading.
The citation and changelog dates must match. A matching file under
`docs/releases/` supplies the GitHub release notes.

Before merge, CI and local verification cover:

- a locked core environment, lint, compatibility corpus, and full core tests;
- the complete optional simulation test suite;
- the supported MCP dependency floor;
- source and wheel builds, distribution validation, bundled templates, and a
  clean core-wheel import.

A release is created only from a merge commit on `main` after main-branch CI
passes. Publishing is triggered by the GitHub release event. The publish
workflow checks out the exact tag, validates it against package metadata,
builds and validates both distributions, and uploads through PyPI Trusted
Publishing.
