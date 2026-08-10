# Architecture

Stella MCP is organized around a behavior-preserving core model, an MCP
protocol layer, and optional simulation capabilities. Public compatibility is
maintained through narrow facade modules while implementation responsibilities
live in focused modules.

## Dependency Boundaries

The package has two unconditional runtime dependencies: `mcp>=2.0.0,<3` and
`jsonschema>=4.20.0`. MCP 2.0.0 is the first stable SDK v2 release and supplies
the dual-era server for MCP 2026-07-28 and supported legacy clients. JSON Schema
validation is declared directly because Stella validates every successful
structured tool result before returning it.

| Layer | Modules | Dependency rule |
| --- | --- | --- |
| Model core | `model_types`, `model`, `model_layout`, `layout_graph`, `layout_router`, `layout_quality`, `layout_pipeline`, `xmile_features`, `xmile_parse`, `xmile_export`, `validator`, `render_svg` | Uses the Python standard library and project modules. It must not require simulation dependencies. |
| MCP | `server`, `mcp_resources`, `tool_schemas`, `tool_handlers`, `tools/*`, `session_store` | Depends on `mcp`, `jsonschema`, and the model core. Importing the server must work with only core dependencies installed. |
| Simulation | `simulate`, `analysis`, `calibrate` | Uses the optional `sim` extra: PySD, NumPy, pandas, and SciPy. Optional packages are imported only on simulation call paths. |

The `sim` extra is an additive capability. Model creation, XMILE import/export,
validation, templates, and SVG rendering remain available without it.

## MCP Request Flow

`server.py` owns protocol registration and converts exceptions into stable tool
errors. Tool definitions and implementations are composed through two public
facades:

1. `tool_schemas.build_tool_definitions()` concatenates each domain's
   `build_tools()` result in the established order, appends workspace lifecycle
   tools, and applies annotations, workspace routing, and output contracts.
2. `tool_handlers.register_tool_handlers()` creates a `HandlerContext`
   containing model and workspace operations, then delegates registration to each
   domain's `register_handlers()` function.
3. The SDK v2 low-level server invokes explicit `async (ctx, params) -> result`
   handlers. `server.call_tool()` resolves the application workspace, acquires
   its lock, dispatches through the registry, and validates successful
   structured output against JSON Schema 2020-12.

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
| `xmile_features.py` | Pre-conversion feature classification and typed errors for supported and preserved-only XMILE constructs. |
| `xmile_parse.py` | File parsing, namespace handling, numeric parsing, compatibility warnings, and retention of unknown supported content. |
| `xmile_export.py` | XML serialization, namespace declarations, preserved fragments, view styles, point lists, and graphical functions. |
| `xmile.py` | Public facade for `StellaModel`, model records, constants, and `parse_stmx`. |
| `xmile_io.py` | Compatibility facade for `model_to_xml`, `parse_stmx_file`, and `gf_eqn_text`. |

`StellaModel` keeps its established layout and XML methods as delegates to the
layout pipeline and `xmile_export`. Callers should continue importing public
model types from `stella_mcp.xmile`; implementation code may import from the
owner modules directly when that clarifies the dependency.

Parser changes must preserve strict and permissive warning behavior. In 0.13,
strict mode rejects arrays, compositional module instances, additional top-level
models, and confirmed reserved identifiers. Permissive mode records deterministic
findings and retains the selected unsupported fragments at their structural level;
those constructs remain preserved-only and are blocked at the shared PySD compile
boundary. Exporter changes must preserve unknown attributes and elements in
supported sections.
Internal stock geometry is center-based. XMILE parsing and export convert the
Stella convention in which stock `x`/`y` is upper-left when the corresponding
dimension is explicit and center-based when it is omitted. Generated connector
routes are logical polylines internally and in SVG; XMILE export inserts Bezier
anchors so Stella's rendered curve follows those analyzed segments.
Imported stock, flow, and auxiliary view-font sizes are retained as typed point
values and drive the same label-box estimator used by layout analysis and SVG.
The compatibility corpus, focused parser/exporter tests, and exact built-in
template exports guard these contracts.

## Application Workspace State

`WorkspaceStore` owns isolated model registries, current-model pointers,
optional caller-selected expiry, bounded expiry/revocation tombstones, and one
`asyncio.Lock` per workspace. Handlers receive narrow operations through
`HandlerContext`; they do not read or mutate registry dictionaries directly.

MCP 2026-07-28 removes protocol sessions, so production state never depends on
`server.request_context`, a transport object, or `id(session)`. Modern clients
create an opaque workspace handle and send `workspace_id` with every stateful
tool call; modern discovery marks the field required for those tools. Supported
legacy stdio discovery keeps it optional, and calls that omit it resolve to the
reserved process-local `legacy` workspace. Unknown, expired, and revoked IDs
produce distinct classified tool errors and never fall back to shared state.

Workspace IDs are routing identifiers, not authorization secrets. This release
ships stdio only. A future multi-user HTTP transport must separately bind
workspace ownership to an authenticated principal and reject cross-principal
replay; credentials must not appear in resource URIs, results, logs, or errors.

Modern model resources encode both identifiers as
`stella://workspaces/{workspace_id}/models/{model_id}` and resolve through the
store without ambient request state. Modern resource listing returns immutable
templates because list requests carry no workspace argument; legacy stdio also
lists models from its compatibility workspace. Resource reads use a
non-mutating model lookup, so reading a named resource cannot change the
workspace's current-model pointer.

## Structured Result Contracts

All 44 tools retain human-readable `content` and declare a JSON Schema 2020-12
`outputSchema` for successful `structuredContent`. Contracts require stable
top-level fields and types while nested Stella snapshot records remain open to
additive fields. The server validates success results before returning them,
and SDK v2 clients validate them again. Classified `is_error` results use the
separate stable error envelope and are not validated against success schemas,
matching the verified SDK v2 client behavior.

## Compatibility Contracts

Version 0.14 retains these public surfaces from 0.13 except for the documented
compatibility failures, SIR identifier migration, and additive evidence/backend
metadata:

- the existing 42 tool names, order, input behavior, annotations, text results,
  and structured field meanings; workspace lifecycle tools are appended;
- imports from `stella_mcp.xmile`, `stella_mcp.xmile_io`,
  `stella_mcp.tool_schemas`, and `stella_mcp.tool_handlers`;
- strict and permissive XMILE modes, with new early rejection for documented
  preserved-only constructs and reserved identifiers;
- generated XMILE and model snapshots; authored layout geometry remains fixed,
  while automatic layout and SVG routing use the new deterministic pipeline;
- simulation, analysis, and calibration behavior when the `sim` extra is
  installed, plus additive PySD version, actual/declared method, feature-preflight,
  and warning metadata;
- stable structured error codes and categories;
- legacy stdio model resource URIs inside the compatibility workspace.

Code Mode, `StellaAPI`, and any server-side code executor are outside this
release and do not appear in the tool catalog.

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
