# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] - 2026-06-11

### Added

- `compare_scenarios` tool: run several named what-if scenarios (each a set
  of constant parameter overrides) against a baseline and report how each
  diverges — per-variable final/max absolute deltas and final percent change.
  Override names are validated up front so a typo fails fast and atomically;
  a per-scenario blow-up is reported as a warning without aborting the others.
  Requires the optional `sim` extra.
- `sensitivity_analysis` tool: sweep parameters one at a time across a range
  (holding the others at their baseline) and report how a chosen output
  metric (`final`, `max`, `min`, `mean`, or `time_to_threshold`) responds,
  with a range slope and a baseline-normalized elasticity for ranking
  parameters by influence. A `max_runs` cap errors rather than truncating an
  oversized sweep, non-finite inputs are rejected, and undefined metrics are
  reported as warnings. Requires the optional `sim` extra.

### Changed

- The PySD model is now compiled once per call and reused across a scenario
  comparison or sensitivity sweep, so a multi-run analysis no longer
  recompiles the XMILE for every run.

## [0.8.0] - 2026-06-11

### Added

- `render_diagram` tool: render the model as an SVG stock-and-flow diagram
  (stocks as rectangles, auxiliaries as circles, flows as valved pipes with
  source/sink clouds, dependency connectors as arcs), returned inline and
  optionally written to a file. Pure stdlib — no rasterization dependency.
  Runs auto-layout first by default.
- Units validation (warning-tier, conservative): `units_missing` flags a
  stock or flow with no units while others define them; `units_inconsistent`
  flags a flow whose units don't read as stock-units-per-time-unit when every
  attached stock shares the same units. Both stay silent when uncertain.
- `unused_variable` validation warning for an auxiliary referenced by no
  equation or connector.
- MCP tool annotations (`readOnlyHint`/`destructiveHint`/`idempotentHint`) on
  every tool.
- MCP resources: `stella://templates/{name}` and `stella://models/{model_id}`.
- MCP prompt `build-stella-model` encoding the recommended modeling workflow.

### Changed

- Auto-layout produces much tighter diagrams. The force-directed engine's
  ideal edge length is now a fixed, readable distance instead of scaling with
  the canvas area (which made small models sprawl across the whole canvas).
  Overlap prevention is now size-aware (larger stocks are kept proportionally
  farther apart) and runs after the canvas-fit rescale, so a downscale can no
  longer compress elements back into overlap.
- `get_model_xml` is now read-only: it exports from a copy, so previewing XML
  no longer rewrites the model's layout state.
- Minimum `mcp` dependency raised to `>=1.7.0` (introduces `ToolAnnotations`);
  a CI job exercises the suite pinned at that floor.

## [0.7.0] - 2026-06-10

### Added

- `simulate` tool: run the model through [PySD](https://pysd.readthedocs.io/)
  and return downsampled time series with per-variable summaries, parameter
  overrides, variable selection, and optional CSV export. PySD is an
  optional extra: `pip install 'stella-mcp[sim]'`. Integration is Euler
  regardless of the model's method setting (a warning is included for
  RK4 models).
- Batch construction tools: `build_model` (create and populate a model in
  one call) and `add_variables` (batch-extend an existing model). Both are
  all-or-nothing; item failures report the failing stage, index, and item
  name in the structured error.
- `delete_model` tool to remove a model from the session.
- Quoted identifiers in equations (`"net growth rate" * Population`) are
  now recognized as variable references by validation and connector sync;
  quoted spans matching no variable produce an
  `unresolved_quoted_reference` warning instead of an undefined-variable
  error.
- `CHANGELOG.md` and a uvx-based client configuration example.

### Fixed

- Reserved-token list now covers the full XMILE v1.0 builtin set and the
  isee Stella extensions, so functions like `SINWAVE`, `ARCTAN`, or
  `CLOCKTIME` are no longer misread as undefined variables (which also
  caused `sync_connectors_from_equations` to fabricate connectors).
- Graphical-function point lists are exported comma-separated per the
  XMILE spec (previously space-separated, which broke downstream XMILE
  readers such as PySD). Import now accepts comma-separated values, the
  `sep` attribute, and the legacy space-separated form — real Stella
  files with comma-separated `ypts` previously lost their graphical
  functions silently on import.
- Graphical-function equations written as `GRAPH(input)` (the documented
  tool-input convention) are exported in spec form — only the input
  expression in `<eqn>` — since both Stella and PySD reject the
  `GRAPH()` wrapper. The in-memory equation and tool input are
  unchanged; only the XMILE output differs.

### Changed

- CI installs from the committed `uv.lock` (`uv sync --locked`) and fails
  when the lockfile drifts.

## [0.6.0] - 2026-05-22

### Added

- `inspect_model` tool returning a complete structured model summary
  (variables, connectors, modules, sim specs, counts, validation).
- Explicit update tools: `set_sim_specs`, `update_stock`, `update_flow`,
  `update_aux` — change individual fields while preserving relationships.
- `sync_connectors_from_equations` tool that adds missing dependency
  connectors inferred from flow and auxiliary equations.
- Structured `structuredContent` payloads on inspection tools
  (`list_models`, `list_templates`, `get_template_info`, `load_template`,
  `save_as_template`, `list_modules`, `list_connectors`, `list_variables`,
  `validate_model`).
- "Recommended Agent Workflow" documentation in the README.

### Changed

- Lint (ruff) enforced in CI alongside the test suite.

## [0.5.0] - 2026-05-13

### Added

- Built-in model templates (exponential growth, SIR, Lotka-Volterra,
  2-box carbon cycle, 2-box ocean nutrient) with `list_templates`,
  `get_template_info`, `load_template`, and `save_as_template` tools.
- Module/group tools: `create_module`, `add_to_module`,
  `remove_from_module`, `rename_module`, `delete_module`,
  `set_module_view`, `set_module_style`, `auto_place_module_boxes`.
- Variable lifecycle tools: `rename_variable` (updates references in
  equations, connectors, and modules) and `delete_variable` (with
  consistency checks).
- Multi-model sessions: tools accept an optional `model_id`;
  `list_models` shows session state.
- XMILE compatibility modes (`permissive`/`strict`) on `read_model`,
  `save_model`, and `get_model_xml`, with round-trip preservation of
  unknown attributes/elements and a compatibility regression corpus
  (`tests/fixtures/compat_corpus/`) run in CI.
- Graphical function support on flows and auxiliaries.
- Structured tool errors with stable `error.code`/`error.category`.

### Changed

- Server split into focused modules (`tool_schemas.py`,
  `tool_handlers.py`, `xmile_io.py`, `equation_parser.py`,
  `templates.py`).

[Unreleased]: https://github.com/bradleylab/stella-mcp/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/bradleylab/stella-mcp/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/bradleylab/stella-mcp/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/bradleylab/stella-mcp/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/bradleylab/stella-mcp/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/bradleylab/stella-mcp/compare/v0.4.0...v0.5.0
