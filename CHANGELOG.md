# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/bradleylab/stella-mcp/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/bradleylab/stella-mcp/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/bradleylab/stella-mcp/compare/v0.4.0...v0.5.0
