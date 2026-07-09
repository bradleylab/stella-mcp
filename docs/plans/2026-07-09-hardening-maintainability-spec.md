# Stella MCP 0.10 Hardening and 0.11 Maintainability Specification

Status: APPROVED FOR EXECUTION

Date: 2026-07-09

Current branch: feat/0.10.0-calibrate

Current release: 0.9.0

Current pull request: #7, feat: add calibrate parameter-fitting tool (0.10.0)

## Goal

Ship a scientifically honest and reliably tested 0.10.0 calibration release,
then make 0.11.0 a behavior-preserving maintenance release that reduces the
cost and risk of extending the MCP surface and XMILE model implementation.

The work is divided into two release tracks:

1. Track A hardens the existing 0.10.0 pull request and releases it.
2. Track B reorganizes the codebase in 0.11.0 without changing public tool
   behavior or XMILE semantics.

Each track is implemented on feature branches. No direct commits to main.
Every proposed commit message must be shown to the operator and approved before
the commit is created.

## Evidence and Current Baseline

The following baseline was measured from feat/0.10.0-calibrate on 2026-07-09:

- Ruff passes.
- A core-only environment runs 222 tests with 3 optional modules skipped.
- An environment with the sim extra runs 280 tests.
- The optional simulation modules contain 58 tests:
  - tests/test_simulate.py: 8 tests
  - tests/test_analysis.py: 25 tests
  - tests/test_calibrate.py: 25 tests
- The CI test-sim job currently executes only tests/test_simulate.py.
- uv lock --check passes.
- The 0.10.0 wheel and source distribution build successfully.
- The MCP surface contains 42 tool schemas and 42 registered handlers.
- Package and module versions are 0.10.0, while CITATION.cff is 0.6.0.
- README requirements say mcp>=1.0.0 while pyproject.toml requires mcp>=1.7.0.
- The calibration schema and docstring say the differential-evolution maxiter
  default is derived from max_nfev, while the implementation uses a fixed
  default of 100.
- The calibration objective is formed from residual multipliers:
  weight * (simulated - observed), but the aggregate root mean square is
  currently returned under the unqualified name rmse.

## Locked Decisions

### 1. Calibration objective reporting

The optimizer continues to minimize weighted sum of squared residuals:

    residual = weight * (simulated - observed)
    weighted_sse = sum(residual ** 2)

The input field remains named weights for compatibility with the existing
unreleased branch. Its documentation must state that each value is a residual
multiplier. A value equal to inverse measurement standard deviation gives the
usual normalized-residual interpretation.

The 0.10.0 structured result replaces the ambiguous aggregate rmse field with:

    "objective": {
      "metric": "weighted_sse",
      "initial": <weighted SSE at the initial parameters>,
      "final": <weighted SSE at the fitted parameters>,
      "weighted_rmse": <sqrt(final weighted SSE / residual count)>
    }

The result also adds:

    "target_metrics": [
      {
        "name": <display name>,
        "units": <exact model units string, including an empty string>,
        "n": <number of observations>,
        "sse": <unweighted final SSE in the target's native squared units>,
        "rmse": <unweighted final RMSE in the target's native units>
      }
    ]

No aggregate physical-unit RMSE is reported across multiple targets. The
weighted_rmse field is explicitly described as a weighted-residual diagnostic,
not a native-unit error metric.

### 2. Optimizer result contract

The result exposes one canonical optimizer object:

    "optimizer": {
      "method": "least_squares" | "differential_evolution",
      "converged": <boolean>,
      "status": <integer or boolean-compatible backend status>,
      "message": <backend termination message>,
      "n_function_evals": <integer>,
      "config": {
        "max_nfev": <integer or null>,
        "maxiter": <effective integer or null>,
        "popsize": <integer or null>,
        "seed": <integer or null>
      }
    }

The old top-level method, converged, and n_function_evals fields are removed
before release because 0.10.0 has not yet been published. The MCP text summary
and all tests are updated to use optimizer.

least_squares config:

- max_nfev is populated.
- maxiter, popsize, and seed are null.

differential_evolution config:

- maxiter is the effective generation cap.
- popsize and seed are populated.
- max_nfev is null.

The fixed default for differential-evolution maxiter is 100. The schema,
implementation, README, and result config must agree on this value.

### 3. Optimizer input validation

Validation occurs in calibrate before scipy is called.

- max_nfev must be an integer greater than or equal to 1 and must not be bool.
- maxiter may be null; otherwise it must be an integer greater than or equal
  to 1 and must not be bool.
- popsize must be an integer greater than or equal to 1 and must not be bool.
- seed must be an integer and must not be bool or null.
- Invalid values raise ValueError so the MCP layer returns invalid_input.
- Method-irrelevant settings are accepted but omitted from the effective
  optimizer config; no warning is emitted because handler defaults make it
  impossible to distinguish an omitted value from an explicit default.

### 4. Optional dependency ownership

Every distribution directly imported by stella_mcp is declared directly.
The sim extra becomes:

    sim = [
      "pysd>=3.14",
      "numpy>=1.23",
      "pandas",
      "scipy>=1.10",
    ]

The NumPy floor matches the installed PySD 3.14.3 distribution metadata.
PySD declares pandas without a floor, so this package does the same rather than
inventing a minimum.

All NumPy, pandas, scipy, and PySD imports remain inside functions so importing
stella_mcp.server with only core dependencies remains supported.

### 5. MCP compatibility

The mcp minimum remains 1.7.0 for 0.10.0 and 0.11.0 unless a separate operator
decision raises it.

Tool outputSchema is not added in this goal. mcp 1.7.0 Tool models do not
support outputSchema. Adding it requires a deliberate minimum-version increase
and client-compatibility review.

### 6. Release metadata

The following values must agree before a release:

- pyproject.toml project version
- stella_mcp.__version__
- CITATION.cff version
- CHANGELOG version heading
- Git release tag without its leading v

The CITATION.cff date-released and CHANGELOG version date must agree. The date
is set in the final release-preparation commit, not guessed earlier.

### 7. Refactor posture

Track B is behavior-preserving. It must not:

- change tool names, input schemas, annotations, text summaries, or structured
  result shapes;
- change model naming, equation parsing, layout, validation, or XMILE output;
- change import paths used by existing users;
- add outputSchema or raise the MCP minimum;
- add new analysis modes.

Public imports from stella_mcp.xmile remain valid through compatibility
re-exports.

## Scope

### In scope for Track A: 0.10.0

- Complete optional-stack CI coverage.
- Calibration metric and optimizer result corrections.
- Optimizer argument validation.
- Direct optional dependency declarations.
- Schema-handler and release-metadata contract tests.
- Package build and clean-wheel checks in CI.
- Publish-time tag/version validation.
- README, CHANGELOG, CITATION, and plan synchronization.
- Merge and release only after all gates pass and the operator approves.

### In scope for Track B: 0.11.0

- Split tool schemas and handlers by domain.
- Add a catalog contract that prevents schema-handler-annotation drift.
- Split StellaModel layout responsibilities from core model operations.
- Split XMILE parsing and exporting while retaining facades.
- Update architecture documentation.
- Preserve all behavior with existing tests and contract snapshots.

### Out of scope

- Grid, Monte Carlo, Sobol, Bayesian, or MCMC analysis.
- Fitting stock initial conditions.
- New optimization methods or objective functions.
- HTTP/SSE transport.
- MCP outputSchema and an MCP minimum-version increase.
- Real-Stella compatibility fixtures that have not been supplied by the
  operator.
- Visual redesign or new rendering formats.

## Track A: 0.10.0 Hardening

### Task A1: Make the optional stack a real CI gate

Files:

- Modify .github/workflows/ci.yml

Changes:

1. Keep the Python 3.10, 3.11, and 3.12 core matrix using dev only. This
   verifies that the server and core tests work without the sim extra.
2. Change test-sim to install dev and sim and run the complete test suite:

       uv sync --locked --extra dev --extra sim
       uv run python -m pytest

3. Keep test-mcp-floor at mcp==1.7.0.
4. Rename the test-sim step to make its coverage explicit.

Acceptance:

- Core matrix reports the optional modules skipped and otherwise passes.
- test-sim runs all 280 or more collected tests with no skipped sim modules.
- A deliberately failing calibration test fails test-sim.

Proposed commit:

    test: run the complete simulation stack in CI

### Task A2: Add tool and release contract tests

Files:

- Modify tests/test_mcp_surface.py
- Create tests/test_release_metadata.py
- Modify pyproject.toml
- Modify uv.lock

Changes:

1. Add test_tool_schemas_match_registered_handlers:

       schema_names = {tool.name for tool in build_tool_definitions()}
       handler_names = set(server_mod._TOOL_HANDLERS)
       assert schema_names == handler_names

2. Add PyYAML>=6 to the dev extra so CITATION.cff is parsed as YAML instead of
   ad hoc text.
3. Add release metadata tests that:
   - read the installed/editable distribution version using
     importlib.metadata.version("stella-mcp");
   - compare it with stella_mcp.__version__;
   - parse CITATION.cff with yaml.safe_load and compare its version;
   - locate the exact CHANGELOG heading for that version;
   - compare CITATION date-released with the CHANGELOG heading date.
4. Keep tag checking in a reusable script for publish time:
   scripts/check_release_metadata.py accepts --expected-tag and validates that
   vX.Y.Z matches the same metadata version.
5. Unit-test the script's success and mismatch cases without network access.

Acceptance:

- Removing a handler or schema causes a contract-test failure.
- Changing any one version source causes a release-metadata failure.
- A mismatched tag exits nonzero with a message naming both values.
- Tests pass on Python 3.10 through 3.12.

Proposed commit:

    test: enforce MCP and release metadata contracts

### Task A3: Correct calibration objective semantics

Files:

- Modify stella_mcp/calibrate.py
- Modify stella_mcp/tool_handlers.py
- Modify tests/test_calibrate.py
- Modify README.md
- Modify CHANGELOG.md

Implementation:

1. Add _variable_units(model, key) returning the exact units field from the
   matching stock, flow, or auxiliary. Return an empty string when the model
   stores an empty string.
2. Add _target_fit_metrics(fit_results, obs) that:
   - interpolates each fitted target onto observation times;
   - computes unweighted residuals;
   - returns n, SSE, and RMSE per target;
   - raises a clear ValueError if a target column is absent or final fitted
     values are non-finite.
3. Rename objective.metric from sse to weighted_sse.
4. Rename objective.rmse to objective.weighted_rmse.
5. Add target_metrics using the locked result contract.
6. Update the MCP text summary to name weighted RMSE and include no ambiguous
   bare RMSE label.
7. Document that weights are residual multipliers and that inverse-sigma
   values give a normalized-residual interpretation.

Tests:

- Single-target unweighted fit: target RMSE equals weighted RMSE.
- Single-target weighted fit: target RMSE remains in native units while
  weighted_rmse changes by the residual multiplier.
- Two targets with different units: each target metric carries the exact
  model units and no aggregate native-unit RMSE is present.
- objective.initial and objective.final remain weighted SSE.
- Non-finite final fitted output produces a controlled invalid-input/internal
  contract chosen from the existing error policy, never invalid JSON.
- Structured MCP output and text summary use the new fields.

Acceptance:

- No structured result field named objective.rmse remains.
- Every target receives an unweighted native-unit metric.
- Existing parameter recovery, bounds, covariance, and CSV tests remain green.

Proposed commit:

    fix: report calibration errors with explicit units and weighting

### Task A4: Validate optimizer controls and expose execution details

Files:

- Modify stella_mcp/calibrate.py
- Modify stella_mcp/tool_schemas.py
- Modify stella_mcp/tool_handlers.py
- Modify tests/test_calibrate.py
- Modify README.md

Implementation:

1. Add _positive_int(value, label) and _seed_int(value).
2. Validate max_nfev, maxiter, popsize, and seed before observation loading or
   model compilation.
3. Set maxiter schema default and documentation to 100.
4. Replace the old top-level method, converged, and n_function_evals fields
   with the locked optimizer object.
5. Capture backend status and message:
   - least_squares: integer result.status and string result.message;
   - differential_evolution: use a stable status representation derived from
     result.success and string result.message.
6. Include only method-relevant effective config values.
7. Update the text handler and README.

Tests:

- Zero, negative, bool, string, list, and null-invalid controls return
  ValueError or MCP invalid_input as specified.
- Default DE output reports maxiter 100, popsize 15, and seed 0.
- Explicit DE controls are echoed exactly.
- least_squares output reports max_nfev and null DE controls.
- Backend status/message and n_function_evals are present.
- Existing convergence and budget-exhaustion tests use optimizer.converged.

Acceptance:

- Schema, docstring, README, implementation, and result agree on defaults.
- No malformed optimizer control reaches scipy as an internal error.

Proposed commit:

    fix: validate and report calibration optimizer controls

### Task A5: Own direct optional dependencies

Files:

- Modify pyproject.toml
- Modify uv.lock
- Modify tests/test_release_metadata.py

Changes:

1. Add numpy>=1.23 and pandas to the sim extra.
2. Retain pysd>=3.14 and scipy>=1.10.
3. Add a metadata test asserting the sim extra contains pysd, numpy, pandas,
   and scipy.
4. Add an import-without-sim regression test in the core environment.

Acceptance:

- uv lock --check passes.
- Core wheel metadata has only mcp as an unconditional runtime dependency.
- sim extra metadata explicitly includes all four direct simulation imports.
- stella_mcp.server imports in a clean core-only environment.

Proposed commit:

    build: declare direct simulation dependencies

### Task A6: Synchronize documentation and release metadata

Files:

- Modify CITATION.cff
- Modify CHANGELOG.md
- Modify README.md
- Modify docs/plans/2026-06-12-feat-0.10.0-calibrate-plan.md
- Add this specification to git

Changes:

1. Set CITATION version to 0.10.0.
2. At the release-preparation checkpoint, set CITATION date-released and the
   CHANGELOG 0.10.0 date to the actual release date.
3. Change README requirements to mcp>=1.7.0.
4. Update README Project Structure to include analysis.py, calibrate.py,
   simulate.py, render_svg.py, mcp_resources.py, and the planned facades.
5. Update Maintainer Release to require:
   - synchronized metadata;
   - uv lock --check;
   - core and sim CI;
   - package job;
   - a release-notes file;
   - main CI success before publishing.
6. Replace the obsolete v0.5.0 example with the current release pattern.
7. Mark the 0.10.0 calibration plan IMPLEMENTED / HARDENING and update its
   maxiter text and review status.
8. Do not modify HANDOFF.md without a separate preview and operator approval.

Acceptance:

- Release metadata tests pass.
- README no longer contains mcp>=1.0.0 or the stale project tree.
- Calibration docs use weighted_sse, weighted_rmse, and target_metrics.

Proposed commit:

    docs: synchronize 0.10.0 release documentation

### Task A7: Add package and publish gates

Files:

- Modify .github/workflows/ci.yml
- Modify .github/workflows/publish.yml
- Modify scripts/check_release_metadata.py

CI package job:

1. Build with uv build.
2. Run twine check against both artifacts.
3. Install the wheel into a clean environment without extras.
4. Import stella_mcp.server.
5. Assert built-wheel metadata and bundled templates through a script/test.

Publish workflow:

1. Resolve the release tag from the published release event.
2. Checkout that exact tag.
3. Run scripts/check_release_metadata.py --expected-tag with that tag.
4. Build and run twine check.
5. Publish only after all checks pass.
6. Keep Trusted Publishing and the pypi environment.
7. Replace unrestricted workflow_dispatch with a required tag input, or remove
   workflow_dispatch. Default decision: keep it only with a required tag input
   and run the same checkout and metadata checks as the release path.

Acceptance:

- A release tag not matching package metadata cannot reach the publish step.
- A malformed distribution cannot reach PyPI.
- Manual publication cannot implicitly publish the current branch.

Proposed commit:

    ci: gate package builds and PyPI publication

### Task A8: Final 0.10.0 verification and release

Required local checks:

    uv lock --check
    uv sync --locked --extra dev
    uv run --no-sync python -m pytest
    uv run --no-sync ruff check .
    uv sync --locked --extra dev --extra sim
    uv run --no-sync python -m pytest
    uv run --with "mcp==1.7.0" --extra dev python -m pytest
    python scripts/sync_compat_corpus_manifest.py --check
    uv build

Additional checks:

- Install the built wheel in a clean core-only environment and import server.
- Install the built wheel with sim and run one analytical simulation and one
  small calibration truth-recovery case.
- Review git diff --check.
- Confirm no secrets, generated data, or local environment files are staged.
- Review PR #7 after the final push.
- Wait for every CI job.
- Obtain operator approval before merging.
- Create v0.10.0 only from merged main.
- Confirm Trusted Publishing success and verify PyPI reports 0.10.0.

Track A completion criteria:

- PR #7 is merged.
- v0.10.0 and PyPI 0.10.0 point to the same source state.
- Core and sim installation paths are verified.
- The release metadata contract passes.

## Track B: 0.11.0 Maintainability

Track B begins from updated main after 0.10.0 is released. It uses separate
feature branches and pull requests so behavior-preserving refactors remain
reviewable.

### Task B1: Split the MCP tool catalog by domain

New package:

    stella_mcp/tools/
      __init__.py
      shared.py
      build.py
      io.py
      inspect.py
      modules.py
      simulation.py

Compatibility facades retained:

- stella_mcp/tool_schemas.py
- stella_mcp/tool_handlers.py

Domain ownership:

- build.py: model creation, batch build, variables, connectors.
- modules.py: module lifecycle and layout-box controls.
- io.py: read/save/render/templates.
- inspect.py: listing, validation, XML preview, model deletion.
- simulation.py: simulate, compare, sensitivity, calibrate.
- shared.py: shared schema fragments, ToolResponse, protocols, batch helpers.

Implementation:

1. Each domain exposes build_tools() and register_handlers(...).
2. tool_schemas.build_tool_definitions concatenates domain tool definitions in
   the existing order and applies annotations.
3. tool_handlers.register_tool_handlers delegates to each domain registrar.
4. Annotation policy remains centralized until outputSchema/MCP-floor work is
   separately approved.
5. Add a snapshot test for ordered tool names and schema equality.
6. Keep all current imports working.

Acceptance:

- Tool names, order, schemas, annotations, text, and structured results are
  unchanged from 0.10.0.
- Existing 280-or-more tests pass without edits except contract/snapshot tests.
- No domain tool file exceeds the project guideline without a documented
  reason.

Proposed commits:

    refactor: split MCP tool schemas by domain
    refactor: split MCP tool handlers by domain

### Task B2: Separate model state from layout operations

New modules:

    stella_mcp/model_types.py
    stella_mcp/model.py
    stella_mcp/model_layout.py

Compatibility facade:

    stella_mcp/xmile.py

Ownership:

- model_types.py: Stock, Flow, Aux, GraphicalFunction, Connector, Module,
  SimSpecs, namespace constants.
- model.py: StellaModel construction, naming, CRUD, variable lifecycle,
  connectors, modules, simulation specifications, compatibility metadata.
- model_layout.py: dependency graph, subsystem positioning, stock sizing,
  auto-layout, flow routing, connector angles, collision detection and
  resolution.
- xmile.py: re-export public types and parse_stmx; preserve existing imports.

Implementation posture:

- Layout helpers become functions receiving StellaModel, or a narrow internal
  helper object. Avoid a new inheritance hierarchy unless delegation proves
  materially more complex.
- StellaModel methods remain as compatibility delegates where callers use
  methods such as _auto_layout.
- Move no XML behavior in this task.

Acceptance:

- Existing imports from stella_mcp.xmile remain valid.
- XML round trips and rendered SVG snapshots are unchanged.
- Layout tests pass without tolerance widening.
- No scientific or analytical behavior changes.

Proposed commits:

    refactor: extract Stella model types and lifecycle
    refactor: isolate model layout operations

### Task B3: Split XMILE parsing and export

New modules:

    stella_mcp/xmile_export.py
    stella_mcp/xmile_parse.py

Compatibility facades:

- stella_mcp/xmile_io.py
- stella_mcp/xmile.py

Ownership:

- xmile_export.py: model_to_xml, namespace declarations, preserved fragments,
  view styles, point lists, graphical functions.
- xmile_parse.py: parse_stmx_file, namespace lookup, numeric parsing,
  compatibility warnings, preserved unknown content.
- xmile_io.py: re-export model_to_xml and parse_stmx_file.

Implementation:

1. Move remaining XML string helpers out of StellaModel.
2. Keep permissive and strict warning behavior identical.
3. Preserve exact output where tests assert text; otherwise preserve semantic
   parse-export-parse equality and extension retention.
4. Add focused parser/exporter unit tests rather than expanding monolithic
   compatibility tests.

Acceptance:

- Compatibility corpus and manifest check pass.
- Built-in templates parse, export, and parse again with equivalent snapshots.
- Unknown attributes/elements remain preserved.
- Strict-mode failures and permissive warnings are unchanged.

Proposed commits:

    refactor: separate XMILE export from the model
    refactor: separate XMILE parsing from export

### Task B4: Encapsulate session state

Files:

- Create stella_mcp/session_store.py
- Modify stella_mcp/server.py
- Modify stella_mcp/mcp_resources.py
- Modify tests/test_server_state_and_gf.py

Implementation:

1. Introduce SessionStore with get, set_current, delete, list, and clear
   operations.
2. Keep the current stdio behavior and test fallback key.
3. Centralize the id(session) assumption and document that HTTP transport
   requires lifecycle cleanup or a transport-provided identity.
4. Add tests for isolation, current-model transitions, deletion, and explicit
   cleanup.
5. Do not add HTTP transport in this task.

Acceptance:

- Existing session behavior and error codes are unchanged.
- Registry internals are no longer accessed outside server/session modules
  except by explicit test hooks.

Proposed commit:

    refactor: encapsulate session model state

### Task B5: Documentation and 0.11.0 release

Files:

- Modify README.md
- Modify CHANGELOG.md
- Modify CITATION.cff
- Modify pyproject.toml
- Modify stella_mcp/__init__.py
- Create docs/architecture.md

Architecture documentation includes:

- dependency boundaries between core, sim, and MCP layers;
- tool-domain ownership;
- model/layout/XMILE module ownership;
- session-state assumptions;
- release and compatibility contracts.

Release checks repeat Track A Task A8. No public behavior changes are claimed
unless an unavoidable difference is separately reviewed and documented.

Track B completion criteria:

- 0.11.0 is released from main.
- Public tool and XMILE behavior remain compatible with 0.10.0.
- Former monolith responsibilities have explicit module owners.
- Architecture and release contracts are documented and CI-enforced.

## Deferred Follow-Up Requiring Operator Input

Real-world compatibility expansion begins only when representative Stella files
are supplied or approved for inclusion. When available:

1. Store immutable source fixtures under tests/fixtures/compat_corpus.
2. Record provenance, Stella version, expected warnings, and redistribution
   permission in the manifest.
3. Add parse-export-parse semantic snapshots.
4. Add strict/permissive expectations.
5. Never edit raw fixture files to make tests pass.

## Goal Checkpoints

The execution goal pauses for operator approval at:

1. Every proposed commit message.
2. Any change to the locked scientific result contract.
3. Any new dependency floor not specified above.
4. Merge of PR #7.
5. Creation of v0.10.0 and PyPI publication.
6. Creation of each Track B branch and pull request.
7. Merge/release of 0.11.0.

The goal is complete only when Track A and Track B completion criteria are met,
or when the operator explicitly narrows the goal.
