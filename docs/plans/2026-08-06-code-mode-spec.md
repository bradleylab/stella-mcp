# Detailed Specification: Code Mode for `stella-mcp`

**Version:** 1.0<br>
**Target repository:** https://github.com/bradleylab/stella-mcp<br>
**Date:** 2026-08-06<br>
**Goal:** Convert the existing multi-tool MCP interface into a primary **code-mode** interface while preserving full backward compatibility. The LLM writes Python against a clean, high-level `stella` API instead of issuing many sequential tool calls.

---

## 1. Design Goals & Principles

1. **Primary interface becomes code** — For any multi-step or compositional work the model should prefer writing Python.
2. **Zero re-implementation of domain logic** — The new API is a thin, Pythonic façade over the existing `StellaModel`, `SessionStore`, validators, simulators, layout, etc.
3. **Full backward compatibility** — All existing individual tools remain available and continue to work exactly as before.
4. **Session-aware & multi-model safe** — The `stella` object is bound to the current MCP session and respects `model_id` scoping.
5. **Safe-by-default execution** — Restricted namespace, no arbitrary imports, no filesystem/network access beyond what the Stella domain already allows, clear timeouts and error reporting.
6. **Token-efficient & expressive** — The model can use loops, conditionals, intermediate variables, list comprehensions, etc., without round-tripping every intermediate result through the LLM context.
7. **Discoverable** — The API surface is documented in the tool description and via an optional `help()` / `dir(stella)` capability.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ MCP Client (Claude Desktop, Cursor, custom agent, etc.)     │
└────────────────────────────┬────────────────────────────────┘
                             │ tools/call
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ stella-mcp server                                           │
│                                                             │
│  Existing tools (create_model, add_stock, …)  ← unchanged   │
│                                                             │
│  NEW: code tool                                             │
│       │                                                     │
│       ▼                                                     │
│  RestrictedExecutor                                         │
│       │                                                     │
│       ▼                                                     │
│  StellaAPI (session-bound)  ──► SessionStore + StellaModel  │
│                                                             │
│  Domain modules (validator, simulate, layout, xmile, …)     │
└─────────────────────────────────────────────────────────────┘
```

- The `code` tool is the **preferred** path for complex work.
- Individual tools remain for simple one-shot operations and for clients that do not yet support code mode.

---

## 3. New Public API Surface (`stella_mcp/api.py`)

Create a single class `StellaAPI` (exposed to the model as the name `stella`).

```python
class StellaAPI:
    """High-level programmatic interface to Stella models.

    Bound to one MCP session. All methods that accept model_id=None
    operate on the current model of the session.
    """

    def __init__(self, session_store: SessionStore, session_key: int):
        ...

    # ── Model lifecycle ──────────────────────────────────────────────
    def create_model(
        self,
        name: str = "Untitled",
        model_id: str | None = None,
        *,
        start: float = 0.0,
        stop: float = 100.0,
        dt: float = 0.25,
        method: str = "Euler",
        **kwargs,
    ) -> str:
        """Create a new empty model and make it current. Returns model_id."""

    def build_model(
        self,
        name: str,
        model_id: str | None = None,
        *,
        sim_specs: dict | None = None,
        stocks: list[dict] | None = None,
        flows: list[dict] | None = None,
        auxs: list[dict] | None = None,
        modules: list[dict] | None = None,
        connectors: list[dict] | None = None,
        auto_sync_connectors: bool = True,
        auto_validate: bool = True,
    ) -> dict:
        """Atomically create and fully populate a model.
        Returns structured summary + validation report.
        """

    def read_model(
        self,
        path: str,
        model_id: str | None = None,
        *,
        compat_mode: str = "permissive",
    ) -> str:
        """Load a .stmx file into the session. Returns model_id."""

    def save_model(
        self,
        path: str,
        model_id: str | None = None,
        *,
        auto_layout: bool = True,
        resolve_layout_violations: bool = False,
        compat_mode: str = "permissive",
    ) -> dict:
        """Write the model to disk as XMILE. Returns layout metrics + warnings."""

    def delete_model(self, model_id: str) -> dict: ...
    def list_models(self) -> list[dict]: ...
    def set_current(self, model_id: str) -> None: ...

    # ── Variables ────────────────────────────────────────────────────
    def add_stock(
        self,
        name: str,
        initial,
        *,
        units: str = "",
        model_id: str | None = None,
        **kwargs,
    ) -> None: ...

    def add_flow(
        self,
        name: str,
        equation: str,
        *,
        from_stock: str | None = None,
        to_stock: str | None = None,
        units: str = "",
        graphical_function: dict | None = None,
        model_id: str | None = None,
        **kwargs,
    ) -> None: ...

    def add_aux(
        self,
        name: str,
        equation: str,
        *,
        units: str = "",
        graphical_function: dict | None = None,
        model_id: str | None = None,
        **kwargs,
    ) -> None: ...

    def update_stock(self, name: str, *, model_id: str | None = None, **fields) -> None: ...
    def update_flow(self, name: str, *, model_id: str | None = None, **fields) -> None: ...
    def update_aux(self, name: str, *, model_id: str | None = None, **fields) -> None: ...
    def rename_variable(self, old_name: str, new_name: str, *, model_id: str | None = None) -> None: ...
    def delete_variable(self, name: str, *, model_id: str | None = None) -> None: ...
    def list_variables(self, *, model_id: str | None = None, type: str | None = None) -> list[dict]: ...

    # ── Connectors & modules ─────────────────────────────────────────
    def add_connector(
        self,
        from_var: str,
        to_var: str,
        *,
        model_id: str | None = None,
        **kwargs,
    ) -> None: ...

    def sync_connectors_from_equations(self, *, model_id: str | None = None) -> dict: ...
    def create_module(self, name: str, variables: list[str] | None = None, *, model_id: str | None = None) -> None: ...
    def add_to_module(self, module: str, variables: list[str], *, model_id: str | None = None) -> None: ...
    def remove_from_module(self, module: str, variables: list[str], *, model_id: str | None = None) -> None: ...
    def rename_module(self, old_name: str, new_name: str, *, model_id: str | None = None) -> None: ...
    def delete_module(self, name: str, *, model_id: str | None = None) -> None: ...
    def set_module_view(self, name: str, *, model_id: str | None = None, **kwargs) -> None: ...
    def set_module_style(self, name: str, *, model_id: str | None = None, **kwargs) -> None: ...
    def auto_place_module_boxes(self, *, model_id: str | None = None) -> None: ...

    # ── Inspection & validation ──────────────────────────────────────
    def inspect(self, model_id: str | None = None) -> dict: ...
    def validate(self, model_id: str | None = None) -> dict:
        """Return structured validation report (errors + warnings)."""
    def get_xml(self, model_id: str | None = None, *, auto_layout: bool = True) -> str: ...
    def render_diagram(self, model_id: str | None = None, **kwargs) -> str:
        """Return SVG string of the stock-and-flow diagram."""

    # ── Simulation & analysis (require [sim] extra) ──────────────────
    def simulate(
        self,
        model_id: str | None = None,
        *,
        overrides: dict | None = None,
        **kwargs,
    ) -> dict: ...

    def compare_scenarios(
        self,
        scenarios: dict[str, dict],
        *,
        model_id: str | None = None,
        **kwargs,
    ) -> dict: ...

    def sensitivity_analysis(
        self,
        parameters: list[str],
        metric: str,
        *,
        model_id: str | None = None,
        **kwargs,
    ) -> dict: ...

    def calibrate(
        self,
        observed: dict | str,
        parameters: list[str],
        *,
        model_id: str | None = None,
        **kwargs,
    ) -> dict: ...

    # ── Templates ────────────────────────────────────────────────────
    def list_templates(self, **filters) -> list[dict]: ...
    def load_template(self, name: str, model_id: str | None = None) -> str: ...
    def save_as_template(
        self,
        name: str,
        *,
        description: str = "",
        tags: list[str] | None = None,
        model_id: str | None = None,
    ) -> None: ...

    # ── Utilities ────────────────────────────────────────────────────
    def help(self, topic: str | None = None) -> str:
        """Return documentation for the whole API or a specific method."""
```

### Implementation notes for the API class

- Every method that mutates state should obtain the model via `self._store.get(self._session_key, model_id)`.
- Methods should raise clear, structured exceptions that the executor can turn into MCP-friendly error objects.
- Prefer returning plain dicts / lists / strings that are JSON-serializable (or convert `StellaModel` summaries via the existing inspect helpers).
- Re-use the exact same validation, connector-sync, layout, and simulation code paths that the current tool handlers use.
- Do **not** reimplement domain logic — call into the existing modules (`validator.py`, `simulate.py`, `layout_pipeline.py`, `xmile_*.py`, etc.).

---

## 4. New MCP Tool: `code`

### Schema

```python
Tool(
    name="code",
    description=(
        "Execute Python code against the high-level Stella API. "
        "This is the preferred way to build, modify, validate, simulate, "
        "or analyze models when more than one simple operation is required.\n\n"
        "The object `stella` is already available in the namespace and is bound "
        "to the current session. You may also use `print()` and basic Python "
        "constructs (loops, conditionals, list comprehensions, etc.).\n\n"
        "Convention: assign any final value you want returned to the variable "
        "`result`. Everything printed to stdout is also returned.\n\n"
        "Example:\n"
        "```python\n"
        "mid = stella.create_model('SIR', start=0, stop=100, dt=0.125)\n"
        "stella.add_stock('Susceptible', 990)\n"
        "stella.add_stock('Infected', 10)\n"
        "stella.add_stock('Recovered', 0)\n"
        "stella.add_flow('Infection', 'beta * Susceptible * Infected / Total', "
        "from_stock='Susceptible', to_stock='Infected')\n"
        "...\n"
        "report = stella.validate()\n"
        "result = report\n"
        "```"
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source code to execute"
            },
            "timeout_seconds": {
                "type": "number",
                "default": 30,
                "description": "Maximum execution time"
            }
        },
        "required": ["code"]
    },
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False)
)
```

### Handler responsibilities

1. Create a `StellaAPI` instance bound to the current session.
2. Build a restricted globals/locals dict:

   ```python
   ns = {
       "stella": stella_api,
       "print": capture_print,          # redirects to StringIO
       "__builtins__": safe_builtins,   # limited set
   }
   ```

3. Execute with a timeout (use `signal`, `multiprocessing`, `concurrent.futures`, or a library such as `RestrictedPython` + timeout).
4. Capture:
   - stdout
   - the value of `result` (if present)
   - any exception + traceback
5. Return a structured MCP result:

   ```json
   {
     "stdout": "...",
     "result": { ... },
     "error": null
   }
   ```

   or on failure:

   ```json
   {
     "stdout": "...",
     "result": null,
     "error": {
       "type": "ValueError",
       "message": "...",
       "traceback": "..."
     }
   }
   ```

### Safety constraints (must implement)

- No `import`, `open`, `exec`, `eval`, `__import__`, `os`, `sys`, `subprocess`, network, etc. unless explicitly allow-listed.
- Prefer `RestrictedPython` or a carefully curated `__builtins__`.
- Enforce a wall-clock timeout (default 30 s, configurable via the tool argument).
- Memory / recursion limits are desirable but secondary for v1.

---

## 5. Integration Points (files to change / add)

| File | Change |
|------|--------|
| `stella_mcp/api.py` | **New** – `StellaAPI` class |
| `stella_mcp/code_executor.py` | **New** – restricted executor + timeout + capture |
| `stella_mcp/tools/code.py` | **New** – schema + handler for the `code` tool |
| `stella_mcp/tool_schemas.py` | Import and include the new tool (keep it near the top of the catalog) |
| `stella_mcp/tool_handlers.py` | Register the new handler |
| `stella_mcp/server.py` | Ensure the tool is wired (usually automatic via the existing registration) |
| `README.md` | Document Code Mode, show examples, update recommended workflow |
| `docs/code-mode.md` | **New** – full design & usage documentation |
| Tests under `tests/` or `evaluation/` | New unit + integration tests for the API and the `code` tool |

**Important:** The existing tool modules (`tools/build.py`, `tools/io.py`, `tools/modules.py`, `tools/simulation.py`, `tools/inspect.py`) should be refactored so that both the classic handlers *and* `StellaAPI` call the same internal functions. This avoids duplication and keeps behaviour identical.

---

## 6. Recommended Agent Workflow (update README + prompt)

### New preferred workflow

1. Prefer the `code` tool for anything that involves more than one or two operations.
2. Use individual tools only for trivial one-shot actions or when the client does not yet support code mode.
3. Inside `code`:
   - Create / load the model
   - Build structure
   - Call `stella.validate()` and fix problems in a loop if needed
   - Simulate / analyse
   - Save

### Suggested system-prompt / tool-description fragment

> When constructing or analysing system-dynamics models, prefer the `code` tool.<br>
> Write clear, sequential Python against the `stella` object.<br>
> Use intermediate variables, loops, and validation checks.<br>
> Assign the final useful object to `result`.

This guidance should also be reflected in the existing `build-stella-model` prompt if one is present.

---

## 7. Compatibility & Migration Strategy

- **Phase 1 (this change):** Add `code` tool + `StellaAPI`. Keep every existing tool unchanged.
- **Phase 2 (optional later):** Mark the most fine-grained tools as secondary in their descriptions (“prefer the `code` tool for multi-step work”).
- **Phase 3 (optional):** Provide a configuration flag or server variant that only exposes the `code` tool (true single-tool Code Mode MCP, similar to Cloudflare’s later pattern).
- Existing clients continue to work with zero changes.

---

## 8. Testing Requirements

### 8.1 Unit tests for `StellaAPI`

- Every public method has a corresponding test that exercises the same path the classic tool used.
- Cover success paths, validation failures, missing models, duplicate names, etc.

### 8.2 Executor tests

- Successful multi-step scripts (create → add variables → validate → simulate → save).
- Validation and simulation success/failure cases.
- Timeout enforcement.
- Attempted unsafe operations are blocked.
- Correct capture of `result` and `stdout`.
- Exception messages and tracebacks are returned cleanly.

### 8.3 Integration / evaluation

- End-to-end scenarios such as:
  - “Build an SIR model, add a vaccination flow, run sensitivity analysis on R0, then save.”
  - “Load an existing model, fix validation errors, compare two scenarios.”
- Compare token usage and success rate of pure tool-calling vs. code-mode on the existing evaluation suite.

### 8.4 Regression

- All existing evaluation tests continue to pass when using the classic tools.

---

## 9. Documentation Deliverables

- `docs/code-mode.md` – full design, security model, API reference, and examples.
- Update main `README.md` with a prominent “Code Mode” section and a side-by-side comparison (classic tools vs. code).
- Add several realistic examples under `examples/code_mode/`.
- Optionally expose `stella.help()` so the model can discover the API at runtime.

---

## 10. Future Extensions (out of scope for v1 but designed for)

- Persistent IPython-style kernel across multiple `code` calls (Prime-Agent / RLM style).
- Ability for the generated code to spawn sub-agents or call other MCP tools.
- Automatic generation of TypeScript type definitions if a JS/TS client ever wants the same pattern.
- Server-side Code Mode that collapses *all* tools into a single `code` tool (Cloudflare single-tool pattern).

---

## 11. Suggested Implementation Order

1. Extract shared domain functions so classic handlers and the new API share them.
2. Implement `StellaAPI` + comprehensive unit tests.
3. Implement restricted executor + the `code` tool.
4. Wire into tool registration and annotations.
5. Write documentation and examples.
6. Run full evaluation suite and measure improvement on multi-step modeling tasks.
7. Merge.

---

## Appendix A – Example Usage (what the model should write)

```python
# Create and fully build an SIR model in one shot
mid = stella.create_model("SIR Epidemic", start=0, stop=120, dt=0.25)

stella.add_stock("Susceptible", 990, units="people")
stella.add_stock("Infected", 10, units="people")
stella.add_stock("Recovered", 0, units="people")

stella.add_aux("Total", "Susceptible + Infected + Recovered", units="people")
stella.add_aux("beta", "0.3")
stella.add_aux("gamma", "0.1")

stella.add_flow(
    "Infection",
    "beta * Susceptible * Infected / Total",
    from_stock="Susceptible",
    to_stock="Infected",
    units="people/day",
)
stella.add_flow(
    "Recovery",
    "gamma * Infected",
    from_stock="Infected",
    to_stock="Recovered",
    units="people/day",
)

stella.sync_connectors_from_equations()
report = stella.validate()
print(report)

if report.get("errors"):
    result = {"status": "validation_failed", "report": report}
else:
    sim = stella.simulate()
    stella.save_model("/tmp/sir.stmx")
    result = {"status": "ok", "validation": report, "simulation_summary": sim}
```

---

## Appendix B – Key Existing Components to Reuse

| Component | Location | Role |
|-----------|----------|------|
| `SessionStore` | `session_store.py` | Per-session model registry and current-model pointer |
| `StellaModel` | `model.py` | Core domain object (stocks, flows, auxs, modules, connectors) |
| Validation | `validator.py` | Semantic checks |
| Simulation | `simulate.py` + `analysis.py` + `calibrate.py` | PySD-backed runs and analysis |
| Layout | `layout_*.py` | Auto-layout pipeline |
| XMILE I/O | `xmile_*.py` | Parse / export |
| Tool modules | `tools/*.py` | Current schemas + handlers (refactor to share logic) |

---

*End of specification.*
