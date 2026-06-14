"""Simulation bridge backed by PySD (optional dependency).

PySD is the established open-source XMILE runner; it is deliberately an
optional extra (``stella-mcp[sim]``) so the core package keeps its single
``mcp`` dependency. Everything here operates on a deep copy of the model
because ``to_xml()`` mutates layout state.
"""

from __future__ import annotations

import copy
import math
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .xmile import StellaModel

DEFAULT_MAX_POINTS = 101


class SimulationDependencyError(RuntimeError):
    """Raised when the optional pysd dependency is not installed."""


def _import_pysd():
    try:
        import pysd
    except ImportError as exc:
        raise SimulationDependencyError(
            "Simulation requires the optional pysd dependency. "
            "Install with: pip install 'stella-mcp[sim]'"
        ) from exc
    return pysd


def _json_safe(value: float) -> float | None:
    return None if (math.isnan(value) or math.isinf(value)) else float(value)


def _series_summary(values: list[float]) -> tuple[dict[str, float | None], bool]:
    """Summarize a series over its finite values only.

    Python's min/max are order-dependent in the presence of NaN, so
    summaries computed over raw values could look finite while hiding bad
    points. Returns (summary, had_non_finite).
    """
    finite = [v for v in values if math.isfinite(v)]
    had_non_finite = len(finite) != len(values)
    if not finite:
        return (
            {"initial": None, "final": None, "min": None, "max": None},
            had_non_finite,
        )
    return (
        {
            "initial": _json_safe(values[0]),
            "final": _json_safe(values[-1]),
            "min": min(finite),
            "max": max(finite),
        },
        had_non_finite,
    )


def _downsample_indices(n: int, max_points: int) -> list[int]:
    """Evenly strided indices keeping the first and last point."""
    if n <= max_points:
        return list(range(n))
    stride = (n - 1) / (max_points - 1)
    indices = {round(i * stride) for i in range(max_points)}
    indices.add(n - 1)
    return sorted(indices)


def _resolve_key(model: StellaModel, name: str) -> str | None:
    """Map a user-supplied name (display or underscore form) to a normalized
    model variable key, or None if it matches no stock/flow/aux."""
    key = model._normalize_name(name)
    for registry in (model.stocks, model.flows, model.auxs):
        if key in registry:
            return key
    return None


def resolve_overrides(
    model: StellaModel, overrides: dict[str, float] | None
) -> dict[str, float]:
    """Validate override names and return them keyed by display name (the form
    PySD's ``params=`` expects), with float values.

    Raises ValueError naming the offending entry and the valid variable names.
    """
    all_keys = [*model.stocks, *model.flows, *model.auxs]
    resolved: dict[str, float] = {}
    for name, value in (overrides or {}).items():
        key = _resolve_key(model, name)
        if key is None:
            candidates = ", ".join(sorted(model._display_name(k) for k in all_keys))
            raise ValueError(
                f"Override '{name}' matches no model variable. "
                f"Valid names: {candidates}"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"override '{name}' must be a finite number")
        resolved[model._display_name(key)] = numeric
    return resolved


def constant_parameter_value(model: StellaModel, key: str | None) -> float | None:
    """The defining numeric constant of a *constant aux or flow*, or None for a
    stock or any non-constant equation.

    Deliberately returns None for stocks: PySD's ``run(params={name: x})`` pins
    a stock to the constant ``x`` for the entire run rather than setting its
    initial condition (verified 2026-06-14), so stocks are not overridable as
    fit parameters. Callers (e.g. calibrate) use a None return to reject a
    variable as non-calibratable. This is intentionally stricter than a stock's
    initial value, which is why it is separate from analysis's
    ``_baseline_param_value``.
    """
    if key is None:
        return None
    if key in model.auxs:
        equation = model.auxs[key].equation
    elif key in model.flows:
        equation = model.flows[key].equation
    else:  # stock or unknown -> not a calibratable constant
        return None
    try:
        return float(equation)
    except (TypeError, ValueError):
        return None


def resolve_report_keys(
    model: StellaModel, include: list[str] | None
) -> list[str]:
    """Resolve the list of variables to report to normalized keys. Defaults to
    all stocks when ``include`` is None. Raises ValueError on an unknown name."""
    if include is None:
        return list(model.stocks)
    report_keys = []
    for name in include:
        key = _resolve_key(model, name)
        if key is None:
            raise ValueError(f"include entry '{name}' matches no model variable")
        report_keys.append(key)
    return report_keys


def method_warnings(model: StellaModel) -> list[str]:
    """Warn when the model's integration method is not Euler (PySD is Euler-only)."""
    method = (model.sim_specs.method or "").strip()
    if method.upper() not in ("", "EULER"):
        return [
            f"Model integration method is '{method}' but PySD integrates with "
            "Euler only; results will differ from Stella for stiff systems."
        ]
    return []


@contextmanager
def _compile_runner(model: StellaModel):
    """Compile ``model`` into a PySD runner once, yielding the runner.

    The model is deep-copied and written to a temp ``.stmx`` (export mutates
    layout state; the XMILE writer also handles the GRAPH(input) -> spec-form
    rewrite for gf-bearing equations). The temp file is kept for the lifetime
    of the context so repeated ``runner.run(params=...)`` calls are safe, then
    removed. PySD is stateless across ``run()`` calls (verified 2026-06-11:
    a run after a different-params run reproduces a fresh-compile result
    byte-for-byte), so callers may loop ``runner.run(params=...)`` with
    different params and reuse a single compiled model across a sweep.
    """
    pysd = _import_pysd()
    sim_model = copy.deepcopy(model)
    handle = tempfile.NamedTemporaryFile(
        suffix=".stmx", mode="w", delete=False, encoding="utf-8"
    )
    tmp_path = Path(handle.name)
    try:
        with handle:
            handle.write(sim_model.to_xml())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            runner = pysd.read_xmile(str(tmp_path))
            yield runner
    finally:
        tmp_path.unlink(missing_ok=True)


def summarize_run(
    results: Any,
    report_keys: list[str],
    model: StellaModel,
    max_points: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Reduce a PySD results DataFrame to downsampled, NaN-aware series.

    Returns ``(series, warnings)``. A report key absent from the results
    columns yields a warning rather than a series entry. Shared by
    ``run_simulation`` and the scenario/sensitivity analyses so every tool
    produces an identical series structure.
    """
    times = [float(t) for t in results.index]
    indices = _downsample_indices(len(times), max_points)

    series: list[dict[str, Any]] = []
    warnings_out: list[str] = []
    for key in report_keys:
        display = model._display_name(key)
        if display not in results.columns:
            warnings_out.append(
                f"Variable '{display}' was not reported by the simulation backend"
            )
            continue
        values = [float(v) for v in results[display]]
        summary, had_non_finite = _series_summary(values)
        if had_non_finite:
            warnings_out.append(
                f"Series '{display}' contains non-finite values (NaN/inf); "
                "summary covers finite points only"
            )
        series.append({
            "name": display,
            "points": [
                {"t": times[i], "value": _json_safe(values[i])} for i in indices
            ],
            "summary": summary,
        })
    return series, warnings_out


def run_simulation(
    model: StellaModel,
    overrides: dict[str, float] | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
    include: list[str] | None = None,
    save_results_csv: str | None = None,
) -> dict[str, Any]:
    """Run the model through PySD and return downsampled, summarized series.

    Parameters
    ----------
    model : StellaModel
        Session model; never mutated (all work happens on a deep copy).
    overrides : dict, optional
        Constant parameter overrides keyed by variable name (display or
        underscore form). Validated against model variables before running.
    max_points : int
        Maximum points per returned series (first and last always kept).
    include : list of str, optional
        Variables to report. Defaults to all stocks.
    save_results_csv : str, optional
        Write the full (non-downsampled) results table to this CSV path.
    """
    if max_points < 2:
        raise ValueError("max_points must be >= 2")

    resolved_overrides = resolve_overrides(model, overrides)
    report_keys = resolve_report_keys(model, include)
    sim_warnings = method_warnings(model)

    with _compile_runner(model) as runner:
        results = runner.run(params=resolved_overrides or None)

    if save_results_csv:
        results.to_csv(save_results_csv, index_label="time")

    series, series_warnings = summarize_run(results, report_keys, model, max_points)
    sim_warnings.extend(series_warnings)

    return {
        "sim_specs": {
            "start": model.sim_specs.start,
            "stop": model.sim_specs.stop,
            "dt": model.sim_specs.dt,
            "method": model.sim_specs.method,
            "time_units": model.sim_specs.time_units,
        },
        "overrides": resolved_overrides,
        "warnings": sim_warnings,
        "series": series,
        "csv_path": save_results_csv,
    }
