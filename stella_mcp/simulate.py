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
    pysd = _import_pysd()

    if max_points < 2:
        raise ValueError("max_points must be >= 2")

    def resolve(name: str) -> str | None:
        """Map a user-supplied name to a normalized model variable key."""
        key = model._normalize_name(name)
        for registry in (model.stocks, model.flows, model.auxs):
            if key in registry:
                return key
        return None

    all_keys = [*model.stocks, *model.flows, *model.auxs]

    resolved_overrides: dict[str, float] = {}
    for name, value in (overrides or {}).items():
        key = resolve(name)
        if key is None:
            candidates = ", ".join(sorted(model._display_name(k) for k in all_keys))
            raise ValueError(
                f"Override '{name}' matches no model variable. "
                f"Valid names: {candidates}"
            )
        resolved_overrides[model._display_name(key)] = float(value)

    if include is not None:
        report_keys = []
        for name in include:
            key = resolve(name)
            if key is None:
                raise ValueError(f"include entry '{name}' matches no model variable")
            report_keys.append(key)
    else:
        report_keys = list(model.stocks)

    sim_warnings: list[str] = []
    method = (model.sim_specs.method or "").strip()
    if method.upper() not in ("", "EULER"):
        sim_warnings.append(
            f"Model integration method is '{method}' but PySD integrates with "
            "Euler only; results will differ from Stella for stiff systems."
        )

    # The XMILE writer handles the GRAPH(input) -> spec-form rewrite for
    # gf-bearing equations; the copy exists because export mutates layout.
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
            results = runner.run(params=resolved_overrides or None)
    finally:
        tmp_path.unlink(missing_ok=True)

    if save_results_csv:
        results.to_csv(save_results_csv, index_label="time")

    times = [float(t) for t in results.index]
    indices = _downsample_indices(len(times), max_points)

    series: list[dict[str, Any]] = []
    for key in report_keys:
        display = model._display_name(key)
        if display not in results.columns:
            sim_warnings.append(
                f"Variable '{display}' was not reported by the simulation backend"
            )
            continue
        column = results[display]
        values = [float(v) for v in column]
        summary, had_non_finite = _series_summary(values)
        if had_non_finite:
            sim_warnings.append(
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
