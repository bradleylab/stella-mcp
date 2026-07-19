"""Scenario comparison and sensitivity analysis over the simulation engine.

Both tools are compositions of ``simulate.run_simulation``'s building blocks:
they reuse one compiled PySD runner (``_compile_runner``) across many
``run(params=...)`` calls — PySD is stateless across runs, so a sweep needs
only a single XMILE compile. Nothing here imports pysd or pandas at module
load: the analyses inherit ``SimulationDependencyError`` from the shared
runner, so importing this module never requires the optional ``sim`` extra.
"""

from __future__ import annotations

import csv
import math
from typing import Any

from .simulate import (
    DEFAULT_MAX_POINTS,
    _compile_runner,
    _resolve_key,
    method_warnings,
    resolve_overrides,
    resolve_report_keys,
    simulation_backend_metadata,
    summarize_run,
)
from .xmile import StellaModel

_METRICS = frozenset({"final", "max", "min", "mean", "time_to_threshold"})


def _require_finite(value: Any, label: str) -> float:
    """Coerce to float and reject non-finite user input (NaN/inf), so garbage
    never silently propagates into a simulation or a sensitivity curve."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be a finite number")
    return numeric


def _sub(a: float | None, b: float | None) -> float | None:
    """Difference that propagates None (a non-finite summary stays None)."""
    return None if (a is None or b is None) else a - b


def _pct(current: float | None, base: float | None) -> float | None:
    """Percent change of ``current`` relative to ``base``; None when undefined
    (missing value or zero baseline — no divide-by-zero)."""
    if current is None or base is None or base == 0:
        return None
    return (current - base) / base * 100.0


def _delta_vs_baseline(
    base_series: list[dict[str, Any]], scen_series: list[dict[str, Any]]
) -> dict[str, dict[str, float | None]]:
    """Per-variable summary deltas of a scenario against the baseline.

    Only variables present in both runs are compared.
    """
    base_by = {s["name"]: s["summary"] for s in base_series}
    deltas: dict[str, dict[str, float | None]] = {}
    for s in scen_series:
        name = s["name"]
        if name not in base_by:
            continue
        b, c = base_by[name], s["summary"]
        deltas[name] = {
            "final_abs": _sub(c["final"], b["final"]),
            "final_pct": _pct(c["final"], b["final"]),
            "max_abs": _sub(c["max"], b["max"]),
        }
    return deltas


def _resolve_scenarios(
    model: StellaModel, scenarios: list[dict[str, Any]], baseline: dict[str, float] | None
) -> tuple[dict[str, float], list[tuple[str, dict[str, float]]]]:
    """Validate scenario structure and resolve every override name up front.

    Fail-fast and atomic: a bad override name (or duplicate scenario name)
    raises before any model is compiled or run, so a typo in the last scenario
    never leaves earlier scenarios half-run. Returns the resolved baseline
    overrides and a list of (name, resolved_overrides) pairs.
    """
    if not scenarios:
        raise ValueError("compare_scenarios requires at least one scenario")

    resolved_baseline = resolve_overrides(model, baseline)
    resolved: list[tuple[str, dict[str, float]]] = []
    seen: set[str] = set()
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError("each scenario must be an object")
        name = scenario.get("name")
        if not name:
            raise ValueError("each scenario needs a non-empty 'name'")
        if name in seen:
            raise ValueError(f"duplicate scenario name '{name}'")
        seen.add(name)
        try:
            overrides = resolve_overrides(model, scenario.get("overrides") or {})
        except ValueError as exc:
            raise ValueError(f"scenario '{name}': {exc}") from exc
        resolved.append((name, overrides))
    return resolved_baseline, resolved


def _write_comparison_csv(
    path: str,
    model: StellaModel,
    report_keys: list[str],
    base_results: Any,
    scenario_results: list[tuple[str, Any]],
) -> None:
    """Write a wide table: time index × one column per (variable, scenario).

    pandas is imported locally — it is a pysd dependency present only when the
    runner compiled successfully, so this never runs without the sim extra.
    """
    import pandas as pd

    wanted = [model._display_name(k) for k in report_keys]

    def _suffixed(results: Any, label: str) -> Any:
        cols = [c for c in wanted if c in results.columns]
        return results[cols].add_suffix(f"__{label}")

    frames = [_suffixed(base_results, "baseline")]
    frames += [_suffixed(res, name) for name, res in scenario_results]
    pd.concat(frames, axis=1).to_csv(path, index_label="time")


def compare_scenarios(
    model: StellaModel,
    scenarios: list[dict[str, Any]],
    baseline: dict[str, float] | None = None,
    include: list[str] | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
    save_comparison_csv: str | None = None,
) -> dict[str, Any]:
    """Run several named override sets against a baseline and report divergence.

    Parameters
    ----------
    model : StellaModel
        Session model; never mutated (the runner works on a deep copy).
    scenarios : list of dict
        Each ``{"name": str, "overrides": {var: number}}``. Names must be
        unique. At least one is required.
    baseline : dict, optional
        Override set the deltas are measured against. Defaults to the
        unmodified model (no overrides).
    include : list of str, optional
        Variables to report and compare. Defaults to all stocks.
    max_points : int
        Maximum points per returned series (first and last always kept).
    save_comparison_csv : str, optional
        Write a wide (variable × scenario) results table to this CSV path.

    Returns a dict with ``sim_specs``, the ``baseline`` run, and per-scenario
    ``series`` + ``delta_vs_baseline`` (final/max absolute and final percent).
    """
    if max_points < 2:
        raise ValueError("max_points must be >= 2")

    resolved_baseline, resolved_scenarios = _resolve_scenarios(model, scenarios, baseline)
    report_keys = resolve_report_keys(model, include)
    base_warnings = method_warnings(model)

    with _compile_runner(model) as runner:
        base_results = runner.run(params=resolved_baseline or None)
        base_series, base_series_warnings = summarize_run(
            base_results, report_keys, model, max_points
        )
        scenario_results: list[tuple[str, Any]] = []
        scenario_payload: list[dict[str, Any]] = []
        for name, overrides in resolved_scenarios:
            results = runner.run(params=overrides or None)
            series, warnings = summarize_run(results, report_keys, model, max_points)
            scenario_results.append((name, results))
            scenario_payload.append({
                "name": name,
                "overrides": overrides,
                "series": series,
                "delta_vs_baseline": _delta_vs_baseline(base_series, series),
                "warnings": warnings,
            })

    if save_comparison_csv:
        _write_comparison_csv(
            save_comparison_csv, model, report_keys, base_results, scenario_results
        )

    return {
        "backend": simulation_backend_metadata(model),
        "sim_specs": {
            "start": model.sim_specs.start,
            "stop": model.sim_specs.stop,
            "dt": model.sim_specs.dt,
            "method": model.sim_specs.method,
            "time_units": model.sim_specs.time_units,
        },
        "baseline": {
            "overrides": resolved_baseline,
            "series": base_series,
            "warnings": base_warnings + base_series_warnings,
        },
        "scenarios": scenario_payload,
        "csv_path": save_comparison_csv,
    }


# =============================================================================
# Sensitivity analysis (one-at-a-time)
# =============================================================================

def _output_series(results: Any, display: str) -> tuple[list[float] | None, list[float] | None]:
    """Full (non-downsampled) (times, values) for one variable, or (None, None)
    when the backend did not report it."""
    if display not in results.columns:
        return None, None
    times = [float(t) for t in results.index]
    values = [float(v) for v in results[display]]
    return times, values


def _time_to_threshold(
    times: list[float], values: list[float], threshold: float
) -> float | None:
    """First time the series crosses ``threshold``.

    Direction is inferred from the first finite value: a series starting below
    the threshold crosses when it first reaches/exceeds it, one starting above
    crosses when it first reaches/falls below. Non-finite points are skipped.
    Returns None if it never crosses.
    """
    rising = None
    for t, v in zip(times, values, strict=True):
        if not math.isfinite(v):
            continue
        if rising is None:
            if v == threshold:
                return float(t)
            rising = v < threshold
        if (rising and v >= threshold) or (not rising and v <= threshold):
            return float(t)
    return None


def _reduce_metric(
    times: list[float] | None,
    values: list[float] | None,
    metric: str,
    threshold: float | None,
) -> float | None:
    """Reduce a full output series to the requested scalar metric.

    max/min/mean cover finite values only (consistent with the simulate
    summaries); ``final`` is None when the last point is non-finite. Returns
    None when the metric is undefined (e.g. an all-NaN series, or a threshold
    never crossed)."""
    if not values:
        return None
    if metric == "final":
        last = values[-1]
        return float(last) if math.isfinite(last) else None
    if metric == "time_to_threshold":
        if threshold is None:
            return None
        return _time_to_threshold(times or [], values, threshold)
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return None
    if metric == "max":
        return max(finite)
    if metric == "min":
        return min(finite)
    if metric == "mean":
        return sum(finite) / len(finite)
    raise ValueError(f"unknown metric '{metric}'")


def _range_sensitivity(points: list[dict[str, Any]]) -> float | None:
    """Average slope of the metric across the swept range:
    ``(metric_hi - metric_lo) / (value_hi - value_lo)`` between the lowest and
    highest swept values whose metric is defined. None if fewer than two such
    points or the endpoints share a value."""
    valid = sorted((p["value"], p["metric"]) for p in points if p["metric"] is not None)
    if len(valid) < 2:
        return None
    (v_lo, m_lo), (v_hi, m_hi) = valid[0], valid[-1]
    if v_hi == v_lo:
        return None
    return (m_hi - m_lo) / (v_hi - v_lo)


def _baseline_param_value(model: StellaModel, key: str | None) -> float | None:
    """The parameter's baseline constant value (its defining equation parsed as
    a float), or None when it is not a simple constant."""
    if key is None:
        return None
    if key in model.auxs:
        equation = model.auxs[key].equation
    elif key in model.flows:
        equation = model.flows[key].equation
    elif key in model.stocks:
        equation = model.stocks[key].initial_value
    else:
        return None
    try:
        return float(equation)
    except (TypeError, ValueError):
        return None


def _elasticity(
    model: StellaModel, name: str, range_sensitivity: float | None, baseline_metric: float | None
) -> float | None:
    """Sensitivity normalized at the baseline: ``slope * p0 / metric0``
    (≈ Δoutput% / Δparam%). None when any term is undefined (non-constant
    parameter, zero baseline metric or parameter)."""
    if range_sensitivity is None or baseline_metric in (None, 0):
        return None
    p0 = _baseline_param_value(model, _resolve_key(model, name))
    if p0 is None or p0 == 0:
        return None
    return range_sensitivity * p0 / baseline_metric


def _expand_param_sweep(
    model: StellaModel, spec: dict[str, Any]
) -> tuple[str, list[float]]:
    """Validate one parameter spec and expand it to the list of swept values.

    Accepts either explicit ``values`` (≥2) or ``start``/``stop``/``steps``
    (steps ≥ 2, start ≠ stop). Resolves the parameter name, raising the
    shared override error (with valid names) on a typo.
    """
    if not isinstance(spec, dict):
        raise ValueError("each parameter must be an object")
    name = spec.get("name")
    if not name:
        raise ValueError("each parameter needs a 'name'")
    if _resolve_key(model, name) is None:
        resolve_overrides(model, {name: 0.0})  # raises naming the valid variables
    if spec.get("values") is not None:
        values = [_require_finite(v, f"parameter '{name}' value") for v in spec["values"]]
        if len(values) < 2:
            raise ValueError(f"parameter '{name}': 'values' needs at least 2 entries")
        return name, values
    start, stop, steps = spec.get("start"), spec.get("stop"), spec.get("steps")
    if start is None or stop is None or steps is None:
        raise ValueError(
            f"parameter '{name}': provide 'values' or all of start/stop/steps"
        )
    start = _require_finite(start, f"parameter '{name}' start")
    stop = _require_finite(stop, f"parameter '{name}' stop")
    steps = int(steps)
    if steps < 2:
        raise ValueError(f"parameter '{name}': 'steps' must be >= 2")
    if start == stop:
        raise ValueError(f"parameter '{name}': 'start' and 'stop' must differ")
    step = (stop - start) / (steps - 1)
    return name, [start + i * step for i in range(steps)]


def _validate_output(
    model: StellaModel, output: dict[str, Any]
) -> tuple[str, str, float | None]:
    """Validate the output spec; return (variable_key, metric, threshold)."""
    if not isinstance(output, dict):
        raise ValueError("output must be an object")
    variable = output.get("variable")
    if not variable:
        raise ValueError("output.variable is required")
    key = _resolve_key(model, variable)
    if key is None:
        raise ValueError(f"output.variable '{variable}' matches no model variable")
    metric = output.get("metric", "final")
    if metric not in _METRICS:
        raise ValueError(
            f"output.metric '{metric}' must be one of {sorted(_METRICS)}"
        )
    threshold = output.get("threshold")
    if metric == "time_to_threshold":
        if threshold is None:
            raise ValueError("output.metric 'time_to_threshold' requires output.threshold")
        threshold = _require_finite(threshold, "output.threshold")
    return key, metric, threshold


def _write_sweep_csv(path: str, metric: str, rows: list[tuple[str, float, float | None]]) -> None:
    """Write the long sweep table (parameter, value, metric) as CSV (stdlib —
    no pandas needed for this shape)."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value", metric])
        for name, value, metric_value in rows:
            writer.writerow([name, value, "" if metric_value is None else metric_value])


def sensitivity_analysis(
    model: StellaModel,
    parameters: list[dict[str, Any]],
    output: dict[str, Any],
    mode: str = "oat",
    max_runs: int = 200,
    include_series: bool = False,
    save_sweep_csv: str | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
) -> dict[str, Any]:
    """One-at-a-time parameter sensitivity of a single output metric.

    Sweeps each parameter across its range holding the others at their model
    baseline, reduces each run's output series to ``output.metric``, and
    reports per-parameter metric curves plus a range slope and a
    baseline-normalized elasticity for ranking.

    Parameters
    ----------
    model : StellaModel
        Session model; never mutated (the runner works on a deep copy).
    parameters : list of dict
        Each ``{"name": str, "start", "stop", "steps"}`` or
        ``{"name": str, "values": [..]}``.
    output : dict
        ``{"variable": str, "metric": "final"|"max"|"min"|"mean"|
        "time_to_threshold", "threshold": number}`` (threshold required only
        for ``time_to_threshold``).
    mode : str
        Only ``"oat"`` (one-at-a-time) is supported; ``"grid"``/``"montecarlo"``
        are reserved.
    max_runs : int
        Hard cap on the total swept runs (excludes the single baseline run);
        the call raises rather than truncating a larger sweep.
    include_series : bool
        Also attach each run's downsampled output series to its point.
    save_sweep_csv : str, optional
        Write the long (parameter, value, metric) table to this CSV path.
    """
    if mode != "oat":
        raise ValueError(
            f"mode '{mode}' not supported; only 'oat' (one-at-a-time) is available"
        )
    if not parameters:
        raise ValueError("sensitivity_analysis requires at least one parameter")

    output_key, metric, threshold = _validate_output(model, output)
    output_display = model._display_name(output_key)
    sweeps = [_expand_param_sweep(model, spec) for spec in parameters]

    total_runs = sum(len(values) for _, values in sweeps)
    if total_runs > max_runs:
        raise ValueError(
            f"sweep needs {total_runs} runs (> max_runs={max_runs}); "
            "reduce steps/values or raise max_runs"
        )

    report_keys = [output_key]
    warnings = method_warnings(model)
    csv_rows: list[tuple[str, float, float | None]] = []

    with _compile_runner(model) as runner:
        base_times, base_values = _output_series(runner.run(params=None), output_display)
        baseline_metric = _reduce_metric(base_times, base_values, metric, threshold)
        if baseline_metric is None:
            warnings.append(
                f"baseline did not report output '{output_display}'"
                if base_values is None
                else f"baseline metric '{metric}' is undefined (non-finite series or "
                "threshold never crossed); elasticity will be null"
            )

        param_payload: list[dict[str, Any]] = []
        for name, values in sweeps:
            points: list[dict[str, Any]] = []
            param_warnings: list[str] = []
            for value in values:
                results = runner.run(params=resolve_overrides(model, {name: value}))
                times, series_values = _output_series(results, output_display)
                if series_values is None:
                    param_warnings.append(
                        f"output variable '{output_display}' not reported at {name}={value}"
                    )
                    metric_value = None
                else:
                    metric_value = _reduce_metric(times, series_values, metric, threshold)
                    if metric_value is None:
                        param_warnings.append(
                            f"{name}={value}: metric '{metric}' is undefined (non-finite "
                            "series or threshold never crossed)"
                        )
                point: dict[str, Any] = {"value": value, "metric": metric_value}
                if include_series and series_values is not None:
                    point["series"] = summarize_run(results, report_keys, model, max_points)[0]
                points.append(point)
                csv_rows.append((name, value, metric_value))

            range_sensitivity = _range_sensitivity(points)
            param_payload.append({
                "name": name,
                "points": points,
                "range_sensitivity": range_sensitivity,
                "elasticity": _elasticity(model, name, range_sensitivity, baseline_metric),
                "warnings": param_warnings,
            })

    if save_sweep_csv:
        _write_sweep_csv(save_sweep_csv, metric, csv_rows)

    output_payload: dict[str, Any] = {"variable": output_display, "metric": metric}
    if metric == "time_to_threshold":
        output_payload["threshold"] = threshold
    return {
        "backend": simulation_backend_metadata(model),
        "output": output_payload,
        "baseline": {"overrides": {}, "metric_value": baseline_metric},
        "parameters": param_payload,
        "total_runs": total_runs,
        "warnings": warnings,
    }
