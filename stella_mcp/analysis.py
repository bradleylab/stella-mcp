"""Scenario comparison and sensitivity analysis over the simulation engine.

Both tools are compositions of ``simulate.run_simulation``'s building blocks:
they reuse one compiled PySD runner (``_compile_runner``) across many
``run(params=...)`` calls — PySD is stateless across runs, so a sweep needs
only a single XMILE compile. Nothing here imports pysd or pandas at module
load: the analyses inherit ``SimulationDependencyError`` from the shared
runner, so importing this module never requires the optional ``sim`` extra.
"""

from __future__ import annotations

from typing import Any

from .simulate import (
    DEFAULT_MAX_POINTS,
    _compile_runner,
    method_warnings,
    resolve_overrides,
    resolve_report_keys,
    summarize_run,
)
from .xmile import StellaModel


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
