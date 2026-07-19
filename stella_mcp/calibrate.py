"""Parameter calibration: fit constant parameters to observed data (PySD-backed).

The inverse of ``run_simulation`` — given an observed time-series, find the
constant parameter values whose simulation best reproduces it. Composed on the
shared ``_compile_runner``: the optimizer compiles the model once and every
objective evaluation is a single ``runner.run(params=...)`` (PySD is stateless
across runs, verified in 0.9.0).

Dependency discipline: numpy, scipy, and pandas are imported *inside* functions,
never at module load, so importing this module never requires the optional
``sim`` extra — exactly like ``analysis.py``. ``csv`` and ``math`` are stdlib.

Only constant auxiliaries and flows are calibratable. Stocks are rejected:
PySD's ``run(params={stock: x})`` pins the stock to a constant for the whole
run instead of setting its initial condition (see
``simulate.constant_parameter_value``), which would silently flatten a dynamic
model.
"""

from __future__ import annotations

import csv
import math
from typing import Any

from .simulate import (
    DEFAULT_MAX_POINTS,
    _compile_runner,
    _resolve_key,
    constant_parameter_value,
    method_warnings,
    resolve_overrides,
    simulation_backend_metadata,
    summarize_run,
)
from .xmile import StellaModel
from .xmile_features import ensure_supported_for_simulation

SEED = 0
_METHODS = frozenset({"least_squares", "differential_evolution"})
_DEFAULT_MAX_NFEV = 1000
_DEFAULT_POPSIZE = 15
# differential_evolution generation cap when `maxiter` is not given. Kept
# independent of `max_nfev` (a least_squares knob) so the two optimizers'
# budgets don't masquerade as one.
_DEFAULT_DE_MAXITER = 100
_AT_BOUND_RTOL = 1e-6
# Base magnitude for a non-finite trial's penalty residual. Grown with distance
# from the last feasible point so the optimizer's Jacobian points back toward
# feasibility instead of seeing a flat plateau (which fakes convergence).
_PENALTY_BASE = 1e6
# Above this JᵀJ condition number, the linearized covariance is untrustworthy.
_COND_LIMIT = 1e12
# Targets must differ by more than this factor to trigger the unweighted-fit warning.
_SCALE_WARN_RATIO = 10.0


def _require_finite(value: Any, label: str) -> float:
    """Coerce to float and reject non-finite or non-numeric user input.

    A non-numeric value (list, dict, ...) raises ``ValueError`` — not the
    ``TypeError`` ``float()`` would — so malformed tool input is classified as
    invalid_input rather than internal_error.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be a finite number")
    return numeric


def _positive_int(value: Any, label: str) -> int:
    """Require a non-boolean integer greater than or equal to one."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be an integer greater than or equal to 1")
    return value


def _seed_int(value: Any) -> int:
    """Require an explicit non-boolean integer seed."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("seed must be an integer and must not be null or boolean")
    return value


def _resolve_key_or_raise(model: StellaModel, name: str) -> str:
    """Resolve a user name to a model variable key, or raise the shared override
    error (naming the valid variables)."""
    key = _resolve_key(model, name)
    if key is None:
        resolve_overrides(model, {name: 0.0})  # raises naming the valid variables
        raise ValueError(f"'{name}' matches no model variable")  # unreachable safety
    return key


# === observation loading + alignment =========================================

def _read_obs_csv(path: str) -> tuple[list[Any], dict[str, list[Any]]]:
    """Read an observations CSV: first column is time, the rest are target
    columns keyed by name. pandas is imported locally (a pysd dependency)."""
    import pandas as pd

    frame = pd.read_csv(path)
    if frame.shape[1] < 2:
        raise ValueError(
            "observations CSV needs a time column plus at least one target column"
        )
    time_col = frame.columns[0]
    times = list(frame[time_col])
    targets = {str(col): list(frame[col]) for col in frame.columns[1:]}
    return times, targets


def _load_observations(spec: Any, model: StellaModel) -> dict[str, Any]:
    """Validate and normalize the observation spec to numpy arrays.

    Accepts inline ``{"time": [...], "targets": {var: [...]}}`` or
    ``{"csv_path": "..."}``. All targets share one strictly-increasing time
    grid. Validation is atomic and fail-fast (before any run): every target
    resolves to a model variable, series lengths match the time grid, and every
    value is finite. Returns ``{"times": ndarray, "targets": [(key, display,
    values)]}``.
    """
    import numpy as np

    if not isinstance(spec, dict):
        raise ValueError("observations must be an object")

    if spec.get("csv_path") is not None:
        raw_times, raw_targets = _read_obs_csv(spec["csv_path"])
    else:
        raw_times = spec.get("time")
        raw_targets = spec.get("targets")
        if raw_times is None or raw_targets is None:
            raise ValueError("observations needs 'time' and 'targets' (or a 'csv_path')")
        if not isinstance(raw_targets, dict) or not raw_targets:
            raise ValueError("observations.targets must be a non-empty object")

    if not isinstance(raw_times, list):
        raise ValueError("observations.time must be an array of numbers")
    times = [_require_finite(t, "observations.time") for t in raw_times]
    if len(times) < 2:
        raise ValueError("observations need at least 2 time points")
    for earlier, later in zip(times[:-1], times[1:], strict=True):
        if not later > earlier:
            raise ValueError("observations.time must be strictly increasing")

    targets: list[tuple[str, str, Any]] = []
    for name, values in raw_targets.items():
        if not isinstance(values, list):
            raise ValueError(f"observations target '{name}' must be an array of numbers")
        key = _resolve_key_or_raise(model, name)
        display = model._display_name(key)
        finite = [_require_finite(v, f"observations target '{name}'") for v in values]
        if len(finite) != len(times):
            raise ValueError(
                f"observations target '{name}' has {len(finite)} values but "
                f"{len(times)} time points"
            )
        targets.append((key, display, np.asarray(finite, dtype=float)))

    return {"times": np.asarray(times, dtype=float), "targets": targets}


def _check_obs_window(obs_times: Any, start: float, stop: float, dt: float) -> list[str]:
    """Reject observation times outside the model's ``[start, stop]`` window.

    The guard is explicit (not a reliance on ``numpy.interp``, which silently
    clamps out-of-range queries to the endpoints — that would be extrapolation).
    Warns when observations are denser than the simulation save step.
    """
    import numpy as np

    tol = abs(dt) * 1e-6
    if obs_times[0] < start - tol or obs_times[-1] > stop + tol:
        raise ValueError(
            f"observation times must lie within the model window [{start}, {stop}]; "
            "calibrate does not extrapolate"
        )
    warnings: list[str] = []
    if obs_times.size >= 2 and float(np.min(np.diff(obs_times))) < dt:
        warnings.append(
            "observations are denser than the simulation save step (dt); the fit "
            "interpolates the simulation onto observation times and may "
            "under-resolve the dynamics — consider a smaller dt"
        )
    return warnings


def _interp_onto(sim_times: Any, sim_values: Any, obs_times: Any) -> Any:
    """Linear-interpolate a simulated series onto the observation times."""
    import numpy as np

    return np.interp(obs_times, sim_times, sim_values)


def _residual_vector(
    results: Any, obs_times: Any, targets: list[tuple[str, float, Any]]
) -> Any | None:
    """Stack weighted ``w·(sim_interp − obs)`` across targets into a 1-D vector.

    ``targets`` is ``[(display, weight, obs_values)]``. Returns None when the
    run failed or a target column is missing, so the caller can apply a penalty.
    """
    import numpy as np

    if results is None:
        return None
    sim_times = np.asarray([float(t) for t in results.index], dtype=float)
    chunks: list[Any] = []
    for display, weight, obs_values in targets:
        if display not in results.columns:
            return None
        sim_values = np.asarray([float(v) for v in results[display]], dtype=float)
        chunks.append(weight * (_interp_onto(sim_times, sim_values, obs_times) - obs_values))
    return np.concatenate(chunks)


def _make_residual_fn(
    runner: Any,
    display_keys: list[str],
    obs_times: Any,
    targets: list[tuple[str, float, Any]],
    n_residuals: int,
):
    """Build the optimizer residual function over one compiled runner.

    Returns ``(residual_fn, state)``. ``state["last_feasible"]`` holds the most
    recent parameter vector that produced a finite simulation (None until one
    does); a non-finite trial returns a distance-aware penalty vector rather
    than NaN — a constant plateau would zero the Jacobian and fake convergence.
    """
    import numpy as np

    state: dict[str, Any] = {"last_feasible": None, "penalty_count": 0}

    def residual_fn(x: Any) -> Any:
        params = {display_keys[i]: float(x[i]) for i in range(len(display_keys))}
        try:
            results = runner.run(params=params)
        except Exception:  # any PySD failure on a bad trial -> penalty, not a crash
            results = None
        res = _residual_vector(results, obs_times, targets)
        if res is None or not np.all(np.isfinite(res)):
            state["penalty_count"] += 1
            last = state["last_feasible"]
            distance = (
                0.0 if last is None
                else float(np.linalg.norm(np.asarray(x, dtype=float) - last))
            )
            return np.full(n_residuals, _PENALTY_BASE * (1.0 + distance))
        state["last_feasible"] = np.asarray(x, dtype=float).copy()
        return res

    return residual_fn, state


def _sse(residuals: Any) -> float:
    """Sum of squared residuals (the scalar differential_evolution minimizes)."""
    import numpy as np

    return float(np.sum(np.asarray(residuals, dtype=float) ** 2))


def _variable_units(model: StellaModel, key: str) -> str:
    """Return the exact units string stored for a resolved model variable."""
    for variables in (model.stocks, model.flows, model.auxs):
        if key in variables:
            return variables[key].units
    raise ValueError(f"calibration target '{key}' matches no model variable")


def _target_fit_metrics(
    fit_results: Any, obs: dict[str, Any], model: StellaModel
) -> list[dict[str, Any]]:
    """Calculate unweighted best-fit errors in each target's native units."""
    import numpy as np

    sim_times = np.asarray([float(t) for t in fit_results.index], dtype=float)
    metrics: list[dict[str, Any]] = []
    for key, display, observed in obs["targets"]:
        if display not in fit_results.columns:
            raise ValueError(f"best-fit simulation is missing calibration target '{display}'")
        try:
            simulated = np.asarray([float(v) for v in fit_results[display]], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"best-fit simulation produced non-numeric values for target '{display}'"
            ) from exc
        fitted = _interp_onto(sim_times, simulated, obs["times"])
        if not np.all(np.isfinite(fitted)):
            raise ValueError(
                f"best-fit simulation produced non-finite values for target '{display}'"
            )
        residuals = fitted - observed
        sse = _sse(residuals)
        metrics.append({
            "name": display,
            "units": _variable_units(model, key),
            "n": int(residuals.size),
            "sse": sse,
            "rmse": math.sqrt(sse / int(residuals.size)),
        })
    return metrics


# === parameter setup + weights ===============================================

def _setup_parameters(
    model: StellaModel, parameters: list[dict[str, Any]], method: str
) -> tuple[list[str], list[str], list[float], list[float], list[float]]:
    """Validate the parameter specs and the eligibility gate; return aligned
    ``(names, display_keys, x0, lower, upper)`` vectors.

    Rejects stocks and non-constant equations (only constant auxes/flows are
    calibratable). ``initial`` defaults to the model's current constant value.
    Bounds are optional for least_squares and required for
    differential_evolution.
    """
    names: list[str] = []
    display_keys: list[str] = []
    x0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    seen: set[str] = set()

    for spec in parameters:
        if not isinstance(spec, dict):
            raise ValueError("each parameter must be an object")
        name = spec.get("name")
        if not name:
            raise ValueError("each parameter needs a 'name'")
        key = _resolve_key_or_raise(model, name)
        # Dedup on the resolved key, not the raw name: 'growth rate' and
        # 'growth_rate' alias the same PySD param, and trial params are keyed by
        # display name — two aliases would silently collapse to one dimension.
        if key in seen:
            raise ValueError(f"duplicate parameter '{name}' (aliases an already-listed variable)")
        seen.add(key)
        constant = constant_parameter_value(model, key)
        if constant is None:
            kind = "a stock" if key in model.stocks else "a non-constant variable"
            raise ValueError(
                f"parameter '{name}' is {kind} and is not calibratable: PySD pins "
                "it to a constant for the whole run rather than fitting an initial "
                "condition or structural equation. Only constant auxiliaries and "
                "flows can be calibrated."
            )

        initial = spec.get("initial")
        x0_i = (
            _require_finite(initial, f"parameter '{name}' initial")
            if initial is not None else constant
        )
        lo_raw, hi_raw = spec.get("min"), spec.get("max")
        lo = _require_finite(lo_raw, f"parameter '{name}' min") if lo_raw is not None else None
        hi = _require_finite(hi_raw, f"parameter '{name}' max") if hi_raw is not None else None
        if lo is not None and hi is not None and not lo < hi:
            raise ValueError(f"parameter '{name}': min must be < max")
        if (lo is not None and x0_i < lo) or (hi is not None and x0_i > hi):
            raise ValueError(f"parameter '{name}': initial {x0_i} is outside [min, max]")
        if method == "differential_evolution" and (lo is None or hi is None):
            raise ValueError(
                f"parameter '{name}': differential_evolution requires finite "
                "min and max bounds"
            )

        names.append(name)
        display_keys.append(model._display_name(key))
        x0.append(x0_i)
        lower.append(lo if lo is not None else -math.inf)
        upper.append(hi if hi is not None else math.inf)

    return names, display_keys, x0, lower, upper


def _resolve_weights(
    model: StellaModel, weights: Any, displays: list[str]
) -> list[float]:
    """Per-target weight list aligned with ``displays`` (default 1.0). Weights
    must be positive and key existing observation targets."""
    if weights is None:
        return [1.0] * len(displays)
    if not isinstance(weights, dict):
        raise ValueError("weights must be an object mapping target name -> positive number")
    by_display: dict[str, float] = {}
    for name, value in weights.items():
        key = _resolve_key(model, name)
        display = model._display_name(key) if key is not None else None
        if display is None or display not in displays:
            raise ValueError(f"weights key '{name}' is not one of the observation targets")
        numeric = _require_finite(value, f"weight '{name}'")
        if numeric <= 0:
            raise ValueError(f"weight '{name}' must be positive")
        by_display[display] = numeric
    return [by_display.get(display, 1.0) for display in displays]


def _multi_scale_warning(targets: list[tuple[str, str, Any]], weights: Any) -> list[str]:
    """Warn when targets span more than an order of magnitude with no weights —
    the unweighted fit would be dominated by the larger-scale target."""
    if weights is not None:
        return []
    import numpy as np

    scales = []
    for _key, _display, values in targets:
        finite = values[np.isfinite(values)]
        mean_abs = float(np.mean(np.abs(finite))) if finite.size else 0.0
        if mean_abs > 0:
            scales.append(mean_abs)
    if len(scales) >= 2 and max(scales) / min(scales) > _SCALE_WARN_RATIO:
        return [
            "observation targets span more than an order of magnitude and no "
            "weights were given; the unweighted fit may be dominated by the "
            "larger-scale target — consider per-target weights"
        ]
    return []


# === uncertainty + bounds ====================================================

def _bound_json(value: float) -> float | None:
    """JSON-safe bound (±inf -> None)."""
    return None if math.isinf(value) else float(value)


def _at_bounds_mask(x: Any, lower: list[float], upper: list[float]) -> list[bool]:
    """Per-parameter flag: is the fitted value sitting on a finite bound?"""
    mask: list[bool] = []
    for xi, lo, hi in zip(x, lower, upper, strict=True):
        at = (
            (math.isfinite(lo) and math.isclose(float(xi), lo, rel_tol=_AT_BOUND_RTOL, abs_tol=1e-12))
            or (math.isfinite(hi) and math.isclose(float(xi), hi, rel_tol=_AT_BOUND_RTOL, abs_tol=1e-12))
        )
        mask.append(at)
    return mask


def _least_squares_std_error(
    result: Any, m: int, n: int, at_bounds: list[bool]
) -> tuple[list[float] | None, list[str]]:
    """Linearized standard errors from the residual Jacobian, or None + reason.

    ``cov = (SSE/(m−n))·(JᵀJ)⁻¹`` with ``SSE = 2·result.cost``; valid only when
    ``m > n``, no parameter is at a bound, and ``JᵀJ`` is well-conditioned.
    """
    import numpy as np

    if m <= n:
        return None, ["std_error unavailable: not enough observations (m <= n parameters)"]
    if any(at_bounds):
        return None, [
            "std_error unavailable: a fitted parameter is at a bound (an active "
            "constraint removes a degree of freedom)"
        ]
    jac = np.asarray(result.jac, dtype=float)
    jtj = jac.T @ jac
    try:
        cond = float(np.linalg.cond(jtj))
    except np.linalg.LinAlgError:
        cond = math.inf
    if not math.isfinite(cond) or cond > _COND_LIMIT:
        return None, [
            "std_error unavailable: Jacobian is rank-deficient or ill-conditioned "
            "(parameters may be correlated/unidentifiable)"
        ]
    sse = float(2.0 * result.cost)
    cov = (sse / (m - n)) * np.linalg.inv(jtj)
    diag = np.diag(cov)
    if np.any(diag < 0):
        return None, ["std_error unavailable: negative covariance diagonal (ill-conditioned fit)"]
    return [float(np.sqrt(d)) for d in diag], []


def _de_rng_kwarg(seed: int) -> dict[str, int]:
    """Pass the seed via scipy's current ``rng`` parameter where available,
    falling back to the legacy ``seed`` on older scipy (deprecation-safe)."""
    import inspect

    from scipy import optimize

    params = inspect.signature(optimize.differential_evolution).parameters
    return {"rng": seed} if "rng" in params else {"seed": seed}


def _write_fit_csv(
    path: str, obs: dict[str, Any], fit_results: Any
) -> None:
    """Long table: time, target, observed, fitted (fitted = best-fit sim
    interpolated onto the observation times)."""
    import numpy as np

    obs_times = obs["times"]
    sim_times = np.asarray([float(t) for t in fit_results.index], dtype=float)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "target", "observed", "fitted"])
        for _key, display, values in obs["targets"]:
            if display in fit_results.columns:
                sim_values = np.asarray([float(v) for v in fit_results[display]], dtype=float)
                fitted = _interp_onto(sim_times, sim_values, obs_times)
            else:
                fitted = [None] * len(obs_times)
            for t, observed, fit_value in zip(obs_times, values, fitted, strict=True):
                writer.writerow([
                    float(t), display, float(observed),
                    None if fit_value is None else float(fit_value),
                ])


# === public entry point ======================================================

def calibrate(
    model: StellaModel,
    observations: Any,
    parameters: list[dict[str, Any]],
    method: str = "least_squares",
    objective: str = "sse",
    weights: Any = None,
    max_nfev: int = _DEFAULT_MAX_NFEV,
    maxiter: int | None = None,
    popsize: int = _DEFAULT_POPSIZE,
    seed: int = SEED,
    return_fit_series: bool = False,
    save_fit_csv: str | None = None,
) -> dict[str, Any]:
    """Fit constant model parameters to an observed time-series.

    Parameters
    ----------
    model : StellaModel
        Session model; never mutated (the runner works on a deep copy).
    observations : dict
        ``{"time": [...], "targets": {var: [...]}}`` (one shared time grid) or
        ``{"csv_path": "..."}``. Observation times must lie within the model's
        ``[start, stop]`` window (no extrapolation).
    parameters : list of dict
        Each ``{"name", "initial"?, "min"?, "max"?}``. Only constant auxiliaries
        and flows are calibratable; stocks and non-constant equations are
        rejected. ``initial`` defaults to the model's current constant value.
    method : str
        ``"least_squares"`` (local, default; yields a linearized ``std_error``)
        or ``"differential_evolution"`` (global, stochastic, requires bounds).
    objective : str
        Only ``"sse"`` is supported; use ``weights`` for per-target scaling.
    weights : dict, optional
        Per-target positive residual multipliers ``w·(sim − obs)``. Values
        equal to inverse measurement standard deviation give normalized
        residuals and the usual statistical ``std_error`` interpretation.
    max_nfev : int
        least_squares function-evaluation cap.
    maxiter, popsize : int
        differential_evolution budget (``maxiter`` defaults to 100;
        ``popsize`` is DE's population multiplier).
    seed : int
        differential_evolution seed (must not be None — keeps DE reproducible).
    return_fit_series : bool
        Also return the best-fit downsampled series for each target.
    save_fit_csv : str, optional
        Write a long (time, target, observed, fitted) CSV.

    Returns the fitted parameters (with bounds, ``at_bound`` flags, and
    linearized ``std_error``), the objective trajectory, convergence state, and
    warnings.
    """
    ensure_supported_for_simulation(model.xmile_feature_report)

    import numpy as np
    from scipy import optimize

    if method not in _METHODS:
        raise ValueError(f"method '{method}' must be one of {sorted(_METHODS)}")
    if objective != "sse":
        raise ValueError(f"objective '{objective}' not supported; only 'sse' is available")
    if not isinstance(parameters, list) or not parameters:
        raise ValueError("calibrate requires at least one parameter (a non-empty array)")
    max_nfev = _positive_int(max_nfev, "max_nfev")
    maxiter = None if maxiter is None else _positive_int(maxiter, "maxiter")
    popsize = _positive_int(popsize, "popsize")
    seed = _seed_int(seed)

    obs = _load_observations(observations, model)
    specs = model.sim_specs
    warnings_out = method_warnings(model)
    warnings_out += _check_obs_window(obs["times"], specs.start, specs.stop, specs.dt)
    warnings_out += _multi_scale_warning(obs["targets"], weights)

    names, display_keys, x0_list, lower, upper = _setup_parameters(model, parameters, method)
    target_displays = [display for _key, display, _values in obs["targets"]]
    weight_list = _resolve_weights(model, weights, target_displays)
    residual_targets = [
        (display, weight_list[i], values)
        for i, (_key, display, values) in enumerate(obs["targets"])
    ]
    n_residuals = len(obs["targets"]) * int(obs["times"].size)
    x0 = np.asarray(x0_list, dtype=float)
    n_params = len(x0_list)

    with _compile_runner(model) as runner:
        residual_fn, state = _make_residual_fn(
            runner, display_keys, obs["times"], residual_targets, n_residuals
        )
        r0 = residual_fn(x0)
        if state["last_feasible"] is None:
            raise ValueError(
                "the initial parameter guess produces a non-finite simulation; "
                "provide an `initial` (or bounds) inside the model's valid region"
            )
        sse_initial = _sse(r0)

        if method == "least_squares":
            result = optimize.least_squares(
                residual_fn, x0, bounds=(lower, upper), max_nfev=max_nfev
            )
            x_fit = np.asarray(result.x, dtype=float)
            nfev = int(result.nfev)
            converged = bool(result.status > 0)
            optimizer_status: int | bool = int(result.status)
            optimizer_message = str(result.message)
            optimizer_config = {
                "max_nfev": max_nfev,
                "maxiter": None,
                "popsize": None,
                "seed": None,
            }
            sse_final = float(2.0 * result.cost)
            at_bounds = _at_bounds_mask(x_fit, lower, upper)
            std_error, cov_warnings = _least_squares_std_error(
                result, n_residuals, n_params, at_bounds
            )
            warnings_out += cov_warnings
        else:  # differential_evolution
            bounds = list(zip(lower, upper, strict=True))
            maxiter_eff = maxiter if maxiter is not None else _DEFAULT_DE_MAXITER
            result = optimize.differential_evolution(
                lambda x: _sse(residual_fn(x)),
                bounds,
                maxiter=maxiter_eff,
                popsize=popsize,
                **_de_rng_kwarg(seed),
            )
            x_fit = np.asarray(result.x, dtype=float)
            nfev = int(result.nfev)
            converged = bool(result.success)
            optimizer_status = bool(result.success)
            optimizer_message = str(result.message)
            optimizer_config = {
                "max_nfev": None,
                "maxiter": maxiter_eff,
                "popsize": popsize,
                "seed": seed,
            }
            sse_final = float(result.fun)
            at_bounds = _at_bounds_mask(x_fit, lower, upper)
            std_error = None

        fit_params = {display_keys[i]: float(x_fit[i]) for i in range(n_params)}
        try:
            fit_results = runner.run(params=fit_params)
        except Exception as exc:
            raise ValueError("best-fit parameter set could not be simulated") from exc
        target_metrics = _target_fit_metrics(fit_results, obs, model)
        fit_series = None
        if return_fit_series:
            report_keys = [key for key, _display, _values in obs["targets"]]
            fit_series, fit_warnings = summarize_run(
                fit_results, report_keys, model, DEFAULT_MAX_POINTS
            )
            warnings_out += fit_warnings

    if save_fit_csv:
        _write_fit_csv(save_fit_csv, obs, fit_results)

    if state["penalty_count"]:
        warnings_out.append(
            f"{state['penalty_count']} trial simulation(s) produced non-finite "
            "residuals during the fit and were penalized"
        )
    if weights is not None and std_error is not None:
        warnings_out.append(
            "std_error is conditioned on the given weights; it is a true standard "
            "error only when the weights are inverse-sigma (measurement-error) scale"
        )

    parameters_payload: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        if at_bounds[i]:
            warnings_out.append(f"parameter '{name}' converged to a bound")
        parameters_payload.append({
            "name": name,
            "initial": x0_list[i],
            "fitted": float(x_fit[i]),
            "std_error": std_error[i] if std_error is not None else None,
            "bounds": [_bound_json(lower[i]), _bound_json(upper[i])],
            "at_bound": at_bounds[i],
        })

    return {
        "backend": simulation_backend_metadata(model),
        "objective": {
            "metric": "weighted_sse",
            "initial": sse_initial,
            "final": sse_final,
            "weighted_rmse": math.sqrt(sse_final / n_residuals),
        },
        "optimizer": {
            "method": method,
            "converged": converged,
            "status": optimizer_status,
            "message": optimizer_message,
            "n_function_evals": nfev,
            "config": optimizer_config,
        },
        "parameters": parameters_payload,
        "targets": target_displays,
        "target_metrics": target_metrics,
        "n_observations": n_residuals,
        "time_units": specs.time_units,
        "fit_series": fit_series,
        "warnings": warnings_out,
        "csv_path": save_fit_csv,
    }
