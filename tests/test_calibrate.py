"""Tests for parameter calibration (pysd-backed).

The whole module is skipped when pysd is not installed, matching
test_simulate.py / test_analysis.py; the sim stack (numpy/scipy/pandas) arrives
with pysd.
"""

import math

import pytest

pysd = pytest.importorskip("pysd")

import numpy as np  # noqa: E402

from stella_mcp.calibrate import (  # noqa: E402
    _check_obs_window,
    _interp_onto,
    _load_observations,
    _make_residual_fn,
    _residual_vector,
    calibrate,
)
from stella_mcp.simulate import _resolve_key, constant_parameter_value  # noqa: E402
from stella_mcp.xmile import StellaModel  # noqa: E402


def _accumulator_model() -> StellaModel:
    """Linear integrator: Accumulator(stop) = rate * stop, so the final value is
    exactly linear in `rate` (slope == stop). Ideal for truth recovery."""
    model = StellaModel("Accumulator")
    model.sim_specs.start, model.sim_specs.stop, model.sim_specs.dt = 0.0, 10.0, 1.0
    model.add_stock("Accumulator", "0")
    model.add_aux("rate", "1")
    model.add_flow("inflow", "rate", to_stock="Accumulator")
    return model


# --- eligibility primitive (BLOCKER fix) -------------------------------------

def test_constant_parameter_value_classifies_variables():
    model = _accumulator_model()
    # constant aux -> its value
    assert constant_parameter_value(model, _resolve_key(model, "rate")) == 1.0
    # stock -> None (pinning a stock would flatten it; not calibratable)
    assert constant_parameter_value(model, _resolve_key(model, "Accumulator")) is None
    # non-constant flow equation ("rate" references an aux) -> None
    assert constant_parameter_value(model, _resolve_key(model, "inflow")) is None
    # a genuinely non-constant aux -> None
    model.add_aux("derived", "rate * 2")
    assert constant_parameter_value(model, _resolve_key(model, "derived")) is None
    # unknown key
    assert constant_parameter_value(model, None) is None


# --- observation loading -----------------------------------------------------

def test_inline_and_csv_observations_match(tmp_path):
    model = _accumulator_model()
    inline = _load_observations(
        {"time": [0, 1, 2], "targets": {"Accumulator": [0.0, 1.0, 2.0]}}, model
    )
    csv_path = tmp_path / "obs.csv"
    csv_path.write_text("time,Accumulator\n0,0.0\n1,1.0\n2,2.0\n", encoding="utf-8")
    from_csv = _load_observations({"csv_path": str(csv_path)}, model)

    assert np.array_equal(inline["times"], from_csv["times"])
    assert [d for _, d, _ in inline["targets"]] == [d for _, d, _ in from_csv["targets"]]
    assert np.array_equal(inline["targets"][0][2], from_csv["targets"][0][2])


def test_observation_target_typo_raises_naming_valid():
    with pytest.raises(ValueError, match="matches no model variable"):
        _load_observations(
            {"time": [0, 1], "targets": {"Acumulator": [0.0, 1.0]}}, _accumulator_model()
        )


def test_observation_validation_guards():
    model = _accumulator_model()
    with pytest.raises(ValueError, match="strictly increasing"):
        _load_observations({"time": [0, 1, 1], "targets": {"Accumulator": [0, 1, 2]}}, model)
    with pytest.raises(ValueError, match="time points"):
        _load_observations({"time": [0, 1, 2], "targets": {"Accumulator": [0, 1]}}, model)
    with pytest.raises(ValueError, match="finite"):
        _load_observations(
            {"time": [0, 1, 2], "targets": {"Accumulator": [0, float("nan"), 2]}}, model
        )
    with pytest.raises(ValueError, match="at least 2"):
        _load_observations({"time": [0], "targets": {"Accumulator": [0]}}, model)


# --- window + interpolation --------------------------------------------------

def test_window_boundary_and_extrapolation_guard():
    times = np.asarray([0.0, 5.0, 10.0])
    # exactly at the window edges -> ok (no extrapolation), returns a list
    assert _check_obs_window(times, 0.0, 10.0, 1.0) == []
    # just past final_time -> raises (numpy.interp would silently clamp)
    with pytest.raises(ValueError, match="extrapolate"):
        _check_obs_window(np.asarray([0.0, 10.0001]), 0.0, 10.0, 1.0)
    with pytest.raises(ValueError, match="extrapolate"):
        _check_obs_window(np.asarray([-0.5, 5.0]), 0.0, 10.0, 1.0)


def test_window_warns_when_obs_denser_than_dt():
    warnings = _check_obs_window(np.asarray([0.0, 0.1, 0.2]), 0.0, 10.0, 1.0)
    assert any("denser" in w for w in warnings)


def test_interp_onto_exact_grid_and_midpoint():
    sim_times = np.asarray([0.0, 1.0, 2.0])
    sim_values = np.asarray([10.0, 20.0, 30.0])
    # exact grid reproduces values
    assert np.array_equal(_interp_onto(sim_times, sim_values, sim_times), sim_values)
    # midpoint linearly interpolated
    assert _interp_onto(sim_times, sim_values, np.asarray([0.5])).tolist() == [15.0]


# --- residual builder + penalty ----------------------------------------------

class _StubRunner:
    """Minimal pysd-runner stand-in: returns a finite frame unless x[0] < 0,
    in which case it injects NaN (an infeasible trial)."""

    def __init__(self, times):
        self._times = times

    def run(self, params):
        import pandas as pd

        x0 = next(iter(params.values()))
        col = np.full(len(self._times), float("nan")) if x0 < 0 else np.asarray(self._times, dtype=float)
        return pd.DataFrame({"X": col}, index=self._times)


def test_residual_vector_weights_and_missing_column():
    import pandas as pd

    times = np.asarray([0.0, 1.0, 2.0])
    frame = pd.DataFrame({"X": [0.0, 1.0, 2.0]}, index=times)
    obs = np.asarray([0.0, 0.5, 1.0])
    res = _residual_vector(frame, times, [("X", 2.0, obs)])
    # weight 2 * (sim - obs) = 2 * (0, 0.5, 1.0)
    assert np.allclose(res, [0.0, 1.0, 2.0])
    # missing target column -> None (caller penalizes)
    assert _residual_vector(frame, times, [("Y", 1.0, obs)]) is None


def test_penalty_is_distance_aware_not_flat():
    times = [0.0, 1.0, 2.0]
    obs = np.asarray([0.0, 1.0, 2.0])
    targets = [("X", 1.0, obs)]
    fn, state = _make_residual_fn(_StubRunner(times), ["X"], np.asarray(times), targets, 3)

    feasible = fn(np.asarray([1.0]))  # finite run -> sets last_feasible
    assert np.all(np.isfinite(feasible))
    assert state["last_feasible"] is not None

    near = fn(np.asarray([-1.0]))  # infeasible, distance 2 from last feasible
    far = fn(np.asarray([-5.0]))   # infeasible, larger distance
    assert np.all(np.isfinite(near)) and np.all(np.isfinite(far))
    assert float(np.linalg.norm(far)) > float(np.linalg.norm(near))  # grows with distance
    assert state["penalty_count"] == 2


# === calibrate core ==========================================================

def _obs_for_rate(truth: float, times=(0, 2, 4, 6, 8, 10)) -> dict:
    """Analytic observed Accumulator series: Euler integral of a constant rate
    gives Accumulator(t) = rate * t exactly at dt=1."""
    return {"time": list(times), "targets": {"Accumulator": [truth * t for t in times]}}


def test_recovers_truth_least_squares():
    result = calibrate(
        _accumulator_model(), _obs_for_rate(3.0), [{"name": "rate", "initial": 1.0}]
    )
    assert result["converged"]
    fitted = result["parameters"][0]["fitted"]
    assert math.isclose(fitted, 3.0, rel_tol=1e-4)
    assert result["objective"]["final"] < 1e-6  # near-perfect fit
    assert result["parameters"][0]["std_error"] is not None  # well-posed


def test_recovers_truth_differential_evolution():
    result = calibrate(
        _accumulator_model(),
        _obs_for_rate(3.0),
        [{"name": "rate", "min": 0.0, "max": 10.0}],
        method="differential_evolution",
        seed=0,
    )
    assert math.isclose(result["parameters"][0]["fitted"], 3.0, rel_tol=1e-3)
    assert result["parameters"][0]["std_error"] is None  # DE has no Jacobian


def test_stock_parameter_rejected():
    with pytest.raises(ValueError, match="not calibratable"):
        calibrate(
            _accumulator_model(),
            {"time": [0, 1], "targets": {"Accumulator": [0, 1]}},
            [{"name": "Accumulator"}],  # a stock
        )


def test_infeasible_initial_guess_raises_not_false_converge():
    model = StellaModel("Div")
    model.sim_specs.start, model.sim_specs.stop, model.sim_specs.dt = 0.0, 5.0, 1.0
    model.add_stock("X", "1")
    model.add_aux("denom", "0")  # zero -> 1/denom is non-finite
    model.add_flow("inflow", "1 / denom", to_stock="X")
    with pytest.raises(ValueError, match="non-finite"):
        calibrate(model, {"time": [0, 1, 2], "targets": {"X": [1, 1, 1]}},
                  [{"name": "denom", "initial": 0.0}])


def _two_param_model() -> StellaModel:
    model = StellaModel("Two")
    model.sim_specs.start, model.sim_specs.stop, model.sim_specs.dt = 0.0, 10.0, 1.0
    model.add_stock("Acc1", "0")
    model.add_stock("Acc2", "0")
    model.add_aux("rate1", "1")
    model.add_aux("rate2", "1")
    model.add_flow("in1", "rate1", to_stock="Acc1")
    model.add_flow("in2", "rate2", to_stock="Acc2")
    return model


def test_multi_target_recovery_with_weights():
    times = [0, 2, 4, 6, 8, 10]
    obs = {"time": times, "targets": {
        "Acc1": [2.0 * t for t in times],
        "Acc2": [5.0 * t for t in times],
    }}
    result = calibrate(
        _two_param_model(), obs,
        [{"name": "rate1", "initial": 1.0}, {"name": "rate2", "initial": 1.0}],
        weights={"Acc1": 1.0, "Acc2": 1.0},
    )
    fitted = {p["name"]: p["fitted"] for p in result["parameters"]}
    assert math.isclose(fitted["rate1"], 2.0, rel_tol=1e-3)
    assert math.isclose(fitted["rate2"], 5.0, rel_tol=1e-3)


def test_multi_scale_targets_warn_without_weights():
    times = [0, 2, 4, 6, 8, 10]
    obs = {"time": times, "targets": {
        "Acc1": [1.0 * t for t in times],       # ~10
        "Acc2": [1000.0 * t for t in times],    # ~10000
    }}
    result = calibrate(
        _two_param_model(), obs,
        [{"name": "rate1", "initial": 1.0}, {"name": "rate2", "initial": 1.0}],
    )
    assert any("order of magnitude" in w for w in result["warnings"])


def test_parameter_pinned_at_bound():
    # truth rate is 3 but max is 2 -> fit pinned at the bound
    result = calibrate(
        _accumulator_model(), _obs_for_rate(3.0),
        [{"name": "rate", "initial": 1.0, "min": 0.0, "max": 2.0}],
    )
    param = result["parameters"][0]
    assert param["at_bound"] is True
    assert math.isclose(param["fitted"], 2.0, rel_tol=1e-3)
    assert param["std_error"] is None  # active bound removes a DOF
    assert any("bound" in w for w in result["warnings"])


def test_max_nfev_exhaustion_reports_not_converged():
    result = calibrate(
        _accumulator_model(), _obs_for_rate(3.0),
        [{"name": "rate", "initial": 1.0}], max_nfev=1,
    )
    assert result["converged"] is False  # did not raise


def test_de_honors_maxiter_budget():
    result = calibrate(
        _accumulator_model(), _obs_for_rate(3.0),
        [{"name": "rate", "min": 0.0, "max": 10.0}],
        method="differential_evolution", maxiter=2, popsize=5, seed=0,
    )
    assert result["n_function_evals"] > 0


def test_bound_and_method_guards():
    model = _accumulator_model()
    obs = _obs_for_rate(3.0)
    with pytest.raises(ValueError, match="requires finite"):
        calibrate(model, obs, [{"name": "rate", "min": 0.0}], method="differential_evolution")
    with pytest.raises(ValueError, match="min must be < max"):
        calibrate(model, obs, [{"name": "rate", "min": 5.0, "max": 5.0}])
    with pytest.raises(ValueError, match="outside"):
        calibrate(model, obs, [{"name": "rate", "initial": 10.0, "min": 0.0, "max": 5.0}])
    with pytest.raises(ValueError, match="method"):
        calibrate(model, obs, [{"name": "rate"}], method="bogus")
    with pytest.raises(ValueError, match="objective"):
        calibrate(model, obs, [{"name": "rate"}], objective="mae")
    with pytest.raises(ValueError, match="seed"):
        calibrate(model, obs, [{"name": "rate"}], seed=None)
    with pytest.raises(ValueError, match="at least one parameter"):
        calibrate(model, obs, [])
    with pytest.raises(ValueError, match="duplicate parameter"):
        calibrate(model, obs, [{"name": "rate"}, {"name": "rate"}])
