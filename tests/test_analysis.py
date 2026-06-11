"""Tests for scenario comparison and sensitivity analysis (pysd-backed).

The whole module is skipped when pysd is not installed, matching
test_simulate.py; the dependency-missing path is covered elsewhere.
"""

import asyncio
import math

import pytest

pysd = pytest.importorskip("pysd")

from stella_mcp import server as server_mod  # noqa: E402
from stella_mcp.analysis import (  # noqa: E402
    _reduce_metric,
    compare_scenarios,
    sensitivity_analysis,
)
from stella_mcp.xmile import StellaModel  # noqa: E402


def _growth_model() -> StellaModel:
    """Exponential growth: Population_{n+1} = Population_n * (1 + dt*rate)."""
    model = StellaModel("Growth")
    model.sim_specs.start, model.sim_specs.stop, model.sim_specs.dt = 0.0, 10.0, 0.25
    model.add_stock("Population", "100")
    model.add_aux("rate", "0.1")
    model.add_flow("growth", "Population * rate", to_stock="Population")
    return model


def _final(series: list[dict], name: str) -> float:
    [match] = [s for s in series if s["name"] == name]
    return match["summary"]["final"]


# --- core comparison ---------------------------------------------------------

def test_scenarios_differ_and_deltas_match_baseline():
    result = compare_scenarios(
        _growth_model(),
        scenarios=[
            {"name": "low", "overrides": {"rate": 0.1}},
            {"name": "high", "overrides": {"rate": 0.2}},
        ],
        include=["Population"],
    )
    base_final = _final(result["baseline"]["series"], "Population")  # default rate 0.1
    high = next(s for s in result["scenarios"] if s["name"] == "high")
    low = next(s for s in result["scenarios"] if s["name"] == "low")
    high_final = _final(high["series"], "Population")

    assert high_final > base_final > 0  # higher rate -> larger population
    # delta_abs is exactly scenario_final - baseline_final
    assert math.isclose(
        high["delta_vs_baseline"]["Population"]["final_abs"],
        high_final - base_final,
        rel_tol=1e-9,
    )
    assert high["delta_vs_baseline"]["Population"]["final_pct"] > 0
    # 'low' equals the model default -> ~zero delta
    assert abs(low["delta_vs_baseline"]["Population"]["final_abs"]) < 1e-6


def test_baseline_defaults_to_unmodified_model():
    result = compare_scenarios(
        _growth_model(),
        scenarios=[{"name": "same", "overrides": {"rate": 0.1}}],
        include=["Population"],
    )
    assert result["baseline"]["overrides"] == {}
    assert abs(result["scenarios"][0]["delta_vs_baseline"]["Population"]["final_abs"]) < 1e-6


def test_explicit_baseline_shifts_the_reference():
    result = compare_scenarios(
        _growth_model(),
        scenarios=[{"name": "high", "overrides": {"rate": 0.2}}],
        baseline={"rate": 0.1},
        include=["Population"],
    )
    assert result["baseline"]["overrides"] == {"rate": 0.1}
    assert result["scenarios"][0]["delta_vs_baseline"]["Population"]["final_abs"] > 0


# --- fail-fast validation ----------------------------------------------------

def test_override_typo_raises_naming_scenario():
    with pytest.raises(ValueError) as exc:
        compare_scenarios(
            _growth_model(),
            scenarios=[
                {"name": "ok", "overrides": {"rate": 0.1}},
                {"name": "bad", "overrides": {"raat": 0.2}},  # typo, second scenario
            ],
        )
    message = str(exc.value)
    assert "bad" in message and "raat" in message


def test_duplicate_scenario_names_raise():
    with pytest.raises(ValueError, match="duplicate scenario name"):
        compare_scenarios(
            _growth_model(),
            scenarios=[
                {"name": "x", "overrides": {"rate": 0.1}},
                {"name": "x", "overrides": {"rate": 0.2}},
            ],
        )


def test_empty_scenarios_raise():
    with pytest.raises(ValueError, match="at least one scenario"):
        compare_scenarios(_growth_model(), scenarios=[])


def test_final_pct_none_when_baseline_final_zero():
    """A stock that stays at 0 -> baseline final 0 -> final_pct is None (no
    divide-by-zero), final_abs still reported."""
    model = StellaModel("Zero")
    model.sim_specs.start, model.sim_specs.stop, model.sim_specs.dt = 0.0, 5.0, 1.0
    model.add_stock("Empty", "0")
    model.add_aux("rate", "0")
    model.add_flow("inflow", "rate", to_stock="Empty")
    result = compare_scenarios(
        model,
        scenarios=[{"name": "still_zero", "overrides": {"rate": 0}}],
        include=["Empty"],
    )
    delta = result["scenarios"][0]["delta_vs_baseline"]["Empty"]
    assert delta["final_abs"] == 0
    assert delta["final_pct"] is None


# --- CSV + tool wiring -------------------------------------------------------

def test_comparison_csv_has_per_scenario_columns(tmp_path):
    csv_path = tmp_path / "cmp.csv"
    compare_scenarios(
        _growth_model(),
        scenarios=[{"name": "high", "overrides": {"rate": 0.2}}],
        include=["Population"],
        save_comparison_csv=str(csv_path),
    )
    assert csv_path.exists()
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "Population__baseline" in header
    assert "Population__high" in header


def test_compare_scenarios_tool(monkeypatch):
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 7001)
    asyncio.run(server_mod.call_tool("build_model", {
        "name": "Growth", "model_id": "g",
        "sim_specs": {"start": 0, "stop": 10, "dt": 0.25},
        "stocks": [{"name": "Population", "initial_value": "100"}],
        "auxs": [{"name": "rate", "equation": "0.1"}],
        "flows": [{"name": "growth", "equation": "Population * rate", "to_stock": "Population"}],
    }))
    result = asyncio.run(server_mod.call_tool("compare_scenarios", {
        "model_id": "g",
        "scenarios": [{"name": "high", "overrides": {"rate": 0.3}}],
        "include": ["Population"],
    }))
    assert not result.isError
    sc = result.structuredContent
    assert sc["model_id"] == "g"
    assert sc["scenarios"][0]["name"] == "high"
    assert sc["scenarios"][0]["delta_vs_baseline"]["Population"]["final_abs"] > 0


# === sensitivity analysis ====================================================

def _accumulator_model() -> StellaModel:
    """Linear integrator: Accumulator(stop) = rate * stop, so the final value
    is exactly linear in `rate` (slope == stop)."""
    model = StellaModel("Accumulator")
    model.sim_specs.start, model.sim_specs.stop, model.sim_specs.dt = 0.0, 10.0, 1.0
    model.add_stock("Accumulator", "0")
    model.add_aux("rate", "1")
    model.add_flow("inflow", "rate", to_stock="Accumulator")
    return model


def test_sensitivity_linear_slope_and_elasticity():
    result = sensitivity_analysis(
        _accumulator_model(),
        parameters=[{"name": "rate", "start": 1, "stop": 5, "steps": 5}],
        output={"variable": "Accumulator", "metric": "final"},
    )
    assert result["total_runs"] == 5
    param = result["parameters"][0]
    metrics = [pt["metric"] for pt in param["points"]]
    assert metrics == sorted(metrics)  # monotonic in rate
    # range_sensitivity is exactly the endpoint slope, and the physics gives 10.
    assert math.isclose(param["range_sensitivity"], (metrics[-1] - metrics[0]) / 4, rel_tol=1e-9)
    assert math.isclose(param["range_sensitivity"], 10.0, rel_tol=1e-3)
    # baseline rate is 1 -> final 10; elasticity = slope * p0/m0 = 10 * 1/10 = 1.
    assert math.isclose(result["baseline"]["metric_value"], 10.0, rel_tol=1e-3)
    assert math.isclose(param["elasticity"], 1.0, rel_tol=1e-3)


def test_metric_reducers():
    times = [0, 1, 2, 3, 4]
    values = [1.0, 5.0, 3.0, 2.0, 4.0]
    assert _reduce_metric(times, values, "final", None) == 4.0
    assert _reduce_metric(times, values, "max", None) == 5.0
    assert _reduce_metric(times, values, "min", None) == 1.0
    assert math.isclose(_reduce_metric(times, values, "mean", None), 3.0)


def test_metric_reducers_skip_non_finite():
    nan = float("nan")
    assert _reduce_metric([0, 1, 2], [1.0, nan, 3.0], "max", None) == 3.0  # NaN skipped
    assert _reduce_metric([0, 1, 2], [1.0, nan, 3.0], "mean", None) == 2.0
    assert _reduce_metric([0, 1], [nan, nan], "max", None) is None  # all-NaN
    assert _reduce_metric([0, 1], [1.0, nan], "final", None) is None  # final non-finite


def test_time_to_threshold():
    assert _reduce_metric([0, 1, 2, 3, 4], [0.0, 2.0, 5.0, 8.0, 10.0],
                          "time_to_threshold", 5.0) == 2.0  # first >= 5
    assert _reduce_metric([0, 1, 2, 3, 4], [0.0, 2.0, 5.0, 8.0, 10.0],
                          "time_to_threshold", 100.0) is None  # never crossed
    assert _reduce_metric([0, 1, 2, 3], [10.0, 8.0, 4.0, 1.0],
                          "time_to_threshold", 5.0) == 2.0  # falling, first <= 5


def test_time_to_threshold_requires_threshold():
    with pytest.raises(ValueError, match="threshold"):
        sensitivity_analysis(
            _accumulator_model(),
            parameters=[{"name": "rate", "values": [1, 2]}],
            output={"variable": "Accumulator", "metric": "time_to_threshold"},
        )


def test_max_runs_guard_raises():
    with pytest.raises(ValueError, match="max_runs"):
        sensitivity_analysis(
            _accumulator_model(),
            parameters=[{"name": "rate", "start": 1, "stop": 10, "steps": 50}],
            output={"variable": "Accumulator", "metric": "final"},
            max_runs=10,
        )


def test_invalid_sweep_specs_raise():
    model = _accumulator_model()
    output = {"variable": "Accumulator", "metric": "final"}
    with pytest.raises(ValueError, match="steps"):
        sensitivity_analysis(model, parameters=[{"name": "rate", "start": 1, "stop": 5, "steps": 1}], output=output)
    with pytest.raises(ValueError, match="differ"):
        sensitivity_analysis(model, parameters=[{"name": "rate", "start": 3, "stop": 3, "steps": 4}], output=output)
    with pytest.raises(ValueError, match="at least 2"):
        sensitivity_analysis(model, parameters=[{"name": "rate", "values": [1]}], output=output)


def test_sensitivity_validation_guards():
    model = _accumulator_model()
    output = {"variable": "Accumulator"}
    with pytest.raises(ValueError, match="at least one parameter"):
        sensitivity_analysis(model, parameters=[], output=output)
    with pytest.raises(ValueError, match="oat"):
        sensitivity_analysis(model, parameters=[{"name": "rate", "values": [1, 2]}], output=output, mode="grid")
    with pytest.raises(ValueError, match="matches no model variable"):
        sensitivity_analysis(model, parameters=[{"name": "rate", "values": [1, 2]}], output={"variable": "Nope"})
    with pytest.raises(ValueError, match="metric"):
        sensitivity_analysis(model, parameters=[{"name": "rate", "values": [1, 2]}],
                             output={"variable": "Accumulator", "metric": "bogus"})
    with pytest.raises(ValueError, match="matches no model variable"):
        sensitivity_analysis(model, parameters=[{"name": "raat", "values": [1, 2]}], output=output)


def test_elasticity_none_when_baseline_metric_zero():
    model = StellaModel("Zero")
    model.sim_specs.start, model.sim_specs.stop, model.sim_specs.dt = 0.0, 10.0, 1.0
    model.add_stock("Acc", "0")
    model.add_aux("rate", "0")  # baseline 0 -> baseline metric 0
    model.add_flow("inflow", "rate", to_stock="Acc")
    result = sensitivity_analysis(
        model,
        parameters=[{"name": "rate", "values": [1, 2, 3]}],
        output={"variable": "Acc", "metric": "final"},
    )
    assert result["baseline"]["metric_value"] == 0
    assert result["parameters"][0]["elasticity"] is None  # zero baseline metric
    assert result["parameters"][0]["range_sensitivity"] is not None  # slope still defined


def test_sweep_csv_has_long_rows(tmp_path):
    csv_path = tmp_path / "sweep.csv"
    sensitivity_analysis(
        _accumulator_model(),
        parameters=[{"name": "rate", "values": [1, 2, 3]}],
        output={"variable": "Accumulator", "metric": "final"},
        save_sweep_csv=str(csv_path),
    )
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "parameter,value,final"
    assert len(lines) == 1 + 3  # header + one row per swept value


def test_sensitivity_tool(monkeypatch):
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 7002)
    asyncio.run(server_mod.call_tool("build_model", {
        "name": "Accumulator", "model_id": "acc",
        "sim_specs": {"start": 0, "stop": 10, "dt": 1.0},
        "stocks": [{"name": "Accumulator", "initial_value": "0"}],
        "auxs": [{"name": "rate", "equation": "1"}],
        "flows": [{"name": "inflow", "equation": "rate", "to_stock": "Accumulator"}],
    }))
    result = asyncio.run(server_mod.call_tool("sensitivity_analysis", {
        "model_id": "acc",
        "parameters": [{"name": "rate", "start": 1, "stop": 5, "steps": 5}],
        "output": {"variable": "Accumulator", "metric": "final"},
    }))
    assert not result.isError
    sc = result.structuredContent
    assert sc["total_runs"] == 5
    assert sc["parameters"][0]["range_sensitivity"] > 0
