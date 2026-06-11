"""Tests for scenario comparison and sensitivity analysis (pysd-backed).

The whole module is skipped when pysd is not installed, matching
test_simulate.py; the dependency-missing path is covered elsewhere.
"""

import asyncio
import math

import pytest

pysd = pytest.importorskip("pysd")

from stella_mcp import server as server_mod  # noqa: E402
from stella_mcp.analysis import compare_scenarios  # noqa: E402
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
