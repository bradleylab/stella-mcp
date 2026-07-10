"""Schemas for simulation, scenario, sensitivity, and calibration tools.

The four analytical tools have detailed literal JSON schemas, so this domain
module intentionally exceeds the project's approximate line guideline. Keeping
the contracts together preserves their public ordering and shared optional-stack
boundary without introducing schema-only submodules.
"""

from __future__ import annotations

from mcp.types import Tool

from .shared import SharedSchemas, build_shared_schemas


def build_tools(shared: SharedSchemas | None = None) -> list[Tool]:
    """Build simulation-domain tool descriptors in public catalog order."""
    model_id_property = (shared or build_shared_schemas()).model_id_property
    return [
        Tool(
            name="simulate",
            description=(
                "Run the model and return downsampled time series with per-"
                "variable summaries (initial/final/min/max). Requires the "
                "optional pysd dependency (pip install 'stella-mcp[sim]'). "
                "Integration is Euler regardless of the model's method setting."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "overrides": {
                        "type": "object",
                        "description": (
                            "Constant parameter overrides keyed by variable name "
                            "(display or underscore form)"
                        ),
                        "additionalProperties": {"type": "number"},
                    },
                    "include": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables to report (default: all stocks)",
                    },
                    "max_points": {
                        "type": "integer",
                        "description": "Maximum points per returned series",
                        "default": 101,
                        "minimum": 2,
                    },
                    "save_results_csv": {
                        "type": "string",
                        "description": "Optional path to write the full results table as CSV",
                    },
                },
            },
        ),
        Tool(
            name="compare_scenarios",
            description=(
                "Run several named what-if scenarios (each a set of constant "
                "parameter overrides) against a baseline and report how each "
                "diverges: per-variable final/max absolute deltas and final "
                "percent change. Requires the optional pysd dependency "
                "(pip install 'stella-mcp[sim]')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "scenarios": {
                        "type": "array",
                        "minItems": 1,
                        "description": "Named override sets to compare (names must be unique)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Unique scenario label",
                                },
                                "overrides": {
                                    "type": "object",
                                    "additionalProperties": {"type": "number"},
                                    "description": (
                                        "Constant parameter overrides for this scenario"
                                    ),
                                },
                            },
                            "required": ["name", "overrides"],
                        },
                    },
                    "baseline": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": (
                            "Override set to measure deltas against "
                            "(default: the unmodified model)"
                        ),
                    },
                    "include": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Variables to report and compare (default: all stocks)"
                        ),
                    },
                    "max_points": {
                        "type": "integer",
                        "description": "Maximum points per returned series",
                        "default": 101,
                        "minimum": 2,
                    },
                    "save_comparison_csv": {
                        "type": "string",
                        "description": (
                            "Optional path to write a wide variable-by-scenario CSV"
                        ),
                    },
                },
                "required": ["scenarios"],
            },
        ),
        Tool(
            name="sensitivity_analysis",
            description=(
                "One-at-a-time sensitivity: sweep each parameter across a range "
                "(holding the others at their baseline) and report how one chosen "
                "output metric responds, with a range slope and a "
                "baseline-normalized elasticity for ranking. Requires the "
                "optional pysd dependency (pip install 'stella-mcp[sim]')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "parameters": {
                        "type": "array",
                        "minItems": 1,
                        "description": "Parameters to sweep, each one at a time",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Parameter variable name",
                                },
                                "start": {
                                    "type": "number",
                                    "description": "Sweep start (use with stop + steps)",
                                },
                                "stop": {
                                    "type": "number",
                                    "description": "Sweep stop (use with start + steps)",
                                },
                                "steps": {
                                    "type": "integer",
                                    "minimum": 2,
                                    "description": "Number of evenly spaced sweep points",
                                },
                                "values": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 2,
                                    "description": (
                                        "Explicit sweep values "
                                        "(alternative to start/stop/steps)"
                                    ),
                                },
                            },
                            "required": ["name"],
                        },
                    },
                    "output": {
                        "type": "object",
                        "description": "The single output metric to track across the sweep",
                        "properties": {
                            "variable": {
                                "type": "string",
                                "description": "Output variable to reduce to a metric",
                            },
                            "metric": {
                                "type": "string",
                                "enum": [
                                    "final",
                                    "max",
                                    "min",
                                    "mean",
                                    "time_to_threshold",
                                ],
                                "default": "final",
                            },
                            "threshold": {
                                "type": "number",
                                "description": (
                                    "Required when metric is time_to_threshold"
                                ),
                            },
                        },
                        "required": ["variable"],
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["oat"],
                        "default": "oat",
                        "description": (
                            "Sweep design; only one-at-a-time is available "
                            "(grid/montecarlo reserved)"
                        ),
                    },
                    "max_runs": {
                        "type": "integer",
                        "default": 200,
                        "minimum": 1,
                        "description": (
                            "Hard cap on total swept runs; the call errors "
                            "rather than truncating a larger sweep"
                        ),
                    },
                    "include_series": {
                        "type": "boolean",
                        "default": False,
                        "description": "Also return each run's downsampled output series",
                    },
                    "save_sweep_csv": {
                        "type": "string",
                        "description": (
                            "Optional path to write the long "
                            "(parameter, value, metric) CSV"
                        ),
                    },
                },
                "required": ["parameters", "output"],
            },
        ),
        Tool(
            name="calibrate",
            description=(
                "Fit constant model parameters to an observed time-series — the "
                "inverse of simulate. Only constant auxiliaries/flows can be "
                "calibrated (stocks are rejected: overriding a stock pins it to a "
                "constant rather than setting its initial value). least_squares "
                "(default) reports a linearized std_error; differential_evolution "
                "is a global, seeded alternative requiring bounds. Observation "
                "times must lie within the model window (no extrapolation). "
                "Requires the optional pysd dependency "
                "(pip install 'stella-mcp[sim]')."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": model_id_property,
                    "observations": {
                        "type": "object",
                        "description": (
                            "Observed data on one shared time grid: inline "
                            "{time, targets} or {csv_path} (first CSV column is time)"
                        ),
                        "properties": {
                            "time": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "description": "Strictly increasing observation times",
                            },
                            "targets": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                },
                                "description": (
                                    "Observed series per model variable "
                                    "(each same length as time)"
                                ),
                            },
                            "csv_path": {
                                "type": "string",
                                "description": (
                                    "Alternative to time/targets: CSV with a time "
                                    "column followed by one column per target"
                                ),
                            },
                        },
                    },
                    "parameters": {
                        "type": "array",
                        "minItems": 1,
                        "description": (
                            "Constant parameters to fit (constant auxiliaries/flows "
                            "only; stocks are rejected)"
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Parameter variable name",
                                },
                                "initial": {
                                    "type": "number",
                                    "description": (
                                        "Initial guess (default: the model's current "
                                        "constant value)"
                                    ),
                                },
                                "min": {
                                    "type": "number",
                                    "description": (
                                        "Lower bound (optional for least_squares, "
                                        "required for differential_evolution)"
                                    ),
                                },
                                "max": {
                                    "type": "number",
                                    "description": "Upper bound (see min)",
                                },
                            },
                            "required": ["name"],
                        },
                    },
                    "method": {
                        "type": "string",
                        "enum": ["least_squares", "differential_evolution"],
                        "default": "least_squares",
                        "description": (
                            "Optimizer: least_squares (local, gives std_error) or "
                            "differential_evolution (global, seeded, needs bounds)"
                        ),
                    },
                    "objective": {
                        "type": "string",
                        "enum": ["sse"],
                        "default": "sse",
                        "description": (
                            "Objective (sum of squared residuals; scale via weights)"
                        ),
                    },
                    "weights": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": (
                            "Optional positive per-target residual multipliers; "
                            "inverse-sigma values give normalized residuals and the "
                            "usual statistical std_error interpretation"
                        ),
                    },
                    "max_nfev": {
                        "type": "integer",
                        "default": 1000,
                        "minimum": 1,
                        "description": "least_squares function-evaluation cap",
                    },
                    "maxiter": {
                        "type": ["integer", "null"],
                        "default": 100,
                        "minimum": 1,
                        "description": (
                            "differential_evolution generation cap (default 100)"
                        ),
                    },
                    "popsize": {
                        "type": "integer",
                        "default": 15,
                        "minimum": 1,
                        "description": "differential_evolution population multiplier",
                    },
                    "seed": {
                        "type": "integer",
                        "default": 0,
                        "description": (
                            "differential_evolution seed (kept non-null for reproducibility)"
                        ),
                    },
                    "return_fit_series": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Also return the best-fit downsampled series per target"
                        ),
                    },
                    "save_fit_csv": {
                        "type": "string",
                        "description": (
                            "Optional path to write a long "
                            "(time, target, observed, fitted) CSV"
                        ),
                    },
                },
                "required": ["observations", "parameters"],
            },
        ),
    ]
