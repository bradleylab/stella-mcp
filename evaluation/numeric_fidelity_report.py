"""Generate reviewed numeric-fidelity evidence from retained parity reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = "0.13.0"
DEFAULT_REVIEWS = Path(__file__).with_name("numeric_reviews_0.13.0.json")
DEFAULT_JSON = PROJECT_ROOT / "results/evaluation/0.13.0-numeric-fidelity.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "docs/evaluation/0.13.0-numeric-fidelity.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return document


def _validate_metric(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric {context}")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Expected finite non-negative {context}")
    return number


def _normalize_alignment(policy: dict[str, Any]) -> dict[str, Any]:
    alignment = policy.get("time_alignment")
    if alignment == "exact; no interpolation":
        alignment = "exact"
    return {
        "time_alignment": alignment,
        "candidate_decimal_places": policy.get("candidate_decimal_places"),
        "interpolation": policy.get("interpolation", "none"),
        "max_time_label_difference": policy.get("max_time_label_difference", 0.0),
        "pass_threshold": policy.get("pass_threshold"),
    }


def generate_numeric_fidelity_report(
    reviews_path: Path = DEFAULT_REVIEWS,
) -> dict[str, Any]:
    """Validate retained parity reports and combine them with manual reviews."""
    reviews = _load_json(reviews_path)
    if reviews.get("schema_version") != 1 or reviews.get("release") != DEFAULT_RELEASE:
        raise ValueError("Unexpected numeric review schema or release")
    case_reviews = reviews.get("cases")
    if not isinstance(case_reviews, list) or not case_reviews:
        raise ValueError("Numeric review file requires cases")

    seen: set[str] = set()
    cases = []
    for review in case_reviews:
        case_id = review.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("Numeric review ids must be non-empty and unique")
        seen.add(case_id)
        review_text = review.get("review")
        category = review.get("review_category")
        if not isinstance(review_text, str) or not review_text.strip():
            raise ValueError(f"Numeric case {case_id} requires a review")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"Numeric case {case_id} requires a review category")

        report_path = PROJECT_ROOT / review["report"]
        report = _load_json(report_path)
        if report.get("schema_version") not in {1, 2}:
            raise ValueError(f"Unsupported parity schema: {report_path}")
        comparison = report.get("comparison")
        if not isinstance(comparison, dict):
            raise ValueError(f"Parity report lacks comparison: {report_path}")
        points = comparison.get("points")
        if isinstance(points, bool) or not isinstance(points, int) or points <= 0:
            raise ValueError(f"Parity report has invalid point count: {report_path}")
        alignment = _normalize_alignment(comparison.get("comparison_policy", {}))
        if alignment["pass_threshold"] is not None:
            raise ValueError(f"Scientific pass threshold is prohibited: {report_path}")
        _validate_metric(
            alignment["max_time_label_difference"],
            context=f"{case_id} maximum time-label difference",
        )

        columns = comparison.get("columns")
        if not isinstance(columns, list) or not columns:
            raise ValueError(f"Parity report has no columns: {report_path}")
        normalized_columns = []
        for column in columns:
            normalized = dict(column)
            for metric in (
                "max_absolute_error",
                "max_absolute_error_time",
                "max_relative_error",
                "max_relative_error_time",
            ):
                _validate_metric(normalized.get(metric), context=f"{case_id} {metric}")
            normalized_columns.append(normalized)
        nonzero_columns = [
            column["reference_column"]
            for column in normalized_columns
            if column["max_absolute_error"] != 0 or column["max_relative_error"] != 0
        ]
        cases.append(
            {
                "id": case_id,
                "parity_report": {
                    "path": report_path.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": _sha256(report_path),
                    "schema_version": report["schema_version"],
                },
                "engines": report["engines"],
                "points": points,
                "alignment": alignment,
                "columns": normalized_columns,
                "nonzero_columns": nonzero_columns,
                "review_category": category,
                "review": review_text,
            }
        )

    return {
        "schema_version": 1,
        "release": DEFAULT_RELEASE,
        "claim_scope": "retained_cases_only",
        "scientific_pass_threshold": None,
        "cases": cases,
    }


def _format_number(value: float | int) -> str:
    return f"{value:.12g}"


def render_numeric_fidelity_markdown(report: dict[str, Any]) -> str:
    """Render the numeric-fidelity review from its machine-readable record."""
    lines = [
        f"# Stella MCP {report['release']} Numeric Fidelity",
        "",
        "These are raw discrepancies for retained cases. No scientific pass threshold is "
        "defined, and no interpolation is performed.",
        "",
        "| Case | Points | Alignment | Max time-label difference | Max absolute error | "
        "Max relative error | Review |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for case in report["cases"]:
        max_absolute = max(column["max_absolute_error"] for column in case["columns"])
        max_relative = max(column["max_relative_error"] for column in case["columns"])
        lines.append(
            f"| `{case['id']}` | {case['points']} | "
            f"{case['alignment']['time_alignment']} | "
            f"{_format_number(case['alignment']['max_time_label_difference'])} | "
            f"{_format_number(max_absolute)} | {_format_number(max_relative)} | "
            f"{case['review_category']} |"
        )
    lines.extend(["", "## Explicit Review", ""])
    for case in report["cases"]:
        nonzero = ", ".join(f"`{name}`" for name in case["nonzero_columns"]) or "none"
        lines.append(f"- `{case['id']}` (nonzero columns: {nonzero}): {case['review']}")
    lines.extend(
        [
            "",
            "The JSON record retains every per-variable maximum, its model time, report hash, "
            "engine version, and alignment policy. The Lotka-Volterra predation result is a "
            "documented backend-semantic limitation, not a tolerated parity result.",
        ]
    )
    return "\n".join(lines)


def write_numeric_fidelity_report(
    report: dict[str, Any],
    output_json: Path = DEFAULT_JSON,
    output_markdown: Path = DEFAULT_MARKDOWN,
) -> None:
    """Write synchronized numeric-fidelity JSON and Markdown."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output_markdown.write_text(render_numeric_fidelity_markdown(report) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = generate_numeric_fidelity_report(args.reviews)
    expected_json = json.dumps(report, indent=2) + "\n"
    expected_markdown = render_numeric_fidelity_markdown(report) + "\n"
    if args.check:
        if not args.output_json.is_file() or not args.output_markdown.is_file():
            return 1
        return int(
            args.output_json.read_text(encoding="utf-8") != expected_json
            or args.output_markdown.read_text(encoding="utf-8") != expected_markdown
        )
    write_numeric_fidelity_report(report, args.output_json, args.output_markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
