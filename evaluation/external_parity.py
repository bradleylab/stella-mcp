"""Generate numeric parity evidence for pinned external corpus fixtures."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .corpus_manifest import DEFAULT_EXTERNAL_CORPUS_MANIFEST, load_external_corpus_manifest
from .desktop_parity import generate_desktop_parity_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LABEL_PATTERN = re.compile(r"[0-9A-Za-z][0-9A-Za-z._-]*")


def generate_external_parity_reports(
    output_dir: Path,
    *,
    version_label: str,
    manifest_path: Path = DEFAULT_EXTERNAL_CORPUS_MANIFEST,
    selected_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Generate PySD CSV and JSON reports for numeric external fixtures."""
    if _LABEL_PATTERN.fullmatch(version_label) is None:
        raise ValueError("version_label must contain only letters, numbers, dot, dash, underscore")
    document = load_external_corpus_manifest(manifest_path)
    fixtures = [fixture for fixture in document["fixtures"] if "numeric" in fixture]
    known_ids = {fixture["id"] for fixture in fixtures}
    if selected_ids:
        unknown = selected_ids - known_ids
        if unknown:
            raise ValueError(f"Unknown numeric external fixture ids: {', '.join(sorted(unknown))}")
        fixtures = [fixture for fixture in fixtures if fixture["id"] in selected_ids]

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for fixture in fixtures:
        fixture_id = fixture["id"]
        numeric = fixture["numeric"]
        alignment = numeric["time_alignment"]
        model_path = manifest_path.parent / fixture["model"]["path"]
        stella_path = manifest_path.parent / numeric["stella_output"]["path"]
        pysd_path = output_dir / f"{version_label}-{fixture_id}-pysd.csv"
        report_path = output_dir / f"{version_label}-{fixture_id}-parity.json"
        report = generate_desktop_parity_report(
            model_path,
            stella_path,
            pysd_path,
            [
                (column["reference"], column["candidate"])
                for column in numeric["columns"]
            ],
            stella_application=numeric["application"],
            stella_version=numeric["application_version"],
            pysd_time=numeric["reference_time"],
            stella_time=numeric["candidate_time"],
            time_alignment=alignment["policy"],
            candidate_decimal_places=alignment.get("candidate_decimal_places"),
        )
        report["fixture"] = {
            "id": fixture_id,
            "source": fixture["source"],
            "constructs": fixture["constructs"],
            "manifest": str(manifest_path.resolve().relative_to(PROJECT_ROOT)),
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        records.append(
            {
                "fixture_id": fixture_id,
                "report_path": report_path,
                "pysd_csv_path": pysd_path,
                "report": report,
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/evaluation")
    parser.add_argument("--version-label", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_EXTERNAL_CORPUS_MANIFEST)
    parser.add_argument("--fixture", action="append", default=[])
    args = parser.parse_args()
    records = generate_external_parity_reports(
        args.output_dir,
        version_label=args.version_label,
        manifest_path=args.manifest,
        selected_ids=set(args.fixture) or None,
    )
    for record in records:
        print(record["report_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
