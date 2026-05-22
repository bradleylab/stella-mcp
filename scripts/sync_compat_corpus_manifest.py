#!/usr/bin/env python3
"""Sync and validate the compatibility corpus manifest.

Usage:
  python scripts/sync_compat_corpus_manifest.py
  python scripts/sync_compat_corpus_manifest.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"fixtures": []}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    fixtures = raw.get("fixtures")
    if fixtures is None:
        raw["fixtures"] = []
        return raw
    if not isinstance(fixtures, list):
        raise ValueError(f"Manifest 'fixtures' must be a list: {path}")
    return raw


def normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    file_name = entry.get("file")
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("Each fixture entry must include non-empty 'file'")
    normalized: dict[str, Any] = {
        "file": file_name.strip(),
        "strict_import": bool(entry.get("strict_import", True)),
    }
    if "expected_warning_contains" in entry:
        warnings = entry["expected_warning_contains"]
        if not isinstance(warnings, list) or not all(isinstance(x, str) for x in warnings):
            raise ValueError(f"'expected_warning_contains' must be list[str] for {file_name}")
        normalized["expected_warning_contains"] = warnings
    if "preserve_markers" in entry:
        markers = entry["preserve_markers"]
        if not isinstance(markers, list) or not all(isinstance(x, str) for x in markers):
            raise ValueError(f"'preserve_markers' must be list[str] for {file_name}")
        normalized["preserve_markers"] = markers
    return normalized


def build_synced_manifest(
    manifest: dict[str, Any],
    fixture_files: list[str],
) -> tuple[dict[str, Any], list[str], list[str]]:
    seen: set[str] = set()
    existing_by_file: dict[str, dict[str, Any]] = {}
    for raw in manifest.get("fixtures", []):
        if not isinstance(raw, dict):
            raise ValueError("All fixture entries must be JSON objects")
        entry = normalize_entry(raw)
        file_name = entry["file"]
        if file_name in seen:
            raise ValueError(f"Duplicate fixture entry in manifest: {file_name}")
        seen.add(file_name)
        existing_by_file[file_name] = entry

    created: list[str] = []
    synced_fixtures: list[dict[str, Any]] = []
    for file_name in sorted(fixture_files):
        if file_name in existing_by_file:
            synced_fixtures.append(existing_by_file[file_name])
        else:
            created.append(file_name)
            synced_fixtures.append({"file": file_name, "strict_import": True})

    removed = sorted(name for name in existing_by_file if name not in set(fixture_files))
    return {"fixtures": synced_fixtures}, created, removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync compatibility corpus manifest")
    parser.add_argument(
        "--corpus-dir",
        default="tests/fixtures/compat_corpus",
        help="Directory containing corpus .stmx fixtures",
    )
    parser.add_argument(
        "--manifest",
        default="tests/fixtures/compat_corpus/manifest.json",
        help="Manifest JSON path",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check only; return non-zero if manifest is out of sync",
    )
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    manifest_path = Path(args.manifest)
    if not corpus_dir.exists():
        print(f"Corpus directory not found: {corpus_dir}", file=sys.stderr)
        return 2

    fixture_files = sorted(p.name for p in corpus_dir.glob("*.stmx") if p.is_file())
    if not fixture_files:
        print(f"No .stmx fixtures found in {corpus_dir}", file=sys.stderr)
        return 2

    current_manifest = load_manifest(manifest_path)
    synced_manifest, created, removed = build_synced_manifest(current_manifest, fixture_files)

    current_json = json.dumps(current_manifest, sort_keys=True)
    synced_json = json.dumps(synced_manifest, sort_keys=True)
    changed = current_json != synced_json

    if args.check:
        if changed:
            print("Manifest is out of sync.")
            if created:
                print("Missing entries:")
                for name in created:
                    print(f"  - {name}")
            if removed:
                print("Stale entries:")
                for name in removed:
                    print(f"  - {name}")
            return 1
        print("Manifest is in sync.")
        return 0

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(synced_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}")
    if created:
        print(f"Added {len(created)} new fixture entries.")
    if removed:
        print(f"Removed {len(removed)} stale fixture entries.")
    if not changed:
        print("No changes were needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
