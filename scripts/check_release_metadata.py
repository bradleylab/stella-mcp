#!/usr/bin/env python3
"""Validate that package, citation, changelog, and release-tag metadata agree."""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG_HEADING = re.compile(
    r"^## \[(?P<version>[^]]+)\] - (?P<date>\d{4}-\d{2}-\d{2})$", re.MULTILINE
)
MODULE_VERSION = re.compile(r'^__version__\s*=\s*["\'](?P<version>[^"\']+)["\']$', re.MULTILINE)


class ReleaseMetadataError(ValueError):
    """Raised when two release metadata sources disagree."""


@dataclass(frozen=True)
class ReleaseMetadata:
    version: str
    release_date: str


def _module_version(root: Path) -> str:
    source = (root / "stella_mcp" / "__init__.py").read_text(encoding="utf-8")
    match = MODULE_VERSION.search(source)
    if match is None:
        raise ReleaseMetadataError("stella_mcp.__version__ is missing")
    return match.group("version")


def validate_release_metadata(
    root: Path = ROOT,
    *,
    expected_tag: str | None = None,
    distribution_version: str | None = None,
) -> ReleaseMetadata:
    """Return canonical release metadata after validating every source."""
    if distribution_version is None:
        distribution_version = importlib.metadata.version("stella-mcp")

    module_version = _module_version(root)
    if module_version != distribution_version:
        raise ReleaseMetadataError(
            "version mismatch: installed distribution is "
            f"{distribution_version}, stella_mcp.__version__ is {module_version}"
        )

    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    if not isinstance(citation, dict):
        raise ReleaseMetadataError("CITATION.cff must contain a YAML object")
    citation_version = str(citation.get("version", ""))
    if citation_version != distribution_version:
        raise ReleaseMetadataError(
            "version mismatch: installed distribution is "
            f"{distribution_version}, CITATION.cff is {citation_version or '<missing>'}"
        )

    citation_date = str(citation.get("date-released", ""))
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    matching_headings = [
        match
        for match in CHANGELOG_HEADING.finditer(changelog)
        if match.group("version") == distribution_version
    ]
    if not matching_headings:
        raise ReleaseMetadataError(
            f"CHANGELOG.md has no exact release heading for {distribution_version}"
        )
    if len(matching_headings) > 1:
        raise ReleaseMetadataError(
            f"CHANGELOG.md has duplicate release headings for {distribution_version}"
        )
    changelog_date = matching_headings[0].group("date")
    if citation_date != changelog_date:
        raise ReleaseMetadataError(
            "release-date mismatch: CITATION.cff is "
            f"{citation_date or '<missing>'}, CHANGELOG.md is {changelog_date}"
        )

    if expected_tag is not None:
        metadata_tag = f"v{distribution_version}"
        if expected_tag != metadata_tag:
            raise ReleaseMetadataError(
                f"release tag mismatch: requested {expected_tag}, metadata requires {metadata_tag}"
            )

    return ReleaseMetadata(version=distribution_version, release_date=changelog_date)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-tag",
        help="release tag that must exactly equal v<metadata version>",
    )
    args = parser.parse_args(argv)

    try:
        metadata = validate_release_metadata(expected_tag=args.expected_tag)
    except (ReleaseMetadataError, OSError, yaml.YAMLError) as exc:
        print(f"release metadata check failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"release metadata valid: v{metadata.version} ({metadata.release_date})",
        file=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
