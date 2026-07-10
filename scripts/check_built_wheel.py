#!/usr/bin/env python3
"""Validate runtime metadata and package data in a built stella-mcp wheel."""

from __future__ import annotations

import argparse
import email.policy
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

from check_release_metadata import ROOT, _module_version
from packaging.requirements import Requirement


class WheelValidationError(ValueError):
    """Raised when a built wheel violates the distribution contract."""


def validate_wheel(wheel: Path, root: Path = ROOT) -> tuple[str, int]:
    """Validate one wheel and return its version and bundled-template count."""
    if wheel.suffix != ".whl":
        raise WheelValidationError(f"expected a .whl file, got {wheel}")

    with ZipFile(wheel) as archive:
        members = set(archive.namelist())
        metadata_members = [name for name in members if name.endswith(".dist-info/METADATA")]
        if len(metadata_members) != 1:
            raise WheelValidationError(
                f"expected one wheel METADATA file, found {len(metadata_members)}"
            )
        metadata = BytesParser(policy=email.policy.default).parsebytes(
            archive.read(metadata_members[0])
        )

    name = str(metadata["Name"])
    if name != "stella-mcp":
        raise WheelValidationError(f"wheel project name is {name}, expected stella-mcp")

    version = str(metadata["Version"])
    source_version = _module_version(root)
    if version != source_version:
        raise WheelValidationError(
            f"wheel version is {version}, stella_mcp.__version__ is {source_version}"
        )

    requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
    unconditional = {
        requirement.name.lower()
        for requirement in requirements
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""})
    }
    sim = {
        requirement.name.lower()
        for requirement in requirements
        if requirement.marker is not None and requirement.marker.evaluate({"extra": "sim"})
    }
    if unconditional != {"mcp"}:
        raise WheelValidationError(
            f"core wheel dependencies must be only mcp; found {sorted(unconditional)}"
        )
    expected_sim = {"numpy", "pandas", "pysd", "scipy"}
    if sim != expected_sim:
        raise WheelValidationError(
            f"sim extra must contain {sorted(expected_sim)}; found {sorted(sim)}"
        )

    source_templates = {
        path.relative_to(root).as_posix()
        for path in (root / "stella_mcp" / "builtin_templates").iterdir()
        if path.is_file() and path.suffix in {".stmx", ".json"}
    }
    missing_templates = source_templates - members
    if missing_templates:
        raise WheelValidationError(
            f"wheel is missing bundled templates: {sorted(missing_templates)}"
        )

    return version, len(source_templates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()

    try:
        version, template_count = validate_wheel(args.wheel)
    except (OSError, WheelValidationError) as exc:
        parser.exit(1, f"wheel validation failed: {exc}\n")

    print(f"wheel valid: stella-mcp {version}, {template_count} bundled template files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
