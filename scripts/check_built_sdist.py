#!/usr/bin/env python3
"""Validate runtime source and metadata in a built stella-mcp sdist."""

from __future__ import annotations

import argparse
import email.policy
import tarfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

from check_release_metadata import ROOT, _module_version


class SdistValidationError(ValueError):
    """Raised when a built source distribution violates the release contract."""


def validate_sdist(sdist: Path, root: Path = ROOT) -> tuple[str, int, int]:
    """Validate one source archive and return version and source-file counts."""
    if sdist.name.endswith(".tar.gz") is False:
        raise SdistValidationError(f"expected a .tar.gz file, got {sdist}")

    with tarfile.open(sdist, mode="r:gz") as archive:
        members = archive.getmembers()
        unsafe = [member.name for member in members if member.issym() or member.islnk()]
        if unsafe:
            raise SdistValidationError(f"sdist must not contain links: {unsafe}")
        paths = {PurePosixPath(member.name) for member in members}
        roots = {path.parts[0] for path in paths if path.parts}
        if len(roots) != 1:
            raise SdistValidationError(f"expected one archive root, found {sorted(roots)}")
        archive_root = next(iter(roots))
        metadata_path = PurePosixPath(archive_root, "PKG-INFO")
        metadata_member = archive.getmember(metadata_path.as_posix())
        metadata_file = archive.extractfile(metadata_member)
        if metadata_file is None:
            raise SdistValidationError("sdist PKG-INFO is not a regular file")
        metadata = BytesParser(policy=email.policy.default).parsebytes(metadata_file.read())

    if str(metadata["Name"]) != "stella-mcp":
        raise SdistValidationError(f"sdist project name is {metadata['Name']}, expected stella-mcp")
    version = str(metadata["Version"])
    source_version = _module_version(root)
    if version != source_version:
        raise SdistValidationError(
            f"sdist version is {version}, stella_mcp.__version__ is {source_version}"
        )

    relative_paths = {PurePosixPath(*path.parts[1:]) for path in paths if len(path.parts) > 1}
    source_modules = {
        PurePosixPath(path.relative_to(root).as_posix())
        for path in (root / "stella_mcp").rglob("*.py")
    }
    missing_modules = source_modules - relative_paths
    if missing_modules:
        raise SdistValidationError(
            f"sdist is missing runtime modules: {sorted(map(str, missing_modules))}"
        )
    source_templates = {
        PurePosixPath(path.relative_to(root).as_posix())
        for path in (root / "stella_mcp" / "builtin_templates").iterdir()
        if path.is_file() and path.suffix in {".stmx", ".json"}
    }
    missing_templates = source_templates - relative_paths
    if missing_templates:
        raise SdistValidationError(
            f"sdist is missing templates: {sorted(map(str, missing_templates))}"
        )

    repository_only = [
        path
        for path in relative_paths
        if path.parts and path.parts[0] in {"docs", "evaluation", "results", "tests"}
    ]
    if repository_only:
        raise SdistValidationError(
            "sdist contains repository-only paths: " + ", ".join(sorted(map(str, repository_only)))
        )
    return version, len(source_modules), len(source_templates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdist", type=Path)
    args = parser.parse_args()

    try:
        version, module_count, template_count = validate_sdist(args.sdist)
    except (OSError, SdistValidationError, tarfile.TarError) as exc:
        parser.exit(1, f"sdist validation failed: {exc}\n")

    print(
        f"sdist valid: stella-mcp {version}, {module_count} runtime modules, "
        f"{template_count} bundled template files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
