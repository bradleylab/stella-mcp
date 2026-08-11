"""Release metadata and package dependency contracts."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path

import yaml
from packaging.requirements import Requirement

import stella_mcp
from scripts.check_release_metadata import main, validate_release_metadata

ROOT = Path(__file__).resolve().parents[1]
RELEASE_DATE = "2026-08-09"


def test_release_metadata_sources_agree():
    distribution_version = importlib.metadata.version("stella-mcp")
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    metadata = validate_release_metadata(ROOT)

    assert distribution_version == stella_mcp.__version__
    assert citation["version"] == distribution_version
    assert f"## [{distribution_version}] - {metadata.release_date}" in changelog
    assert str(citation["date-released"]) == metadata.release_date


def test_release_metadata_uses_published_date():
    metadata = validate_release_metadata(ROOT)

    assert metadata.release_date == RELEASE_DATE


def test_changelog_links_skip_unpublished_0_13_tag():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert (
        "[Unreleased]: https://github.com/bradleylab/stella-mcp/compare/v0.14.0...HEAD"
    ) in changelog
    assert (
        "[0.14.0]: https://github.com/bradleylab/stella-mcp/compare/v0.12.0...v0.14.0"
    ) in changelog
    assert "[0.13.0]: https://" not in changelog
    assert "v0.13.0..." not in changelog


def test_release_metadata_cli_accepts_matching_tag(capsys):
    version = importlib.metadata.version("stella-mcp")

    assert main(["--expected-tag", f"v{version}"]) == 0
    assert f"release metadata valid: v{version}" in capsys.readouterr().out


def test_release_metadata_cli_rejects_mismatched_tag(capsys):
    version = importlib.metadata.version("stella-mcp")

    assert main(["--expected-tag", "v99.99.99"]) == 1
    error = capsys.readouterr().err
    assert "requested v99.99.99" in error
    assert f"metadata requires v{version}" in error


def test_distribution_dependency_contract():
    requirements = [Requirement(value) for value in importlib.metadata.requires("stella-mcp") or []]
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

    assert unconditional == {"jsonschema", "mcp"}
    assert sim == {"numpy", "pandas", "pysd", "scipy"}

    [mcp_requirement] = [
        requirement for requirement in requirements if requirement.name.lower() == "mcp"
    ]
    assert mcp_requirement.specifier == Requirement("mcp>=2.0.0,<3").specifier


def test_server_import_does_not_load_simulation_dependencies():
    blocker = """
import importlib.abc
import sys

class BlockSimulationImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'numpy', 'pandas', 'pysd', 'scipy'}:
            raise ImportError(f'blocked optional dependency: {fullname}')
        return None

sys.meta_path.insert(0, BlockSimulationImports())
import stella_mcp.server
"""

    completed = subprocess.run(
        [sys.executable, "-c", blocker],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_local_workflow_state_is_ignored_and_untracked():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    local_paths = {
        ".agents/",
        ".beads/",
        ".claude/",
        ".codex/",
        ".elves/",
        ".elves-session.json",
        "HANDOFF.md",
        "docs/brainstorms/",
        "docs/evaluation/",
        "docs/plans/",
        "docs/releases/",
        "docs/superpowers/",
        "evaluation/",
        "results/",
    }

    assert local_paths <= set(gitignore)

    completed = subprocess.run(
        ["git", "ls-files", "--", *sorted(local_paths)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout == ""
