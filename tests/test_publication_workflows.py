"""Static security contracts for the release artifact and PyPI workflows."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
RELEASE_BUILD = WORKFLOWS / "release-build.yml"
CI = WORKFLOWS / "ci.yml"
PUBLISH = WORKFLOWS / "publish.yml"

REVIEWED_ACTIONS = {
    "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803",
    "astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b",
    "actions/upload-artifact@b7c566a772e6b6bfb58ed0dc250532a479d7789f",
    "actions/download-artifact@70fc10c6e5e1ce46ad2ea6f2b72d43f7d47b13c3",
    "pypa/gh-action-pypi-publish@ed0c53931b1dc9bd32cbe73a98c7f6766f8a527e",
}
FULL_ACTION_SHA = re.compile(r"^[^./][^@]+@[0-9a-f]{40}$")
ARTIFACT_NAME = "stella-mcp-distributions"


def _load(path: Path) -> dict[str, Any]:
    """Load workflow YAML without YAML 1.1 coercing the ``on`` key."""

    document = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(document, dict)
    return document


def _external_action_uses(node: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "uses" and isinstance(value, str) and not value.startswith("./"):
                refs.append(value)
            refs.extend(_external_action_uses(value))
    elif isinstance(node, list):
        for value in node:
            refs.extend(_external_action_uses(value))
    return refs


def _step_index(steps: list[dict[str, Any]], name: str) -> int:
    return next(index for index, step in enumerate(steps) if step.get("name") == name)


def _script_lines(step: dict[str, Any]) -> list[str]:
    return [line.strip() for line in step["run"].splitlines() if line.strip()]


def _run_scripts(node: Any) -> list[str]:
    scripts: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "run" and isinstance(value, str):
                scripts.append(value)
            scripts.extend(_run_scripts(value))
    elif isinstance(node, list):
        for value in node:
            scripts.extend(_run_scripts(value))
    return scripts


def test_all_release_path_actions_use_reviewed_full_commit_shas() -> None:
    for path in (RELEASE_BUILD, CI, PUBLISH):
        refs = _external_action_uses(_load(path))
        assert refs, f"{path.name} must use at least one reviewed action"
        assert all(FULL_ACTION_SHA.fullmatch(ref) for ref in refs), refs
        assert set(refs) <= REVIEWED_ACTIONS


def test_untrusted_contexts_are_not_interpolated_into_release_shell_scripts() -> None:
    for path in (RELEASE_BUILD, CI, PUBLISH):
        assert all("${{" not in script for script in _run_scripts(_load(path))), path.name


def test_reusable_build_is_unprivileged_and_retains_digest_bound_artifacts() -> None:
    workflow = _load(RELEASE_BUILD)
    assert set(workflow["on"]) == {"workflow_call"}
    assert "id-token" not in RELEASE_BUILD.read_text(encoding="utf-8")
    outputs = workflow["on"]["workflow_call"]["outputs"]
    assert outputs["artifact_digest"]["value"] == "${{ jobs.build.outputs.artifact_digest }}"
    assert outputs["manifest_digest"]["value"] == "${{ jobs.build.outputs.manifest_digest }}"

    build = workflow["jobs"]["build"]
    assert build["permissions"] == {"contents": "read"}
    assert build["outputs"] == {
        "artifact_digest": "${{ steps.upload.outputs['artifact-digest'] }}",
        "manifest_digest": "${{ steps.manifest.outputs.manifest_digest }}",
    }
    steps = build["steps"]
    upload = next(
        step
        for step in steps
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert upload["with"]["name"] == ARTIFACT_NAME
    assert upload["with"]["if-no-files-found"] == "error"

    manifest = next(step for step in steps if step.get("name") == "Create artifact manifest")
    assert _script_lines(manifest) == [
        "cd dist",
        "sha256sum *.whl *.tar.gz > SHA256SUMS",
        "sha256sum --check SHA256SUMS",
        "manifest_digest=\"$(sha256sum SHA256SUMS | cut -d ' ' -f 1)\"",
        "printf 'manifest_digest=%s\\n' \"$manifest_digest\" >> \"$GITHUB_OUTPUT\"",
    ]
    assert set(upload["with"]["path"].splitlines()) == {
        "dist/*.whl",
        "dist/*.tar.gz",
        "dist/SHA256SUMS",
    }
    assert _step_index(steps, "Validate installed MCP package") < _step_index(
        steps, "Create artifact manifest"
    )
    metadata = next(step for step in steps if step.get("name") == "Validate release tag and metadata")
    assert metadata["env"] == {"EXPECTED_TAG": "${{ inputs.expected_tag }}"}
    assert "${{" not in metadata["run"]


def test_ci_calls_reusable_build_and_verifies_same_run_artifact_without_oidc() -> None:
    workflow = _load(CI)
    assert "id-token" not in CI.read_text(encoding="utf-8")
    assert workflow["permissions"] == {"contents": "read"}
    for job in workflow["jobs"].values():
        if "permissions" in job:
            assert job["permissions"] == {"contents": "read"}

    package = workflow["jobs"]["package"]
    assert package["uses"] == "./.github/workflows/release-build.yml"
    consumer = workflow["jobs"]["verify-release-artifact"]
    assert consumer["needs"] == "package"
    assert "environment" not in consumer

    steps = consumer["steps"]
    download = next(step for step in steps if step["uses"].startswith("actions/download-artifact@"))
    assert download["with"]["name"] == ARTIFACT_NAME
    assert not ({"run-id", "repository", "github-token"} & set(download["with"]))
    assert download["with"]["digest-mismatch"] == "error"
    assert download["with"]["skip-decompress"] == "true"
    verify = next(step for step in steps if step.get("name") == "Verify artifact manifest")
    assert verify["env"] == {
        "EXPECTED_ARTIFACT_DIGEST": "${{ needs.package.outputs.artifact_digest }}",
        "EXPECTED_MANIFEST_DIGEST": "${{ needs.package.outputs.manifest_digest }}",
    }
    assert _script_lines(verify) == [
        'archive_path="$(find release-archive -maxdepth 1 -type f -print -quit)"',
        'test -n "$archive_path"',
        "printf '%s  %s\\n' \"$EXPECTED_ARTIFACT_DIGEST\" \"$archive_path\" | sha256sum --check -",
        "mkdir release-dist",
        'unzip -q "$archive_path" -d release-dist',
        "cd release-dist",
        "printf '%s  SHA256SUMS\\n' \"$EXPECTED_MANIFEST_DIGEST\" | sha256sum --check -",
        "sha256sum --check SHA256SUMS",
    ]
    assert not any("pypa/gh-action-pypi-publish" in ref for ref in _external_action_uses(consumer))


def test_publish_has_one_protected_oidc_boundary_and_no_manual_trigger() -> None:
    workflow = _load(PUBLISH)
    assert workflow["on"] == {"release": {"types": ["published"]}}
    assert set(workflow["jobs"]) == {"build", "pypi"}

    build = workflow["jobs"]["build"]
    assert build["if"] == "${{ github.event.release.prerelease == false }}"
    assert build["uses"] == "./.github/workflows/release-build.yml"
    assert build["permissions"] == {"contents": "read"}
    assert build["with"]["ref"] == "${{ github.event.release.tag_name }}"
    assert build["with"]["expected_tag"] == "${{ github.event.release.tag_name }}"

    pypi = workflow["jobs"]["pypi"]
    assert pypi["needs"] == "build"
    assert pypi["environment"] == "pypi"
    assert pypi["permissions"] == {"id-token": "write"}

    steps = pypi["steps"]
    download = next(step for step in steps if step["uses"].startswith("actions/download-artifact@"))
    assert download["with"]["name"] == ARTIFACT_NAME
    assert not ({"run-id", "repository", "github-token"} & set(download["with"]))
    assert download["with"]["digest-mismatch"] == "error"
    assert download["with"]["skip-decompress"] == "true"
    assert _step_index(steps, "Download validated distributions") < _step_index(
        steps, "Verify artifact manifest"
    ) < _step_index(steps, "Publish with PyPI Trusted Publishing")
    verify = next(step for step in steps if step.get("name") == "Verify artifact manifest")
    assert verify["env"] == {
        "EXPECTED_ARTIFACT_DIGEST": "${{ needs.build.outputs.artifact_digest }}",
        "EXPECTED_MANIFEST_DIGEST": "${{ needs.build.outputs.manifest_digest }}",
    }
    assert _script_lines(verify) == [
        'archive_path="$(find release-archive -maxdepth 1 -type f -print -quit)"',
        'test -n "$archive_path"',
        "printf '%s  %s\\n' \"$EXPECTED_ARTIFACT_DIGEST\" \"$archive_path\" | sha256sum --check -",
        "mkdir release-dist",
        'unzip -q "$archive_path" -d release-dist',
        "cd release-dist",
        "printf '%s  SHA256SUMS\\n' \"$EXPECTED_MANIFEST_DIGEST\" | sha256sum --check -",
        "sha256sum --check SHA256SUMS",
    ]

    remove = steps[_step_index(steps, "Remove manifest from publication directory")]
    assert _script_lines(remove) == ["rm release-dist/SHA256SUMS"]
    assert _step_index(steps, "Remove manifest from publication directory") + 1 == _step_index(
        steps, "Publish with PyPI Trusted Publishing"
    )

    publisher = steps[_step_index(steps, "Publish with PyPI Trusted Publishing")]
    assert publisher["uses"] == (
        "pypa/gh-action-pypi-publish@ed0c53931b1dc9bd32cbe73a98c7f6766f8a527e"
    )
    assert publisher["with"] == {"packages-dir": "release-dist"}
