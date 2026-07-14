"""Tests for compatibility corpus manifest sync helper script."""

import json
import subprocess
import sys
from pathlib import Path


def test_manifest_sync_tool_check_and_write(tmp_path):
    """Script should detect out-of-sync manifest and then repair it."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "sample.stmx").write_text("<xmile/>", encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"fixtures":[]}\n', encoding="utf-8")

    script = str(
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "sync_compat_corpus_manifest.py"
    )

    check = subprocess.run(
        [
            sys.executable,
            script,
            "--check",
            "--corpus-dir",
            str(corpus_dir),
            "--manifest",
            str(manifest_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 1
    assert "out of sync" in check.stdout.lower()

    write = subprocess.run(
        [
            sys.executable,
            script,
            "--corpus-dir",
            str(corpus_dir),
            "--manifest",
            str(manifest_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert write.returncode == 0

    recheck = subprocess.run(
        [
            sys.executable,
            script,
            "--check",
            "--corpus-dir",
            str(corpus_dir),
            "--manifest",
            str(manifest_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert recheck.returncode == 0
    assert "in sync" in recheck.stdout.lower()


def test_manifest_sync_preserves_valid_provenance(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "stella_saved.stmx").write_text("<xmile/>", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    provenance = {
        "origin": "stella_mcp_generated_then_stella_saved",
        "application": "Stella Professional",
        "application_version": "4.1.1",
        "saved_at": "2026-07-13",
        "source": "evaluation/scenarios.json#fixture",
        "acceptance": {
            "opened_without_repair": True,
            "run_completed": True,
            "run_final_time": 4,
            "visual_review": "pass",
            "notes": "Expected objects were visible.",
        },
    }
    manifest_path.write_text(
        json.dumps(
            {
                "fixtures": [
                    {
                        "file": "stella_saved.stmx",
                        "strict_import": True,
                        "provenance": provenance,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    script = str(
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "sync_compat_corpus_manifest.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            script,
            "--corpus-dir",
            str(corpus_dir),
            "--manifest",
            str(manifest_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written["fixtures"][0]["provenance"] == provenance
