"""Tests for `python3 -m workloads.simbench_bw.validate --schema-check PATH`.

PATH may be a single `manifest.json` / `measurement_manifest.json` file
or a directory whose tree is walked recursively. Behaviour clauses:

  * Single valid manifest file -> exit 0 + `ok`.
  * Directory with one or more valid manifests -> exit 0 + `ok`.
  * Any manifest fails the schema -> exit 1 with one
    `schema_validation_error:` line per failure.
  * Empty directory (no manifest.json / measurement_manifest.json) ->
    exit 2 with `schema_check_no_manifests`.
  * Missing path -> exit 2 with `schema_check_path_missing`.
  * Unsupported single file (basename not manifest.json /
    measurement_manifest.json) -> exit 2 with
    `schema_check_unsupported_file`.

The tests build their manifest fixtures in a temporary directory so they do
not depend on historical generated result trees.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

SAMPLE_MANIFEST_BODY = {
    "schema_version": 2,
    "slug": "rmsnorm_fwd_bf16",
    "mode": "timing",
    "warm_up_iters": 5,
    "repeat_iters": 30,
    "cv_pct": 1.25,
    "cuda_version": "13.2",
    "driver_version": "595.58.03",
    "gpu_sku": "rtx_pro_6000_blackwell",
    "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
    "sm_count": 96,
    "total_global_mem_bytes": 1024,
    "device_name": "RTX PRO 6000 Blackwell",
    "compute_capability_major": 12,
    "compute_capability_minor": 0,
    "binary_sha256": "0" * 64,
    "compiler_flags": ["-gencode", "arch=compute_120,code=sm_120"],
    "kb_capability_card": (
        "references/NVIDIA_Manuals/blackwell_compatibility/"
        "01_blackwell-architecture-compatibility.md"
    ),
    "timestamp_utc": "2026-05-03T00:00:00Z",
}


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", "-m", "workloads.simbench_bw.validate", *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def _write_manifest(path: Path, *, slug: str = "rmsnorm_fwd_bf16") -> None:
    body = dict(SAMPLE_MANIFEST_BODY)
    body["slug"] = slug
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))


def test_positive_single_valid_manifest_file(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)
    result = _run("--schema-check", str(manifest))
    assert result.returncode == 0, (
        f"expected exit 0 on valid manifest; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "ok" in result.stdout
    assert "1 manifest(s) validated" in result.stdout


def test_positive_directory_with_multiple_manifests(tmp_path: Path) -> None:
    for slug in ("alpha", "beta", "gamma"):
        slug_dir = tmp_path / slug / "results" / slug
        _write_manifest(slug_dir / "manifest.json", slug=slug)
    result = _run("--schema-check", str(tmp_path))
    assert result.returncode == 0, (
        f"expected exit 0 on directory of valid manifests; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "ok" in result.stdout
    assert "3 manifest(s) validated" in result.stdout


def test_positive_directory_with_measurement_manifest_name(tmp_path: Path) -> None:
    manifest = tmp_path / "measurement_manifest.json"
    _write_manifest(manifest)
    result = _run("--schema-check", str(tmp_path))
    assert result.returncode == 0, (
        f"expected exit 0 on directory with measurement_manifest.json; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "ok" in result.stdout


def test_negative_malformed_manifest_missing_required_field(tmp_path: Path) -> None:
    # Schema v2 requires `slug`. Drop it and assert exit 1 with a
    # schema_validation_error.
    body = dict(SAMPLE_MANIFEST_BODY)
    del body["slug"]
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(body))
    result = _run("--schema-check", str(target))
    assert result.returncode == 1, (
        f"expected exit 1 on malformed manifest; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "schema_validation_error:" in result.stderr
    assert "slug" in result.stderr


def test_negative_empty_directory(tmp_path: Path) -> None:
    result = _run("--schema-check", str(tmp_path))
    assert result.returncode == 2, (
        f"expected exit 2 on empty directory; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "schema_check_no_manifests" in result.stderr


def test_negative_path_missing(tmp_path: Path) -> None:
    bad = tmp_path / "does_not_exist"
    result = _run("--schema-check", str(bad))
    assert result.returncode == 2, (
        f"expected exit 2 on missing path; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "schema_check_path_missing" in result.stderr


def test_negative_unsupported_file(tmp_path: Path) -> None:
    other = tmp_path / "not_a_manifest.json"
    other.write_text(json.dumps({"hello": "world"}))
    result = _run("--schema-check", str(other))
    assert result.returncode == 2, (
        f"expected exit 2 on unsupported file basename; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "schema_check_unsupported_file" in result.stderr


def test_existing_artifact_path_behavior_preserved() -> None:
    # The previously existing positional-artifact CLI must still work for
    # supported artifacts (regression guard for the new --schema-check
    # branch). roster.toml is the simplest available case.
    roster = REPO_ROOT / "workloads" / "simbench_bw" / "roster.toml"
    if not roster.is_file():
        pytest.skip("workloads/simbench_bw/roster.toml not present")
    result = _run(str(roster))
    assert result.returncode == 0, (
        f"expected exit 0 on positional roster.toml validation; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "ok" in result.stdout
