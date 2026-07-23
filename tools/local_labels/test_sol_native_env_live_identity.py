"""Regression tests for the live SOL compiler-input identity."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import jsonschema

from tools.local_labels import sol_reconstitution
from tools.sol_baselines_harness import native_env


REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_FINGERPRINT = (
    REPO_ROOT / "data/local_labels/campaign/sol_native_env_fingerprint_sm86.json"
)
HISTORICAL_FINGERPRINT_SHA256 = (
    "04940920fa0ed2df7f827c055eccd589230c2352c36515dab6ad049458d58169"
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
    )


def _write_cutlass_tree(root: Path, *, header_value: int = 1) -> Path:
    header = root / "include" / "cutlass" / "live_identity.h"
    utility = root / "tools" / "util" / "include" / "utility.h"
    header.parent.mkdir(parents=True)
    utility.parent.mkdir(parents=True)
    header.write_text(f"#define SOL_LIVE_IDENTITY {header_value}\n")
    utility.write_text("#define SOL_UTILITY_IDENTITY 1\n")
    return header


def test_in_repo_cutlass_identity_is_stable_across_checkout_roots(
    tmp_path: Path, monkeypatch,
) -> None:
    first_checkout = tmp_path / "checkout-a"
    second_checkout = tmp_path / "checkout-b"
    _write_cutlass_tree(first_checkout / "third_party" / "cutlass")
    _write_cutlass_tree(second_checkout / "third_party" / "cutlass")
    monkeypatch.delenv("CUTLASS_DIR", raising=False)

    monkeypatch.setattr(native_env, "REPO_ROOT", first_checkout)
    first = native_env.cutlass_tree_hash()
    monkeypatch.setattr(native_env, "REPO_ROOT", second_checkout)
    second = native_env.cutlass_tree_hash()

    assert first != native_env.UNKNOWN
    assert second != native_env.UNKNOWN
    assert first == second


def test_in_repo_cutlass_worktree_header_mutation_changes_fingerprint(
    tmp_path: Path, monkeypatch,
) -> None:
    """The identity follows the header nvcc reads, not ``HEAD:<tree>``."""
    cutlass = tmp_path / "third_party" / "cutlass"
    header = _write_cutlass_tree(cutlass)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "test")
    _git(tmp_path, "add", "third_party/cutlass")
    _git(tmp_path, "commit", "-qm", "baseline")

    monkeypatch.setattr(native_env, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("CUTLASS_DIR", raising=False)
    before = native_env.cutlass_tree_hash()
    assert before != native_env.UNKNOWN

    # Leave the Git index and HEAD untouched.  The old HEAD-tree fast path
    # returned the same value here even though nvcc would consume these bytes.
    header.write_text("#define SOL_LIVE_IDENTITY 2\n")
    after = native_env.cutlass_tree_hash()

    assert after != native_env.UNKNOWN
    assert after != before
    assert _git_status(tmp_path) == " M third_party/cutlass/include/cutlass/live_identity.h"


def test_external_cutlass_identity_remains_absolute_path_sensitive(
    tmp_path: Path, monkeypatch,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    first_external = tmp_path / "external-a"
    second_external = tmp_path / "external-b"
    _write_cutlass_tree(first_external)
    _write_cutlass_tree(second_external)
    monkeypatch.setattr(native_env, "REPO_ROOT", checkout)

    monkeypatch.setenv("CUTLASS_DIR", str(first_external))
    first = native_env.cutlass_tree_hash()
    monkeypatch.setenv("CUTLASS_DIR", str(second_external))
    second = native_env.cutlass_tree_hash()

    assert first != native_env.UNKNOWN
    assert second != native_env.UNKNOWN
    assert first != second


def test_historical_and_live_fingerprint_authorities_are_disjoint_and_valid():
    historical_payload = HISTORICAL_FINGERPRINT.read_bytes()
    assert hashlib.sha256(historical_payload).hexdigest() == (
        HISTORICAL_FINGERPRINT_SHA256
    )
    campaign_manifest = (
        REPO_ROOT / "evidence/manifests/campaign.sha256"
    ).read_text().splitlines()
    assert (
        f"{HISTORICAL_FINGERPRINT_SHA256}  "
        "./sol_native_env_fingerprint_sm86.json"
    ) in campaign_manifest

    live_path = sol_reconstitution.SOL_NATIVE_ENV_FINGERPRINT
    assert live_path != HISTORICAL_FINGERPRINT
    live = json.loads(live_path.read_text())
    schema = json.loads((
        REPO_ROOT / "workloads/simbench_bw/schemas"
        / "sol_native_env_fingerprint.live.v2.schema.json"
    ).read_text())
    jsonschema.validate(live, schema)
    assert live["native_env_hash"] == hashlib.sha256(json.dumps(
        live["components"], sort_keys=True, separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")).hexdigest()
    assert sol_reconstitution.certified_native_env_hash(live_path) == (
        live["native_env_hash"]
    )


def _git_status(repo: Path) -> str:
    return subprocess.run(
        ["git", "status", "--short"], cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.rstrip("\n")
