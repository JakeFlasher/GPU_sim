"""Focused tests for the shared fail-closed path-clearance authority."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.local_labels.protected_paths import (
    EXTERNAL_PROTECTED_ABSOLUTE,
    REPO_PROTECTED_RELATIVE,
    ProtectedPathError,
    assert_external_output,
    assert_write_allowed,
    canonicalize_candidate,
    discover_repo_root,
    is_inside_git_worktree,
    is_protected_path,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_discovers_repository_root_and_anchors_relative_candidates() -> None:
    assert discover_repo_root(Path(__file__)) == REPO_ROOT
    assert canonicalize_candidate(
        "results/nvbit_capture_staging/probe.json", repo_root=REPO_ROOT
    ) == REPO_ROOT / "results/nvbit_capture_staging/probe.json"


def test_every_declared_repository_protected_root_rejects_descendants() -> None:
    for relative in REPO_PROTECTED_RELATIVE:
        candidate = REPO_ROOT / relative / "nested" / "output.json"
        assert is_protected_path(candidate, repo_root=REPO_ROOT)
        with pytest.raises(ProtectedPathError, match="protected root"):
            assert_write_allowed(candidate, repo_root=REPO_ROOT, operation="test write")
    for root in EXTERNAL_PROTECTED_ABSOLUTE:
        with pytest.raises(ProtectedPathError, match="protected root"):
            assert_write_allowed(root / "nested/output.json", repo_root=REPO_ROOT)


def test_mutating_an_ancestor_of_a_protected_root_is_rejected() -> None:
    with pytest.raises(ProtectedPathError, match="data/annex"):
        assert_write_allowed(REPO_ROOT / "data", repo_root=REPO_ROOT, operation="remove")


def test_complete_portable_results_tree_is_protected() -> None:
    for suffix in ("slug", "slug/sku/sm_86/trace_timing_shape", "new/future/file"):
        with pytest.raises(ProtectedPathError):
            assert_write_allowed(
                REPO_ROOT / "workloads/portable_lcd/results" / suffix,
                repo_root=REPO_ROOT,
            )


def test_active_capture_staging_destination_is_allowed() -> None:
    candidate = REPO_ROOT / "results/nvbit_capture_staging/cell/trace"
    assert assert_write_allowed(candidate, repo_root=REPO_ROOT) == candidate


def test_nearest_existing_parent_resolution_rejects_symlink_escape(
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "repo"
    protected = fake_root / "data" / "annex"
    staging = fake_root / "staging"
    protected.mkdir(parents=True)
    staging.mkdir()
    (staging / "alias").symlink_to(protected, target_is_directory=True)
    with pytest.raises(ProtectedPathError, match="protected root"):
        assert_write_allowed(
            staging / "alias" / "not-created" / "payload.json",
            repo_root=fake_root,
        )
    (protected / "escape").symlink_to(staging, target_is_directory=True)
    with pytest.raises(ProtectedPathError, match="protected root"):
        assert_write_allowed(
            protected / "escape" / "not-created" / "payload.json",
            repo_root=fake_root,
        )


def test_toolchain_pb_exception_is_narrow_and_explicit(tmp_path: Path) -> None:
    prefix_file = REPO_ROOT / ".toolchain/pb/conda-meta/new-record.json"
    with pytest.raises(ProtectedPathError):
        assert_write_allowed(prefix_file, repo_root=REPO_ROOT)
    assert (
        assert_write_allowed(
            prefix_file,
            repo_root=REPO_ROOT,
            allow_toolchain_pb_ensure=True,
            operation="pinned environment ensure",
        )
        == prefix_file
    )
    with pytest.raises(ProtectedPathError):
        assert_write_allowed(
            REPO_ROOT / ".toolchain/other/file",
            repo_root=REPO_ROOT,
            allow_toolchain_pb_ensure=True,
        )
    fake_root = tmp_path / "repo"
    outside = tmp_path / "outside"
    (fake_root / ".toolchain").mkdir(parents=True)
    outside.mkdir()
    (fake_root / ".toolchain/pb").symlink_to(
        outside, target_is_directory=True)
    with pytest.raises(ProtectedPathError):
        assert_write_allowed(
            fake_root / ".toolchain/pb/conda-meta/new-record.json",
            repo_root=fake_root,
            allow_toolchain_pb_ensure=True,
        )


def test_explicit_output_rejects_canonical_and_other_worktree_paths() -> None:
    canonical = REPO_ROOT / "workloads/portable_lcd/PRESERVED_BINARIES.md"
    with pytest.raises(ProtectedPathError, match="canonical authority"):
        assert_external_output(
            canonical,
            repo_root=REPO_ROOT,
            canonical_paths=(canonical,),
        )
    with pytest.raises(ProtectedPathError, match="Git worktree"):
        assert_external_output(
            REPO_ROOT / "scratch/preserved.md",
            repo_root=REPO_ROOT,
        )


def test_explicit_output_accepts_external_temporary_destination(tmp_path: Path) -> None:
    candidate = tmp_path / "preserved.json"
    assert assert_external_output(candidate, repo_root=REPO_ROOT) == candidate


def test_git_worktree_detection_distinguishes_repo_and_external_tmp(
    tmp_path: Path,
) -> None:
    assert is_inside_git_worktree(REPO_ROOT / "future/output.json")
    external = tmp_path / "external/output.json"
    assert not is_inside_git_worktree(external)

    repo = tmp_path / "repo"
    linked = tmp_path / "linked"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "test"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    (repo / "seed").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "seed"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "seed"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-q", "--detach", str(linked)],
        check=True,
    )
    assert is_inside_git_worktree(linked / "future/output.json")
