"""Focused tests for the shared fail-closed path-clearance authority."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.local_labels import protected_paths
from tools.local_labels.protected_paths import (
    EXTERNAL_PROTECTED_ABSOLUTE,
    GIT_REPOSITORY_BARE,
    GIT_REPOSITORY_NOT_FOUND,
    GIT_REPOSITORY_STATUS_UNAVAILABLE,
    GIT_REPOSITORY_WORKTREE,
    REPO_PROTECTED_RELATIVE,
    ProtectedPathError,
    assert_external_output,
    assert_write_allowed,
    canonicalize_candidate,
    discover_repo_root,
    git_repository_status,
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert is_inside_git_worktree(REPO_ROOT / "future/output.json")
    assert git_repository_status(REPO_ROOT / "future/output.json") == (
        GIT_REPOSITORY_WORKTREE
    )
    external = tmp_path / "external/output.json"
    assert not is_inside_git_worktree(external)
    assert git_repository_status(external) == GIT_REPOSITORY_NOT_FOUND

    repo = tmp_path / "repo"
    linked = tmp_path / "linked"
    bare = tmp_path / "bare.git"
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
    assert git_repository_status(repo / "future/output.json") == (
        GIT_REPOSITORY_WORKTREE
    )
    assert git_repository_status(linked / "future/output.json") == (
        GIT_REPOSITORY_WORKTREE
    )
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    tracer = bare / "tracer_tool.so"
    tracer.write_bytes(b"tracer")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(bare))
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text("#!/bin/sh\nprintf 'false\\nfalse\\n'\n", encoding="utf-8")
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))
    assert not is_inside_git_worktree(tracer)
    assert git_repository_status(tracer) == GIT_REPOSITORY_BARE


def test_git_repository_status_preserves_malformed_probe_as_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        protected_paths.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["git"],
            returncode=0,
            stdout="false\nfalse\n",
            stderr="",
        ),
    )

    assert (
        git_repository_status(tmp_path / "tracer_tool.so")
        == GIT_REPOSITORY_STATUS_UNAVAILABLE
    )


def test_git_worktree_probe_execution_error_fails_closed_on_consumer_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_probe(*_args, **_kwargs):
        raise OSError("injected Git execution failure")

    monkeypatch.setattr(protected_paths.subprocess, "run", fail_probe)
    candidate = tmp_path / "external" / "result.json"

    assert git_repository_status(candidate) == GIT_REPOSITORY_STATUS_UNAVAILABLE
    with pytest.raises(ProtectedPathError, match="cannot determine Git-worktree"):
        is_inside_git_worktree(candidate)
    with pytest.raises(ProtectedPathError, match="cannot determine Git-worktree"):
        assert_external_output(candidate, repo_root=REPO_ROOT)


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr"),
    (
        (0, "false\nfalse\n", ""),
        (1, "", "injected rev-parse failure"),
        (128, "", "fatal: ambiguous probe failure"),
    ),
    ids=("malformed-success", "generic-nonzero", "unrecognized-128"),
)
def test_git_worktree_indeterminate_results_fail_closed_on_consumer_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stdout: str,
    stderr: str,
) -> None:
    monkeypatch.setattr(
        protected_paths.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["git"],
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        ),
    )
    candidate = tmp_path / "external" / "result.json"

    assert git_repository_status(candidate) == GIT_REPOSITORY_STATUS_UNAVAILABLE
    with pytest.raises(ProtectedPathError, match="cannot determine Git-worktree"):
        is_inside_git_worktree(candidate)
    with pytest.raises(ProtectedPathError, match="cannot determine Git-worktree"):
        assert_external_output(candidate, repo_root=REPO_ROOT)
