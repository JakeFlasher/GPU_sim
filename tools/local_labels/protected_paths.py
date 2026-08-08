"""Fail-closed path clearance for repository writers.

The local-label and NVBit tools have several machine-local output modes, but
they share protected evidence trees that must never be treated as scratch.
This module provides the small, read-only decision surface writers call before
their first ``mkdir``, removal, temporary-file creation, subprocess launch, or
write.

Clearance resolves the nearest existing parent before appending a nonexistent
suffix.  Consequently a symlinked parent cannot make a lexically harmless path
escape into a protected tree.  Both descendants *and ancestors* of a protected
root are rejected: removing ``data`` would be just as destructive as removing
``data/annex`` directly.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable


REPO_PROTECTED_RELATIVE = (
    ".humanize",
    "data/annex",
    "data/local_labels/campaign",
    "tools/sol_execbench/data",
    "references",
    ".toolchain",
    "evidence/round28",
    "workloads/portable_lcd/results",
)

EXTERNAL_PROTECTED_ABSOLUTE = (
    Path("/home/jakeshea/non_root_profiler/cross_arch_gpu_profiler_suite/.humanize"),
    Path("/home/jakeshea/sm86_reference_capture_r27"),
)

GIT_REPOSITORY_NOT_FOUND = "not_repository"
GIT_REPOSITORY_WORKTREE = "worktree"
GIT_REPOSITORY_BARE = "bare"
GIT_REPOSITORY_STATUS_UNAVAILABLE = "status_unavailable"
TRUSTED_GIT_EXECUTABLE = Path("/usr/bin/git")
_TRUSTED_GIT_EXECUTABLE = str(TRUSTED_GIT_EXECUTABLE)


class ProtectedPathError(ValueError):
    """A requested mutation overlaps protected state or an unsafe location."""


def _git_env() -> dict[str, str]:
    return retained_git_environment()


def retained_git_environment() -> dict[str, str]:
    """Return the complete nonambient environment for read-only Git authority."""

    return {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LANGUAGE": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }


def run_retained_git(
    retained_git: object,
    *,
    repo_root: Path | str,
    arguments: tuple[str, ...],
    text: bool = False,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Run one Git builtin through a retained executable capability."""

    root = Path(repo_root).resolve(strict=False)
    descriptor = getattr(retained_git, "descriptor", None)
    proc_path = getattr(retained_git, "proc_path", None)
    verify_unchanged = getattr(retained_git, "verify_unchanged", None)
    if (
        type(descriptor) is not int
        or descriptor < 0
        or type(proc_path) is not str
        or not proc_path
        or not callable(verify_unchanged)
    ):
        raise ProtectedPathError("retained Git capability is invalid")
    verify_unchanged()
    completed = subprocess.run(
        (
            proc_path,
            "--no-pager",
            "--no-optional-locks",
            "-C",
            str(root),
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            *arguments,
        ),
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        cwd="/",
        env=retained_git_environment(),
        pass_fds=(descriptor,),
    )
    verify_unchanged()
    return completed


def _nearest_existing_parent(path: Path) -> Path:
    cursor = path
    while not cursor.exists() and not cursor.is_symlink():
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    if cursor.is_file() or cursor.is_symlink():
        return cursor.parent
    return cursor


def discover_repo_root(start: Path | str | None = None) -> Path:
    """Return the canonical Git worktree root containing ``start``.

    ``start`` may name a not-yet-created output.  Git is invoked from its
    nearest existing directory with repository-local environment overrides
    removed so a caller cannot accidentally redirect this probe.
    """

    candidate = Path(start) if start is not None else Path(__file__)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    probe = _nearest_existing_parent(candidate)
    try:
        completed = subprocess.run(
            [_TRUSTED_GIT_EXECUTABLE, "-C", str(probe), "rev-parse", "--show-toplevel"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_git_env(),
        )
    except OSError as exc:
        raise ProtectedPathError(f"cannot execute Git to discover repository: {exc}") from exc
    if completed.returncode != 0 or not completed.stdout.strip():
        raise ProtectedPathError(
            f"path is not inside a discoverable Git worktree: {candidate}"
        )
    return Path(completed.stdout.strip()).resolve(strict=True)


def canonicalize_candidate(path: Path | str, *, repo_root: Path | str) -> Path:
    """Resolve an output path without requiring its final components to exist."""

    root = Path(repo_root).resolve(strict=True)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        return candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ProtectedPathError(f"cannot canonicalize output path {path!r}: {exc}") from exc


def _lexical_candidate(path: Path | str, *, repo_root: Path) -> Path:
    """Normalize ``.``/``..`` without following any filesystem symlink."""

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return Path(os.path.abspath(os.fspath(candidate)))


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def protected_roots(*, repo_root: Path | str) -> tuple[Path, ...]:
    """Return every canonical repository-local and external protected root."""

    root = Path(repo_root).resolve(strict=True)
    local = tuple((root / rel).resolve(strict=False) for rel in REPO_PROTECTED_RELATIVE)
    external = tuple(path.resolve(strict=False) for path in EXTERNAL_PROTECTED_ABSOLUTE)
    return (*local, *external)


def _lexical_protected_roots(*, repo_root: Path) -> tuple[Path, ...]:
    local = tuple(_lexical_candidate(rel, repo_root=repo_root)
                  for rel in REPO_PROTECTED_RELATIVE)
    external = tuple(Path(os.path.abspath(os.fspath(path)))
                     for path in EXTERNAL_PROTECTED_ABSOLUTE)
    return (*local, *external)


def _pb_ensure_exception_is_safe(
    *, repo_root: Path, lexical_candidate: Path, canonical_candidate: Path,
) -> bool:
    lexical_toolchain = _lexical_candidate(".toolchain", repo_root=repo_root)
    lexical_pb = _lexical_candidate(".toolchain/pb", repo_root=repo_root)
    canonical_toolchain = (repo_root / ".toolchain").resolve(strict=False)
    canonical_pb = (repo_root / ".toolchain/pb").resolve(strict=False)
    return (
        _contains(repo_root, canonical_toolchain)
        and _contains(lexical_toolchain, lexical_pb)
        and _contains(lexical_pb, lexical_candidate)
        and _contains(canonical_toolchain, canonical_pb)
        and _contains(canonical_pb, canonical_candidate)
    )


def _overlapping_roots(
    candidate: Path,
    *,
    repo_root: Path,
    allow_toolchain_pb_ensure: bool,
    roots: Iterable[Path] | None = None,
    pb_root: Path | None = None,
) -> list[Path]:
    pb_root = pb_root or (repo_root / ".toolchain" / "pb").resolve(strict=False)
    pb_exception = allow_toolchain_pb_ensure and _contains(pb_root, candidate)
    overlaps: list[Path] = []
    selected_roots = tuple(roots) if roots is not None else protected_roots(repo_root=repo_root)
    toolchain_root = pb_root.parent
    for protected in selected_roots:
        if protected == toolchain_root and pb_exception:
            continue
        if _contains(protected, candidate) or _contains(candidate, protected):
            overlaps.append(protected)
    return overlaps


def is_protected_path(path: Path | str, *, repo_root: Path | str) -> bool:
    """Return whether mutating ``path`` could mutate protected state."""

    root = Path(repo_root).resolve(strict=True)
    lexical = _lexical_candidate(path, repo_root=root)
    candidate = canonicalize_candidate(path, repo_root=root)
    lexical_roots = _lexical_protected_roots(repo_root=root)
    lexical_pb = _lexical_candidate(".toolchain/pb", repo_root=root)
    return bool(
        _overlapping_roots(
            lexical,
            repo_root=root,
            allow_toolchain_pb_ensure=False,
            roots=lexical_roots,
            pb_root=lexical_pb,
        )
        or _overlapping_roots(
            candidate,
            repo_root=root,
            allow_toolchain_pb_ensure=False,
        )
    )


def assert_write_allowed(
    path: Path | str,
    *,
    repo_root: Path | str,
    operation: str = "write",
    allow_toolchain_pb_ensure: bool = False,
) -> Path:
    """Return the canonical path or raise before the caller mutates anything.

    ``allow_toolchain_pb_ensure`` is the sole-writer exception.  It applies
    only to descendants of ``.toolchain/pb`` and is intended exclusively for
    the explicit pinned-environment ``--ensure`` operation.
    """

    root = Path(repo_root).resolve(strict=True)
    lexical = _lexical_candidate(path, repo_root=root)
    candidate = canonicalize_candidate(path, repo_root=root)
    pb_exception = allow_toolchain_pb_ensure and _pb_ensure_exception_is_safe(
        repo_root=root,
        lexical_candidate=lexical,
        canonical_candidate=candidate,
    )
    lexical_overlaps = _overlapping_roots(
        lexical,
        repo_root=root,
        allow_toolchain_pb_ensure=pb_exception,
        roots=_lexical_protected_roots(repo_root=root),
        pb_root=_lexical_candidate(".toolchain/pb", repo_root=root),
    )
    canonical_overlaps = _overlapping_roots(
        candidate,
        repo_root=root,
        allow_toolchain_pb_ensure=pb_exception,
    )
    overlaps = sorted({*lexical_overlaps, *canonical_overlaps}, key=str)
    if overlaps:
        rendered = ", ".join(str(item) for item in overlaps)
        raise ProtectedPathError(
            f"refusing to {operation} {candidate}: overlaps protected root(s) {rendered}"
        )
    return candidate


def git_repository_status(path: Path | str) -> str:
    """Classify Git membership, preserving unavailable or malformed probes."""

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    probe = _nearest_existing_parent(candidate.resolve(strict=False))
    env = _git_env()
    env.update({"LANG": "C", "LANGUAGE": "C", "LC_ALL": "C"})
    try:
        completed = subprocess.run(
            [
                _TRUSTED_GIT_EXECUTABLE,
                "-C",
                str(probe),
                "rev-parse",
                "--is-inside-work-tree",
                "--is-bare-repository",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
    except OSError:
        return GIT_REPOSITORY_STATUS_UNAVAILABLE
    stdout = completed.stdout.strip().splitlines()
    if completed.returncode == 0:
        if stdout == ["true", "false"]:
            return GIT_REPOSITORY_WORKTREE
        if stdout == ["false", "true"]:
            return GIT_REPOSITORY_BARE
        return GIT_REPOSITORY_STATUS_UNAVAILABLE
    stderr = completed.stderr.strip()
    if completed.returncode == 128 and stderr.startswith(
        "fatal: not a git repository"
    ):
        return GIT_REPOSITORY_NOT_FOUND
    return GIT_REPOSITORY_STATUS_UNAVAILABLE


def is_inside_git_worktree(path: Path | str) -> bool:
    """Return whether ``path`` resolves beneath any Git worktree.

    A recognized non-repository or bare-repository result is a definitive
    negative.  Execution failures, nonzero probe errors, and malformed output
    are indeterminate and therefore raise instead of clearing a write.
    """

    status = git_repository_status(path)
    if status == GIT_REPOSITORY_WORKTREE:
        return True
    if status in {GIT_REPOSITORY_NOT_FOUND, GIT_REPOSITORY_BARE}:
        return False
    raise ProtectedPathError(
        f"cannot determine Git-worktree containment for {Path(path)}"
    )


def assert_external_output(
    path: Path | str,
    *,
    repo_root: Path | str,
    canonical_paths: Iterable[Path | str] = (),
    operation: str = "emit",
) -> Path:
    """Clear an explicit verification output outside every Git worktree.

    Canonical authority paths are rejected explicitly, even if a symlink or an
    unusual worktree layout would otherwise obscure their identity.
    """

    root = Path(repo_root).resolve(strict=True)
    candidate = assert_write_allowed(path, repo_root=root, operation=operation)
    canonical = {
        canonicalize_candidate(item, repo_root=root) for item in canonical_paths
    }
    if candidate in canonical:
        raise ProtectedPathError(
            f"refusing to {operation} canonical authority via explicit-output mode: "
            f"{candidate}"
        )
    if is_inside_git_worktree(candidate):
        raise ProtectedPathError(
            f"refusing to {operation} explicit output inside a Git worktree: {candidate}"
        )
    return candidate
