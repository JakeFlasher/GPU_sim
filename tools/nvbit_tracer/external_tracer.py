"""Fail-closed authority for tracers used to mint new NVBit evidence.

A capture may name its tracer through an explicit argument pair or through the
matching environment pair below.  There is deliberately no repository-local,
installed, PATH, or ignored-artifact discovery fallback.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from tools.local_labels.immutable_artifact import (
    ImmutableArtifactError,
    RetainedPathIdentity,
    open_retained_regular,
)
from tools.local_labels.protected_paths import (
    GIT_REPOSITORY_NOT_FOUND,
    GIT_REPOSITORY_STATUS_UNAVAILABLE,
    git_repository_status,
    is_inside_git_worktree,
)


EXTERNAL_TRACER_PATH_ENV = "NVBIT_EXTERNAL_TRACER_SO"
EXTERNAL_TRACER_SHA256_ENV = "NVBIT_EXTERNAL_TRACER_SHA256"
EXTERNAL_TRACER_SOURCE_KIND = "external_non_git_retained_descriptor_v1"
EXTERNAL_TRACER_MAX_BYTES = 256 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ExternalTracerAuthorityError(ValueError):
    """The requested tracer cannot authorize a new evidence capture."""


@dataclass(frozen=True)
class ExternalTracerAuthority:
    path: Path
    sha256: str
    source_kind: str = EXTERNAL_TRACER_SOURCE_KIND
    path_identity: RetainedPathIdentity | None = None


class _RetainedTracer(Protocol):
    path: Path
    sha256: str
    path_identity: RetainedPathIdentity


def _reject_claude_worktree_path(candidate: Path, *, label: str) -> None:
    if any(
        candidate.parts[index : index + 2] == (".claude", "worktrees")
        for index in range(len(candidate.parts) - 1)
    ):
        raise ExternalTracerAuthorityError(
            f"{label} must not come from a Claude worktree"
        )


def _resolve_symlink_free_absolute(candidate: Path, *, label: str) -> Path:
    if not candidate.is_absolute():
        raise ExternalTracerAuthorityError(f"{label} must be an absolute path")
    _reject_claude_worktree_path(candidate, label=label)
    lexical = Path(candidate.anchor)
    try:
        for component in candidate.parts[1:]:
            lexical /= component
            if lexical.is_symlink():
                raise ExternalTracerAuthorityError(
                    f"{label} path contains a symlink: {lexical}"
                )
        resolved = candidate.resolve(strict=True)
        _reject_claude_worktree_path(resolved, label=label)
        return resolved
    except OSError as exc:
        raise ExternalTracerAuthorityError(
            f"{label} authority path is unavailable: {exc}"
        ) from exc


def require_external_non_git_root(
    path: Path | str,
    *,
    repo_root: Path,
    label: str = "external root",
) -> Path:
    """Return one symlink-free external directory outside every Git tree."""

    candidate = Path(path)
    resolved = _resolve_symlink_free_absolute(candidate, label=label)
    try:
        resolved_repo = Path(repo_root).resolve(strict=True)
    except OSError as exc:
        raise ExternalTracerAuthorityError(
            f"repository root authority path is unavailable: {exc}"
        ) from exc
    try:
        resolved.relative_to(resolved_repo)
    except ValueError:
        pass
    else:
        raise ExternalTracerAuthorityError(
            f"{label} must be external to the repository"
        )

    try:
        metadata = resolved.lstat()
    except OSError as exc:
        raise ExternalTracerAuthorityError(f"{label} is unavailable: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ExternalTracerAuthorityError(f"{label} must be a directory")

    for ancestor in (resolved, *resolved.parents):
        marker = ancestor / ".git"
        try:
            marker.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ExternalTracerAuthorityError(
                f"{label} Git-worktree status is unavailable: {ancestor}"
            ) from exc
        raise ExternalTracerAuthorityError(
            f"{label} must come from a non-Git build tree"
        )

    repository_status = git_repository_status(resolved)
    if repository_status == GIT_REPOSITORY_STATUS_UNAVAILABLE:
        raise ExternalTracerAuthorityError(
            f"{label} Git-repository status is unavailable"
        )
    if repository_status != GIT_REPOSITORY_NOT_FOUND or is_inside_git_worktree(
        resolved
    ):
        raise ExternalTracerAuthorityError(
            f"{label} must come from a non-Git build tree"
        )
    return resolved


def require_external_non_git_artifact(
    path: Path | str,
    *,
    root: Path | str,
    repo_root: Path,
    label: str = "external artifact",
    max_bytes: int = EXTERNAL_TRACER_MAX_BYTES,
) -> Path:
    """Return one nonempty regular artifact beneath an authorized root."""

    if type(max_bytes) is not int or max_bytes <= 0:
        raise ExternalTracerAuthorityError(
            f"{label} byte ceiling must be a positive integer"
        )
    resolved_root = require_external_non_git_root(
        root, repo_root=repo_root, label=f"{label} root"
    )
    resolved = _resolve_symlink_free_absolute(Path(path), label=label)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ExternalTracerAuthorityError(
            f"{label} must be beneath its external root"
        ) from exc
    if not relative.parts:
        raise ExternalTracerAuthorityError(f"{label} must name a file")

    try:
        metadata = resolved.lstat()
    except OSError as exc:
        raise ExternalTracerAuthorityError(f"{label} is unavailable: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
        raise ExternalTracerAuthorityError(
            f"{label} must be a nonempty regular file"
        )
    if metadata.st_nlink != 1:
        raise ExternalTracerAuthorityError(
            f"{label} must not be a hard-linked alias"
        )
    if metadata.st_size > max_bytes:
        raise ExternalTracerAuthorityError(
            f"{label} exceeds the retained-source byte ceiling: "
            f"size={metadata.st_size}, max={max_bytes}"
        )
    return resolved


def _explicit_authority_values(
    path: Path | str | None,
    sha256: str | None,
    *,
    environment: Mapping[str, str],
) -> tuple[Path | str, str]:
    if (path is None) != (sha256 is None):
        raise ExternalTracerAuthorityError(
            "external tracer path and SHA-256 must be supplied together"
        )
    if path is not None and sha256 is not None:
        return path, sha256

    environment_path = environment.get(EXTERNAL_TRACER_PATH_ENV)
    environment_sha256 = environment.get(EXTERNAL_TRACER_SHA256_ENV)
    if not environment_path or not environment_sha256:
        raise ExternalTracerAuthorityError(
            "new capture requires --nvbit-tracer plus --nvbit-tracer-sha256 "
            f"or the explicit {EXTERNAL_TRACER_PATH_ENV}/"
            f"{EXTERNAL_TRACER_SHA256_ENV} environment pair"
        )
    return environment_path, environment_sha256


def require_external_tracer_authority(
    path: Path | str | None = None,
    sha256: str | None = None,
    *,
    repo_root: Path,
    environment: Mapping[str, str] | None = None,
) -> ExternalTracerAuthority:
    """Validate one exact non-Git tracer outside ``repo_root``.

    The lexical path must be absolute and symlink-free.  The resolved file may
    not be inside this repository, any other Git repository/worktree, or a
    Claude worktree.  The digest is authority supplied here and is rechecked
    against the retained file descriptor by ``verify_retained_tracer_authority``.
    """

    raw_path, raw_sha256 = _explicit_authority_values(
        path,
        sha256,
        environment=os.environ if environment is None else environment,
    )
    if type(raw_sha256) is not str or _SHA256.fullmatch(raw_sha256) is None:
        raise ExternalTracerAuthorityError(
            "external tracer SHA-256 must be exact lowercase 64-hex"
        )

    candidate = Path(raw_path)
    resolved = require_external_non_git_artifact(
        candidate,
        root=candidate.parent,
        repo_root=repo_root,
        label="NVBit tracer",
        max_bytes=EXTERNAL_TRACER_MAX_BYTES,
    )
    try:
        with open_retained_regular(
            resolved, max_bytes=EXTERNAL_TRACER_MAX_BYTES
        ) as retained:
            repeated = require_external_non_git_artifact(
                resolved,
                root=resolved.parent,
                repo_root=repo_root,
                label="NVBit tracer",
                max_bytes=EXTERNAL_TRACER_MAX_BYTES,
            )
            if repeated != resolved:
                raise ExternalTracerAuthorityError(
                    "NVBit tracer path changed during authority validation"
                )
            path_identity = retained.path_identity
    except ExternalTracerAuthorityError:
        raise
    except ImmutableArtifactError as exc:
        raise ExternalTracerAuthorityError(
            f"NVBit tracer identity could not be retained: {exc}"
        ) from exc
    return ExternalTracerAuthority(
        path=resolved,
        sha256=raw_sha256,
        path_identity=path_identity,
    )


def verify_retained_tracer_authority(
    retained_tracer: _RetainedTracer,
    authority: ExternalTracerAuthority,
) -> None:
    """Bind an opened retained tracer capability to its declared authority."""

    if not isinstance(authority, ExternalTracerAuthority):
        raise TypeError("authority must be an ExternalTracerAuthority")
    if retained_tracer.path != authority.path:
        raise ExternalTracerAuthorityError(
            "retained tracer path differs from external authority"
        )
    if authority.source_kind != EXTERNAL_TRACER_SOURCE_KIND:
        raise ExternalTracerAuthorityError(
            "external tracer authority source kind is unsupported"
        )
    if authority.path_identity is None:
        raise ExternalTracerAuthorityError(
            "external tracer authority lacks retained path identity"
        )
    if retained_tracer.path_identity != authority.path_identity:
        raise ExternalTracerAuthorityError(
            "retained tracer path identity differs from external authority"
        )
    if retained_tracer.sha256 != authority.sha256:
        raise ExternalTracerAuthorityError(
            "retained external tracer digest mismatch: "
            f"expected {authority.sha256}, got {retained_tracer.sha256}"
        )
