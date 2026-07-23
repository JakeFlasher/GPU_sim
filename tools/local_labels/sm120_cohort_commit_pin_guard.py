"""sm_120 L2 cohort commit-pin guard (AC-2.5 strengthening).

Verifies that a pinned (commit_sha, path, expected_sha256) tuple still
holds. Three checks:

1. `commit_sha` is a real commit object in the current git repository.
2. The content of `path` at that commit hashes to `expected_sha256`.
3. The current working-tree content of `path` also hashes to
   `expected_sha256`.

Either drift (commit drift, working-tree drift) rejects the pin.

The Round 0 contract pins the frozen roster's commit SHA and the
SHA256 of its content; downstream rounds invoke this guard before any
NCU sweep to refuse measurement when the roster has drifted.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

__version__ = "1.0.0"


class CommitPinError(Exception):
    """Raised when the commit-pin guard rejects. Carries the per-pin
    defects so multiple pins can be reported in one CLI run."""

    def __init__(self, defects: list[str]) -> None:
        super().__init__(f"{len(defects)} commit-pin defects")
        self.defects = list(defects)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _commit_exists(commit_sha: str, *, repo: Path) -> bool:
    try:
        subprocess.check_output(
            ["git", "rev-parse", "--verify", f"{commit_sha}^{{commit}}"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def _content_at_commit(commit_sha: str, path: str, *, repo: Path) -> bytes | None:
    """Return the bytes of `path` as of `commit_sha`, or None if not
    present at that commit."""
    try:
        return subprocess.check_output(
            ["git", "show", f"{commit_sha}:{path}"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return None


def verify_pin(
    commit_sha: str,
    repo_relative_path: str,
    expected_sha256: str,
    *,
    repo: Path,
) -> list[str]:
    """Verify a (commit, path, sha256) pin. Returns a list of defect
    strings (empty when all three checks pass)."""
    defects: list[str] = []
    expected = expected_sha256.lower()

    if not _commit_exists(commit_sha, repo=repo):
        defects.append(
            f"commit_pin: commit {commit_sha!r} does not exist in the "
            f"repository at {repo}"
        )
        return defects

    blob = _content_at_commit(commit_sha, repo_relative_path, repo=repo)
    if blob is None:
        defects.append(
            f"commit_pin: path {repo_relative_path!r} not present at "
            f"commit {commit_sha!r}"
        )
        return defects

    commit_tree_sha = _sha256_bytes(blob)
    if commit_tree_sha != expected:
        defects.append(
            f"commit_pin: commit-tree SHA256 of {repo_relative_path!r} at "
            f"{commit_sha!r} is {commit_tree_sha}; expected {expected}"
        )

    abs_path = repo / repo_relative_path
    if not abs_path.is_file():
        defects.append(
            f"commit_pin: working-tree path {repo_relative_path!r} does "
            f"not exist at {abs_path}"
        )
        return defects

    working_sha = _sha256_file(abs_path)
    if working_sha != expected:
        defects.append(
            f"commit_pin: working-tree SHA256 of {repo_relative_path!r} "
            f"is {working_sha}; expected {expected}"
        )

    return defects


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--commit-sha",
        required=True,
        help="The pinned commit SHA (full or unambiguous prefix).",
    )
    p.add_argument(
        "--path",
        required=True,
        help="Repository-relative path to verify.",
    )
    p.add_argument(
        "--expected-sha256",
        required=True,
        help="Expected SHA256 of the file's content.",
    )
    p.add_argument(
        "--repo",
        default=None,
        type=Path,
        help="Repository root (defaults to the script's containing repo).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo = args.repo if args.repo else Path(__file__).resolve().parents[2]
    defects = verify_pin(
        args.commit_sha, args.path, args.expected_sha256, repo=repo
    )
    if defects:
        for d in defects:
            print(f"DEFECT: {d}", file=sys.stderr)
        print(
            f"FAIL: commit-pin guard rejected {args.path!r} "
            f"({len(defects)} defects)",
            file=sys.stderr,
        )
        return 2
    print(
        f"OK: commit-pin guard verified {args.path!r} at "
        f"{args.commit_sha[:12]} -> {args.expected_sha256[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
