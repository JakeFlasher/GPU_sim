"""Deterministic path-scoped guard for the committed label-artifact tree.

Brackets a test sweep (or any command) with a pre/post integrity check
over the committed ``data/local_labels/`` tree:

  * pre: sha256 every git-tracked file under the scope + record the
    untracked (non-ignored) file set;
  * run the bracketed command;
  * post: recompute both and emit a typed verdict. Any changed /
    added / removed tracked file or any new untracked file under the
    scope is committed-artifact drift (``typed_reason:
    committed_artifact_tree_drift``) — test pollution the sweep must
    not produce.

This replaces a bare ``git status`` check: the verdict is path-scoped
(only the guarded scope), content-based (sha256 over bytes, so a
rewrite-then-revert still verifies byte-identity), and enumerates the
exact offending paths. Gitignored scratch outputs (e.g.
``data/local_labels/uncommitted/``, ``*.tmp``, ``*.log``) never trip
the untracked detection — ``git ls-files --others --exclude-standard``
excludes them by construction.

CLI::

    # one-shot bracketing (the canonical sweep usage)
    python tools/local_labels/artifact_tree_guard.py run \
        [--repo-root R] [--scope data/local_labels] \
        [--verdict-out VERDICT.json] -- <command> [args...]

    # split form for callers that need the snapshot on disk
    python tools/local_labels/artifact_tree_guard.py snapshot \
        --output SNAPSHOT.json [--repo-root R] [--scope ...]
    python tools/local_labels/artifact_tree_guard.py verify \
        --against SNAPSHOT.json [--verdict-out ...] [--repo-root R]

Exit codes (run / verify): 0 = tree clean AND bracketed command
succeeded; 2 = tree clean but the bracketed command exited non-zero
(e.g. a sweep with pre-existing documented test failures — the guard
verdict is still ``clean``); 3 = committed-artifact drift detected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SCOPE = "data/local_labels"

# Sentinel hash value for a tracked path whose file is missing from
# the working tree (deleted but not yet committed).
ABSENT = "__absent_from_working_tree__"

EXIT_CLEAN = 0
EXIT_COMMAND_FAILED = 2
EXIT_TREE_DRIFT = 3

TYPED_REASON_DRIFT = "committed_artifact_tree_drift"


def _git_lines(args: list[str], *, repo_root: Path) -> list[str]:
    out = subprocess.check_output(
        ["git", "-C", str(repo_root), *args],
    )
    return [
        line for line in out.decode("utf-8").split("\0") if line
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_tree_state(
    *, repo_root: Path, scope: str = DEFAULT_SCOPE,
) -> dict[str, Any]:
    """Snapshot the guarded scope: sha256 per tracked file + the
    untracked (non-ignored) path set."""
    tracked_paths = _git_lines(
        ["ls-files", "-z", "--", scope], repo_root=repo_root,
    )
    tracked_sha256: dict[str, str] = {}
    for rel in tracked_paths:
        path = repo_root / rel
        tracked_sha256[rel] = (
            _sha256_file(path) if path.is_file() else ABSENT
        )
    untracked = sorted(
        _git_lines(
            ["ls-files", "-z", "--others", "--exclude-standard",
             "--", scope],
            repo_root=repo_root,
        ),
    )
    return {
        "schema_version": 1,
        "scope": scope,
        "tracked_file_count": len(tracked_sha256),
        "tracked_sha256": tracked_sha256,
        "untracked": untracked,
    }


def compare_tree_states(
    pre: dict[str, Any], post: dict[str, Any],
) -> dict[str, Any]:
    """Typed verdict over two snapshots of the same scope."""
    pre_hashes: dict[str, str] = pre["tracked_sha256"]
    post_hashes: dict[str, str] = post["tracked_sha256"]
    changed_tracked = sorted(
        rel for rel, digest in pre_hashes.items()
        if rel in post_hashes and post_hashes[rel] != digest
    )
    removed_tracked = sorted(set(pre_hashes) - set(post_hashes))
    added_tracked = sorted(set(post_hashes) - set(pre_hashes))
    new_untracked = sorted(
        set(post["untracked"]) - set(pre["untracked"]),
    )
    clean = not (
        changed_tracked or removed_tracked or added_tracked
        or new_untracked
    )
    return {
        "schema_version": 1,
        "scope": pre["scope"],
        "status": "clean" if clean else "dirty",
        "typed_reason": None if clean else TYPED_REASON_DRIFT,
        "pre_tracked_file_count": pre["tracked_file_count"],
        "post_tracked_file_count": post["tracked_file_count"],
        "changed_tracked": changed_tracked,
        "added_tracked": added_tracked,
        "removed_tracked": removed_tracked,
        "new_untracked": new_untracked,
    }


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _emit_verdict(
    verdict: dict[str, Any], verdict_out: str | None,
) -> None:
    rendered = json.dumps(verdict, indent=2, sort_keys=True)
    if verdict_out:
        _write_json(verdict, Path(verdict_out))
    print(rendered)


def run_guarded_command(
    *,
    repo_root: Path,
    scope: str,
    command: list[str],
    verdict_out: str | None,
) -> int:
    """Pre-snapshot -> run command -> post-compare -> typed verdict."""
    pre = collect_tree_state(repo_root=repo_root, scope=scope)
    completed = subprocess.run(command)  # noqa: S603 — caller's command
    post = collect_tree_state(repo_root=repo_root, scope=scope)
    verdict = compare_tree_states(pre, post)
    verdict["command"] = command
    verdict["command_exit_code"] = completed.returncode
    _emit_verdict(verdict, verdict_out)
    if verdict["status"] != "clean":
        return EXIT_TREE_DRIFT
    if completed.returncode != 0:
        return EXIT_COMMAND_FAILED
    return EXIT_CLEAN


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-root", default=str(REPO_ROOT))
    common.add_argument("--scope", default=DEFAULT_SCOPE)

    p_snapshot = sub.add_parser(
        "snapshot", parents=[common],
        help="write a snapshot of the guarded scope",
    )
    p_snapshot.add_argument("--output", required=True)

    p_verify = sub.add_parser(
        "verify", parents=[common],
        help="compare the current tree against a snapshot",
    )
    p_verify.add_argument("--against", required=True)
    p_verify.add_argument("--verdict-out", default=None)

    p_run = sub.add_parser(
        "run", parents=[common],
        help="bracket a command with pre/post snapshots",
    )
    p_run.add_argument("--verdict-out", default=None)
    p_run.add_argument(
        "command", nargs=argparse.REMAINDER,
        help="command to bracket (prefix with -- to separate)",
    )

    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root)

    if args.mode == "snapshot":
        state = collect_tree_state(
            repo_root=repo_root, scope=args.scope,
        )
        _write_json(state, Path(args.output))
        print(
            f"[artifact_tree_guard] snapshot scope={args.scope} "
            f"tracked={state['tracked_file_count']} "
            f"untracked={len(state['untracked'])} -> {args.output}",
        )
        return EXIT_CLEAN

    if args.mode == "verify":
        pre = json.loads(
            Path(args.against).read_text(encoding="utf-8"),
        )
        post = collect_tree_state(
            repo_root=repo_root, scope=pre["scope"],
        )
        verdict = compare_tree_states(pre, post)
        _emit_verdict(verdict, args.verdict_out)
        return (
            EXIT_CLEAN if verdict["status"] == "clean"
            else EXIT_TREE_DRIFT
        )

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("run mode requires a command after --")
    return run_guarded_command(
        repo_root=repo_root,
        scope=args.scope,
        command=command,
        verdict_out=args.verdict_out,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
