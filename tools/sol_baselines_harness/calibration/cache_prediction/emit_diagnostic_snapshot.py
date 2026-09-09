"""Operator CLI driver for the intermediate-baseline diagnostic.

Thin argparse wrapper around `emit_intermediate_baseline()` that
fills operator-friendly defaults (canonical manifest paths,
current commit SHA via `git rev-parse HEAD`, default output path
under `.humanize/diagnostic-baselines/intermediate-baseline-<sha>.toml`)
so that the M2 diagnostic snapshot is one command from the repo
root.

Supported operator invocation (from the repo root):

    .venv/bin/python tools/sol_baselines_harness/calibration/cache_prediction/emit_diagnostic_snapshot.py

The direct script-path form is the documented contract: the
module inserts `<repo_root>/tools` onto `sys.path` at import
time, which makes the `sol_baselines_harness` package discoverable
to the script body. The `python -m
sol_baselines_harness.calibration.cache_prediction.emit_diagnostic_snapshot`
form does NOT work from the repo root because
`sol_baselines_harness` lives under `tools/` and is not on
Python's module-discovery path before the script body runs;
operators MUST use the direct script path.

The CLI prints a structured snapshot summary (output path,
denominator-box partition counts, dual-gate verdicts) and exits
with the canonical CLI exit-code ladder:

  * 0 — both contract gate AND user-acceptance gate pass.
  * 2 — contract / schema gate fails.
  * 3 — user-acceptance compound gate fails.
  * 4 — `IntermediateBaselineError` (IO, output-path, or
    round-trip serialization failure).
  * 5 — `git rev-parse HEAD` failed and no
    `--implementation-commit-sha` was provided.

This module is calibration-only and never imported by the
deployed harness path. The NCU quarantine scanner enforces that
contract.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[3]
if str(_REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tools"))

from sol_baselines_harness.calibration.cache_prediction.intermediate_baseline import (  # noqa: E402
    CLI_EXIT_BOTH_GATES_PASS,
    CLI_EXIT_SCHEMA_OR_CONTRACT_GATE_FAIL,
    CLI_EXIT_USER_ACCEPTANCE_GATE_FAIL,
    IntermediateBaselineError,
    IntermediateBaselineResult,
    emit_intermediate_baseline,
)


CLI_EXIT_INTERMEDIATE_BASELINE_ERROR: int = 4
CLI_EXIT_GIT_REV_PARSE_FAILED: int = 5


def _resolve_commit_sha(
    repo_root: Path,
    runner: Callable[..., subprocess.CompletedProcess] = (
        subprocess.run
    ),
) -> str:
    """Resolve the current HEAD's commit SHA via
    `git -C <repo_root> rev-parse HEAD`. Raises `RuntimeError`
    if the call fails (non-zero exit, missing git binary, or
    `repo_root` is not a git checkout).
    """
    try:
        completed = runner(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        stderr = (
            exc.stderr.strip()
            if isinstance(exc, subprocess.CalledProcessError)
            and exc.stderr
            else str(exc)
        )
        raise RuntimeError(
            f"git rev-parse HEAD failed in {repo_root!s}: "
            f"{stderr}",
        ) from exc
    sha = completed.stdout.strip()
    if not sha:
        raise RuntimeError(
            f"git rev-parse HEAD returned empty output in "
            f"{repo_root!s}",
        )
    return sha


def _print_summary(
    result: IntermediateBaselineResult,
    stream=None,
) -> None:
    out = stream if stream is not None else sys.stdout
    print(f"diagnostic snapshot: {result.output_path}", file=out)
    print(f"  total_problems = {result.total_problems}", file=out)
    print(f"  measured = {result.measured}", file=out)
    print(
        f"  typed_unavailable = {result.typed_unavailable}",
        file=out,
    )
    print(
        f"  diagnostic_failed = {result.diagnostic_failed}",
        file=out,
    )
    print(f"  corpus_pending = {result.corpus_pending}", file=out)
    print(
        f"  host_arch_unsupported = "
        f"{result.host_arch_unsupported}",
        file=out,
    )
    print(
        f"  passing_user_target = {result.passing_user_target}",
        file=out,
    )
    print(
        f"  passing_tiered_gate = {result.passing_tiered_gate}",
        file=out,
    )
    print(
        f"  passing_compound_gate = "
        f"{result.passing_compound_gate}",
        file=out,
    )
    print(
        f"  contract_gate_passed = "
        f"{result.contract_gate_passed}",
        file=out,
    )
    print(
        f"  user_compound_gate_passed = "
        f"{result.user_compound_gate_passed}",
        file=out,
    )
    print(f"  cli_exit_code = {result.cli_exit_code}", file=out)


def emit_diagnostic_snapshot(
    args: argparse.Namespace,
    *,
    sha_resolver: Callable[[Path], str] = _resolve_commit_sha,
    emitter: Callable[..., IntermediateBaselineResult] = (
        emit_intermediate_baseline
    ),
    stdout=None,
    stderr=None,
) -> int:
    """Run the operator-facing emission flow:
      1. Resolve `repo_root` (default: in-tree project root).
      2. Resolve `implementation_commit_sha` via `git rev-parse
         HEAD` if not provided.
      3. Resolve `output_path` via the intermediate-baseline
         module's default when not provided.
      4. Call `emitter(...)` and print the structured summary.

    Returns the CLI exit code per the documented ladder
    (0 / 2 / 3 from `cli_exit_code`; 4 on
    `IntermediateBaselineError`; 5 on git rev-parse failure).

    `sha_resolver`, `emitter`, `stdout`, `stderr` are injection
    seams for tests; production callers leave them at the
    default values.
    """
    repo_root = args.repo_root if args.repo_root is not None else (
        _REPO_ROOT
    )
    commit_sha = args.implementation_commit_sha
    err = stderr if stderr is not None else sys.stderr
    if commit_sha is None:
        try:
            commit_sha = sha_resolver(repo_root)
        except RuntimeError as exc:
            print(f"error: {exc}", file=err)
            return CLI_EXIT_GIT_REV_PARSE_FAILED

    try:
        result = emitter(
            output_path=args.output_path,
            repo_root=repo_root,
            implementation_commit_sha=commit_sha,
            denominator_floor=args.denominator_floor,
            accuracy_threshold=args.accuracy_threshold,
            host_arch_manifest_path=args.host_arch_manifest_path,
            calibration_validation_split_path=(
                args.calibration_validation_split_path
            ),
        )
    except IntermediateBaselineError as exc:
        print(
            f"error: intermediate baseline emission failed: "
            f"{exc.typed_reason}: {exc.message}",
            file=err,
        )
        return CLI_EXIT_INTERMEDIATE_BASELINE_ERROR

    _print_summary(result, stream=stdout)
    return result.cli_exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Emit the intermediate-baseline diagnostic snapshot "
            "and print partition + dual-gate verdicts. Exits "
            "with the canonical CLI exit-code ladder."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "explicit output path for the snapshot TOML "
            "(default: "
            ".humanize/diagnostic-baselines/intermediate-baseline-"
            "<commit_sha>.toml)"
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "repository root (default: in-tree project root)"
        ),
    )
    parser.add_argument(
        "--implementation-commit-sha",
        type=str,
        default=None,
        help=(
            "commit SHA to record in the snapshot's "
            "[meta.implementation_commit_sha] (default: HEAD "
            "via `git rev-parse HEAD` in the resolved repo root)"
        ),
    )
    parser.add_argument(
        "--denominator-floor",
        type=float,
        default=0.01,
        help="denominator floor for rel-err computation",
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=0.10,
        help=(
            "per-cell rel-err threshold for the user-acceptance "
            "compound gate (default: 0.10, the strict <10%% "
            "uniform threshold per DEC-1)"
        ),
    )
    parser.add_argument(
        "--host-arch-manifest-path",
        type=Path,
        default=None,
        help=(
            "explicit host-arch availability manifest path "
            "(default: in-tree "
            "tools/sol_baselines_harness/calibration/"
            "cache_prediction/host_arch_availability_v1.toml)"
        ),
    )
    parser.add_argument(
        "--calibration-validation-split-path",
        type=Path,
        default=None,
        help=(
            "explicit calibration/validation split manifest "
            "path (default: in-tree "
            "tools/sol_baselines_harness/calibration/"
            "cache_prediction/calibration_validation_split_v1.toml)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(
        argv if argv is not None else sys.argv[1:],
    )
    return emit_diagnostic_snapshot(args)


__all__ = [
    "CLI_EXIT_BOTH_GATES_PASS",
    "CLI_EXIT_GIT_REV_PARSE_FAILED",
    "CLI_EXIT_INTERMEDIATE_BASELINE_ERROR",
    "CLI_EXIT_SCHEMA_OR_CONTRACT_GATE_FAIL",
    "CLI_EXIT_USER_ACCEPTANCE_GATE_FAIL",
    "emit_diagnostic_snapshot",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
