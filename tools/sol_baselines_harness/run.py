#!/usr/bin/env python3
"""Drive one or more baselines through the official sol-execbench CLI.

Reads ``solution.json`` from each ``--baseline`` directory, resolves
the matching benchmark ``definition.json`` under
``tools/sol_execbench/data/benchmark/<subset>/<NNN_name>/``, and invokes the
upstream CLI as a native subprocess via
``baselines_harness.native_dispatch``. The CLI is invoked with
``--json`` and the JSONL ``Trace`` lines that come back are parsed
into one report row per workload.

Per-workload report row schema (one entry per JSONL output trace):

    {
      "slug": "problem_001_fused_add_rmsnorm_h2048",
      "language": "cuda_cpp",
      "workload_index": 0,
      "status": "PASSED" | "REWARD_HACK" | "COMPILE_ERROR" | ...,
      "latency_ms": <float | null>,
      "reference_latency_ms": <float | null>,
      "speedup_factor": <float | null>,
      "max_relative_error": <float | null>,
      "max_absolute_error": <float | null>,
      "has_nan": <bool | null>,
      "has_inf": <bool | null>,
      "hardware": <string | null>,
      "under_contention": <bool>,
      "definition": "001_fused_add_rmsnorm_h2048"
    }

The four correctness fields propagate from ``trace.evaluation.correctness``
(``None`` when the evaluator did not report — e.g. COMPILE_ERROR /
RUNTIME_ERROR rows where the Pydantic validator forbids correctness).

Workers > 1 tags every row with ``"under_contention": true`` — the
upstream evaluator only schedules one workload per worker, but
multiple workers contending on the same GPU corrupt latency
measurements (and ``reference_latency_ms`` for that matter). Default
``--workers 1`` keeps every row contention-free.

Usage:
    python tools/sol_baselines_harness/run.py \\
        --baseline baselines/FIB/problem_001_fused_add_rmsnorm_h2048/cuda_cpp

    # Dry-run: print the constructed command and exit.
    python tools/sol_baselines_harness/run.py \\
        --baseline baselines/.../cuda_cpp \\
        --print-cmd
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent
_REPO_ROOT = _THIS_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from tools.sol_baselines_harness._bootstrap import install_package_aliases

install_package_aliases()

from baselines_scaffold.generate import (  # noqa: E402
    DEFAULT_BENCHMARK_ROOT,
    SUBSET_MAP,
)
from baselines_harness.native_dispatch import (  # noqa: E402
    DispatchError,
    DispatchRequest,
    build_command,
    run as dispatch_run,
)
from baselines_harness.normalise_benchmark import (  # noqa: E402
    needs_staging,
    stage_benchmark,
)
from baselines_harness.parse_trace import parse_jsonl_lines  # noqa: E402


def _positive_workers(value: str) -> int:
    """Argparse type for ``--workers``: positive integer only.

    Rejects non-integer text and any integer < 1 with a clear
    actionable message: ``--workers N`` with ``N<1`` or a non-integer
    value is rejected before the subprocess fires.
    """
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--workers must be a positive integer; got {value!r}."
        ) from exc
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"--workers must be >= 1; got {n}."
        )
    return n


def _resolve_problem_dir(
    solution_dict: Dict[str, Any],
    baseline_dir: Path,
    benchmark_root: Path,
) -> Path:
    """Resolve the matching ``<subset>/<NNN_name>/`` benchmark directory."""
    parents = list(baseline_dir.resolve().parents)
    if len(parents) < 2:
        raise SystemExit(
            f"error: baseline {baseline_dir} does not match "
            f"baselines/<category>/<slug>/<language>/ layout."
        )
    category = parents[1].name
    subset = SUBSET_MAP.get(category)
    if subset is None:
        raise SystemExit(
            f"error: unknown category {category!r} for baseline {baseline_dir}."
        )
    bench_name = solution_dict.get("definition", "")
    if not bench_name:
        raise SystemExit(
            f"error: solution.json at {baseline_dir} is missing the "
            f"'definition' field."
        )
    bench_dir = benchmark_root / subset / bench_name
    if not bench_dir.is_dir():
        raise SystemExit(
            f"error: benchmark directory not found at {bench_dir}. Run "
            f"./tools/sol_execbench/scripts/download_data.sh to populate "
            f"tools/sol_execbench/data/benchmark/{subset}/, or check the "
            f"solution's definition field against the downloader name."
        )
    return bench_dir


def _row_from_trace(
    trace: Any,
    baseline_dir: Path,
    workload_index: int,
    under_contention: bool,
) -> Dict[str, Any]:
    """Build a per-workload report row from one parsed Trace."""
    slug = baseline_dir.parent.name
    language = baseline_dir.name
    row: Dict[str, Any] = {
        "slug": slug,
        "language": language,
        "workload_index": workload_index,
        "status": (
            trace.evaluation.status.value
            if trace.evaluation is not None
            else None
        ),
        "latency_ms": None,
        "reference_latency_ms": None,
        "speedup_factor": None,
        # Correctness metrics propagated from `trace.evaluation.correctness`
        # so per-workload precision evidence lives in machine
        # artifacts. Defaults are ``None`` so downstream consumers
        # can distinguish "evaluator didn't report" from "evaluator
        # reported zero".
        "max_relative_error": None,
        "max_absolute_error": None,
        "has_nan": None,
        "has_inf": None,
        "hardware": None,
        "under_contention": under_contention,
        "definition": trace.definition,
    }
    if trace.evaluation is None:
        return row
    if trace.evaluation.performance is not None:
        perf = trace.evaluation.performance
        row["latency_ms"] = perf.latency_ms
        row["reference_latency_ms"] = perf.reference_latency_ms
        row["speedup_factor"] = perf.speedup_factor
    if trace.evaluation.correctness is not None:
        corr = trace.evaluation.correctness
        row["max_relative_error"] = corr.max_relative_error
        row["max_absolute_error"] = corr.max_absolute_error
        row["has_nan"] = corr.has_nan
        row["has_inf"] = corr.has_inf
    row["hardware"] = trace.evaluation.environment.hardware
    # Propagate the upstream library version map so the report's
    # cache-key derivation can see Torch / CUDA / library versions.
    # Without this, `report._environment_libs_from_rows()` falls
    # back to `{}` and the `environment_libs` component of the cache
    # key never invalidates replays on an upstream version bump.
    row["environment_libs"] = dict(trace.evaluation.environment.libs)
    return row


def _count_expected_workloads(problem_dir: Path) -> int:
    """Count non-empty lines in ``workload.jsonl`` for accounting.

    Each non-empty / non-comment line in ``workload.jsonl`` corresponds
    to one workload the evaluator runs and (when successful) one row
    the harness should see. Used to detect partial output runs where
    the upstream CLI emitted some but not all rows.
    """
    workload_path = problem_dir / "workload.jsonl"
    if not workload_path.is_file():
        raise SystemExit(
            f"error: workload.jsonl not found at {workload_path}."
        )
    count = 0
    for raw in workload_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        count += 1
    return count


@dataclass(frozen=True)
class BaselineRunOutcome:
    """Container for one baseline's run result, including fail-closed signals."""

    rows: List[Dict[str, Any]]
    parse_errors: List[Dict[str, Any]]
    expected_rows: int
    returncode: int
    stderr_tail: str
    failure_reasons: List[str]

    def is_clean(self) -> bool:
        """True when every gate (upstream exit, row count, parse) is clean."""
        return not self.failure_reasons


def run_baseline(
    baseline_dir: Path,
    benchmark_root: Path,
    workers: int,
    timeout_seconds: Optional[float],
    print_cmd: bool,
) -> BaselineRunOutcome:
    """Run one baseline; return a structured outcome.

    The harness fails closed (non-empty ``failure_reasons``) when ANY of:

    - The upstream sol-execbench CLI exited non-zero.
    - Zero rows were parsed when at least one workload was expected.
    - Parse errors prevented complete per-workload row accounting.
    - The parsed row count does not match the benchmark's ``workload.jsonl``
      line count (under-runs and over-runs both surface).

    Known non-PASSED statuses themselves do not count as harness failure —
    they are the harness's primary signal and are written out so the
    deferred-ledger writer can branch on them. A parsed row with no
    evaluation status is malformed and does count as failure, as do
    silently-missing rows or upstream crashes: the deferred ledger requires
    per-pilot evidence rather than environment-driven empty runs.
    """
    solution_path = baseline_dir / "solution.json"
    if not solution_path.is_file():
        raise SystemExit(
            f"error: solution.json not found at {solution_path}. Run "
            f"scaffold.py + pack.py first."
        )
    solution_dict = json.loads(solution_path.read_text(encoding="utf-8"))
    canonical_problem_dir = _resolve_problem_dir(
        solution_dict, baseline_dir, benchmark_root
    )
    # Stage a Pydantic-compatible copy when the canonical definition
    # carries empty optional strings (upstream divergence at
    # sol_execbench/core/data/definition.py:166). The canonical path
    # still appears in parent READMEs (architecture-portability
    # discipline); only the harness dispatch sees the staged copy.
    if needs_staging(canonical_problem_dir):
        problem_dir = stage_benchmark(canonical_problem_dir)
    else:
        problem_dir = canonical_problem_dir
    expected_rows = _count_expected_workloads(problem_dir)
    # The upstream ``tools/sol_execbench`` Click CLI does not define a
    # ``--workers`` option, so forwarding ``--workers N`` here would
    # fail at argument parsing before any workload is evaluated.
    # Multi-worker parallelism for B200 phase-2 fan-out will be
    # implemented at the harness orchestration layer (running
    # multiple ``tools/sol_execbench`` subprocesses concurrently across
    # baselines), not by passing a flag the upstream CLI rejects.
    # The ``workers`` parameter is preserved as the harness's
    # public contract — callers still pass it, the value is
    # recorded in per-row evidence via ``under_contention``, and a
    # future harness-level parallel-runner can consume it without
    # an API break.
    request = DispatchRequest(
        problem_dir=problem_dir,
        solution_path=solution_path,
        extra_args=(),
        timeout_seconds=timeout_seconds,
    )
    if print_cmd:
        cmd = build_command(request)
        print(" ".join(cmd))
        return BaselineRunOutcome(
            rows=[],
            parse_errors=[],
            expected_rows=expected_rows,
            returncode=0,
            stderr_tail="",
            failure_reasons=[],
        )
    try:
        result = dispatch_run(request)
    except DispatchError as exc:
        raise SystemExit(f"error: {exc}")
    rows: List[Dict[str, Any]] = []
    errs: List[Dict[str, Any]] = []
    workload_index = 0
    for line_no, trace, raw in parse_jsonl_lines(result.stdout.splitlines()):
        if trace is None:
            errs.append({"line_no": line_no, "raw": raw})
            continue
        if trace.is_workload_trace():
            # Workload-only traces describe the problem's input space;
            # the harness skips them in the per-row report — they are
            # informational only and not gating.
            continue
        rows.append(
            _row_from_trace(
                trace,
                baseline_dir=baseline_dir,
                workload_index=workload_index,
                under_contention=(workers > 1),
            )
        )
        workload_index += 1
    failure_reasons = _collect_failure_reasons(
        result.returncode, rows, errs, expected_rows
    )
    return BaselineRunOutcome(
        rows=rows,
        parse_errors=errs,
        expected_rows=expected_rows,
        returncode=result.returncode,
        stderr_tail=result.stderr[-4000:] if result.stderr else "",
        failure_reasons=failure_reasons,
    )


def _collect_failure_reasons(
    returncode: int,
    rows: List[Dict[str, Any]],
    parse_errors: List[Dict[str, Any]],
    expected_rows: int,
) -> List[str]:
    """Decide which gates failed for this run.

    Order matters only for messaging; the harness fails closed on ANY
    non-empty reason list.
    """
    reasons: List[str] = []
    if returncode != 0:
        reasons.append(
            f"upstream sol-execbench CLI exited with returncode={returncode}"
        )
    if expected_rows > 0 and not rows:
        reasons.append(
            f"expected {expected_rows} workload row(s) from workload.jsonl; "
            f"parsed 0"
        )
    if expected_rows > 0 and rows and len(rows) != expected_rows:
        reasons.append(
            f"workload row count mismatch: expected {expected_rows} from "
            f"workload.jsonl, parsed {len(rows)}"
        )
    if any(not isinstance(row.get("status"), str) for row in rows):
        reasons.append(
            "one or more parsed Trace rows is missing an evaluation status"
        )
    if parse_errors:
        reasons.append(
            f"{len(parse_errors)} unparseable JSONL line(s) prevented "
            f"complete Trace validation"
        )
    return reasons


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/sol_baselines_harness/run.py",
        description=(
            "Drive one baseline through the official sol-execbench CLI; "
            "emit one report row per workload."
        ),
    )
    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help=(
            "Path to baselines/<cat>/<slug>/<lang>/ containing solution.json."
        ),
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
        help="Root containing <subset>/<NNN_name>/definition.json.",
    )
    parser.add_argument(
        "--workers",
        type=_positive_workers,
        default=1,
        help=(
            "Parallel worker count. Default 1 keeps timing contention-"
            "free; values > 1 tag every row with under_contention=true."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Subprocess timeout passed to the dispatch run.",
    )
    parser.add_argument(
        "--print-cmd",
        action="store_true",
        help=(
            "Print the constructed sol-execbench invocation and exit "
            "without running it. Useful when the host toolchain isn't "
            "ready yet."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional path to write the JSON report to. Default: print "
            "to stdout."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    outcome = run_baseline(
        baseline_dir=args.baseline,
        benchmark_root=args.benchmark_root,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
        print_cmd=args.print_cmd,
    )
    if args.print_cmd:
        return 0
    report = {
        "baseline": str(args.baseline),
        "workers": args.workers,
        "expected_rows": outcome.expected_rows,
        "returncode": outcome.returncode,
        "rows": outcome.rows,
        "parse_errors": outcome.parse_errors,
        "failure_reasons": outcome.failure_reasons,
    }
    if outcome.failure_reasons and outcome.stderr_tail:
        report["stderr_tail"] = outcome.stderr_tail
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if outcome.failure_reasons:
        # Print a short banner to stderr so CI logs surface the gate
        # failure even when the JSON report goes elsewhere.
        sys.stderr.write(
            "harness failed gate(s):\n"
            + "\n".join(f"  - {r}" for r in outcome.failure_reasons)
            + "\n"
        )
        return 2
    non_passed = [
        r for r in outcome.rows if r.get("status") and r["status"] != "PASSED"
    ]
    return 0 if not non_passed else 1


if __name__ == "__main__":
    sys.exit(main())
