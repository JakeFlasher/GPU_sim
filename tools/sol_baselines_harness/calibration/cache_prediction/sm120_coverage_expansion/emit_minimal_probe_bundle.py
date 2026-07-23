"""Emit a minimal harness trace bundle from sol-execbench-only
correctness evidence.

The full harness (``scripts/baselines_harness/run.py``) drives
``sol-execbench`` AND the five counter-free analytic surfaces,
producing a rich
``scripts/baselines_harness/.harness_traces/<slug>__<lang>/``
bundle with per-surface manifests. The
``compile_verification_probe`` module then walks those bundles
to extract per-slug ``compile_verified`` +
``sol_execbench_correct_verified`` booleans for the ladder
reducer.

For new ports whose correctness has been verified via
``sol-execbench`` alone (skipping the surface stack), this
module emits the MINIMAL bundle subset the probe needs:

* ``manifest.json`` carrying ``arch_tag="sm_120"`` plus a
  documented ``provenance.source = "sol_execbench_only"``
  marker so downstream consumers can see the bundle was NOT
  produced by the full harness surface stack.
* ``trace.json`` carrying one row per ``sol-execbench``
  workload with the row's ``status`` (the probe's
  correctness path checks ``status == "PASSED"`` on every
  row to set ``sol_execbench_correct_verified=True``).

The probe's compile-evidence check accepts either a surface
with ``status=ok`` OR at least one trace row (sol-execbench
emits a workload row only after a successful build, so any
row is transitive evidence of a clean compile). Minimal
bundles therefore promote the ladder reducer's per-cell state
to ``sol_execbench_correct`` without re-running the full
surface stack.

CLI:

    python -m baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.emit_minimal_probe_bundle \\
        --baseline baselines/L1/<slug>/cuda_cpp \\
        [--baseline ...]
        [--traces-root scripts/baselines_harness/.harness_traces]

For each baseline, the helper runs ``sol-execbench`` (the
in-tree CLI installed in the workspace venv), captures the
per-workload JSON output, and writes
``<traces-root>/<slug>__<lang>/manifest.json`` and
``<traces-root>/<slug>__<lang>/trace.json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_TRACES_ROOT = (
    _REPO_ROOT / "scripts" / "baselines_harness" / ".harness_traces"
)
_BENCHMARK_ROOT = _REPO_ROOT / "sol-execbench" / "data" / "benchmark"
_SOL_EXECBENCH_BIN = _REPO_ROOT / ".venv" / "bin" / "sol-execbench"


def _slug_and_language_from_baseline(baseline_dir: Path) -> tuple[str, str]:
    """``baselines/<cat>/<slug>/<lang>/`` -> (slug, language).

    Raises ``ValueError`` when the path shape does not match the
    canonical ``baselines/<cat>/<slug>/<lang>/`` layout."""
    parts = baseline_dir.resolve().parts
    if len(parts) < 4 or parts[-4] != "baselines":
        raise ValueError(
            f"baseline_dir {baseline_dir!r} does not match the canonical "
            "baselines/<cat>/<slug>/<lang>/ layout"
        )
    language = parts[-1]
    slug = parts[-2]
    return slug, language


def _benchmark_subdir_for_slug(slug: str) -> Path:
    """Resolve ``sol-execbench/data/benchmark/<subset>/<NNN_name>/``
    for the given slug.

    The slug is ``problem_<NNN>_<rest>``; the benchmark dir name is
    ``<NNN>_<rest>``. The subset (L1, L2, Quant, FIB) is discovered
    by walking the benchmark root."""
    if not slug.startswith("problem_"):
        raise ValueError(
            f"slug {slug!r} does not start with the canonical 'problem_' "
            "prefix"
        )
    benchmark_name = slug[len("problem_"):]
    for subset in ("L1", "L2", "Quant", "FIB"):
        candidate = _BENCHMARK_ROOT / subset / benchmark_name
        if (candidate / "definition.json").is_file():
            return candidate
    raise FileNotFoundError(
        f"benchmark definition.json not found under "
        f"{_BENCHMARK_ROOT}/{{L1,L2,Quant,FIB}}/{benchmark_name}/"
    )


class MinimalProbeBundleError(RuntimeError):
    """Raised when the helper cannot honestly attest a port's
    sol-execbench correctness for the probe.

    Fail-closed contract (closed-vocabulary tokens recorded in
    ``MINIMAL_PROBE_BUNDLE_FAILURE_VOCABULARY``):
      - ``sol_execbench_subprocess_failed`` — non-zero exit.
      - ``sol_execbench_emitted_zero_rows`` — no parseable JSON.
      - ``sol_execbench_row_status_not_passed`` — at least one
        row's ``evaluation.status`` is not ``"PASSED"``. The
        bundle would silently over-promote the cell's ladder
        state, so the helper refuses to write it.

    Why fail-closed here matters: the helper is the evidence
    bridge between sol-execbench correctness and the harness
    compile-verification probe. Any non-PASSED row, if quietly
    accepted, would still cause the downstream
    ``_bundle_correctness_evidence`` check to demote the cell —
    but only after the bundle is emitted. The helper CLI would
    print ``emitted:`` and exit 0, masking the partial-failure
    signal from operators driving R22+ the calibration workstream batches.
    """


MINIMAL_PROBE_BUNDLE_FAILURE_VOCABULARY: tuple[str, ...] = (
    "sol_execbench_subprocess_failed",
    "sol_execbench_emitted_zero_rows",
    "sol_execbench_row_status_not_passed",
)


def _run_sol_execbench(
    *,
    benchmark_dir: Path,
    solution_path: Path,
) -> list[dict[str, Any]]:
    """Run ``sol-execbench --json`` and return one parsed dict per
    workload row.

    Fail-closed: raises ``MinimalProbeBundleError`` if the
    subprocess exits non-zero, emits zero parseable JSON rows,
    or any row's ``evaluation.status`` is not ``"PASSED"``. The
    helper never emits a bundle for a partially-passing port.
    """
    argv = [
        str(_SOL_EXECBENCH_BIN),
        str(benchmark_dir),
        "--solution",
        str(solution_path),
        "--json",
    ]
    proc = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    if proc.returncode != 0:
        raise MinimalProbeBundleError(
            "sol_execbench_subprocess_failed: "
            f"returncode={proc.returncode} for {benchmark_dir.name}; "
            f"stderr_tail={proc.stderr[-400:]!r}"
        )
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        raise MinimalProbeBundleError(
            "sol_execbench_emitted_zero_rows: no parseable JSON "
            f"rows for {benchmark_dir.name}; "
            f"stderr_tail={proc.stderr[-400:]!r}"
        )
    non_passed = [
        idx
        for idx, row in enumerate(rows)
        if row.get("evaluation", {}).get("status") != "PASSED"
    ]
    if non_passed:
        raise MinimalProbeBundleError(
            "sol_execbench_row_status_not_passed: "
            f"{len(non_passed)} of {len(rows)} rows had "
            f"status != PASSED for {benchmark_dir.name}; "
            f"failing-row indices: {non_passed[:8]}"
        )
    return rows


def _row_to_trace_entry(
    *,
    sol_execbench_row: dict[str, Any],
    slug: str,
    language: str,
    workload_index: int,
) -> dict[str, Any]:
    """Convert one sol-execbench --json line into the trace.json
    row shape the probe + downstream consumers expect."""
    evaluation = sol_execbench_row.get("evaluation", {})
    correctness = evaluation.get("correctness", {}) or {}
    performance = evaluation.get("performance", {}) or {}
    environment = evaluation.get("environment", {}) or {}
    return {
        "slug": slug,
        "language": language,
        "workload_index": workload_index,
        "status": evaluation.get("status", ""),
        "latency_ms": performance.get("latency_ms"),
        "reference_latency_ms": performance.get("reference_latency_ms"),
        "speedup_factor": performance.get("speedup_factor"),
        "max_relative_error": correctness.get("max_relative_error"),
        "max_absolute_error": correctness.get("max_absolute_error"),
        "has_nan": correctness.get("has_nan", False),
        "has_inf": correctness.get("has_inf", False),
        "hardware": environment.get("hardware", ""),
        "under_contention": False,
        "definition": sol_execbench_row.get("definition", ""),
        "environment_libs": environment.get("libs", {}),
    }


def _emit_bundle_for_baseline(
    *,
    baseline_dir: Path,
    traces_root: Path,
    captured_at_utc: str,
) -> Path:
    """Emit minimal manifest.json + trace.json under
    ``traces_root/<slug>__<language>/``. Returns the bundle dir.
    Raises if the baseline_dir layout is wrong or sol-execbench
    fails."""
    slug, language = _slug_and_language_from_baseline(baseline_dir)
    solution_path = baseline_dir / "solution.json"
    if not solution_path.is_file():
        raise FileNotFoundError(
            f"solution.json not found at {solution_path}"
        )
    benchmark_dir = _benchmark_subdir_for_slug(slug)

    rows = _run_sol_execbench(
        benchmark_dir=benchmark_dir, solution_path=solution_path,
    )
    bundle_dir = traces_root / f"{slug}__{language}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    trace = {
        "baseline": {
            "slug": slug,
            "language": language,
            "solution_path": str(
                solution_path.resolve().relative_to(_REPO_ROOT)
            ),
        },
        "rows": [
            _row_to_trace_entry(
                sol_execbench_row=row,
                slug=slug,
                language=language,
                workload_index=index,
            )
            for index, row in enumerate(rows)
        ],
        "environment_libs": (
            rows[0].get("evaluation", {})
                   .get("environment", {})
                   .get("libs", {})
            if rows else {}
        ),
    }
    (bundle_dir / "trace.json").write_text(
        json.dumps(trace, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "arch_tag": "sm_120",
        "language": language,
        "slug": slug,
        "provenance": {
            "source": "sol_execbench_only",
            "note": (
                "Minimal probe-compatible bundle emitted by "
                "emit_minimal_probe_bundle.py. The full harness "
                "surface stack (NVBit / CUPTI / nsys / static / "
                "sanitizer) was NOT invoked; only sol-execbench "
                "correctness was verified. Downstream consumers "
                "that need surface artifacts should re-run the "
                "full harness via scripts/baselines_harness/run.py."
            ),
            "captured_at_utc": captured_at_utc,
        },
        "dispatch_command": {
            "argv": [
                str(_SOL_EXECBENCH_BIN),
                str(benchmark_dir),
                "--solution",
                str(solution_path),
                "--json",
            ],
        },
        "host": {
            "arch_tag": "sm_120",
            "captured_at_utc": captured_at_utc,
        },
        "surfaces": {},
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return bundle_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--baseline",
        action="append",
        type=Path,
        required=True,
        help=(
            "Path to baselines/<cat>/<slug>/<lang>/ containing "
            "solution.json. Repeat once per baseline."
        ),
    )
    parser.add_argument(
        "--traces-root",
        type=Path,
        default=_DEFAULT_TRACES_ROOT,
    )
    parser.add_argument(
        "--captured-at-utc",
        type=str,
        default=None,
        help=(
            "ISO-8601 UTC timestamp recorded in each emitted "
            "manifest.json. Default: current UTC at invocation."
        ),
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    captured = args.captured_at_utc or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()

    for baseline in args.baseline:
        try:
            bundle = _emit_bundle_for_baseline(
                baseline_dir=baseline,
                traces_root=args.traces_root,
                captured_at_utc=captured,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"error emitting bundle for {baseline}: {exc}",
                file=sys.stderr,
            )
            return 2
        print(f"emitted: {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
