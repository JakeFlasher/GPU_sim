"""Run ``sol-execbench`` on each pilot cell and emit
``sm120_pilot_reverification_v1.toml`` with the per-cell
verdict.

This is the pilot port + correctness deliverable for the
12-cell pilot cohort: re-verify compile + correctness against
the current
sm_120 toolchain, record per-cell outcomes, and surface
quarantine reasons via the closed diagnostic vocabulary.

The 12 pilot cells are inherited (they already have ``*.cu`` +
``solution.json`` files under ``baselines/<cat>/<slug>/<lang>/``).
This artifact records the **re-verification verdict**, NOT
new-port credit; the artifact's per-cell ``role`` field is
``pilot_reverification`` to make this explicit.

Each per-cell row carries:

- ``slug`` (e.g. ``problem_024_rmsnorm_h2048``)
- ``language`` (``cuda_cpp`` / ``cutlass``)
- ``benchmark_subset`` (``L1`` / ``L2`` / ``Quant`` /
  ``FlashInfer-Bench``)
- ``benchmark_dir`` (repo-relative path to the sol-execbench
  problem dir)
- ``solution_path`` (repo-relative path to ``solution.json``)
- ``verdict`` (``passed`` / ``correctness_failed`` /
  ``compile_failed`` / ``error``)
- ``workload_count`` (number of workload entries enumerated)
- ``passed_workload_count`` (subset where ``status == "PASSED"``)
- ``command_argv`` (the exact sol-execbench invocation)
- ``captured_at_utc`` (per-cell timestamp)
- ``host_arch`` (always ``"sm_120"`` for this artifact)
- ``role`` (always ``"pilot_reverification"``)
- ``quarantine_reason`` (closed vocabulary; empty when verdict
  is ``passed``)

CLI:

    python -m baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.emit_pilot_reverification \\
        [--output ... ] [--captured-at-utc ISO8601 ]
        [--sol-execbench-bin /path/to/sol-execbench]
        [--cutlass-dir /path/to/cutlass]
        [--timeout-s 90]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final


_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_OUTPUT = (
    _REPO_ROOT
    / "scripts"
    / "baselines_harness"
    / "calibration"
    / "cache_prediction"
    / "sm120_pilot_reverification_v1.toml"
)
_DEFAULT_SOL_EXECBENCH_BIN = _REPO_ROOT / ".venv" / "bin" / "sol-execbench"
_DEFAULT_CUTLASS_DIR = _REPO_ROOT / "third_party" / "cutlass"

REVERIFICATION_SCHEMA_VERSION: Final[str] = "1.0"
REVERIFICATION_HOST_ARCH: Final[str] = "sm_120"
REVERIFICATION_ROLE: Final[str] = "pilot_reverification"

# Closed vocabulary of verdict tokens.
VERDICT_VOCABULARY: Final[tuple[str, ...]] = (
    "passed",
    "correctness_failed",
    "compile_failed",
    "error",
)

# Closed vocabulary of quarantine-reason tokens. Empty string
# is the implicit "no quarantine" value when verdict is
# ``passed``.
QUARANTINE_REASON_VOCABULARY: Final[tuple[str, ...]] = (
    "sol_execbench_correctness_failed",
    "sol_execbench_compile_failed",
    "sol_execbench_invocation_error",
    "sol_execbench_timeout",
)

# Per-cell pilot list (Codex-selected via the analyze-routed
# pilot-selection step). Pinned here so the artifact emission
# is reproducible from this module alone. Each entry
# is (slug, language, baselines_category, benchmark_subset).
# Note that ``baselines_category`` and ``benchmark_subset``
# differ for the FIB category: ``baselines/FIB/`` corresponds
# to ``sol-execbench/data/benchmark/FlashInfer-Bench/``.
_PILOT_LIST: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("problem_024_rmsnorm_h2048", "cuda_cpp", "FIB", "FlashInfer-Bench"),
    ("problem_002_fused_add_rmsnorm_h4096", "cuda_cpp", "FIB", "FlashInfer-Bench"),
    ("problem_069_rms_norm", "cuda_cpp", "L1", "L1"),
    ("problem_025_video_latent_gelu_activation", "cuda_cpp", "L1", "L1"),
    ("problem_068_gelu_approximate_feedforward_backward", "cuda_cpp", "L2", "L2"),
    ("problem_045_fused_linear_gelu_grn_linear", "cuda_cpp", "L1", "L1"),
    ("problem_030_flux_concatenated_sequence_processing_with_split", "cuda_cpp", "L2", "L2"),
    ("problem_010_attention_value_projection_with_transpose", "cuda_cpp", "L1", "L1"),
    ("problem_063_attention_output_reshape_and_projection", "cuda_cpp", "L1", "L1"),
    ("problem_051_attention_qkv_with_qk_norm_single_kernel_backward", "cuda_cpp", "L1", "L1"),
    ("problem_021_cross_attention_text_video_conditioning_backward", "cuda_cpp", "L2", "L2"),
    ("problem_093_grouped_topk_moe_routing_backward", "cuda_cpp", "L1", "L1"),
)


@dataclass(frozen=True, slots=True)
class ReverificationRow:
    slug: str
    language: str
    benchmark_subset: str
    benchmark_dir: str
    solution_path: str
    verdict: str
    workload_count: int
    passed_workload_count: int
    command_argv: tuple[str, ...]
    captured_at_utc: str
    host_arch: str
    role: str
    quarantine_reason: str


def _benchmark_def_name(slug: str) -> str:
    """Convert a slug like ``problem_004_gemm_n128_k2048`` into
    the sol-execbench benchmark dir basename
    (``004_gemm_n128_k2048``).
    """
    if slug.startswith("problem_"):
        return slug[len("problem_"):]
    return slug


def _resolve_paths(
    *,
    slug: str,
    language: str,
    baselines_category: str,
    benchmark_subset: str,
    repo_root: Path,
) -> tuple[Path, Path] | None:
    """Resolve absolute paths to the benchmark dir and the
    solution.json for a pilot cell. Returns None when either
    path does not exist on disk."""
    benchmark_dir = (
        repo_root
        / "sol-execbench"
        / "data"
        / "benchmark"
        / benchmark_subset
        / _benchmark_def_name(slug)
    )
    solution_path = (
        repo_root
        / "baselines"
        / baselines_category
        / slug
        / language
        / "solution.json"
    )
    if not benchmark_dir.is_dir():
        return None
    if not solution_path.is_file():
        return None
    return benchmark_dir, solution_path


def _classify_verdict(
    *, returncode: int, parsed_rows: list[dict]
) -> tuple[str, str]:
    """Classify a sol-execbench run into a (verdict,
    quarantine_reason) pair.

    sol-execbench emits one JSON object per workload (when
    --json). A row's ``evaluation.status`` field is ``"PASSED"``
    on success or a failure token otherwise. A non-zero exit
    typically indicates compile failure or invocation error;
    distinguish them by checking whether any row was emitted.
    """
    if returncode != 0 and not parsed_rows:
        return "compile_failed", "sol_execbench_compile_failed"
    if returncode != 0:
        # Some rows emitted, but the run exited non-zero — treat
        # as invocation error.
        return "error", "sol_execbench_invocation_error"
    passed = sum(
        1
        for row in parsed_rows
        if isinstance(row, dict)
        and row.get("evaluation", {}).get("status") == "PASSED"
    )
    if not parsed_rows:
        return "error", "sol_execbench_invocation_error"
    if passed == len(parsed_rows):
        return "passed", ""
    return "correctness_failed", "sol_execbench_correctness_failed"


def _run_sol_execbench(
    *,
    sol_execbench_bin: Path,
    benchmark_dir: Path,
    solution_path: Path,
    cutlass_dir: Path,
    timeout_s: int,
) -> tuple[int, list[dict], list[str]]:
    """Invoke sol-execbench as a subprocess. Returns
    (returncode, parsed_rows, argv).

    Stdout is parsed as one JSON object per line; lines that do
    not parse cleanly are silently skipped (sol-execbench
    intermixes JSON rows with progress text)."""
    argv = [
        str(sol_execbench_bin),
        str(benchmark_dir),
        "--solution",
        str(solution_path),
        "--json",
        "--compile-timeout",
        str(timeout_s),
        "--timeout",
        str(timeout_s),
    ]
    env = dict(os.environ)
    env["CUTLASS_DIR"] = str(cutlass_dir)
    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s * 4,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return -1, [], argv
    rows: list[dict] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return result.returncode, rows, argv


def _reverify_one(
    *,
    slug: str,
    language: str,
    baselines_category: str,
    benchmark_subset: str,
    repo_root: Path,
    sol_execbench_bin: Path,
    cutlass_dir: Path,
    timeout_s: int,
) -> ReverificationRow:
    """Re-verify one pilot cell and return its row."""
    captured = datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat()
    paths = _resolve_paths(
        slug=slug,
        language=language,
        baselines_category=baselines_category,
        benchmark_subset=benchmark_subset,
        repo_root=repo_root,
    )
    if paths is None:
        return ReverificationRow(
            slug=slug,
            language=language,
            benchmark_subset=benchmark_subset,
            benchmark_dir="",
            solution_path="",
            verdict="error",
            workload_count=0,
            passed_workload_count=0,
            command_argv=(),
            captured_at_utc=captured,
            host_arch=REVERIFICATION_HOST_ARCH,
            role=REVERIFICATION_ROLE,
            quarantine_reason="sol_execbench_invocation_error",
        )
    benchmark_dir, solution_path = paths
    returncode, rows, argv = _run_sol_execbench(
        sol_execbench_bin=sol_execbench_bin,
        benchmark_dir=benchmark_dir,
        solution_path=solution_path,
        cutlass_dir=cutlass_dir,
        timeout_s=timeout_s,
    )
    verdict, quarantine_reason = _classify_verdict(
        returncode=returncode, parsed_rows=rows
    )
    if returncode == -1:
        verdict, quarantine_reason = (
            "error",
            "sol_execbench_timeout",
        )
    passed = sum(
        1
        for row in rows
        if isinstance(row, dict)
        and row.get("evaluation", {}).get("status") == "PASSED"
    )
    return ReverificationRow(
        slug=slug,
        language=language,
        benchmark_subset=benchmark_subset,
        benchmark_dir=str(benchmark_dir.relative_to(repo_root)),
        solution_path=str(solution_path.relative_to(repo_root)),
        verdict=verdict,
        workload_count=len(rows),
        passed_workload_count=passed,
        command_argv=tuple(argv),
        captured_at_utc=captured,
        host_arch=REVERIFICATION_HOST_ARCH,
        role=REVERIFICATION_ROLE,
        quarantine_reason=quarantine_reason,
    )


def reverify_pilot_cohort(
    *,
    sol_execbench_bin: Path = _DEFAULT_SOL_EXECBENCH_BIN,
    cutlass_dir: Path = _DEFAULT_CUTLASS_DIR,
    timeout_s: int = 90,
    repo_root: Path = _REPO_ROOT,
    pilot_list: tuple[
        tuple[str, str, str, str], ...
    ] = _PILOT_LIST,
) -> list[ReverificationRow]:
    """Run sol-execbench on every pilot cell. Returns a list of
    ``ReverificationRow`` (sorted by slug for deterministic
    output)."""
    rows: list[ReverificationRow] = []
    for (slug, language, baselines_category, benchmark_subset) in (
        pilot_list
    ):
        row = _reverify_one(
            slug=slug,
            language=language,
            baselines_category=baselines_category,
            benchmark_subset=benchmark_subset,
            repo_root=repo_root,
            sol_execbench_bin=sol_execbench_bin,
            cutlass_dir=cutlass_dir,
            timeout_s=timeout_s,
        )
        rows.append(row)
        print(
            f"  {slug} ({language}): verdict={row.verdict}, "
            f"workloads={row.passed_workload_count}/"
            f"{row.workload_count}",
            file=sys.stderr,
        )
    rows.sort(key=lambda r: (r.slug, r.language))
    return rows


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def render_reverification_toml(
    *,
    captured_at_utc: str,
    rows: list[ReverificationRow],
) -> str:
    """Render the reverification artifact TOML."""
    aggregate_passed = sum(1 for r in rows if r.verdict == "passed")
    aggregate_correctness_failed = sum(
        1 for r in rows if r.verdict == "correctness_failed"
    )
    aggregate_compile_failed = sum(
        1 for r in rows if r.verdict == "compile_failed"
    )
    aggregate_error = sum(1 for r in rows if r.verdict == "error")
    payload_sha = hashlib.sha256(
        "\n".join(
            f"{r.slug}|{r.language}|{r.verdict}|"
            f"{r.passed_workload_count}/{r.workload_count}"
            for r in rows
        ).encode("utf-8")
    ).hexdigest()
    lines: list[str] = []
    lines.append(
        "# sm_120 pilot-cell reverification artifact."
    )
    lines.append("#")
    lines.append(
        "# Records the per-cell sol-execbench compile +"
    )
    lines.append(
        "# correctness verdict for the 12 pilot cells selected"
    )
    lines.append(
        "# in the analyze-routed pilot-selection step. These"
    )
    lines.append(
        "# cells already exist on disk; this artifact is the"
    )
    lines.append(
        "# reverification cohort, NOT new-port credit toward the"
    )
    lines.append(
        "# 30-new-hand-port floor (role=pilot_reverification per"
    )
    lines.append("# row).")
    lines.append("")
    lines.append("[meta]")
    lines.append(
        f'schema_version = "{REVERIFICATION_SCHEMA_VERSION}"'
    )
    lines.append(f'host_arch = "{REVERIFICATION_HOST_ARCH}"')
    lines.append(f'captured_at_utc = "{captured_at_utc}"')
    lines.append(f"row_count = {len(rows)}")
    lines.append(f"aggregate_passed = {aggregate_passed}")
    lines.append(
        f"aggregate_correctness_failed = "
        f"{aggregate_correctness_failed}"
    )
    lines.append(
        f"aggregate_compile_failed = {aggregate_compile_failed}"
    )
    lines.append(f"aggregate_error = {aggregate_error}")
    lines.append(f'payload_sha256 = "{payload_sha}"')
    lines.append("")
    for row in rows:
        lines.append("[[entries]]")
        lines.append(f'slug = "{_toml_escape(row.slug)}"')
        lines.append(f'language = "{_toml_escape(row.language)}"')
        lines.append(
            f'benchmark_subset = "{_toml_escape(row.benchmark_subset)}"'
        )
        lines.append(
            f'benchmark_dir = "{_toml_escape(row.benchmark_dir)}"'
        )
        lines.append(
            f'solution_path = "{_toml_escape(row.solution_path)}"'
        )
        lines.append(f'verdict = "{row.verdict}"')
        lines.append(f"workload_count = {row.workload_count}")
        lines.append(
            f"passed_workload_count = {row.passed_workload_count}"
        )
        if row.command_argv:
            argv_repr = ", ".join(
                f'"{_toml_escape(tok)}"' for tok in row.command_argv
            )
            lines.append(f"command_argv = [{argv_repr}]")
        else:
            lines.append("command_argv = []")
        lines.append(
            f'captured_at_utc = "{row.captured_at_utc}"'
        )
        lines.append(f'host_arch = "{row.host_arch}"')
        lines.append(f'role = "{row.role}"')
        lines.append(
            f'quarantine_reason = "{_toml_escape(row.quarantine_reason)}"'
        )
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def emit_pilot_reverification(
    *,
    output: Path,
    captured_at_utc: str,
    sol_execbench_bin: Path = _DEFAULT_SOL_EXECBENCH_BIN,
    cutlass_dir: Path = _DEFAULT_CUTLASS_DIR,
    timeout_s: int = 90,
) -> Path:
    """Run sol-execbench on every pilot cell + write the artifact."""
    rows = reverify_pilot_cohort(
        sol_execbench_bin=sol_execbench_bin,
        cutlass_dir=cutlass_dir,
        timeout_s=timeout_s,
    )
    toml_text = render_reverification_toml(
        captured_at_utc=captured_at_utc, rows=rows
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(toml_text, encoding="utf-8")
    return output


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run sol-execbench on each pilot cell and emit the "
            "reverification artifact."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=_DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--captured-at-utc", type=str, default=None,
    )
    parser.add_argument(
        "--sol-execbench-bin",
        type=Path,
        default=_DEFAULT_SOL_EXECBENCH_BIN,
    )
    parser.add_argument(
        "--cutlass-dir", type=Path, default=_DEFAULT_CUTLASS_DIR,
    )
    parser.add_argument("--timeout-s", type=int, default=90)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    captured = args.captured_at_utc or datetime.now(
        timezone.utc
    ).replace(microsecond=0).isoformat()
    output = emit_pilot_reverification(
        output=args.output,
        captured_at_utc=captured,
        sol_execbench_bin=args.sol_execbench_bin,
        cutlass_dir=args.cutlass_dir,
        timeout_s=args.timeout_s,
    )
    print(f"emitted: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
