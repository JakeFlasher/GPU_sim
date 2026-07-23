"""3-run NCU repeatability driver for the pilot cohort.

Wraps the existing per-cell NCU harvester
(``calibration.cache_prediction.ncu_harvester.harvest_ncu_label_with_provenance``)
in a 3-run-then-CV envelope: per cell, invoke NCU 3 times,
collect the LSU-pipe read-only L1 hit rate
(``l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct``)
and the read-only L2 hit rate
(``lts__t_sector_op_read_hit_rate.pct`` — derived from the
sum counters when the .pct column is absent on this NCU
version) per run, compute per-metric mean + coefficient of
variation, and quarantine cells whose relative CV exceeds
0.05 with the typed ``ncu_repeatability_failed`` token.

The driver follows the same closed-vocabulary / per-cell
status surface the Tier-2 sidecar harvester established in
``harvest_tier2_pilot.py``:

* ``NCU_FAILURE_REASON_VOCABULARY`` — closed token set.
* ``NcuRepeatabilityOutcome`` — typed return from the 3-run
  invocation: completed, per-metric run lists, mean/CV,
  failure reason.
* ``run_pilot_ncu_repeatability`` — per-cohort driver.
* ``emit_ncu_repeatability_status`` — status TOML emitter.

The 3-run invocation is pluggable via ``run_fn`` so unit
tests inject a synthetic runner and never spawn real NCU
subprocesses; the default ``live_ncu_run_fn`` is the
production path that calls
``harvest_ncu_label_with_provenance``.

CLAUDE.md Critical Rule 1: NCU is calibration-only on sm_86
and sm_120. The deployed framework path and any B200
invocation MUST NOT touch NCU. This module sits under
``calibration/cache_prediction/`` and is opt-in via
``emit_pilot_ncu_repeatability`` only.

CLAUDE.md Critical Rule 2: every NCU invocation MUST pass
``--clock-control none`` (already pinned by the underlying
``ncu_harvester._build_invocation``).

Metric-family contract: only the read-only LSU-pipe family
(``*_op_read_*`` or the L1 LSU-pipe variant). NEVER the
store-conflating ``*_t_sector_hit_rate.pct`` family
(CLAUDE.md Critical Rule 1's calibration-only carveout
mandates read-only sector hit-rate metrics).
"""

from __future__ import annotations

import math
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Final


_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_HARVEST_OUTPUT_ROOT = (
    _REPO_ROOT
    / "scripts"
    / "baselines_harness"
    / ".harness_traces"
)

NCU_REPEATABILITY_SCHEMA_VERSION: Final[str] = "1.1"
NCU_REPEATABILITY_HOST_ARCH: Final[str] = "sm_120"

# Per-cell coefficient-of-variation cap. The 0.05 threshold
# is pinned by the project's NCU 3-run repeatability
# contract: cells whose relative CV exceeds 5% on either
# the L1 or the L2 metric are NOT admitted to the
# downstream calibration ledger (cells get the diagnostic
# typed reason ``ncu_repeatability_failed`` and stop at the
# ladder's ``sol_execbench_correct`` rung).
NCU_REPEATABILITY_CV_CAP: Final[float] = 0.05

# Closed vocabulary of failure-reason tokens. Empty string
# is the implicit "no failure" value when ``completed=True``.
NCU_FAILURE_REASON_VOCABULARY: Final[tuple[str, ...]] = (
    "ncu_not_yet_run",
    "ncu_subprocess_failed",
    "ncu_timeout",
    "ncu_metric_missing",
    "ncu_zero_launches_captured",
    "ncu_repeatability_failed",
    "ncu_kernel_name_not_pinned",
)


# NULL sentinels for NCU evidence on failed / unattempted
# rows. The float sentinel is NaN (TOML rejects bare NaN,
# so the emitter renders it as the same -1.0 the diagnostics
# sidecar uses; downstream readers test for the magic value
# via constants). Matches the per-row sentinel discipline in
# ``harvest_tier2_pilot.py``.
_NULL_FLOAT: Final[float] = -1.0
_NULL_INT: Final[int] = -1


@dataclass(frozen=True, slots=True)
class NcuRepeatabilityOutcome:
    """Typed return from a 3-run NCU invocation for a single
    pilot cell. ``run_fn`` returns one of these per cell.

    ``l1_runs`` and ``l2_runs`` are the per-run hit-rate
    measurements (length 3 on success; empty on failure).
    The driver computes per-metric mean + CV after
    ``run_fn`` returns to keep the typed-failure decision
    explicit.

    ``attempted`` records whether the run_fn actually
    spawned NCU. Pre-spawn demotions (e.g. the pilot's
    `(slug, language)` is absent from the kernel-name
    filter map, so the live wrapper returns
    ``ncu_kernel_name_not_pinned`` BEFORE any
    subprocess fires) MUST set ``attempted=False`` so the
    downstream diagnostics consumer's
    ``ncu_attempted`` field stays semantically accurate
    (a no-spawn cell was not actually profiled). Post-
    spawn failures (subprocess error, timeout, parser
    error on a real NCU CSV, empty stdout) set
    ``attempted=True`` because NCU did fire.

    The default ``True`` preserves existing synthetic
    test fixtures that supply ``completed=True`` without
    naming attempted; only the explicit
    pre-spawn-demote paths in production code set
    ``attempted=False``.
    """

    completed: bool
    failure_reason: str
    l1_runs: tuple[float, ...] = field(default_factory=tuple)
    l2_runs: tuple[float, ...] = field(default_factory=tuple)
    attempted: bool = True


class NcuRepeatabilityOutcomeInvalidError(ValueError):
    """Raised when a ``run_fn`` returns an outcome that
    violates the closed contract:

    - ``completed=True`` requires both run tuples to carry
      EXACTLY ``NCU_REPEATABILITY_REQUIRED_RUNS`` (= 3)
      measurements; the contract is a 3-run repeatability
      sweep, NOT an n-run streaming average.
    - ``completed=False`` requires a ``failure_reason``
      member of ``NCU_FAILURE_REASON_VOCABULARY``.
    """


# Required NCU run count per metric for a completed cell.
# Pinned at 3 by the project's NCU 3-run repeatability
# contract; admitting 2-run or 4-run completed outcomes
# would silently change the CV's degrees-of-freedom
# assumption and break the downstream CV-gate semantics.
NCU_REPEATABILITY_REQUIRED_RUNS: Final[int] = 3


def _validate_outcome(outcome: NcuRepeatabilityOutcome) -> None:
    if outcome.completed:
        if outcome.failure_reason:
            raise NcuRepeatabilityOutcomeInvalidError(
                f"completed=True requires failure_reason='', got "
                f"{outcome.failure_reason!r}"
            )
        if len(outcome.l1_runs) != NCU_REPEATABILITY_REQUIRED_RUNS:
            raise NcuRepeatabilityOutcomeInvalidError(
                f"completed=True requires exactly "
                f"{NCU_REPEATABILITY_REQUIRED_RUNS} L1 runs; got "
                f"{len(outcome.l1_runs)}"
            )
        if len(outcome.l2_runs) != NCU_REPEATABILITY_REQUIRED_RUNS:
            raise NcuRepeatabilityOutcomeInvalidError(
                f"completed=True requires exactly "
                f"{NCU_REPEATABILITY_REQUIRED_RUNS} L2 runs; got "
                f"{len(outcome.l2_runs)}"
            )
        return
    if not outcome.failure_reason:
        raise NcuRepeatabilityOutcomeInvalidError(
            "completed=False requires a typed failure_reason; "
            "got empty string"
        )
    if outcome.failure_reason not in NCU_FAILURE_REASON_VOCABULARY:
        raise NcuRepeatabilityOutcomeInvalidError(
            f"failure_reason={outcome.failure_reason!r} not in "
            f"NCU_FAILURE_REASON_VOCABULARY "
            f"{sorted(NCU_FAILURE_REASON_VOCABULARY)}"
        )


@dataclass(frozen=True, slots=True)
class NcuRepeatabilityStatusRow:
    slug: str
    language: str
    attempted: bool
    completed: bool
    failure_reason: str
    # Per-metric 3-run evidence. NULL sentinels on failure.
    ncu_l1_runs: tuple[float, ...]
    ncu_l2_runs: tuple[float, ...]
    ncu_l1_mean: float
    ncu_l1_cv: float
    ncu_l2_mean: float
    ncu_l2_cv: float
    # ncu_run_count records how many runs the driver actually
    # collected per metric. Always 3 on success; 0 on failure.
    ncu_run_count: int
    captured_at_utc: str


def _stub_ncu_run_fn(
    *,
    slug: str,
    language: str,
) -> NcuRepeatabilityOutcome:
    """Default ``run_fn`` — records ``ncu_not_yet_run`` for
    every cell. Production callers pass ``live_ncu_run_fn``
    via the CLI driver. ``attempted=False`` because no NCU
    subprocess fires in stub mode."""
    del slug, language
    return NcuRepeatabilityOutcome(
        completed=False,
        failure_reason="ncu_not_yet_run",
        attempted=False,
    )


def _relative_cv(samples: tuple[float, ...]) -> float:
    """Return the relative coefficient of variation
    ``stdev / mean`` over a sample tuple. Returns 0.0 when
    every sample is exactly 0.0 (the predictor pipeline
    treats a uniformly-zero hit rate as trivially repeatable
    rather than divide-by-zero / NaN).

    Uses the sample stdev (n-1 denominator) to align with
    the project's existing per-cell CV convention.
    """
    if not samples:
        return _NULL_FLOAT
    if len(samples) == 1:
        # A single sample's stdev is 0 by definition; CV is 0.
        return 0.0
    mean = statistics.fmean(samples)
    if mean == 0.0:
        if all(s == 0.0 for s in samples):
            return 0.0
        # Non-zero samples with zero mean (cancellation) is
        # an edge case the repeatability gate cannot honestly
        # score. Surface as the NaN-equivalent sentinel so
        # the caller demotes via ``ncu_metric_missing``.
        return _NULL_FLOAT
    return statistics.stdev(samples) / mean


def _build_failure_row(
    *,
    slug: str,
    language: str,
    attempted: bool,
    failure_reason: str,
    timestamp: str,
) -> NcuRepeatabilityStatusRow:
    return NcuRepeatabilityStatusRow(
        slug=slug,
        language=language,
        attempted=attempted,
        completed=False,
        failure_reason=failure_reason,
        ncu_l1_runs=(),
        ncu_l2_runs=(),
        ncu_l1_mean=_NULL_FLOAT,
        ncu_l1_cv=_NULL_FLOAT,
        ncu_l2_mean=_NULL_FLOAT,
        ncu_l2_cv=_NULL_FLOAT,
        ncu_run_count=0,
        captured_at_utc=timestamp,
    )


def run_pilot_ncu_repeatability(
    pilot_cells: list[tuple[str, str]],
    *,
    run_fn: Callable[..., NcuRepeatabilityOutcome] = _stub_ncu_run_fn,
    cv_cap: float = NCU_REPEATABILITY_CV_CAP,
    captured_at_utc: str | None = None,
) -> list[NcuRepeatabilityStatusRow]:
    """Drive the 3-run NCU repeatability harvest over a pilot
    cohort.

    Per-cell semantics:

    1. Invoke ``run_fn`` (default ``_stub_ncu_run_fn``)
       against the cell; it returns an
       ``NcuRepeatabilityOutcome`` with 3 L1 measurements +
       3 L2 measurements OR a typed failure reason.
    2. Validate the outcome against
       ``NCU_FAILURE_REASON_VOCABULARY`` (raises
       ``NcuRepeatabilityOutcomeInvalidError`` on
       wiring-error bad outcomes).
    3. On success, compute per-metric mean + relative CV.
    4. If either L1 CV or L2 CV exceeds ``cv_cap``
       (default 0.05), demote the row to
       ``failure_reason="ncu_repeatability_failed"`` with
       NULL evidence sentinels.
    5. Driver continues on per-cell failures (no
       batch-level abort).

    Args:
        pilot_cells: list of (slug, language) tuples.
        run_fn: pluggable NCU runner. The default stub
            records ``ncu_not_yet_run``; pass
            ``live_ncu_run_fn`` for the real
            NCU-via-sol-execbench invocation.
        cv_cap: CV gate threshold; defaults to 0.05.
        captured_at_utc: optional ISO-8601 UTC timestamp
            stamped on every row.

    Returns:
        List of ``NcuRepeatabilityStatusRow`` sorted by
        (slug, language).
    """
    rows: list[NcuRepeatabilityStatusRow] = []
    for slug, language in pilot_cells:
        timestamp = captured_at_utc or datetime.now(
            timezone.utc
        ).replace(microsecond=0).isoformat()
        try:
            outcome = run_fn(slug=slug, language=language)
        except Exception as exc:  # noqa: BLE001
            # An exception escaping run_fn means the
            # subprocess machinery itself broke (not a typed
            # cell failure). Treat as attempted since the
            # crash happens past the pre-spawn gate; the
            # batch continues to the next cell.
            outcome = NcuRepeatabilityOutcome(
                completed=False,
                failure_reason="ncu_subprocess_failed",
                attempted=True,
            )
            print(
                f"  {slug} ({language}): ncu run_fn raised "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        _validate_outcome(outcome)

        if not outcome.completed:
            rows.append(
                _build_failure_row(
                    slug=slug,
                    language=language,
                    attempted=outcome.attempted,
                    failure_reason=outcome.failure_reason,
                    timestamp=timestamp,
                )
            )
            continue

        l1_mean = statistics.fmean(outcome.l1_runs)
        l2_mean = statistics.fmean(outcome.l2_runs)
        l1_cv = _relative_cv(outcome.l1_runs)
        l2_cv = _relative_cv(outcome.l2_runs)
        # CV sentinel propagation: if either CV is the
        # NULL float (non-zero samples with zero mean), the
        # cell's repeatability cannot be honestly scored.
        # The cell WAS attempted (run_fn produced 3 samples
        # before the post-hoc CV sanity check); the
        # downstream non-admission is the gate's job.
        if l1_cv == _NULL_FLOAT or l2_cv == _NULL_FLOAT:
            rows.append(
                _build_failure_row(
                    slug=slug,
                    language=language,
                    attempted=True,
                    failure_reason="ncu_metric_missing",
                    timestamp=timestamp,
                )
            )
            continue
        # CV gate. Same logic: the cell was attempted with 3
        # real runs; the gate non-admits because the jitter
        # is too high.
        if l1_cv > cv_cap or l2_cv > cv_cap:
            rows.append(
                _build_failure_row(
                    slug=slug,
                    language=language,
                    attempted=True,
                    failure_reason="ncu_repeatability_failed",
                    timestamp=timestamp,
                )
            )
            continue

        rows.append(
            NcuRepeatabilityStatusRow(
                slug=slug,
                language=language,
                attempted=True,
                completed=True,
                failure_reason="",
                ncu_l1_runs=tuple(outcome.l1_runs),
                ncu_l2_runs=tuple(outcome.l2_runs),
                ncu_l1_mean=l1_mean,
                ncu_l1_cv=l1_cv,
                ncu_l2_mean=l2_mean,
                ncu_l2_cv=l2_cv,
                ncu_run_count=len(outcome.l1_runs),
                captured_at_utc=timestamp,
            )
        )
    rows.sort(key=lambda r: (r.slug, r.language))
    return rows


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _format_float(value: float) -> str:
    """Render a float for TOML emission. NaN / sentinel
    values surface as the NULL sentinel ``-1.0`` so TOML
    parsers accept the artifact."""
    if math.isnan(value):
        return f"{_NULL_FLOAT}"
    return f"{value!r}"


def _format_float_list(values: tuple[float, ...]) -> str:
    if not values:
        return "[]"
    return "[" + ", ".join(_format_float(v) for v in values) + "]"


def render_ncu_repeatability_status_toml(
    *,
    captured_at_utc: str,
    rows: list[NcuRepeatabilityStatusRow],
    cv_cap: float = NCU_REPEATABILITY_CV_CAP,
) -> str:
    """Render the per-cohort NCU repeatability status TOML."""
    aggregate_attempted = sum(1 for r in rows if r.attempted)
    aggregate_completed = sum(1 for r in rows if r.completed)
    aggregate_failed = sum(
        1 for r in rows if r.attempted and not r.completed
    )
    aggregate_cv_quarantined = sum(
        1 for r in rows if r.failure_reason == "ncu_repeatability_failed"
    )
    aggregate_not_yet_run = sum(
        1 for r in rows if r.failure_reason == "ncu_not_yet_run"
    )
    aggregate_skipped_kernel_not_pinned = sum(
        1 for r in rows
        if r.failure_reason == "ncu_kernel_name_not_pinned"
    )
    lines: list[str] = []
    lines.append("# sm_120 pilot NCU 3-run repeatability status.")
    lines.append("#")
    lines.append(
        "# Per-cell outcomes from the 3-run NCU harvest pipeline"
    )
    lines.append(
        "# against the pilot cohort. Closed-vocabulary failure"
    )
    lines.append("# tokens; CV gate enforced at the recorded cap.")
    lines.append("")
    lines.append("[meta]")
    lines.append(
        f'schema_version = "{NCU_REPEATABILITY_SCHEMA_VERSION}"'
    )
    lines.append(f'host_arch = "{NCU_REPEATABILITY_HOST_ARCH}"')
    lines.append(f'captured_at_utc = "{captured_at_utc}"')
    lines.append(f"row_count = {len(rows)}")
    lines.append(f"aggregate_attempted = {aggregate_attempted}")
    lines.append(f"aggregate_completed = {aggregate_completed}")
    lines.append(f"aggregate_failed = {aggregate_failed}")
    lines.append(
        f"aggregate_cv_quarantined = {aggregate_cv_quarantined}"
    )
    lines.append(f"aggregate_not_yet_run = {aggregate_not_yet_run}")
    lines.append(
        f"aggregate_skipped_kernel_not_pinned = "
        f"{aggregate_skipped_kernel_not_pinned}"
    )
    lines.append(f"cv_cap = {cv_cap!r}")
    lines.append("")
    for row in rows:
        lines.append("[[entries]]")
        lines.append(f'slug = "{_toml_escape(row.slug)}"')
        lines.append(f'language = "{_toml_escape(row.language)}"')
        lines.append(f"attempted = {str(row.attempted).lower()}")
        lines.append(f"completed = {str(row.completed).lower()}")
        lines.append(
            f'failure_reason = "{_toml_escape(row.failure_reason)}"'
        )
        lines.append(
            f"ncu_l1_runs = {_format_float_list(row.ncu_l1_runs)}"
        )
        lines.append(
            f"ncu_l2_runs = {_format_float_list(row.ncu_l2_runs)}"
        )
        lines.append(f"ncu_l1_mean = {_format_float(row.ncu_l1_mean)}")
        lines.append(f"ncu_l1_cv = {_format_float(row.ncu_l1_cv)}")
        lines.append(f"ncu_l2_mean = {_format_float(row.ncu_l2_mean)}")
        lines.append(f"ncu_l2_cv = {_format_float(row.ncu_l2_cv)}")
        lines.append(f"ncu_run_count = {row.ncu_run_count}")
        lines.append(
            f'captured_at_utc = "{row.captured_at_utc}"'
        )
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def emit_ncu_repeatability_status(
    pilot_cells: list[tuple[str, str]],
    *,
    output: Path,
    captured_at_utc: str,
    run_fn: Callable[..., NcuRepeatabilityOutcome] = _stub_ncu_run_fn,
    cv_cap: float = NCU_REPEATABILITY_CV_CAP,
) -> Path:
    rows = run_pilot_ncu_repeatability(
        pilot_cells,
        run_fn=run_fn,
        cv_cap=cv_cap,
        captured_at_utc=captured_at_utc,
    )
    toml_text = render_ncu_repeatability_status_toml(
        captured_at_utc=captured_at_utc,
        rows=rows,
        cv_cap=cv_cap,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(toml_text, encoding="utf-8")
    return output


def live_ncu_run_fn(
    *,
    slug: str,
    language: str,
    launch_skip_count: int = 1,
) -> NcuRepeatabilityOutcome:
    """Opt-in live 3-run NCU repeatability invocation for a
    pilot cell. Invokes the existing
    ``ncu_harvester.harvest_ncu_label_with_provenance`` three
    times wrapping ``sol-execbench`` (with
    ``--target-processes all`` injected so NCU follows the
    sol-execbench worker subprocess).

    Pre-conditions checked at call time:
      - ``(slug, language)`` is in the pilot pinned lookup
        (re-uses the Tier-2 driver's ``_PILOT_PATH_LOOKUP``).
      - The pilot cell's ``solution.json`` exists on disk.
      - The benchmark problem dir exists on disk.

    Any pre-condition failure returns a typed
    ``ncu_subprocess_failed`` outcome (NEVER raises so the
    cohort driver records a closed-vocabulary row and
    continues).

    NOTE: This live path may exit ``ncu_subprocess_failed``
    on hosts where NCU's ``--target-processes all`` permission
    is not granted, or where wrapping a CPython parent leads
    to PyTorch worker isolation that loses the kernel
    launches. The closed-vocabulary outcome surfaces that
    honestly to the per-cohort status emitter.
    """
    # Re-use the Tier-2 driver's pilot lookup so the two
    # cohorts stay in lockstep on which cells are "pilot".
    from baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.harvest_tier2_pilot import (  # noqa: E402
        _PILOT_PATH_LOOKUP,
        _benchmark_def_name,
    )
    from baselines_harness.calibration.cache_prediction.sm120_coverage_expansion.pilot_kernel_filter import (  # noqa: E402
        lookup_pilot_kernel_name_base,
    )
    from baselines_harness.calibration.cache_prediction.ncu_harvester import (  # noqa: E402
        NcuHarvesterError,
        _build_invocation,
        _run_checked_ncu,
    )
    from baselines_harness.calibration.cache_prediction.label_parser import (  # noqa: E402
        NcuLabelParserError,
        parse_ncu_csv,
    )

    # Pre-spawn gates: every branch below returns BEFORE
    # any NCU subprocess fires; outcomes MUST carry
    # ``attempted=False`` so the cohort driver records the
    # cell as unattempted (matches the downstream
    # diagnostics consumer's ``ncu_attempted`` semantic).
    if (slug, language) not in _PILOT_PATH_LOOKUP:
        return NcuRepeatabilityOutcome(
            completed=False,
            failure_reason="ncu_subprocess_failed",
            attempted=False,
        )
    # Per-cell kernel-name filter is mandatory for the
    # pilot path. ATen-dispatched cells (cuBLAS / cuDNN
    # library kernels with shape-dependent selection) have
    # no stable kernel name to pin; demote them BEFORE
    # spawning NCU so the cohort driver records a typed
    # closed-vocabulary failure rather than burning wall
    # time capturing a non-deterministic library kernel.
    kernel_name_base = lookup_pilot_kernel_name_base(slug, language)
    if kernel_name_base is None:
        return NcuRepeatabilityOutcome(
            completed=False,
            failure_reason="ncu_kernel_name_not_pinned",
            attempted=False,
        )
    baselines_category, benchmark_subset = _PILOT_PATH_LOOKUP[
        (slug, language)
    ]
    solution_path = (
        _REPO_ROOT
        / "baselines"
        / baselines_category
        / slug
        / language
        / "solution.json"
    )
    benchmark_dir = (
        _REPO_ROOT
        / "sol-execbench"
        / "data"
        / "benchmark"
        / benchmark_subset
        / _benchmark_def_name(slug)
    )
    sol_execbench_bin = _REPO_ROOT / ".venv" / "bin" / "sol-execbench"
    if not solution_path.is_file():
        return NcuRepeatabilityOutcome(
            completed=False,
            failure_reason="ncu_subprocess_failed",
            attempted=False,
        )
    if not benchmark_dir.is_dir():
        return NcuRepeatabilityOutcome(
            completed=False,
            failure_reason="ncu_subprocess_failed",
            attempted=False,
        )
    if not sol_execbench_bin.is_file():
        return NcuRepeatabilityOutcome(
            completed=False,
            failure_reason="ncu_subprocess_failed",
            attempted=False,
        )

    binary_args: tuple[str, ...] = (
        str(benchmark_dir),
        "--solution",
        str(solution_path),
        "--json",
    )
    argv = _build_invocation(
        binary=str(sol_execbench_bin),
        launch_skip_count=launch_skip_count,
        binary_args=binary_args,
        # --target-processes all so NCU follows
        # sol-execbench's worker subprocess (the worker is
        # where the actual GPU launches happen; the CLI
        # parent only orchestrates).
        target_processes="all",
        # --kernel-name pins the captured launch to the
        # pilot's actual GPU kernel (rejects torch init,
        # curand RNG, and other library kernels).
        kernel_name_filter=kernel_name_base,
    )
    # Post-spawn paths: NCU subprocess actually fires for
    # every iteration below; outcomes MUST carry
    # ``attempted=True`` so the cohort driver records the
    # cell as profiled, even when the run ultimately fails.
    l1_runs: list[float] = []
    l2_runs: list[float] = []
    for _ in range(3):
        try:
            completed = _run_checked_ncu(argv)
        except NcuHarvesterError:
            return NcuRepeatabilityOutcome(
                completed=False,
                failure_reason="ncu_subprocess_failed",
                attempted=True,
            )
        # CSV-is-source-of-truth: NCU exits non-zero when its
        # wrapped target (sol-execbench) returns non-zero, but
        # the CSV in stdout can still carry a valid captured
        # launch. Mirrors the Tier-2 sidecar-is-source-of-truth
        # pattern from harvest_tier2_pilot.py. Treat a
        # non-empty stdout as candidate evidence regardless of
        # the wrapped target's exit code; only fall through to
        # ncu_subprocess_failed when NCU emitted nothing.
        if not completed.stdout.strip():
            return NcuRepeatabilityOutcome(
                completed=False,
                failure_reason="ncu_subprocess_failed",
                attempted=True,
            )
        try:
            record = parse_ncu_csv(
                completed.stdout,
                ncu_metric_version="unknown",
                binary_sha256="",
                launch_skip_count=launch_skip_count,
                warmup_protocol="unknown",
            )
        except NcuLabelParserError as exc:
            # Parser-side failures: a missing required metric
            # column maps to the typed ``ncu_metric_missing``
            # token. Other parser violations (CSV malformed,
            # denominator zero, combined-metric-forbidden,
            # hit-rate-above-100) propagate as
            # ``ncu_subprocess_failed`` since they reflect
            # malformed NCU output the cohort consumer cannot
            # honestly interpret.
            typed = getattr(exc, "typed_reason", "")
            if typed == "ncu_label_metric_missing":
                return NcuRepeatabilityOutcome(
                    completed=False,
                    failure_reason="ncu_metric_missing",
                    attempted=True,
                )
            return NcuRepeatabilityOutcome(
                completed=False,
                failure_reason="ncu_subprocess_failed",
                attempted=True,
            )
        # NcuLabelRecord exposes the LSU-pipe L1 global-load
        # hit rate and the read-only L2 hit rate as fractions
        # in [0.0, 1.0].
        l1_runs.append(float(record.l1_global_load_hit_rate))
        l2_runs.append(float(record.l2_read_hit_rate))
    return NcuRepeatabilityOutcome(
        completed=True,
        failure_reason="",
        l1_runs=tuple(l1_runs),
        l2_runs=tuple(l2_runs),
    )
