"""Intermediate-baseline diagnostic emitter.

The operator-callable helper wraps the existing diagnostic
pipeline — `build_accuracy_diagnostic_report()` →
`serialize_diagnostic_report()` →
`parse_diagnostic_report()` — and writes a snapshot TOML to the
configured `output_path`. The helper round-trips the parsed
report against the originally built report, captures the
`[denominator_box]` partition counts plus the dual-gate verdict,
and computes the CLI-equivalent exit code (0 / 2 / 3) so
downstream operators and reviewers can confirm the snapshot's
acceptance state without re-emitting the artifact.

This module is calibration-only and never imported by the
deployed harness path. The NCU quarantine scanner enforces that
contract.
"""

from __future__ import annotations

from sol_baselines_harness.calibration.cache_prediction.retirement import (
    refuse_legacy_ncu_calibration,
    retired_legacy_ncu_calibration,
)

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from sol_baselines_harness.calibration.cache_prediction.accuracy_diagnostic import (
    CanonicalSolbenchProblem,
    DiagnosticReport,
    build_accuracy_diagnostic_report,
    parse_diagnostic_report,
    serialize_diagnostic_report,
)


INTERMEDIATE_BASELINE_TYPED_REASONS: frozenset[str] = frozenset({
    "intermediate_baseline_io_error",
    "intermediate_baseline_round_trip_mismatch",
    "intermediate_baseline_output_path_invalid",
})


CLI_EXIT_BOTH_GATES_PASS: int = 0
CLI_EXIT_SCHEMA_OR_CONTRACT_GATE_FAIL: int = 2
CLI_EXIT_USER_ACCEPTANCE_GATE_FAIL: int = 3


class IntermediateBaselineError(Exception):
    """Typed error raised by `emit_intermediate_baseline()`.

    Mirrors the typed-error pattern used throughout the
    calibration package. `typed_reason` is drawn from the
    closed set `INTERMEDIATE_BASELINE_TYPED_REASONS`.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        if typed_reason not in INTERMEDIATE_BASELINE_TYPED_REASONS:
            raise ValueError(
                f"unknown typed_reason {typed_reason!r}; expected "
                f"one of "
                f"{sorted(INTERMEDIATE_BASELINE_TYPED_REASONS)}",
            )
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class IntermediateBaselineResult:
    """Snapshot summary returned by
    `emit_intermediate_baseline()`.

    Carries the on-disk artifact path, the nine
    `[denominator_box]` partition + gate counts, the two
    dual-gate verdict booleans, and the CLI-equivalent exit
    code (0 / 2 / 3) computed from the parsed report's
    `overall_summary.accuracy_gate_passed` and
    `overall_user_target_summary.gate_compound_passed` fields.

    Reviewers can use the dataclass shape to verify the
    snapshot's acceptance state without re-emitting the
    artifact: `cli_exit_code` is the canonical input downstream
    rounds replay through the standalone CLI.
    """

    output_path: Path
    total_problems: int
    measured: int
    typed_unavailable: int
    diagnostic_failed: int
    corpus_pending: int
    host_arch_unsupported: int
    passing_user_target: int
    passing_tiered_gate: int
    passing_compound_gate: int
    contract_gate_passed: bool
    user_compound_gate_passed: bool
    cli_exit_code: int


def _default_output_path(
    repo_root: Path, implementation_commit_sha: str,
) -> Path:
    return (
        repo_root / ".humanize" / "diagnostic-baselines"
        / f"intermediate-baseline-{implementation_commit_sha}.toml"
    )


def _cli_exit_code_for_report(report: DiagnosticReport) -> int:
    if not report.overall_summary.accuracy_gate_passed:
        return CLI_EXIT_SCHEMA_OR_CONTRACT_GATE_FAIL
    if not report.overall_user_target_summary.gate_compound_passed:
        return CLI_EXIT_USER_ACCEPTANCE_GATE_FAIL
    return CLI_EXIT_BOTH_GATES_PASS


def emit_intermediate_baseline(
    output_path: Path | None = None,
    *,
    repo_root: Path | None = None,
    baselines_root: Path | None = None,
    ledger_path: Path | None = None,
    implementation_commit_sha: str | None = None,
    denominator_floor: float = 0.01,
    accuracy_threshold: float = 0.10,
    predictor_callback: Callable[..., Any] | None = None,
    canonical_problems: Sequence[CanonicalSolbenchProblem] | None = None,
    nvbit_so_sha256: str | None = None,
    trace_budget_bytes: int | None = None,
    timeout_seconds: int | None = None,
    host_arch_manifest_path: Path | None = None,
    calibration_validation_split_path: Path | None = None,
    default_output_root: Path | None = None,
    serializer: Callable[[DiagnosticReport], str] | None = None,
) -> IntermediateBaselineResult:
    """Build the diagnostic, serialize it to TOML, write it to
    `output_path`, and verify the round-trip.

    The build step accepts every injection seam
    `build_accuracy_diagnostic_report()` exposes. The
    `serializer` keyword exists so tests can inject a
    deliberately-broken serializer to exercise the round-trip
    failure path; production callers should leave it as `None`.

    When `output_path` is `None`, the default location is
    `<default_output_root>/.humanize/diagnostic-baselines/intermediate-baseline-<implementation_commit_sha>.toml`,
    where `default_output_root` defaults to `repo_root` (which
    in turn defaults to the in-tree project root derived from
    this module's path). The parent directory is created if
    absent. Operator-local: the `.humanize/` directory is
    gitignored.

    The `default_output_root` seam exists so tests can redirect
    the resolved default path to a per-test temporary tree
    without disrupting the diagnostic build (which still uses
    `repo_root` to find the canonical manifests + ledger).

    Returns an `IntermediateBaselineResult`; raises
    `IntermediateBaselineError(typed_reason, message)` on IO,
    output-path, or round-trip failures.
    """

    if predictor_callback is not None:
        refuse_legacy_ncu_calibration('intermediate_baseline.predictor_callback')
    report = build_accuracy_diagnostic_report(
        repo_root=repo_root,
        baselines_root=baselines_root,
        ledger_path=ledger_path,
        implementation_commit_sha=implementation_commit_sha,
        denominator_floor=denominator_floor,
        accuracy_threshold=accuracy_threshold,
        predictor_callback=predictor_callback,
        canonical_problems=canonical_problems,
        nvbit_so_sha256=nvbit_so_sha256,
        trace_budget_bytes=trace_budget_bytes,
        timeout_seconds=timeout_seconds,
        host_arch_manifest_path=host_arch_manifest_path,
        calibration_validation_split_path=(
            calibration_validation_split_path
        ),
    )

    if output_path is None:
        if default_output_root is not None:
            resolved_default_root = default_output_root
        elif repo_root is not None:
            resolved_default_root = repo_root
        else:
            resolved_default_root = (
                Path(__file__).resolve().parents[4]
            )
        output_path = _default_output_path(
            resolved_default_root,
            report.meta.implementation_commit_sha,
        )

    output_path = Path(output_path)
    parent = output_path.parent
    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise IntermediateBaselineError(
                "intermediate_baseline_output_path_invalid",
                f"cannot create parent directory {parent!s}: "
                f"{exc}",
            ) from exc

    serializer_fn = (
        serializer if serializer is not None
        else serialize_diagnostic_report
    )
    text = serializer_fn(report)

    try:
        output_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise IntermediateBaselineError(
            "intermediate_baseline_io_error",
            f"cannot write baseline to {output_path!s}: {exc}",
        ) from exc

    reparsed = parse_diagnostic_report(text)
    if reparsed != report:
        raise IntermediateBaselineError(
            "intermediate_baseline_round_trip_mismatch",
            f"serialize -> parse round-trip diverged for "
            f"baseline at {output_path!s}; the serializer or "
            f"parser dropped or mutated a field",
        )

    box = report.denominator_box
    contract_passed = report.overall_summary.accuracy_gate_passed
    user_compound_passed = (
        report.overall_user_target_summary.gate_compound_passed
    )
    return IntermediateBaselineResult(
        output_path=output_path,
        total_problems=box.total_problems,
        measured=box.measured,
        typed_unavailable=box.typed_unavailable,
        diagnostic_failed=box.diagnostic_failed,
        corpus_pending=box.corpus_pending,
        host_arch_unsupported=box.host_arch_unsupported,
        passing_user_target=box.passing_user_target,
        passing_tiered_gate=box.passing_tiered_gate,
        passing_compound_gate=box.passing_compound_gate,
        contract_gate_passed=contract_passed,
        user_compound_gate_passed=user_compound_passed,
        cli_exit_code=_cli_exit_code_for_report(report),
    )


__all__ = [
    "CLI_EXIT_BOTH_GATES_PASS",
    "CLI_EXIT_SCHEMA_OR_CONTRACT_GATE_FAIL",
    "CLI_EXIT_USER_ACCEPTANCE_GATE_FAIL",
    "INTERMEDIATE_BASELINE_TYPED_REASONS",
    "IntermediateBaselineError",
    "IntermediateBaselineResult",
    "emit_intermediate_baseline",
]
