"""Standalone CLI for the Tier-2 accuracy diagnostic.

Reads an already-emitted diagnostic TOML report, validates every
section against the schema, runs the report-level reconciliation
(`validate_diagnostic_report`), and returns an exit code that
reflects the dual-gate verdict (contractual tiered gate + user-
acceptance compound gate). Does not re-launch GPU kernels,
invoke NCU, or touch the deployed predictor; consumes an
artifact produced by `build_accuracy_diagnostic_report()`.

Usage:

    python accuracy_diagnostic_cli.py <report.toml>

Exit codes:

  0  Both the contractual tiered gate AND the user-acceptance
     compound gate pass on the report.
  2  Schema validation failed OR the contractual tiered gate
     failed (the contract gate's failure mode is the older
     "gate failed" path; the report has at least one cohort
     row whose `accuracy_gate_passed` is False).
  3  Schema validation passed AND the contractual tiered gate
     passed BUT the user-acceptance compound gate failed
     (`overall_user_target_summary.gate_compound_passed=False`
     equivalently `meta.gate_compound=False`).

The dual-gate semantics give downstream automation a way to
distinguish "the contract still passes; the user-acceptance
overlay is the bottleneck" (exit 3) from "the report itself is
broken" (exit 2). The earlier single-binary-gate CLI returned
0 / 1 / 2; the dual-gate surface preserves the schema-failure
path at 2 and reserves 3 for the user-acceptance miss.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[3]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from baselines_harness.calibration.cache_prediction.accuracy_diagnostic import (  # noqa: E402
    AccuracyGateError,
    PerProblemRecordSchemaError,
    parse_diagnostic_report,
)


# Exit-code constants exported for downstream consumers + tests.
EXIT_BOTH_GATES_PASS: int = 0
EXIT_SCHEMA_OR_CONTRACT_GATE_FAIL: int = 2
EXIT_USER_ACCEPTANCE_GATE_FAIL: int = 3


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an emitted accuracy diagnostic TOML report "
            "and return an exit code reflecting the dual-gate "
            "verdict: 0 = both gates pass, 2 = schema or "
            "contractual tiered gate fails, 3 = contract passes "
            "but user-acceptance / compound gate fails."
        ),
    )
    parser.add_argument(
        "report_path",
        type=Path,
        help="path to the diagnostic TOML report",
    )
    args = parser.parse_args(argv)

    if not args.report_path.is_file():
        print(
            f"error: report file not found: {args.report_path}",
            file=sys.stderr,
        )
        return EXIT_SCHEMA_OR_CONTRACT_GATE_FAIL

    try:
        text = args.report_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(
            f"error: failed to read report file: {exc}",
            file=sys.stderr,
        )
        return EXIT_SCHEMA_OR_CONTRACT_GATE_FAIL

    try:
        report = parse_diagnostic_report(text)
    except tomllib.TOMLDecodeError as exc:
        print(f"malformed TOML in report file: {exc}", file=sys.stderr)
        return EXIT_SCHEMA_OR_CONTRACT_GATE_FAIL
    except PerProblemRecordSchemaError as exc:
        print(
            f"schema validation failed: {exc.typed_reason}: "
            f"{exc.message}",
            file=sys.stderr,
        )
        return EXIT_SCHEMA_OR_CONTRACT_GATE_FAIL
    except AccuracyGateError as exc:
        print(
            f"gate validation failed at parse time: "
            f"{exc.typed_reason}: {exc.message}",
            file=sys.stderr,
        )
        return EXIT_SCHEMA_OR_CONTRACT_GATE_FAIL

    contract_gate_passed = report.overall_summary.accuracy_gate_passed
    user_compound_gate_passed = (
        report.overall_user_target_summary.gate_compound_passed
    )

    print(f"report: {args.report_path}")
    print(
        f"  total problems: {report.coverage_summary.total_problems}"
    )
    print(
        f"  measured cells: {report.coverage_summary.measured_cell_count}"
    )
    print(
        f"  corpus_expansion_pending: "
        f"{report.coverage_summary.corpus_expansion_pending_cells}"
    )
    print(
        f"  diagnostic_failed: "
        f"{report.coverage_summary.diagnostic_failed_cells}"
    )
    print(
        f"  typed_unavailable: "
        f"{report.coverage_summary.typed_unavailable_cells}"
    )
    print(
        f"  host_arch_unsupported: "
        f"{report.coverage_summary.host_arch_unsupported_cells}"
    )
    print(
        f"  measured_pass_user_target: "
        f"{report.overall_user_target_summary.measured_pass_count}"
    )
    print(
        f"  measured_fail_user_target: "
        f"{report.overall_user_target_summary.measured_fail_count}"
    )
    print(f"  contract tiered gate passed: {contract_gate_passed}")
    print(f"  user-acceptance compound gate passed: {user_compound_gate_passed}")

    if not contract_gate_passed:
        return EXIT_SCHEMA_OR_CONTRACT_GATE_FAIL
    if not user_compound_gate_passed:
        return EXIT_USER_ACCEPTANCE_GATE_FAIL
    return EXIT_BOTH_GATES_PASS


def main(argv: list[str] | None = None) -> int:
    return _main(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
