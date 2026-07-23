"""Lockstep tests asserting the local-label driver's NCU metric name
matches the contract pinned at
``workloads/simbench_bw/comparison_contract.toml``.

The driver invokes NCU with a hardcoded ``--metrics`` argument; the
comparison runner consumes admitted-cell records whose ``profiler_version``
must match the contract's pinned ``ncu_metric.name`` value. A drift
between the hardcoded driver string and the contract pin would silently
produce admitted cells that the runner cannot consume (typed
``comparison_contract_drift``).

This module's tests:

- Read the contract's pinned ``ncu_metric.name`` value from
  ``comparison_contract.toml`` and assert that exact literal appears in
  ``tools/local_labels/label_driver.sh``'s NCU invocation argv (the
  driver's ``--metrics`` argument string).
- Assert the formerly-pinned typo
  ``lts__t_sectors_lookup_hit_rate.pct`` is absent from the driver,
  the parser, and every committed test file under
  ``tools/local_labels/``. The typo'd form was an unintended pin in an
  earlier contract revision; NCU 2026.1.0's `--query-metrics` exposes
  no metric named `lts__t_sectors_lookup_hit_rate`, so the typo'd
  pin produced an "n/a" metric value and blocked label admission.
  The corrected pin is the canonical Ratio name `lts__t_sector_hit_rate`
  (accessed in percent form via `.pct`). This scanner module
  legitimately names the typo'd literal as a string constant and is
  excluded from its own scan.
- Assert the parser's module docstring or CLI help text references the
  current contract-pinned name (so a future reader who consults the
  parser's prose is not misdirected to the typo'd form).

Per CLAUDE.md's "Two-Device Local Development Model", on-disk per-launch
sidecars under ``data/local_labels/sm_*/`` may carry historical
metric-name values as historical evidence; this scan is bounded to
source files that the driver+parser+tests load at runtime.
"""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_LABELS_DIR = _REPO_ROOT / "tools" / "local_labels"
_DRIVER = _LOCAL_LABELS_DIR / "label_driver.sh"
_PARSER = _LOCAL_LABELS_DIR / "parse_ncu_metrics.py"
_CONTRACT_TOML = (
    _REPO_ROOT
    / "workloads"
    / "simbench_bw"
    / "comparison_contract.toml"
)

# The typo'd metric literal an earlier contract revision pinned (the
# string `lts__t_sectors_lookup_hit_rate.pct`). NCU 2026.1.0's
# `--query-metrics` lists no metric of that name and returns "n/a"
# when asked for it, so the typo'd pin produced unusable label data.
# The lockstep test asserts this string is absent from the live source
# surface. Constructed via concatenation so a substring search of this
# very file cannot mistake the constant for an embedded source
# occurrence.
_TYPOD_METRIC_LITERAL = "lts__t_sectors_lookup" + "_hit_rate.pct"

# Backwards-compatible alias for the existing test method names that
# refer to "legacy". The semantic is "the metric literal that used to
# appear in the driver/parser/tests but is now forbidden because it
# does not resolve to a real NCU metric".
_LEGACY_METRIC_LITERAL = _TYPOD_METRIC_LITERAL


def _load_contract_metric_name() -> str:
    with _CONTRACT_TOML.open("rb") as fh:
        contract = tomllib.load(fh)
    return contract["ncu_metric"]["name"]


class ContractPinnedMetricNameAppearsInDriverTest(unittest.TestCase):
    """The driver's ``--metrics`` argument string must contain the
    contract's pinned ``ncu_metric.name`` value verbatim. A drift
    between the two surfaces is exactly the regression this test
    catches before the runner is invoked downstream."""

    def test_driver_invokes_ncu_with_contract_pinned_metric_name(self) -> None:
        contract_metric = _load_contract_metric_name()
        driver_text = _DRIVER.read_text(encoding="utf-8")
        self.assertIn(
            contract_metric,
            driver_text,
            msg=(
                f"contract metric name {contract_metric!r} not found in "
                f"label_driver.sh; the driver's --metrics arg must match "
                f"the contract pin"
            ),
        )

    def test_parser_module_carries_contract_pinned_metric_name(self) -> None:
        contract_metric = _load_contract_metric_name()
        parser_text = _PARSER.read_text(encoding="utf-8")
        self.assertIn(
            contract_metric,
            parser_text,
            msg=(
                f"parse_ncu_metrics.py does not reference the contract "
                f"metric name {contract_metric!r}; the parser is "
                f"metric-name-agnostic but its docstring or CLI help "
                f"text must mirror the canonical metric so a reader is "
                f"not misdirected to a legacy form"
            ),
        )


class LegacyMetricLiteralAbsentFromSourceTest(unittest.TestCase):
    """The legacy metric literal must not appear in the driver, the
    parser, or any committed test file under ``tools/local_labels/``.
    The on-disk per-launch sidecars under ``data/local_labels/`` may
    carry the legacy value as historical evidence; this scan is bounded
    to source files that the driver+parser+tests load at runtime."""

    def test_legacy_literal_absent_from_driver(self) -> None:
        text = _DRIVER.read_text(encoding="utf-8")
        self.assertNotIn(
            _LEGACY_METRIC_LITERAL,
            text,
            msg=(
                f"legacy literal {_LEGACY_METRIC_LITERAL!r} found in "
                f"label_driver.sh; replace it with the contract-pinned "
                f"name from comparison_contract.toml"
            ),
        )

    def test_legacy_literal_absent_from_parser(self) -> None:
        text = _PARSER.read_text(encoding="utf-8")
        self.assertNotIn(
            _LEGACY_METRIC_LITERAL,
            text,
            msg=(
                f"legacy literal {_LEGACY_METRIC_LITERAL!r} found in "
                f"parse_ncu_metrics.py; update the docstring and CLI "
                f"help to the contract-pinned name"
            ),
        )

    def test_legacy_literal_absent_from_local_labels_test_files(self) -> None:
        offenders: list[str] = []
        self_path = Path(__file__).resolve()
        for test_file in sorted(_LOCAL_LABELS_DIR.glob("test_*.py")):
            if test_file.resolve() == self_path:
                # This scanner module legitimately names the legacy
                # literal as a string constant; exclude it from its
                # own scan so the constant does not flag itself.
                continue
            text = test_file.read_text(encoding="utf-8")
            if _LEGACY_METRIC_LITERAL in text:
                offenders.append(test_file.name)
        self.assertEqual(
            offenders,
            [],
            msg=(
                f"legacy literal {_LEGACY_METRIC_LITERAL!r} found in "
                f"test files: {offenders!r}; update fixtures to the "
                f"contract-pinned name"
            ),
        )


if __name__ == "__main__":
    unittest.main()
