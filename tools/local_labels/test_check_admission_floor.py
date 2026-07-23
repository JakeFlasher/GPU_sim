"""Tests for ``tools/local_labels/check_admission_floor.py``.

The tool is a standalone CLI surface the operator runs on either
host (the current 3090 Ti host pre-pivot, the receiving RTX 5060
laptop post-pivot) to verify the per-arch and cross-arch
admission state of ``data/local_labels/label_admitted.json``
against the contract's smoke-test floor.

Test surface:

- ``--arch sm_86`` with the current file: exit 0 (5 admitted cells).
- ``--arch sm_120`` with the current file: exit non-zero (0 cells).
- ``--cross-arch`` with the current file: exit non-zero (sm_120 below
  floor).
- ``--admitted-path <missing>``: exit non-zero with a typed diagnostic.
- ``--admitted-path <invalid>``: exit non-zero with a schema-validation
  failure.
- ``--quiet`` emits only the verdict token; default mode emits a
  one-line per-arch report.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_TOOL = _HERE / "check_admission_floor.py"
_DEFAULT_ADMITTED = _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"


def _run_tool(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_TOOL), *args],
        capture_output=True, text=True, check=False,
        cwd=str(_REPO_ROOT),
    )


def _make_admitted_record(slug: str, arch: str, *,
                          n_repeats: int = 5,
                          stddev_l2_sector: float = 0.005,
                          cache_hit_rate_l2_sector_pct: float = 50.0,
                          evidence_kind: str = "live") -> dict:
    """Mint a strict-v5 admitted-cell record matching
    ``local_label_admitted.schema.json``'s ``$defs.admitted_cell``."""
    return {
        "schema_version": 5,
        "slug": slug,
        "shape_point": "trace_00",
        "binary_sha256": "a" * 64,
        "source_commit": "b" * 40,
        "kernel_signature_sha256": "c" * 64,
        "arch": arch,
        "ncu_metric_version": "ncu_2026.1.x",
        "profiler_version": "ncu_2026.1.x",
        "cache_hit_rate_l2_sector_pct": cache_hit_rate_l2_sector_pct,
        "cache_hit_rate_l1tex_sector_pct": None,
        "l1tex_label_status": "mean_absent_label_missing",
        "n_repeats": n_repeats,
        "stddev_l2_sector": stddev_l2_sector,
        "evidence_kind": evidence_kind,
    }


def _write_admitted(td: Path, *, sm86: int = 0, sm120: int = 0) -> Path:
    """Write a synthetic ``label_admitted.json`` at ``td/admitted.json``
    with the requested per-arch counts. Returns the path."""
    cells: list[dict] = []
    for i in range(sm86):
        cells.append(_make_admitted_record(f"sm86_kernel_{i}", "sm_86"))
    for i in range(sm120):
        cells.append(_make_admitted_record(f"sm120_kernel_{i}", "sm_120"))
    payload = {"schema_version": 5, "cells": cells}
    path = td / "admitted.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


class ToolFileExistsTest(unittest.TestCase):
    def test_tool_exists(self) -> None:
        self.assertTrue(_TOOL.is_file(), msg=str(_TOOL))


class CurrentRepoStateAdmissionFloorTest(unittest.TestCase):
    """Exercise the tool against the live
    ``data/local_labels/label_admitted.json`` to verify the receiving
    3090 Ti host's current state. The committed file carries sm_86
    admitted cells at ``source_commit=7189cda...`` (live evidence's
    pre-sweep HEAD) plus 5 historical activation labels at
    ``source_commit=0c16e335...``; current git HEAD does NOT match
    either, so the live-state tests pass ``--all-source-commits`` to
    preserve their original semantic ('the file has >=5 sm_86
    admitted cells regardless of source_commit'). The new
    consumer-aligned default behavior (filter to current HEAD) is
    covered by the synthetic-fixture tests in
    ``SourceCommitDefaultSemanticTest`` below."""

    def test_default_admitted_path_arch_sm86_floor_5_exits_zero(self) -> None:
        proc = _run_tool(
            "--arch", "sm_86", "--floor", "5", "--all-source-commits",
        )
        self.assertEqual(
            proc.returncode, 0,
            msg=f"stdout={proc.stdout!r} stderr={proc.stderr!r}",
        )

    def test_default_admitted_path_arch_sm120_floor_5_exits_zero(self) -> None:
        # The sm_120-only decoupled framework's amended cohort
        # admitted 5 sm_120 cells (commits 34d4aef1 + 448a691e):
        # layernorm/timing, rmsnorm/timing, silu/timing,
        # softmax_safe/timing, swiglu_xl. The default sm_120
        # floor 5 check therefore now PASSES against the live
        # file. The legacy assumption "0 sm_120 admitted cells"
        # was a pre-loop world; the synthetic-fixture test
        # below
        # (`SyntheticFixtureAdmissionFloorTest::test_sm120_does_not_meet_floor_when_synthetic_file_has_no_sm120_cells`)
        # preserves the 0-cells-fails semantic with a
        # tmpdir fixture.
        proc = _run_tool(
            "--arch", "sm_120", "--floor", "5", "--all-source-commits",
        )
        self.assertEqual(
            proc.returncode, 0,
            msg=(
                "tool rejected sm_120 floor 5 against the current file, "
                "but the file now has 5 sm_120 admitted cells "
                "(amended cohort: layernorm/rmsnorm/silu/softmax_safe "
                "timing_shape + swiglu_fwd_bf16_sm80/trace_l2_accuracy_xl). "
                f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
            ),
        )

    def test_default_cross_arch_floor_5_exits_zero(self) -> None:
        # Cross-arch floor 5 against the live file: sm_86 has
        # 44 admitted (legacy) + sm_120 has 5 admitted (the
        # decoupled framework's amended cohort); both arches
        # meet the floor so the cross-arch check now PASSES.
        # The legacy "0 sm_120 cells → cross-arch fails" world
        # is preserved by the synthetic-fixture test that uses
        # a tmpdir manifest with 0 sm_120 cells.
        proc = _run_tool(
            "--cross-arch", "--floor", "5", "--all-source-commits",
        )
        self.assertEqual(proc.returncode, 0)


class SyntheticFixtureAdmissionFloorTest(unittest.TestCase):
    """Exercise the tool against synthetic ``label_admitted.json``
    fixtures so the test does not depend on the live host's file."""

    def test_sm120_meets_floor_when_synthetic_file_has_5_sm120_cells(self) -> None:
        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            path = _write_admitted(td, sm86=5, sm120=5)
            proc = _run_tool(
                "--arch", "sm_120", "--floor", "5",
                "--admitted-path", str(path),
                "--all-source-commits",
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_cross_arch_meets_floor_when_both_arches_at_5(self) -> None:
        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            path = _write_admitted(td, sm86=5, sm120=5)
            proc = _run_tool(
                "--cross-arch", "--floor", "5",
                "--admitted-path", str(path),
                "--all-source-commits",
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_sm86_below_floor_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            path = _write_admitted(td, sm86=3, sm120=5)
            proc = _run_tool(
                "--arch", "sm_86", "--floor", "5",
                "--admitted-path", str(path),
                "--all-source-commits",
            )
            self.assertNotEqual(proc.returncode, 0)

    def test_synthetic_cells_do_not_count_toward_admission_floor(self) -> None:
        """The comparison-runner enforces a live-only smoke-test
        floor (the admitted-cell schema documents that synthetic
        cells must not satisfy the smoke-test gate). The
        operator self-check tool must apply the same filter so
        the verdict matches what the comparison bundle would
        report; otherwise an admitted file with 4 live + 2
        synthetic sm_86 cells would self-check as `admitted`
        (6 >= 5) while the actual comparison bundle would
        report `below_floor` (live-only count = 4 < 5).

        Pre-fix: 3 live + 2 synthetic sm_86 cells admit at
        floor=5 (counts mix live + synthetic to 5).
        Post-fix: same fixture rejects (live-only count = 3 <
        5)."""

        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            cells: list[dict] = []
            for i in range(3):
                cells.append(
                    _make_admitted_record(
                        f"sm86_live_{i}", "sm_86",
                        evidence_kind="live",
                    )
                )
            for i in range(2):
                cells.append(
                    _make_admitted_record(
                        f"sm86_synth_{i}", "sm_86",
                        evidence_kind="synthetic",
                    )
                )
            payload = {"schema_version": 5, "cells": cells}
            path = td / "admitted.json"
            with path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.write("\n")
            proc = _run_tool(
                "--arch", "sm_86", "--floor", "5",
                "--admitted-path", str(path),
                "--all-source-commits",
            )
            self.assertNotEqual(
                proc.returncode, 0,
                msg=(
                    "self-check tool admitted floor=5 against a 3 "
                    "live + 2 synthetic sm_86 fixture (live-only "
                    "count = 3 < 5). The comparison-runner would "
                    "report below_floor; the self-check verdict "
                    "must match.\nstdout: " + proc.stdout +
                    "\nstderr: " + proc.stderr
                ),
            )

    def test_duplicate_cell_identity_at_distinct_source_commit_does_not_overcount(self) -> None:
        """The materializer
        (`select_best_config_labels.py --preserve-historical-source-commit`)
        intentionally retains historical/superseded live records
        alongside HEAD-coherent records for the same
        ``(arch, slug, shape_point)``: the production
        ``label_admitted.json`` carries 5 historical activation
        labels at ``source_commit=0c16e335...`` AS WELL AS the
        HEAD-coherent labels at ``source_commit=7189cda...``.
        ``comparison_runner`` joins on the full 7-tuple
        (which includes ``source_commit``), so historical
        labels surface as
        ``comparison_runner_join_key_unmatched`` rather than
        contributing to the per-arch headline live count.

        The operator self-check must mirror that grain: count
        UNIQUE ``(arch, slug, shape_point)`` cell identities,
        not raw rows. Otherwise a corpus with 4 distinct
        cells + 1 historical duplicate of one of them at a
        different source_commit would self-check as 5 (admits
        at floor=5) while the runner's headline count would
        be 4 (rejects)."""

        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            cells: list[dict] = []
            for i in range(4):
                cells.append(
                    _make_admitted_record(
                        f"sm86_kernel_{i}", "sm_86",
                        evidence_kind="live",
                    )
                )
            duplicate = _make_admitted_record(
                "sm86_kernel_0", "sm_86",
                evidence_kind="live",
            )
            duplicate["source_commit"] = "d" * 40
            cells.append(duplicate)
            payload = {"schema_version": 5, "cells": cells}
            path = td / "admitted.json"
            with path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.write("\n")
            proc = _run_tool(
                "--arch", "sm_86", "--floor", "5",
                "--admitted-path", str(path),
                "--all-source-commits",
            )
            self.assertNotEqual(
                proc.returncode, 0,
                msg=(
                    "self-check tool admitted floor=5 against a 4 "
                    "distinct sm_86 cells + 1 historical duplicate "
                    "of one of them; the runner's headline count "
                    "would be 4 (joins on full 7-tuple including "
                    "source_commit). Self-check must mirror by "
                    "deduping (arch, slug, shape_point) cell "
                    "identities.\nstdout: " + proc.stdout +
                    "\nstderr: " + proc.stderr
                ),
            )

    def test_distinct_cell_identities_at_floor_admits(self) -> None:
        """Positive control: 5 cell identities each with a
        distinct ``slug`` and the same arch admit at floor=5.
        Pre-fix and post-fix both pass; pins the contract that
        the dedup does NOT collapse cells with the same arch
        but different slug/shape_point."""

        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            path = _write_admitted(td, sm86=5, sm120=0)
            proc = _run_tool(
                "--arch", "sm_86", "--floor", "5",
                "--admitted-path", str(path),
                "--all-source-commits",
            )
            self.assertEqual(
                proc.returncode, 0,
                msg=(
                    "self-check tool rejected 5 distinct sm_86 "
                    "cells at floor=5; dedup must not collapse "
                    "distinct identities.\nstdout: " + proc.stdout +
                    "\nstderr: " + proc.stderr
                ),
            )

    def test_live_only_sm86_meets_floor_after_synthetic_filter(self) -> None:
        """Positive control: an admitted file with 5 live sm_86
        cells AND 100 synthetic sm_86 cells must still admit at
        floor=5. The synthetic cells are silently dropped from
        the per-arch count; the live count alone is the
        admission signal."""

        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            cells: list[dict] = []
            for i in range(5):
                cells.append(
                    _make_admitted_record(
                        f"sm86_live_{i}", "sm_86",
                        evidence_kind="live",
                    )
                )
            for i in range(100):
                cells.append(
                    _make_admitted_record(
                        f"sm86_synth_{i}", "sm_86",
                        evidence_kind="synthetic",
                    )
                )
            payload = {"schema_version": 5, "cells": cells}
            path = td / "admitted.json"
            with path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.write("\n")
            proc = _run_tool(
                "--arch", "sm_86", "--floor", "5",
                "--admitted-path", str(path),
                "--all-source-commits",
            )
            self.assertEqual(
                proc.returncode, 0,
                msg=(
                    "self-check tool rejected a 5-live + 100-synthetic "
                    "sm_86 fixture at floor=5; live-only count >= "
                    "floor must admit.\nstdout: " + proc.stdout +
                    "\nstderr: " + proc.stderr
                ),
            )


class SourceCommitDefaultSemanticTest(unittest.TestCase):
    """Closes the operator-DX finding raised in Codex Round-22
    review: the documented operator command
    ``check_admission_floor.py --arch sm_86 --floor 5`` MUST
    admit on a committed ``label_admitted.json`` whose cells
    carry historical sweep source_commits (the common operator
    scenario: inspecting the committed corpus on any HEAD that
    isn't the original sweep commit).

    The prior default (auto-derive current HEAD + filter) was a
    false-negative on this scenario because no committed cells
    matched the current HEAD. Codex Round-22 review reverted the
    default to count across all source_commits; the
    consumer-runtime-mirror discipline (the prior R33
    BL-20260509 update) is still accessible via the explicit
    ``--current-head`` flag for operators who run the self-check
    immediately after a fresh sweep.
    """

    def test_default_counts_across_all_source_commits(self) -> None:
        """Post Codex Round-22 fix: 5 sm_86 cells all at
        synthetic ``source_commit='b'*40`` admit at floor=5 under
        the default (no flag) because the default counts every
        unique (arch, slug, shape_point) live cell identity
        regardless of source_commit. The documented operator
        command works on the committed corpus."""

        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            path = _write_admitted(td, sm86=5, sm120=0)
            proc = _run_tool(
                "--arch", "sm_86", "--floor", "5",
                "--admitted-path", str(path),
            )
            self.assertEqual(
                proc.returncode, 0,
                msg=(
                    "default-mode self-check rejected 5 sm_86 "
                    "cells at synthetic source_commit 'b'*40; the "
                    "default must count across all source_commits "
                    "so the documented operator command works on "
                    "the committed corpus.\nstdout: " + proc.stdout +
                    "\nstderr: " + proc.stderr
                ),
            )

    def test_all_source_commits_flag_is_explicit_synonym_for_default(self) -> None:
        """``--all-source-commits`` is the explicit synonym for
        the default behavior post Codex Round-22 fix. The flag
        is retained for documentation and back-compat with
        existing operator runbooks that pass it explicitly."""

        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            path = _write_admitted(td, sm86=5, sm120=0)
            proc = _run_tool(
                "--arch", "sm_86", "--floor", "5",
                "--admitted-path", str(path),
                "--all-source-commits",
            )
            self.assertEqual(
                proc.returncode, 0,
                msg=(
                    "--all-source-commits rejected 5 sm_86 cells "
                    "at floor=5; the flag is the explicit synonym "
                    "for the default and must admit identically."
                    "\nstdout: " + proc.stdout +
                    "\nstderr: " + proc.stderr
                ),
            )

    def test_current_head_flag_filters_to_current_HEAD(self) -> None:
        """``--current-head`` opts into the consumer-runtime-
        mirror discipline (the prior R33 BL-20260509 default).
        On a fixture whose cells don't match current HEAD, this
        flag REJECTS. Pairs with the default-admits positive
        test above."""

        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            path = _write_admitted(td, sm86=5, sm120=0)
            proc = _run_tool(
                "--arch", "sm_86", "--floor", "5",
                "--admitted-path", str(path),
                "--current-head",
            )
            self.assertNotEqual(
                proc.returncode, 0,
                msg=(
                    "--current-head admitted 5 sm_86 cells at "
                    "synthetic source_commit 'b'*40; current "
                    "HEAD does not match so the consumer-runtime-"
                    "mirror discipline must REJECT.\nstdout: "
                    + proc.stdout + "\nstderr: " + proc.stderr
                ),
            )

    def test_explicit_source_commit_filters_to_specific_sweep(self) -> None:
        """Explicit ``--source-commit <SHA>`` filters live cells
        to the given SHA. This preserves the post-sweep
        verification use case: the operator passes the
        pre-sweep HEAD SHA explicitly and the helper filters
        on that SHA, regardless of what the current HEAD is."""

        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            path = _write_admitted(td, sm86=5, sm120=0)
            proc = _run_tool(
                "--arch", "sm_86", "--floor", "5",
                "--admitted-path", str(path),
                "--source-commit", "b" * 40,
            )
            self.assertEqual(
                proc.returncode, 0,
                msg=(
                    "explicit --source-commit 'b'*40 should match "
                    "the synthetic fixture's cells "
                    "(_make_admitted_record default source_commit "
                    "is 'b'*40); expected admit at 5.\nstdout: "
                    + proc.stdout + "\nstderr: " + proc.stderr
                ),
            )

    def test_source_commit_and_all_source_commits_are_mutually_exclusive(self) -> None:
        """Mutex: passing both ``--source-commit`` and
        ``--all-source-commits`` is a usage error. argparse
        emits a usage-error to stderr (rc=2)."""

        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            path = _write_admitted(td, sm86=5, sm120=0)
            proc = _run_tool(
                "--arch", "sm_86", "--floor", "5",
                "--admitted-path", str(path),
                "--source-commit", "b" * 40,
                "--all-source-commits",
            )
            self.assertNotEqual(
                proc.returncode, 0,
                msg=(
                    "passing both --source-commit and "
                    "--all-source-commits should be a usage error; "
                    f"got rc={proc.returncode}.\nstdout: "
                    f"{proc.stdout!r} stderr: {proc.stderr!r}"
                ),
            )


class FailClosedDiagnosticsTest(unittest.TestCase):
    def test_missing_admitted_path_exits_nonzero_with_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            missing = td / "does_not_exist.json"
            proc = _run_tool(
                "--arch", "sm_86", "--floor", "5",
                "--admitted-path", str(missing),
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("does_not_exist.json", proc.stderr)

    def test_schema_invalid_admitted_file_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            invalid = td / "invalid.json"
            # Drop a required field to fail the strict schema.
            cell = _make_admitted_record("bad_kernel", "sm_86")
            del cell["profiler_version"]
            invalid.write_text(
                json.dumps({"schema_version": 5, "cells": [cell]}),
                encoding="utf-8",
            )
            proc = _run_tool(
                "--arch", "sm_86", "--floor", "5",
                "--admitted-path", str(invalid),
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn(
                "profiler_version",
                proc.stdout + proc.stderr,
                msg=(
                    "tool did not name the missing required field in "
                    "its diagnostic; expected 'profiler_version' to "
                    "appear in stderr or stdout"
                ),
            )


class SchemaPathSchemaErrorTest(unittest.TestCase):
    """review-fix-14 regression: passing ``--schema-path`` at a
    syntactically-valid JSON file that is NOT a valid JSON Schema
    (e.g., ``{"not": 3}`` — the ``"not"`` keyword requires a
    schema, not an integer) MUST fail closed with rc=2 and a typed
    ``check_admission_floor:`` diagnostic naming the schema path.

    The runbook documents rc=2 for "schema/io errors". Without
    catching ``jsonschema.SchemaError`` around
    ``jsonschema.validate(...)``, the exception propagates as an
    uncaught Python traceback with rc=1, breaking the documented
    exit-code contract."""

    def test_invalid_schema_at_schema_path_exits_2_with_typed_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            invalid_schema_path = td / "not_a_schema.json"
            invalid_schema_path.write_text(
                json.dumps({"not": 3}), encoding="utf-8",
            )
            proc = _run_tool(
                "--arch", "sm_86", "--floor", "5",
                "--schema-path", str(invalid_schema_path),
            )
            self.assertEqual(
                proc.returncode, 2,
                msg=(
                    f"expected rc=2 for invalid JSON Schema at "
                    f"--schema-path; got rc={proc.returncode}. "
                    f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
                ),
            )
            self.assertNotIn(
                "Traceback", proc.stderr,
                msg=(
                    "stderr contains a Python traceback; the "
                    "tool must catch jsonschema.SchemaError and "
                    "emit a typed diagnostic instead. "
                    f"stderr={proc.stderr!r}"
                ),
            )
            self.assertIn(
                "check_admission_floor:", proc.stderr,
                msg=(
                    "expected typed 'check_admission_floor:' "
                    "diagnostic prefix on stderr. "
                    f"stderr={proc.stderr!r}"
                ),
            )
            self.assertIn(
                str(invalid_schema_path), proc.stderr,
                msg=(
                    "expected diagnostic to name the schema path. "
                    f"stderr={proc.stderr!r}"
                ),
            )


class QuietModeTest(unittest.TestCase):
    def test_quiet_mode_emits_only_verdict_token(self) -> None:
        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            path = _write_admitted(td, sm86=5, sm120=5)
            proc = _run_tool(
                "--cross-arch", "--floor", "5", "--quiet",
                "--admitted-path", str(path),
                "--all-source-commits",
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            stdout_lines = [
                line for line in proc.stdout.splitlines() if line.strip()
            ]
            self.assertEqual(
                len(stdout_lines), 1,
                msg=(
                    f"--quiet emitted {len(stdout_lines)} stdout lines, "
                    f"expected exactly 1: {stdout_lines!r}"
                ),
            )

    def test_default_mode_emits_per_arch_report(self) -> None:
        with tempfile.TemporaryDirectory() as td_str:
            td = Path(td_str)
            path = _write_admitted(td, sm86=5, sm120=5)
            proc = _run_tool(
                "--cross-arch", "--floor", "5",
                "--admitted-path", str(path),
                "--all-source-commits",
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("sm_86", proc.stdout)
            self.assertIn("sm_120", proc.stdout)


class CountVariantIdentityTest(unittest.TestCase):
    """`_count_per_arch` keys register-cap build variants by a mixed-arity
    identity: base cells keep the 3-tuple (byte-identical), and two canonical
    cap variants of one (arch, slug, shape_point) count as DISTINCT admitted
    cells, while a noncanonical alias / inconsistent pair fails closed."""

    @staticmethod
    def _caf():
        import check_admission_floor as caf
        return caf

    @staticmethod
    def _variant_cell(*, register_cap=None, variant_id=None,
                      binary="a" * 64, ksig="c" * 64) -> dict:
        rec = _make_admitted_record("register_spill_local_mem", "sm_86")
        rec["binary_sha256"] = binary
        rec["kernel_signature_sha256"] = ksig
        if register_cap is not None:
            rec["register_cap"] = register_cap
        if variant_id is not None:
            rec["variant_id"] = variant_id
        return rec

    def test_two_canonical_caps_count_as_distinct(self) -> None:
        caf = self._caf()
        admitted = {"cells": [
            self._variant_cell(register_cap=64, variant_id="register_cap_64",
                               binary="a" * 64, ksig="1" * 64),
            self._variant_cell(register_cap=128, variant_id="register_cap_128",
                               binary="b" * 64, ksig="2" * 64),
        ]}
        self.assertEqual(caf._count_per_arch(admitted)["sm_86"], 2)

    def test_noncanonical_alias_does_not_inflate(self) -> None:
        # The canonical cap-64 cell counts; the register_cap_064 alias is
        # failed closed (skipped), so it can never become a 2nd admitted cell.
        caf = self._caf()
        admitted = {"cells": [
            self._variant_cell(register_cap=64, variant_id="register_cap_64"),
            self._variant_cell(register_cap=64, variant_id="register_cap_064"),
        ]}
        self.assertEqual(caf._count_per_arch(admitted)["sm_86"], 1)

    def test_register_cap_only_and_variant_id_only_do_not_double_count(self) -> None:
        # Both normalize to the same canonical variant_id => one cell.
        caf = self._caf()
        admitted = {"cells": [
            self._variant_cell(register_cap=64),
            self._variant_cell(variant_id="register_cap_64"),
        ]}
        self.assertEqual(caf._count_per_arch(admitted)["sm_86"], 1)

    def test_base_cells_count_unchanged(self) -> None:
        caf = self._caf()
        admitted = {"cells": [
            _make_admitted_record("slug_a", "sm_86"),
            _make_admitted_record("slug_b", "sm_86"),
        ]}
        self.assertEqual(caf._count_per_arch(admitted)["sm_86"], 2)

    def test_inconsistent_pair_is_failed_closed(self) -> None:
        caf = self._caf()
        admitted = {"cells": [
            self._variant_cell(register_cap=64, variant_id="register_cap_32"),
        ]}
        self.assertEqual(caf._count_per_arch(admitted).get("sm_86", 0), 0)

    def test_negative_register_cap_is_failed_closed(self) -> None:
        # The shared non-negative domain: a negative cap is skipped from the
        # count (same normalizer the producer + aggregator use).
        caf = self._caf()
        admitted = {"cells": [self._variant_cell(register_cap=-1)]}
        self.assertEqual(caf._count_per_arch(admitted).get("sm_86", 0), 0)

    def test_base_and_variant_of_same_shape_are_distinct(self) -> None:
        # A base cell and a cap variant of the same (arch, slug, shape_point)
        # are distinct identities (3-tuple vs 4-tuple).
        caf = self._caf()
        admitted = {"cells": [
            _make_admitted_record("register_spill_local_mem", "sm_86"),
            self._variant_cell(register_cap=64, variant_id="register_cap_64"),
        ]}
        self.assertEqual(caf._count_per_arch(admitted)["sm_86"], 2)


if __name__ == "__main__":
    unittest.main()
