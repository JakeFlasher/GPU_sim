"""Tests for the v3 calibration-evaluation sidecar selection
rule in `workloads/simbench_bw/comparison_runner.py` (the direct
CLI surface; the headline-runner wrapper has its own equivalent
helper covered by
`tools/local_labels/test_run_headline_cohort_v3_selection.py`).
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SUITE_ROOT = _HERE.parent
sys.path.insert(0, str(_SUITE_ROOT))

import comparison_runner as cr  # noqa: E402  type: ignore[import-not-found]


class ComparisonRunnerV3SelectionTest(unittest.TestCase):
    """The direct comparison-runner CLI must honor the same
    v3 calibration sidecar selection rule as the headline wrapper:

      * explicit override always honored,
      * --target-arch sm_120 defaults to
        data/local_labels/calibration_evaluation_v3.json,
      * --target-arch sm_120 with v3 absent fails closed with
        a typed diagnostic naming calibration_head_v3.json,
      * any other --target-arch falls back to v1 when present;
        otherwise None.
    """

    def _make_repo(self, tmp_root: Path) -> Path:
        (tmp_root / "data" / "local_labels").mkdir(
            parents=True, exist_ok=True,
        )
        return tmp_root

    def test_explicit_override_always_honored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            repo = self._make_repo(tmp_root)
            override = tmp_root / "operator_chosen.json"
            override.write_text("{}", encoding="utf-8")
            for arch in (None, "sm_86", "sm_120"):
                resolved = cr._resolve_calibration_evaluation_sidecar_path(
                    explicit_override=override,
                    target_arch=arch,
                    repo_root=repo,
                )
                self.assertEqual(resolved, override)

    def test_active_sm120_cohort_defaults_to_v4_when_present(self) -> None:
        """AC-6.4: active sm_120 path defaults to v4 (replaced v3
        as the active default in Round 20)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            repo = self._make_repo(tmp_root)
            v4 = repo / "data" / "local_labels" / "calibration_evaluation_v4.json"
            v4.write_text("{}", encoding="utf-8")
            resolved = cr._resolve_calibration_evaluation_sidecar_path(
                explicit_override=None,
                target_arch="sm_120",
                repo_root=repo,
            )
            self.assertEqual(resolved, v4)

    def test_active_sm120_cohort_fails_closed_when_v4_absent(self) -> None:
        """AC-6.4: the active sm_120 path is strictly
        fail-closed — it does NOT fall back to v3 (retired), v2,
        or v1 when v4 is absent."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            repo = self._make_repo(tmp_root)
            v3 = repo / "data" / "local_labels" / "calibration_evaluation_v3.json"
            v1 = repo / "data" / "local_labels" / "calibration_evaluation_v1.json"
            v3.write_text("{}", encoding="utf-8")
            v1.write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError) as ctx:
                cr._resolve_calibration_evaluation_sidecar_path(
                    explicit_override=None,
                    target_arch="sm_120",
                    repo_root=repo,
                )
            msg = str(ctx.exception)
            self.assertIn("calibration_head_v4.json", msg)
            self.assertIn("sm_120", msg)
            self.assertNotIn("fell back to v1", msg.lower())

    def test_target_arch_sm86_default_loads_v1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            repo = self._make_repo(tmp_root)
            v1 = repo / "data" / "local_labels" / "calibration_evaluation_v1.json"
            v1.write_text("{}", encoding="utf-8")
            resolved = cr._resolve_calibration_evaluation_sidecar_path(
                explicit_override=None,
                target_arch="sm_86",
                repo_root=repo,
            )
            self.assertEqual(resolved, v1)

    def test_no_target_arch_no_v1_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            repo = self._make_repo(tmp_root)
            resolved = cr._resolve_calibration_evaluation_sidecar_path(
                explicit_override=None,
                target_arch=None,
                repo_root=repo,
            )
            self.assertIsNone(resolved)

    def test_no_target_arch_with_v1_still_returns_none(self) -> None:
        # Back-compat preservation: when no --target-arch is
        # passed, the comparison runner historically ran without
        # a calibration gate (the resolution returned None). The
        # selection rule preserves this default — only an
        # explicit --target-arch sm_120 OR --target-arch sm_86
        # opts INTO a default calibration sidecar.
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            repo = self._make_repo(tmp_root)
            v1 = repo / "data" / "local_labels" / "calibration_evaluation_v1.json"
            v1.write_text("{}", encoding="utf-8")
            resolved = cr._resolve_calibration_evaluation_sidecar_path(
                explicit_override=None,
                target_arch=None,
                repo_root=repo,
            )
            self.assertIsNone(
                resolved,
                msg=(
                    "no --target-arch must NOT auto-default to v1; "
                    "the legacy synthetic-fixture tests rely on this "
                    "behavior to keep the calibration gate off when "
                    "no arch is specified."
                ),
            )


if __name__ == "__main__":
    unittest.main()
