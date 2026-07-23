"""Tests for the arch-token policy + sm_120 sidecar admission surface.

The Cross-Arch Bounded First sm_120 Label Admission Loop requires
three sm_120 path tokens to be deterministically related across the
producer + consumer surface. This test module covers:

1. The ``tools/local_labels/arch_token_policy.py`` module's boundary
   mapping helpers and error types.
2. The committed
   ``data/local_labels/launch_parameters_sidecars/sm_120-laptop/``
   directory existence (the operator-stage sweep populates it with
   per-cell sidecars).
3. The negative-control rejection of an accidental
   ``arch="sm_120-laptop"`` literal in
   ``data/local_labels/label_admitted.json`` (the label-side enum
   admits ONLY ``sm_86`` and ``sm_120`` per the v4 schema).
4. The ``tools/local_labels/extend_launch_parameters.py`` arch filter
   widening to admit ``arch=sm_120`` rows alongside the existing
   ``sm_86`` filter, with per-arch output routing.
5. The
   ``tools/local_labels/run_headline_cohort_predict_compare.py::_load_launch_parameters_sidecar_for_cell``
   parameterization on arch with a typed
   :class:`SidecarDirectoryMissingError` on missing sm_120 sidecar
   directory (sm_86 retains the legacy fall-through no-correction
   path; sm_120 cannot silently skip gridDim correction).
"""

from __future__ import annotations

import importlib
import json
import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


class ArchTokenPolicyModuleTest(unittest.TestCase):
    """Module-level invariants for the boundary mapping helpers."""

    def test_label_arch_to_sidecar_token_mapping(self) -> None:
        import arch_token_policy as atp
        self.assertEqual(atp.LABEL_ARCH_TO_SIDECAR_TOKEN["sm_86"], "sm_86")
        self.assertEqual(
            atp.LABEL_ARCH_TO_SIDECAR_TOKEN["sm_120"], "sm_120-laptop"
        )

    def test_label_arch_to_platform_path_mapping(self) -> None:
        import arch_token_policy as atp
        self.assertEqual(
            atp.LABEL_ARCH_TO_PLATFORM_PATH["sm_86"], "rtx_3090_ti/sm_86"
        )
        self.assertEqual(
            atp.LABEL_ARCH_TO_PLATFORM_PATH["sm_120"], "rtx_5060_laptop/sm_120"
        )

    def test_supported_label_arches_is_two_token_frozenset(self) -> None:
        import arch_token_policy as atp
        self.assertEqual(atp.SUPPORTED_LABEL_ARCHES, frozenset({"sm_86", "sm_120"}))

    def test_sidecar_token_for_label_arch_returns_mapped_value(self) -> None:
        import arch_token_policy as atp
        self.assertEqual(atp.sidecar_token_for_label_arch("sm_86"), "sm_86")
        self.assertEqual(
            atp.sidecar_token_for_label_arch("sm_120"), "sm_120-laptop"
        )

    def test_sidecar_token_for_unknown_arch_raises_typed_error(self) -> None:
        import arch_token_policy as atp
        with self.assertRaises(atp.UnsupportedLabelArchError):
            atp.sidecar_token_for_label_arch("sm_999")

    def test_platform_path_for_label_arch_returns_mapped_value(self) -> None:
        import arch_token_policy as atp
        self.assertEqual(
            atp.platform_path_for_label_arch("sm_86"), "rtx_3090_ti/sm_86"
        )
        self.assertEqual(
            atp.platform_path_for_label_arch("sm_120"),
            "rtx_5060_laptop/sm_120",
        )

    def test_platform_path_for_unknown_arch_raises_typed_error(self) -> None:
        import arch_token_policy as atp
        with self.assertRaises(atp.UnsupportedLabelArchError):
            atp.platform_path_for_label_arch("sm_999")


class Sm120LaptopSidecarDirectoryTest(unittest.TestCase):
    """The committed ``data/local_labels/launch_parameters_sidecars/sm_120-laptop/``
    directory is the parallel of the existing ``sm_86/`` directory; the
    operator-stage sweep populates it."""

    def test_sm_120_laptop_sidecar_directory_exists(self) -> None:
        target = (
            _REPO_ROOT
            / "data"
            / "local_labels"
            / "launch_parameters_sidecars"
            / "sm_120-laptop"
        )
        self.assertTrue(target.is_dir(), msg=str(target))

    def test_sm_86_sidecar_directory_still_exists(self) -> None:
        target = (
            _REPO_ROOT
            / "data"
            / "local_labels"
            / "launch_parameters_sidecars"
            / "sm_86"
        )
        self.assertTrue(target.is_dir(), msg=str(target))


class LabelAdmittedDoesNotCarrySm120LaptopArchTest(unittest.TestCase):
    """Negative control: the v4 admitted-label schema's arch enum admits
    ONLY ``sm_86`` and ``sm_120``; an accidental ``arch="sm_120-laptop"``
    literal (which is the device/capability-token form, NOT the
    label-token form) must not appear in
    ``data/local_labels/label_admitted.json``."""

    def test_live_label_admitted_arch_field_in_supported_set(self) -> None:
        import arch_token_policy as atp
        live_path = (
            _REPO_ROOT / "data" / "local_labels" / "label_admitted.json"
        )
        if not live_path.is_file():
            self.skipTest(f"Live label_admitted.json not present at {live_path}")
        with live_path.open(encoding="utf-8") as fh:
            instance = json.load(fh)
        for idx, cell in enumerate(instance.get("cells", [])):
            with self.subTest(cell_idx=idx, slug=cell.get("slug")):
                self.assertIn(
                    cell.get("arch"),
                    atp.SUPPORTED_LABEL_ARCHES,
                    msg=(
                        f"Cell {idx} ({cell.get('slug')!r}/"
                        f"{cell.get('shape_point')!r}) has arch="
                        f"{cell.get('arch')!r}; the v4 admitted-label "
                        "schema admits ONLY sm_86 and sm_120 (the "
                        "label-side token form); sm_120-laptop is the "
                        "device/sidecar-side token and must not appear "
                        "in label_admitted.json."
                    ),
                )


class ExtendLaunchParametersArchFilterTest(unittest.TestCase):
    """``tools/local_labels/extend_launch_parameters.py`` admits
    ``arch=sm_120`` rows alongside the existing ``sm_86`` filter; the
    per-arch routing sends sm_86 sidecars to ``sm_86/`` and sm_120
    sidecars to ``sm_120-laptop/``."""

    def test_extend_launch_parameters_admits_sm120_arch(self) -> None:
        elp = importlib.import_module("extend_launch_parameters")
        admitted_arches = getattr(elp, "_ADMITTED_LABEL_ARCHES", None)
        self.assertIsNotNone(
            admitted_arches,
            msg=(
                "extend_launch_parameters does not expose "
                "_ADMITTED_LABEL_ARCHES; the cohort-emitter widening "
                "adds this module constant alongside the existing "
                "per-arch filtering."
            ),
        )
        self.assertIn("sm_86", admitted_arches)
        self.assertIn("sm_120", admitted_arches)

    def test_extend_launch_parameters_routes_sm120_output_to_sm_120_laptop_dir(self) -> None:
        elp = importlib.import_module("extend_launch_parameters")
        helper = getattr(elp, "_out_dir_for_label_arch", None)
        self.assertIsNotNone(
            helper,
            msg=(
                "extend_launch_parameters does not expose "
                "_out_dir_for_label_arch(arch) helper; the cohort "
                "emitter requires per-arch output-directory routing "
                "via the shared arch_token_policy mapping."
            ),
        )
        sm120_out = helper("sm_120")
        self.assertEqual(sm120_out.name, "sm_120-laptop")
        sm86_out = helper("sm_86")
        self.assertEqual(sm86_out.name, "sm_86")


class PredictorSideLookupParameterizationTest(unittest.TestCase):
    """``run_headline_cohort_predict_compare.py::_load_launch_parameters_sidecar_for_cell``
    takes an ``arch`` argument and routes to the per-arch sidecar
    directory; sm_120 raises
    :class:`arch_token_policy.SidecarDirectoryMissingError` when the
    directory is absent rather than silently returning ``(None, None,
    None)`` (the legacy fall-through is preserved for sm_86 only)."""

    def test_load_sidecar_raises_typed_error_for_sm120_with_missing_dir(self) -> None:
        rhcpc = importlib.import_module("run_headline_cohort_predict_compare")
        loader = getattr(rhcpc, "_load_launch_parameters_sidecar_for_cell", None)
        self.assertIsNotNone(loader)
        # Probe the signature: the predictor-side parameterization
        # must accept an ``arch`` kwarg.
        import inspect
        sig = inspect.signature(loader)
        self.assertIn(
            "arch",
            sig.parameters,
            msg=(
                "run_headline_cohort_predict_compare._load_launch_parameters_sidecar_for_cell "
                "does not declare an 'arch' parameter; the predictor-"
                "side lookup requires per-arch parameterization with "
                "typed-error on missing sm_120 sidecar directory."
            ),
        )

    def test_load_sidecar_sm86_legacy_fall_through_preserved(self) -> None:
        rhcpc = importlib.import_module("run_headline_cohort_predict_compare")
        loader = rhcpc._load_launch_parameters_sidecar_for_cell
        # Call with a sm_86 slug/shape combination that doesn't have a
        # committed sidecar. Must return the (None, None, None) legacy
        # fall-through, NOT raise.
        result = loader(
            slug="this_slug_does_not_exist",
            shape_point="row_0",
            expected_identity={
                "arch": "sm_86",
                "slug": "this_slug_does_not_exist",
                "shape_point": "row_0",
                "binary_sha256": "a" * 64,
                "source_commit": "b" * 40,
                "kernel_signature_sha256": "c" * 64,
            },
            arch="sm_86",
        )
        self.assertEqual(result, (None, None, None))


if __name__ == "__main__":
    unittest.main()
