"""Regression test for v5 fitter typed_unavailable bundle classification.

Pins the contract: when a bundle has
``result_kind == "typed_unavailable"`` with a non-empty
``typed_reason`` field, the v5 fitter's per-cell classifier MUST
propagate that bundle ``typed_reason`` as the cell's
``typed_reason`` in the skipped list. The downstream
``canonical_pre_calibration_features_absent`` symptom (the
canonical block is None because the bundle never produced a
scalar prediction) must NOT mask the root cause that the staging
manifest already records as
``BUNDLE_TYPED_UNAVAILABLE:<typed_reason>``.

Failure mode the test guards against: a future refactor restores
the symptom-only classification, breaking machine-readable
parity between the staging manifest and the v5 head's skipped
reasons. Codex review surfaced this exact gap on the 3 sm_120
``cache_model_underconfigured`` cells.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.local_labels.fit_monotonic_calibration_head_v5 import (  # noqa: E402
    _build_cell_records,
)


def _write_bundle(path: Path, payload: dict) -> Path:
    import json
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TypedUnavailableBundlePropagationTest(unittest.TestCase):
    """A bundle's ``typed_reason`` must surface verbatim on the
    v5 head's skipped row when ``result_kind=="typed_unavailable"``."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.tmp = Path(self.tmpdir.name)
        self.family_to_bucket = {
            "memory": "memory_stress",
            "compute": "tensor_compute",
            "activation": "activation_norm",
        }

    def _label_cell(self, arch: str, slug: str, shape: str) -> dict:
        return {
            "arch": arch,
            "slug": slug,
            "shape_point": shape,
            # The closed-identity bundle join keys on binary_sha256 (keeps
            # sibling register-cap variants distinct); the fixture cells pin
            # the same digest the fixture bundles' MPI carries.
            "binary_sha256": "0" * 64,
            "cache_hit_rate_l2_sector_pct": 42.0,
        }

    def _bundle(
        self, arch: str, slug: str, shape: str,
        *, result_kind: str,
        typed_reason: str | None = None,
        canonical_features: dict | None = None,
    ) -> dict:
        return {
            "manifest_pairing_index": {
                "arch": arch, "slug": slug, "shape_point": shape,
                # The closed-identity join keys on binary_sha256; pin the
                # same digest the fixture label cells carry.
                "binary_sha256": "0" * 64,
            },
            "result_kind": result_kind,
            "typed_reason": typed_reason,
            "canonical_pre_calibration_features": canonical_features,
        }

    def test_typed_unavailable_bundle_propagates_root_cause(self) -> None:
        """When bundle.result_kind=='typed_unavailable' with a real
        typed_reason, the skipped row carries THAT reason (not the
        downstream canonical-features-absent symptom)."""
        bundle = self._bundle(
            "sm_120", "demo_kernel", "trace_xl",
            result_kind="typed_unavailable",
            typed_reason="cache_model_underconfigured",
        )
        bundle_path = _write_bundle(self.tmp / "b.json", bundle)
        cells = [self._label_cell("sm_120", "demo_kernel", "trace_xl")]
        admitted, skipped = _build_cell_records(
            label_cells=cells,
            bundle_paths=[bundle_path],
            family_to_bucket=self.family_to_bucket,
        )
        self.assertEqual(admitted, [])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(
            skipped[0]["typed_reason"], "cache_model_underconfigured",
            f"v5 fitter must propagate bundle.typed_reason verbatim "
            f"when bundle.result_kind=='typed_unavailable'; got "
            f"{skipped[0]['typed_reason']!r} instead",
        )

    def test_typed_unavailable_without_reason_falls_back(self) -> None:
        """A typed_unavailable bundle missing typed_reason (defensive
        case) falls through to the canonical-features-absent symptom
        so the cell still gets a non-empty skipped row."""
        bundle = self._bundle(
            "sm_120", "demo_kernel", "trace_xl",
            result_kind="typed_unavailable",
            typed_reason="",
        )
        bundle_path = _write_bundle(self.tmp / "b.json", bundle)
        cells = [self._label_cell("sm_120", "demo_kernel", "trace_xl")]
        _, skipped = _build_cell_records(
            label_cells=cells,
            bundle_paths=[bundle_path],
            family_to_bucket=self.family_to_bucket,
        )
        self.assertEqual(len(skipped), 1)
        self.assertEqual(
            skipped[0]["typed_reason"],
            "canonical_pre_calibration_features_absent",
        )

    def test_scalar_bundle_unaffected(self) -> None:
        """A normal scalar bundle (result_kind!='typed_unavailable')
        with canonical features admits as before — the new branch
        does NOT affect the admit path."""
        features = {
            "predicted_hit_rate_uncorrected": 0.5,
            "unique_sector_count": 1024,
            "sector_size_bytes": 32,
            "l2_capacity_bytes": 6 * 1024 * 1024,
            "footprint_ratio": 0.5,
            "sharing_locality_score": 0.3,
        }
        bundle = self._bundle(
            "sm_120", "demo_kernel", "trace_xl",
            result_kind="scalar",
            canonical_features=features,
        )
        bundle_path = _write_bundle(self.tmp / "b.json", bundle)
        cells = [self._label_cell("sm_120", "demo_kernel", "trace_xl")]
        admitted, skipped = _build_cell_records(
            label_cells=cells,
            bundle_paths=[bundle_path],
            family_to_bucket=self.family_to_bucket,
        )
        self.assertEqual(skipped, [])
        self.assertEqual(len(admitted), 1)
        self.assertEqual(admitted[0]["arch"], "sm_120")

    def test_scalar_bundle_missing_features_still_typed_absent(
        self,
    ) -> None:
        """A scalar (non-typed-unavailable) bundle with missing
        canonical features still falls to the
        canonical_pre_calibration_features_absent symptom — the new
        branch only catches typed_unavailable bundles."""
        bundle = self._bundle(
            "sm_120", "demo_kernel", "trace_xl",
            result_kind="scalar",
            canonical_features=None,
        )
        bundle_path = _write_bundle(self.tmp / "b.json", bundle)
        cells = [self._label_cell("sm_120", "demo_kernel", "trace_xl")]
        _, skipped = _build_cell_records(
            label_cells=cells,
            bundle_paths=[bundle_path],
            family_to_bucket=self.family_to_bucket,
        )
        self.assertEqual(len(skipped), 1)
        self.assertEqual(
            skipped[0]["typed_reason"],
            "canonical_pre_calibration_features_absent",
        )


class ArchlessBundleArchResolutionTest(unittest.TestCase):
    """The join must resolve an archless-MPI bundle's arch from its canonical
    top-level ``device`` (the SAME resolver the cohort-comparison joiner uses),
    so a bundle is never admitted for the wrong arch in a multi-arch fit, and
    two bundles for the same cell fail closed instead of silently overwriting.

    Real predictor bundles carry the closed 5-tuple manifest_pairing_index with
    NO arch field; the arch is carried only by ``device``. These fixtures mirror
    that real shape (archless MPI + canonical device), unlike the arch-in-MPI
    fixtures above."""

    _FEATURES = {
        "predicted_hit_rate_uncorrected": 0.5,
        "unique_sector_count": 1024,
        "sector_size_bytes": 32,
        "l2_capacity_bytes": 6 * 1024 * 1024,
        "footprint_ratio": 0.5,
        "sharing_locality_score": 0.3,
    }

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.tmp = Path(self.tmpdir.name)
        self.fam = {
            "memory": "memory_stress",
            "compute": "tensor_compute",
            "activation": "activation_norm",
        }

    def _archless_bundle(self, device: str, slug: str, shape: str) -> dict:
        return {
            "manifest_pairing_index": {
                "slug": slug, "shape_point": shape,
                "binary_sha256": "0" * 64, "source_commit": "a" * 40,
                "kernel_signature_sha256": "1" * 64,
            },
            "device": device,
            "result_kind": "scalar",
            "canonical_pre_calibration_features": dict(self._FEATURES),
        }

    def _label_cell(self, arch: str, slug: str, shape: str) -> dict:
        return {
            "arch": arch, "slug": slug, "shape_point": shape,
            # Match the fixture bundles' MPI binary_sha256 (the closed-
            # identity join keys on it).
            "binary_sha256": "0" * 64,
            "cache_hit_rate_l2_sector_pct": 42.0,
        }

    def test_archless_bundle_with_canonical_device_admits_to_its_arch(
        self,
    ) -> None:
        """The real-bundle shape: an archless-MPI bundle whose canonical device
        is the RTX 3090 Ti resolves to sm_86 and admits to the sm_86 cell."""
        bp = _write_bundle(
            self.tmp / "b.json",
            self._archless_bundle("NVIDIA GeForce RTX 3090 Ti", "demo", "trace_xl"),
        )
        admitted, skipped = _build_cell_records(
            label_cells=[self._label_cell("sm_86", "demo", "trace_xl")],
            bundle_paths=[bp], family_to_bucket=self.fam,
        )
        self.assertEqual(skipped, [])
        self.assertEqual(len(admitted), 1)
        self.assertEqual(admitted[0]["arch"], "sm_86")

    def test_multi_arch_fit_resolves_each_arch_no_cross_contamination(
        self,
    ) -> None:
        """In a multi-arch fit (no --arch) where sm_86 and sm_120 cells share a
        (slug, shape_point), the RTX 3090 Ti bundle admits ONLY to the sm_86
        cell and the RTX 5060 bundle ONLY to the sm_120 cell — neither arch's
        features contaminate the other."""
        b86 = _write_bundle(
            self.tmp / "b86.json",
            self._archless_bundle(
                "NVIDIA GeForce RTX 3090 Ti", "shared", "trace_xl"),
        )
        b120 = _write_bundle(
            self.tmp / "b120.json",
            self._archless_bundle(
                "NVIDIA GeForce RTX 5060 Laptop GPU", "shared", "trace_xl"),
        )
        cells = [
            self._label_cell("sm_86", "shared", "trace_xl"),
            self._label_cell("sm_120", "shared", "trace_xl"),
        ]
        admitted, skipped = _build_cell_records(
            label_cells=cells, bundle_paths=[b86, b120],
            family_to_bucket=self.fam,
        )
        self.assertEqual(skipped, [])
        self.assertEqual(len(admitted), 2)
        self.assertEqual({c["arch"] for c in admitted}, {"sm_86", "sm_120"})

    def test_cross_arch_only_bundle_does_not_admit_other_arch(self) -> None:
        """When only the sm_86 bundle is present, a sm_120 cell sharing the
        (slug, shape_point) is SKIPPED (predictor_bundle_absent), NOT admitted
        with the sm_86 bundle's features (the cross-arch contamination this
        round closes)."""
        b86 = _write_bundle(
            self.tmp / "b86.json",
            self._archless_bundle(
                "NVIDIA GeForce RTX 3090 Ti", "shared", "trace_xl"),
        )
        admitted, skipped = _build_cell_records(
            label_cells=[self._label_cell("sm_120", "shared", "trace_xl")],
            bundle_paths=[b86], family_to_bucket=self.fam,
        )
        self.assertEqual(admitted, [])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["typed_reason"], "predictor_bundle_absent")

    def test_duplicate_same_cell_bundles_fail_closed(self) -> None:
        """Two bundles resolving to the SAME (arch, slug, shape_point) — e.g. a
        superseded same-cell bundle a glob also matched — fail closed with a
        ValueError instead of silently overwriting one with the other."""
        b1 = _write_bundle(
            self.tmp / "b1.json",
            self._archless_bundle(
                "NVIDIA GeForce RTX 3090 Ti", "dup", "trace_xl"),
        )
        b2 = _write_bundle(
            self.tmp / "b2.json",
            self._archless_bundle(
                "NVIDIA GeForce RTX 3090 Ti", "dup", "trace_xl"),
        )
        with self.assertRaises(ValueError) as ctx:
            _build_cell_records(
                label_cells=[self._label_cell("sm_86", "dup", "trace_xl")],
                bundle_paths=[b1, b2], family_to_bucket=self.fam,
            )
        self.assertIn("duplicate predictor bundle key", str(ctx.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
