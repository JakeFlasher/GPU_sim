"""Tests for the local-label repeatability aggregator.

Synthesizes per-launch sidecar files in a temporary quarantine root and
exercises every contract case for the per-cell admissibility verdict:

- positive: an N=5 cell with stddev within threshold and consistent
  ``binary_sha256`` is admitted.
- negative: an N=4 cell is excluded with typed
  ``label_repeatability_missing``.
- negative: an N=5 cell with L2 stddev > 3 percentage points is
  excluded with typed ``label_repeatability_failed``.
- negative: an N=5 cell with L1/TEX stddev > 3 percentage points is
  excluded with typed ``label_repeatability_failed``.
- negative: an N=5 cell whose imported sidecars disagree on
  ``binary_sha256`` is excluded with typed
  ``label_provenance_inconsistent`` regardless of stddev.
- positive: the three output indices
  (``label_repeatability.json``, ``label_admitted.json``,
  ``label_excluded.json``) are written to disk with the expected
  shape.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import repeatability_aggregator as ra  # noqa: E402


_VALID_BINARY_A = "a" * 64
_VALID_BINARY_B = "b" * 64
_VALID_GPU = "c" * 64
# Contract-matching profiler version banner; the aggregator
# normalizes this to "ncu_2026.1.x" per the contract pin in
# workloads/simbench_bw/comparison_contract.toml. Tests that need to
# exercise drift override this default explicitly.
_CONTRACT_PROFILER_VERSION = (
    "Version 2026.1.1.0 (build 37634170) (public-release)"
)


def _sidecar(
    *,
    repeat_index: int,
    launch_index: int = 0,
    cache_hit_rate_l2_sector: float,
    cache_hit_rate_l1tex_sector: float | None = 0.71,
    binary_sha256: str = _VALID_BINARY_A,
    device: str = "sm_86",
    slug: str = "gemm_hmma_bf16_sm80",
    shape_point: str = "row_0",
    evidence_class: str | None = "campaign",
) -> dict:
    """Build a schema-valid sidecar dict.

    The layer-aware admitted-cell migration makes L1/TEX evidence
    optional at the importer layer (the L1 fields are a coupled
    pair via dependentRequired). The helper defaults to a populated
    L1/TEX field so positive-control tests exercise the
    full-evidence happy path; tests that need to drop the L1/TEX
    evidence pass ``cache_hit_rate_l1tex_sector=None``. The L1
    admission floor (n_l1tex_repeats >= repeats_required) is
    enforced by the aggregator, not the importer."""
    sidecar = {
        "schema_version": 5,
        "device": device,
        "slug": slug,
        "shape_point": shape_point,
        "repeat_index": repeat_index,
        "launch_index": launch_index,
        "binary_sha256": binary_sha256,
        "profiler_family": "nsight_compute",
        "profiler_version": _CONTRACT_PROFILER_VERSION,
        "cuda_version": "13.0",
        "driver_version": "560.35.03",
        "gpu_uuid_sha256": _VALID_GPU,
        "replay_mode": "kernel",
        "aggregation": "kernel",
        "warmup_count": 1,
        "cache_hit_rate_l2_sector": cache_hit_rate_l2_sector,
        "confidence_interval_l2_sector": {
            "lower": cache_hit_rate_l2_sector,
            "upper": cache_hit_rate_l2_sector,
            "confidence_level": 0.95,
            "units": "sector_hit_rate_fraction",
            "aggregation": "kernel",
        },
    }
    if evidence_class is not None:
        sidecar["evidence_class"] = evidence_class
    if cache_hit_rate_l1tex_sector is not None:
        sidecar["cache_hit_rate_l1tex_sector"] = cache_hit_rate_l1tex_sector
        sidecar["confidence_interval_l1tex_sector"] = {
            "lower": cache_hit_rate_l1tex_sector,
            "upper": cache_hit_rate_l1tex_sector,
            "confidence_level": 0.95,
            "units": "sector_hit_rate_fraction",
            "aggregation": "kernel",
        }
    return sidecar


def _write_cell(
    root: Path,
    sidecars: list[dict],
    *,
    device: str = "sm_86",
    slug: str = "gemm_hmma_bf16_sm80",
    shape_point: str = "row_0",
    write_manifests: bool = True,
    manifest_source_commit: str = "f" * 40,
    manifest_kernel_signature: str = "1" * 64,
) -> Path:
    """Write the given sidecars under
    ``root/<device>/<slug>/<shape_point>/label_<repeat>_<launch>.json``
    and (by default) a per-launch ``label_run_manifest_<repeat>_<launch>.json``
    alongside each sidecar. Returns the cell directory.

    The manifest's ``binary_sha256`` mirrors each sidecar's value so
    the provenance gate's mismatch check stays on the happy path
    when callers don't want to exercise the drift surface."""
    cell_dir = root / device / slug / shape_point
    cell_dir.mkdir(parents=True, exist_ok=True)
    for sidecar in sidecars:
        repeat = sidecar["repeat_index"]
        launch = sidecar["launch_index"]
        path = cell_dir / f"label_{repeat}_{launch}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(sidecar, fh)
        if write_manifests:
            manifest = {
                "schema_version": 1,
                "source_commit": manifest_source_commit,
                "binary_sha256": sidecar["binary_sha256"],
                "kernel_signature_sha256": manifest_kernel_signature,
                "arch": device,
                "slug": slug,
                "shape_point": shape_point,
                "repeat_index": repeat,
                "launch_index": launch,
            }
            manifest_path = (
                cell_dir / f"label_run_manifest_{repeat}_{launch}.json"
            )
            with manifest_path.open("w", encoding="utf-8") as fh:
                json.dump(manifest, fh)
    return cell_dir


class CellAdmissionTest(unittest.TestCase):
    def test_n5_within_threshold_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sidecars = [
                _sidecar(repeat_index=i, cache_hit_rate_l2_sector=0.40 + 0.005 * i)
                for i in range(5)
            ]
            _write_cell(root, sidecars)
            repeatability, admitted, excluded, _ = ra.aggregate(root)
            self.assertEqual(len(admitted["cells"]), 1)
            self.assertEqual(len(excluded["cells"]), 0)
            cell = admitted["cells"][0]
            self.assertEqual(cell["arch"], "sm_86")
            self.assertEqual(cell["n_repeats"], 5)
            self.assertLess(cell["stddev_l2_sector"], 0.03)

    def test_missing_evidence_class_is_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sidecars = [
                _sidecar(
                    repeat_index=i,
                    cache_hit_rate_l2_sector=0.40 + 0.005 * i,
                    evidence_class=None,
                )
                for i in range(5)
            ]
            _write_cell(root, sidecars)
            _, admitted, _, _ = ra.aggregate(root)
            self.assertEqual(admitted["cells"][0]["evidence_kind"], "synthetic")
            self.assertNotIn("evidence_class", admitted["cells"][0])

    def test_mixed_evidence_classes_never_become_live(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sidecars = [
                _sidecar(
                    repeat_index=i,
                    cache_hit_rate_l2_sector=0.40 + 0.005 * i,
                    evidence_class="campaign" if i < 4 else "diagnostic",
                )
                for i in range(5)
            ]
            _write_cell(root, sidecars)
            _, admitted, _, _ = ra.aggregate(root)
            self.assertEqual(admitted["cells"][0]["evidence_kind"], "synthetic")
            self.assertNotIn("evidence_class", admitted["cells"][0])

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sidecars = [
                _sidecar(
                    repeat_index=i,
                    cache_hit_rate_l2_sector=0.40 + 0.005 * i,
                    cache_hit_rate_l1tex_sector=0.70 + 0.004 * i,
                )
                for i in range(5)
            ]
            _write_cell(root, sidecars)
            _, admitted, excluded, _ = ra.aggregate(root)
            self.assertEqual(len(admitted["cells"]), 1)
            self.assertEqual(len(excluded["cells"]), 0)

    def test_all_noncampaign_class_is_synthetic_and_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sidecars = [
                _sidecar(
                    repeat_index=i,
                    cache_hit_rate_l2_sector=0.40 + 0.005 * i,
                    evidence_class="calibration",
                )
                for i in range(5)
            ]
            _write_cell(root, sidecars)
            _, admitted, _, _ = ra.aggregate(root)
            self.assertEqual(admitted["cells"][0]["evidence_kind"], "synthetic")
            self.assertEqual(admitted["cells"][0]["evidence_class"], "calibration")

    def test_campaign_and_missing_class_never_become_live(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sidecars = [
                _sidecar(
                    repeat_index=i,
                    cache_hit_rate_l2_sector=0.40 + 0.005 * i,
                    evidence_class="campaign" if i < 4 else None,
                )
                for i in range(5)
            ]
            _write_cell(root, sidecars)
            _, admitted, _, _ = ra.aggregate(root)
            self.assertEqual(admitted["cells"][0]["evidence_kind"], "synthetic")
            self.assertNotIn("evidence_class", admitted["cells"][0])

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sidecars = [
                _sidecar(repeat_index=i, cache_hit_rate_l2_sector=0.40 + 0.005 * i)
                for i in range(4)
            ]
            _write_cell(root, sidecars)
            _, admitted, excluded, _ = ra.aggregate(root)
            self.assertEqual(len(admitted["cells"]), 0)
            self.assertEqual(len(excluded["cells"]), 1)
            cell = excluded["cells"][0]
            self.assertEqual(cell["typed_reason"], "label_repeatability_missing")
            self.assertEqual(cell["n_repeats"], 4)

    def test_n5_high_l2_stddev_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Spread the L2 values across 0.10 fraction units; stddev
            # well above the 0.03 threshold.
            sidecars = [
                _sidecar(repeat_index=i, cache_hit_rate_l2_sector=0.20 + 0.10 * i)
                for i in range(5)
            ]
            _write_cell(root, sidecars)
            _, admitted, excluded, _ = ra.aggregate(root)
            self.assertEqual(len(admitted["cells"]), 0)
            self.assertEqual(len(excluded["cells"]), 1)
            cell = excluded["cells"][0]
            self.assertEqual(cell["typed_reason"], "label_repeatability_failed")
            self.assertGreater(cell["stddev_l2_sector"], 0.03)

    def test_n5_stddev_just_under_threshold_admitted(self) -> None:
        # Boundary case: a cell whose sample stddev sits just under the
        # 0.03 threshold must be admitted. The contract reads
        # ``stddev <= 3 percentage points``, so the boundary is
        # admissible. Direct equality tests are FP-fragile (Python's
        # `statistics.stdev` rounding shifts a target of exactly 0.03
        # to 0.030000000000000006), so this pair of tests exercises the
        # boundary by picking a target slightly inside and slightly
        # outside.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            import math

            target_stddev = 0.029
            d = target_stddev / math.sqrt(5 / 8)
            mean = 0.50
            values = [mean - d, mean - d / 2, mean, mean + d / 2, mean + d]
            sidecars = [
                _sidecar(repeat_index=i, cache_hit_rate_l2_sector=v)
                for i, v in enumerate(values)
            ]
            _write_cell(root, sidecars)
            repeatability, admitted, excluded, _ = ra.aggregate(root)
            self.assertEqual(len(admitted["cells"]), 1)
            self.assertEqual(len(excluded["cells"]), 0)
            cell = repeatability["cells"][0]
            self.assertLess(cell["stddev_l2_sector"], 0.03)
            self.assertTrue(cell["admission"]["admitted"])

    def test_n5_stddev_just_over_threshold_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            import math

            target_stddev = 0.031
            d = target_stddev / math.sqrt(5 / 8)
            mean = 0.50
            values = [mean - d, mean - d / 2, mean, mean + d / 2, mean + d]
            sidecars = [
                _sidecar(repeat_index=i, cache_hit_rate_l2_sector=v)
                for i, v in enumerate(values)
            ]
            _write_cell(root, sidecars)
            _, admitted, excluded, _ = ra.aggregate(root)
            self.assertEqual(len(admitted["cells"]), 0)
            self.assertEqual(len(excluded["cells"]), 1)
            cell = excluded["cells"][0]
            self.assertEqual(cell["typed_reason"], "label_repeatability_failed")
            self.assertGreater(cell["stddev_l2_sector"], 0.03)

    def test_n5_high_l1tex_stddev_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sidecars = [
                _sidecar(
                    repeat_index=i,
                    cache_hit_rate_l2_sector=0.40 + 0.005 * i,
                    cache_hit_rate_l1tex_sector=0.20 + 0.10 * i,
                )
                for i in range(5)
            ]
            _write_cell(root, sidecars)
            _, admitted, excluded, _ = ra.aggregate(root)
            self.assertEqual(len(admitted["cells"]), 0)
            self.assertEqual(len(excluded["cells"]), 1)
            self.assertEqual(
                excluded["cells"][0]["typed_reason"],
                "label_repeatability_failed",
            )

    def test_n5_inconsistent_binary_sha256_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sidecars = [
                _sidecar(repeat_index=0, cache_hit_rate_l2_sector=0.42, binary_sha256=_VALID_BINARY_A),
                _sidecar(repeat_index=1, cache_hit_rate_l2_sector=0.42, binary_sha256=_VALID_BINARY_A),
                _sidecar(repeat_index=2, cache_hit_rate_l2_sector=0.42, binary_sha256=_VALID_BINARY_A),
                _sidecar(repeat_index=3, cache_hit_rate_l2_sector=0.42, binary_sha256=_VALID_BINARY_B),
                _sidecar(repeat_index=4, cache_hit_rate_l2_sector=0.42, binary_sha256=_VALID_BINARY_B),
            ]
            _write_cell(root, sidecars)
            _, admitted, excluded, _ = ra.aggregate(root)
            self.assertEqual(len(admitted["cells"]), 0)
            self.assertEqual(len(excluded["cells"]), 1)
            cell = excluded["cells"][0]
            # Per-launch sidecars carry mixed binary_sha256; the
            # default _write_cell mirrors each sidecar's hash into its
            # own per-launch manifest, so the cell's manifests
            # disagree on the binary_sha256 invariant first. The
            # provenance gate's `label_provenance_duplicate` token
            # fires before the legacy `label_provenance_inconsistent`
            # path on the existing summary's binary_sha256_consistent
            # check, which is now subsumed by the manifest-level
            # invariant. Either token is a valid signal of the same
            # underlying defect; the new gate is strictly stricter.
            self.assertEqual(cell["typed_reason"], "label_provenance_duplicate")

    def test_provenance_inconsistent_takes_priority_over_stddev(self) -> None:
        # A cell with both inconsistent binaries AND high stddev should
        # surface the provenance reason first because that is the more
        # fundamental data-integrity violation.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sidecars = [
                _sidecar(repeat_index=0, cache_hit_rate_l2_sector=0.10, binary_sha256=_VALID_BINARY_A),
                _sidecar(repeat_index=1, cache_hit_rate_l2_sector=0.30, binary_sha256=_VALID_BINARY_A),
                _sidecar(repeat_index=2, cache_hit_rate_l2_sector=0.50, binary_sha256=_VALID_BINARY_B),
                _sidecar(repeat_index=3, cache_hit_rate_l2_sector=0.70, binary_sha256=_VALID_BINARY_B),
                _sidecar(repeat_index=4, cache_hit_rate_l2_sector=0.90, binary_sha256=_VALID_BINARY_B),
            ]
            _write_cell(root, sidecars)
            _, _, excluded, _ = ra.aggregate(root)
            # The manifest invariant gate fires before the cross-repeat
            # stddev gate — provenance disagreement is the
            # higher-priority data-integrity defect.
            self.assertEqual(
                excluded["cells"][0]["typed_reason"],
                "label_provenance_duplicate",
            )

    def test_multiple_launches_per_repeat_aggregated(self) -> None:
        # Three launches per repeat across 5 repeats. Each repeat
        # collapses to its mean before stddev is taken across repeats.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sidecars: list[dict] = []
            for repeat in range(5):
                for launch in range(3):
                    sidecars.append(
                        _sidecar(
                            repeat_index=repeat,
                            launch_index=launch,
                            cache_hit_rate_l2_sector=0.40 + 0.005 * repeat,
                        )
                    )
            _write_cell(root, sidecars)
            repeatability, admitted, excluded, _ = ra.aggregate(root)
            self.assertEqual(len(admitted["cells"]), 1)
            self.assertEqual(repeatability["cells"][0]["n_sidecars"], 15)
            self.assertEqual(repeatability["cells"][0]["n_repeats"], 5)


class IndexFileTest(unittest.TestCase):
    def test_main_writes_three_index_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sidecars = [
                _sidecar(repeat_index=i, cache_hit_rate_l2_sector=0.40 + 0.005 * i)
                for i in range(5)
            ]
            _write_cell(root, sidecars)
            self.assertEqual(ra.main(["--root", str(root)]), 0)
            for filename in (
                "label_repeatability.json",
                "label_admitted.json",
                "label_excluded.json",
            ):
                with self.subTest(file=filename):
                    self.assertTrue(
                        (root / filename).exists(),
                        msg=f"{filename} not emitted",
                    )

    def test_main_separate_out_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "quarantine"
            out = Path(td) / "indices"
            sidecars = [
                _sidecar(repeat_index=i, cache_hit_rate_l2_sector=0.40 + 0.005 * i)
                for i in range(5)
            ]
            _write_cell(root, sidecars)
            self.assertEqual(
                ra.main(["--root", str(root), "--out", str(out)]), 0
            )
            self.assertTrue((out / "label_repeatability.json").exists())
            self.assertTrue((out / "label_admitted.json").exists())
            self.assertTrue((out / "label_excluded.json").exists())

    def test_admitted_index_carries_stddevs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sidecars = [
                _sidecar(
                    repeat_index=i,
                    cache_hit_rate_l2_sector=0.40 + 0.005 * i,
                    cache_hit_rate_l1tex_sector=0.70 + 0.003 * i,
                )
                for i in range(5)
            ]
            _write_cell(root, sidecars)
            ra.main(["--root", str(root)])
            with (root / "label_admitted.json").open() as fh:
                admitted = json.load(fh)
            self.assertEqual(len(admitted["cells"]), 1)
            cell = admitted["cells"][0]
            self.assertIn("stddev_l2_sector", cell)
            self.assertIn("stddev_l1tex_sector", cell)
            self.assertLess(cell["stddev_l2_sector"], 0.03)
            self.assertLess(cell["stddev_l1tex_sector"], 0.03)


class RejectedSidecarTest(unittest.TestCase):
    def test_invalid_sidecar_recorded_in_repeatability_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            valid = [
                _sidecar(repeat_index=i, cache_hit_rate_l2_sector=0.40 + 0.005 * i)
                for i in range(5)
            ]
            cell_dir = _write_cell(root, valid)

            # Drop a malformed sidecar in the same cell directory.
            malformed_sidecar = _sidecar(
                repeat_index=99, cache_hit_rate_l2_sector=0.42
            )
            malformed_sidecar["event_id"] = "lts__t_sector_hits"
            with (cell_dir / "label_99_0.json").open("w", encoding="utf-8") as fh:
                json.dump(malformed_sidecar, fh)

            repeatability, admitted, _, rejected = ra.aggregate(root)
            self.assertEqual(len(admitted["cells"]), 1)
            self.assertEqual(len(rejected), 1)
            self.assertEqual(
                rejected[0][1].typed_reason, "forbidden_field_name"
            )
            # The malformed sidecar must NOT contribute to the cell's
            # repeat count.
            cell_summary = repeatability["cells"][0]
            self.assertEqual(cell_summary["n_repeats"], 5)


class L1TexLayerAwareAdmissionTest(unittest.TestCase):
    """The layer-aware migration relaxes the importer to admit
    L2-valid/L1-missing per-repeat sidecars. The aggregator
    absorbs those sidecars into a v4 admitted record with null
    L1 mean + ``l1tex_label_status="mean_absent_label_missing"``;
    L2 admission continues normally for the same cell.

    L1 admission gate: the L1 mean + stddev are populated ONLY
    when ``n_l1tex_repeats >= repeats_required`` (the same floor
    that drives L2 admission). Partial L1 evidence (e.g.,
    1-of-5 or 4-of-5 L1-bearing repeats) must NOT promote to
    ``mean_present`` because the L1 stddev is undefined for n<2
    and degenerate for n<repeats_required, and an under-evidenced
    L1 mean would falsely contribute to the L1 relative error in
    downstream cohort artifacts. Half-pair sidecars (L1 metric
    without CI or vice versa) remain rejected via the schema's
    dependentRequired arm."""

    def test_l2_only_sidecars_admit_as_mean_absent_label_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # All five sidecars carry only L2 (no L1/TEX). Each cell
            # repeat contributes a valid L2 metric; no sidecar is
            # rejected at the importer layer.
            sidecars = [
                _sidecar(
                    repeat_index=i,
                    cache_hit_rate_l2_sector=0.40 + 0.005 * i,
                    cache_hit_rate_l1tex_sector=None,
                )
                for i in range(5)
            ]
            _write_cell(root, sidecars)

            repeatability, admitted, _excluded, rejected = ra.aggregate(root)
            # No sidecar rejected at the importer layer.
            self.assertEqual(rejected, [])
            # The cell admits at v5 with null L1 mean + the
            # absent-label-missing status.
            self.assertEqual(len(admitted["cells"]), 1)
            cell = admitted["cells"][0]
            self.assertEqual(cell["schema_version"], 5)
            self.assertEqual(cell["n_repeats"], 5)
            self.assertIsNone(cell["cache_hit_rate_l1tex_sector_pct"])
            self.assertEqual(
                cell["l1tex_label_status"],
                "mean_absent_label_missing",
            )
            # L2 admission continues normally.
            self.assertGreater(cell["cache_hit_rate_l2_sector_pct"], 0.0)

    def test_l1tex_half_pair_sidecar_rejected(self) -> None:
        """Sidecars with cache_hit_rate_l1tex_sector but no
        confidence_interval_l1tex_sector (or vice versa) remain
        rejected via the schema's dependentRequired arm. The pair
        is coupled; a half-pair is malformed even though the
        importer admits the absent-both case."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Four valid sidecars + one half-pair (L1 metric but no CI).
            valid = [
                _sidecar(repeat_index=i, cache_hit_rate_l2_sector=0.40 + 0.005 * i)
                for i in range(4)
            ]
            half_pair = _sidecar(
                repeat_index=4,
                cache_hit_rate_l2_sector=0.42,
                cache_hit_rate_l1tex_sector=0.55,
            )
            del half_pair["confidence_interval_l1tex_sector"]
            _write_cell(root, valid + [half_pair])

            repeatability, admitted, excluded, rejected = ra.aggregate(root)
            # The half-pair is rejected at the importer layer.
            self.assertEqual(len(rejected), 1)
            _, rejection_result = rejected[0]
            self.assertEqual(
                rejection_result.typed_reason,
                "schema_validation_failed",
            )
            # The cell has only 4 valid repeats; that falls below the
            # n>=5 admission floor, so no admitted cell is produced.
            self.assertEqual(len(admitted["cells"]), 0)

    def _stage_partial_l1_cell(
        self, root: Path, *, n_l1_bearing: int,
    ) -> None:
        """Stage 5 L2-valid sidecars where ``n_l1_bearing`` of them
        also carry a ``cache_hit_rate_l1tex_sector`` value. Used by
        the partial-L1 admission tests below."""
        sidecars = []
        for i in range(5):
            l1_value: float | None
            if i < n_l1_bearing:
                l1_value = 0.55 + 0.001 * i
            else:
                l1_value = None
            sidecars.append(
                _sidecar(
                    repeat_index=i,
                    cache_hit_rate_l2_sector=0.40 + 0.005 * i,
                    cache_hit_rate_l1tex_sector=l1_value,
                )
            )
        _write_cell(root, sidecars)

    def test_partial_l1_1_of_5_repeats_admits_as_mean_absent(
        self,
    ) -> None:
        """Codex Round-8 review reproducer: 1-of-5 L1-bearing
        repeats out of 5 L2-valid repeats must NOT promote to
        ``mean_present`` because the L1 repeat floor is not met.
        The L2 admission stays correct (n_repeats=5); the L1 side
        emits null + ``mean_absent_label_missing`` so the comparison
        runner emits the L1 typed-unavailable path."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._stage_partial_l1_cell(root, n_l1_bearing=1)

            repeatability, admitted, _excluded, rejected = ra.aggregate(root)
            # No importer-layer rejection.
            self.assertEqual(rejected, [])
            # L2 admits at the n>=5 floor.
            self.assertEqual(len(admitted["cells"]), 1)
            cell = admitted["cells"][0]
            self.assertEqual(cell["n_repeats"], 5)
            self.assertGreater(cell["cache_hit_rate_l2_sector_pct"], 0.0)
            # L1 side: under-evidenced, must be absent.
            self.assertIsNone(cell["cache_hit_rate_l1tex_sector_pct"])
            self.assertEqual(
                cell["l1tex_label_status"],
                "mean_absent_label_missing",
            )
            self.assertIsNone(cell["stddev_l1tex_sector"])
            # The repeatability summary records n_l1tex_repeats so
            # the operator can audit the L1 admission decision.
            self.assertEqual(
                repeatability["cells"][0]["n_l1tex_repeats"], 1,
            )

    def test_partial_l1_4_of_5_repeats_admits_as_mean_absent(
        self,
    ) -> None:
        """4-of-5 L1-bearing repeats is still below the n>=5 L1
        repeat floor; the L1 side must remain absent even though
        only one repeat is missing the L1 value. The floor is the
        same as the L2 admission floor — partial evidence does not
        admit on either side."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._stage_partial_l1_cell(root, n_l1_bearing=4)

            repeatability, admitted, _excluded, rejected = ra.aggregate(root)
            self.assertEqual(rejected, [])
            self.assertEqual(len(admitted["cells"]), 1)
            cell = admitted["cells"][0]
            self.assertEqual(cell["n_repeats"], 5)
            self.assertIsNone(cell["cache_hit_rate_l1tex_sector_pct"])
            self.assertEqual(
                cell["l1tex_label_status"],
                "mean_absent_label_missing",
            )
            self.assertIsNone(cell["stddev_l1tex_sector"])
            self.assertEqual(
                repeatability["cells"][0]["n_l1tex_repeats"], 4,
            )

    def test_full_l1_5_of_5_repeats_admits_as_mean_present(
        self,
    ) -> None:
        """5-of-5 L1-bearing repeats meets the floor; the cell
        admits as v4 with ``mean_present`` and a non-null L1 mean +
        valid stddev. Positive control for the L1 admission gate."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._stage_partial_l1_cell(root, n_l1_bearing=5)

            _repeatability, admitted, _excluded, rejected = ra.aggregate(root)
            self.assertEqual(rejected, [])
            self.assertEqual(len(admitted["cells"]), 1)
            cell = admitted["cells"][0]
            self.assertEqual(cell["l1tex_label_status"], "mean_present")
            self.assertIsNotNone(cell["cache_hit_rate_l1tex_sector_pct"])
            self.assertGreater(
                cell["cache_hit_rate_l1tex_sector_pct"], 0.0,
            )
            self.assertIsNotNone(cell["stddev_l1tex_sector"])

    def test_high_variance_l1_5_of_5_rejected_by_stddev_gate(
        self,
    ) -> None:
        """With 5-of-5 L1-bearing repeats but unstable values, the
        existing L1 stddev gate (threshold 0.03 fraction units)
        rejects the cell with ``label_repeatability_failed``. The
        repeat-floor gate is independent of and must not bypass the
        stddev gate."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # L2 is stable (under 0.03 stddev) so the L2 path passes
            # its own gate; L1 spans 0.10 to 0.90 which gives a
            # stddev > 0.03 (the contract threshold), so the
            # repeatability gate fails for L1 first.
            sidecars = []
            l1_values = [0.10, 0.30, 0.50, 0.70, 0.90]
            for i in range(5):
                sidecars.append(
                    _sidecar(
                        repeat_index=i,
                        cache_hit_rate_l2_sector=0.40 + 0.005 * i,
                        cache_hit_rate_l1tex_sector=l1_values[i],
                    )
                )
            _write_cell(root, sidecars)

            repeatability, admitted, excluded, _rejected = ra.aggregate(root)
            # The cell is excluded; no admitted record.
            self.assertEqual(len(admitted["cells"]), 0)
            self.assertEqual(len(excluded["cells"]), 1)
            excluded_cell = excluded["cells"][0]
            self.assertEqual(
                excluded_cell["typed_reason"],
                "label_repeatability_failed",
            )
            # The L1 stddev is the cause; the L2 path stays stable.
            self.assertIn("L1/TEX", excluded_cell["detail"])


class EmptyRootTest(unittest.TestCase):
    def test_empty_root_yields_empty_indices(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repeatability, admitted, excluded, rejected = ra.aggregate(root)
            self.assertEqual(repeatability["cells"], [])
            self.assertEqual(admitted["cells"], [])
            self.assertEqual(excluded["cells"], [])
            self.assertEqual(rejected, [])

    def test_missing_root_yields_empty_indices(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "does_not_exist"
            repeatability, admitted, excluded, rejected = ra.aggregate(root)
            self.assertEqual(repeatability["cells"], [])
            self.assertEqual(admitted["cells"], [])
            self.assertEqual(excluded["cells"], [])
            self.assertEqual(rejected, [])


class BuildVariantAdmissionTest(unittest.TestCase):
    """The optional register-cap build-variant identity on the cell key.

    Two cap variants of the same (arch, slug, shape_point) aggregate and
    admit as DISTINCT cells (no false ``label_provenance_duplicate`` on
    their differing binary / kernel signatures); a base cell's admitted
    record is byte-identical (no variant fields); intra-variant provenance
    disagreement is still rejected.
    """

    _DEVICE = "sm_86"
    _SLUG = "register_spill_local_mem"
    _SHAPE = "trace_00"

    def _write_variant_cell(
        self,
        root: Path,
        *,
        register_cap: int,
        binary_sha256: str,
        kernel_signature: str,
        n: int = 5,
        l2: float = 0.5,
        manifest_kernel_signatures: list[str] | None = None,
        emit_register_cap: bool = True,
        emit_variant_id: bool = True,
        variant_id_value: str | None = None,
        dir_label: str | None = None,
    ) -> Path:
        """Write a variant cell: n sidecars + per-launch run-manifests, each
        (by default) carrying a consistent ``register_cap`` + ``variant_id``,
        into a per-variant directory so two variants of the same shape do not
        collide on disk. ``manifest_kernel_signatures`` (per-repeat) overrides
        the single signature to exercise intra-cell provenance disagreement.
        ``emit_register_cap`` / ``emit_variant_id`` drop a field to exercise
        the derive-the-other path; ``variant_id_value`` overrides the emitted
        variant_id to exercise the mismatch path."""
        variant_id = (
            variant_id_value
            if variant_id_value is not None
            else f"register_cap_{register_cap}"
        )
        cell_dir = (
            root / self._DEVICE / self._SLUG / self._SHAPE
            / (dir_label or variant_id or f"register_cap_{register_cap}")
        )
        cell_dir.mkdir(parents=True, exist_ok=True)

        def _variant_fields() -> dict:
            fields: dict = {}
            if emit_register_cap:
                fields["register_cap"] = register_cap
            if emit_variant_id:
                fields["variant_id"] = variant_id
            return fields

        for r in range(n):
            sidecar = _sidecar(
                repeat_index=r,
                cache_hit_rate_l2_sector=l2,
                cache_hit_rate_l1tex_sector=None,
                binary_sha256=binary_sha256,
                device=self._DEVICE,
                slug=self._SLUG,
                shape_point=self._SHAPE,
            )
            sidecar.update(_variant_fields())
            (cell_dir / f"label_{r}_0.json").write_text(
                json.dumps(sidecar), encoding="utf-8"
            )
            ksig = (
                manifest_kernel_signatures[r]
                if manifest_kernel_signatures is not None
                else kernel_signature
            )
            manifest = {
                "schema_version": 1,
                "source_commit": "f" * 40,
                "binary_sha256": binary_sha256,
                "kernel_signature_sha256": ksig,
                "arch": self._DEVICE,
                "slug": self._SLUG,
                "shape_point": self._SHAPE,
                "repeat_index": r,
                "launch_index": 0,
                **_variant_fields(),
            }
            (cell_dir / f"label_run_manifest_{r}_0.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
        return cell_dir

    def test_two_caps_same_shape_admit_as_distinct_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_variant_cell(
                root, register_cap=64,
                binary_sha256="a" * 64, kernel_signature="1" * 64,
            )
            self._write_variant_cell(
                root, register_cap=32,
                binary_sha256="b" * 64, kernel_signature="2" * 64,
            )
            _repeatability, admitted, excluded, _ = ra.aggregate(root)
            cells = admitted["cells"]
            self.assertEqual(len(cells), 2, (admitted, excluded))
            by_cap = {c["register_cap"]: c for c in cells}
            self.assertEqual(set(by_cap), {64, 32})
            self.assertEqual(by_cap[64]["variant_id"], "register_cap_64")
            self.assertEqual(by_cap[32]["variant_id"], "register_cap_32")
            self.assertEqual(by_cap[64]["binary_sha256"], "a" * 64)
            self.assertEqual(by_cap[32]["binary_sha256"], "b" * 64)
            self.assertEqual(by_cap[64]["kernel_signature_sha256"], "1" * 64)
            self.assertEqual(by_cap[32]["kernel_signature_sha256"], "2" * 64)
            # The collapse-into-one-cell bug would have tripped this.
            excl_reasons = [c.get("typed_reason") for c in excluded["cells"]]
            self.assertNotIn("label_provenance_duplicate", excl_reasons)
            self.assertNotIn("label_schema_admission_failed", excl_reasons)

    def test_base_cell_has_no_variant_fields_and_admits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_cell(
                root,
                [
                    _sidecar(
                        repeat_index=r,
                        cache_hit_rate_l2_sector=0.5,
                        cache_hit_rate_l1tex_sector=None,
                    )
                    for r in range(5)
                ],
            )
            _repeatability, admitted, _excluded, _ = ra.aggregate(root)
            self.assertEqual(len(admitted["cells"]), 1)
            record = admitted["cells"][0]
            self.assertNotIn("register_cap", record)
            self.assertNotIn("variant_id", record)

    def test_intra_variant_provenance_disagreement_still_excluded(self) -> None:
        # The variant key does NOT weaken intra-cell provenance: two manifests
        # in the SAME variant cell disagreeing on a cell-invariant field
        # (kernel_signature) still produce label_provenance_duplicate.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_variant_cell(
                root, register_cap=64,
                binary_sha256="a" * 64, kernel_signature="1" * 64,
                manifest_kernel_signatures=["1" * 64, "1" * 64, "1" * 64,
                                            "1" * 64, "9" * 64],
            )
            _repeatability, admitted, excluded, _ = ra.aggregate(root)
            self.assertEqual(admitted["cells"], [])
            self.assertEqual(len(excluded["cells"]), 1)
            self.assertEqual(
                excluded["cells"][0]["typed_reason"], "label_provenance_duplicate"
            )
            self.assertEqual(
                excluded["cells"][0]["variant_id"], "register_cap_64"
            )

    def test_admitted_schema_accepts_variant_and_base_records(self) -> None:
        # Direct consumer-contract check: the on-disk admitted schema accepts a
        # base record (no variant fields) AND a variant record (register_cap +
        # variant_id), proving the additive widening is correct.
        import jsonschema

        schema = ra._load_admitted_record_schema()
        base = {
            "schema_version": 5,
            "slug": self._SLUG,
            "shape_point": self._SHAPE,
            "binary_sha256": "a" * 64,
            "source_commit": "f" * 40,
            "kernel_signature_sha256": "1" * 64,
            "arch": "sm_86",
            "ncu_metric_version": ra.PINNED_NCU_METRIC_VERSION,
            "profiler_version": ra.PINNED_NCU_METRIC_VERSION,
            "cache_hit_rate_l2_sector_pct": 50.0,
            "cache_hit_rate_l1tex_sector_pct": None,
            "l1tex_label_status": "mean_absent_label_missing",
            "n_repeats": 5,
            "stddev_l2_sector": 0.0,
            "stddev_l1tex_sector": None,
            "evidence_kind": "live",
        }
        jsonschema.validate(instance=base, schema=schema)
        variant = dict(base)
        variant["register_cap"] = 64
        variant["variant_id"] = "register_cap_64"
        jsonschema.validate(instance=variant, schema=schema)
        # A malformed variant_id is rejected by the schema pattern.
        bad = dict(variant)
        bad["variant_id"] = "maxrreg_64"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=bad, schema=schema)
        # A noncanonical leading-zero alias is rejected by the canonical
        # output pattern (belt-and-suspenders for the normalizer).
        alias = dict(variant)
        alias["variant_id"] = "register_cap_064"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=alias, schema=schema)
        # The canonical zero form is accepted.
        zero = dict(variant)
        zero["register_cap"] = 0
        zero["variant_id"] = "register_cap_0"
        jsonschema.validate(instance=zero, schema=schema)

    def test_inconsistent_register_cap_variant_id_fails_closed(self) -> None:
        # The exact contradictory pair: register_cap=64 but
        # variant_id=register_cap_32. The cell must be EXCLUDED with a typed
        # reason and emit NO admitted record (otherwise grouping and the
        # admitted register_cap describe two different build variants).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_variant_cell(
                root,
                register_cap=64,
                binary_sha256="a" * 64,
                kernel_signature="1" * 64,
                variant_id_value="register_cap_32",
            )
            _repeatability, admitted, excluded, _ = ra.aggregate(root)
            self.assertEqual(admitted["cells"], [])
            self.assertEqual(len(excluded["cells"]), 1)
            self.assertEqual(
                excluded["cells"][0]["typed_reason"],
                "label_variant_identity_inconsistent",
            )

    def test_register_cap_only_admits_with_derived_variant_id(self) -> None:
        # A record carrying only register_cap (no variant_id) admits, and the
        # admitted record derives a consistent variant_id.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_variant_cell(
                root,
                register_cap=64,
                binary_sha256="a" * 64,
                kernel_signature="1" * 64,
                emit_variant_id=False,
                dir_label="cap_only_64",
            )
            _repeatability, admitted, _excluded, _ = ra.aggregate(root)
            self.assertEqual(len(admitted["cells"]), 1)
            record = admitted["cells"][0]
            self.assertEqual(record["register_cap"], 64)
            self.assertEqual(record["variant_id"], "register_cap_64")

    def test_variant_id_only_admits_with_derived_register_cap(self) -> None:
        # A record carrying only variant_id (no register_cap) admits, and the
        # admitted record derives the integer register_cap.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_variant_cell(
                root,
                register_cap=64,
                binary_sha256="a" * 64,
                kernel_signature="1" * 64,
                emit_register_cap=False,
                dir_label="vid_only_64",
            )
            _repeatability, admitted, _excluded, _ = ra.aggregate(root)
            self.assertEqual(len(admitted["cells"]), 1)
            record = admitted["cells"][0]
            self.assertEqual(record["register_cap"], 64)
            self.assertEqual(record["variant_id"], "register_cap_64")

    def test_noncanonical_alias_both_present_fails_closed(self) -> None:
        # register_cap=64 with the noncanonical leading-zero alias
        # variant_id=register_cap_064 must be EXCLUDED (not admitted under a
        # second alias for the same physical cap).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_variant_cell(
                root,
                register_cap=64,
                binary_sha256="a" * 64,
                kernel_signature="1" * 64,
                variant_id_value="register_cap_064",
            )
            _repeatability, admitted, excluded, _ = ra.aggregate(root)
            self.assertEqual(admitted["cells"], [])
            self.assertEqual(len(excluded["cells"]), 1)
            self.assertEqual(
                excluded["cells"][0]["typed_reason"],
                "label_variant_identity_inconsistent",
            )

    def test_noncanonical_alias_variant_id_only_fails_closed(self) -> None:
        # variant_id=register_cap_064 with NO register_cap field must also be
        # excluded: the alias is not the canonical register_cap_64 string.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_variant_cell(
                root,
                register_cap=64,
                binary_sha256="a" * 64,
                kernel_signature="1" * 64,
                emit_register_cap=False,
                variant_id_value="register_cap_064",
            )
            _repeatability, admitted, excluded, _ = ra.aggregate(root)
            self.assertEqual(admitted["cells"], [])
            self.assertEqual(len(excluded["cells"]), 1)
            self.assertEqual(
                excluded["cells"][0]["typed_reason"],
                "label_variant_identity_inconsistent",
            )


if __name__ == "__main__":
    unittest.main()
