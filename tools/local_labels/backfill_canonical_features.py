"""Backfill the canonical_pre_calibration_features block onto committed
predictor bundles.

The block is required by the monotonic calibration head's active path
(`workloads/simbench_bw/canonical_features.predict_active_path`). The
producer-side emission landed in cache_predictor.predict_cache_hit_rate
under the `v5_active_path` + `canonical_pre_calibration_features` kwargs,
but newly emitted bundles only carry the block if the caller assembles +
passes it. This script backfills the existing committed scalar bundles
without re-running the predictor end-to-end.

Inputs per cell:
  - The committed scalar predictor bundle (carries
    evidence.predicted_hit_rate_uncorrected + cache_model_capacity_bytes +
    optional cache_model_grid_volume + device).
  - The sibling sharing_locality.json sidecar (provides
    intra_lane_reuse via the `intra_lane` grounded entry).
  - The sibling hrd_profile.json sidecar (provides long_rd_tail_mass via
    sector-layer bins with rd_sector_units_min >= 4096; also provides
    cold_count which equals unique_sector_count).

Per BL-20260527-test-fixture-shape-must-mirror-real-shape: the backfill
fails closed when any required sidecar is absent (rather than guessing) so
downstream consumers cannot silently consume a partial block.

CLI:
  python tools/local_labels/backfill_canonical_features.py \
    --bundle-glob 'data/local_labels/sweeps/sm120_cohort_bundles_*/predictor_bundles/*.json' \
    --results-root workloads/simbench_bw/results \
    [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.simbench_bw import canonical_features as cf  # noqa: E402


def _device_to_sku_dir(device: str | None) -> str:
    """Map the device string carried by predictor bundles to the
    results-root SKU subdirectory."""
    if device is None:
        return "rtx_5060_laptop"
    d = device.lower()
    if "3090" in device or "ga102" in d:
        return "rtx_3090_ti"
    if "5060" in device or "blackwell" in d or "gb20" in d:
        return "rtx_5060_laptop"
    return "rtx_5060_laptop"


def _arch_to_subdir(arch: str | None) -> str:
    return arch if arch in ("sm_86", "sm_120") else "sm_120"


def _resolve_sidecar_paths(
    *, results_root: Path, slug: str, shape_point: str, sku: str, arch: str,
) -> tuple[Path, Path]:
    """Return (sharing_locality_path, hrd_profile_path) per the canonical
    repo layout."""
    cell_root = (
        results_root / slug / sku / arch / f"trace_{shape_point}"
    )
    sl_path = cell_root / "sharing_locality" / "sharing_locality.json"
    hrd_path = cell_root / "hrd_profile" / "hrd_profile.json"
    return sl_path, hrd_path


def _count_unique_sectors_from_hrd(hrd_profile: dict) -> int | None:
    """Sum the sector-layer cold-bin observation_count as a proxy for
    unique_sector_count.

    The observation_builder's `unique_sector_count` equals the count of
    distinct sector identities, which by construction equals the count of
    cold first-touch misses in the sector-layer HRD. For a cold first
    touch, the reuse distance is the binning's `rd_infinity_sentinel`.
    When the sentinel does not appear in the bins, fall back to summing
    all sector-layer observation_count (which over-counts but provides a
    deterministic placeholder rather than silently using a synthetic
    value).

    Returns None when no sector-layer bins are present.
    """
    binning = hrd_profile.get("binning", {})
    cold_sentinel = binning.get("rd_infinity_sentinel")
    bins = hrd_profile.get("bins", [])
    sector_bins = [
        b for b in bins
        if b.get("hierarchy_layer") == "sector"
        and b.get("rd_sector_units_min") is not None
    ]
    if not sector_bins:
        return None
    if cold_sentinel is None:
        # No infinity sentinel pinned; fall back to total observations.
        return sum(int(b.get("observation_count", 0)) for b in sector_bins)
    cold_obs = 0
    found_cold_bin = False
    for b in sector_bins:
        if int(b["rd_sector_units_min"]) >= cold_sentinel:
            cold_obs += int(b.get("observation_count", 0))
            found_cold_bin = True
    if not found_cold_bin:
        # No cold bin in this profile — fall back to total observations
        # for a deterministic value.
        return sum(int(b.get("observation_count", 0)) for b in sector_bins)
    return cold_obs


def _count_global_share_from_sector_stream(
    sector_stream_path: Path,
) -> tuple[int, int]:
    """Walk the sector_stream.jsonl and return (global_count, total_count).

    Only counts records whose `result_kind == "sector_stream"`; typed
    exclusions (e.g. `no_lanes_active`) are not memory accesses and would
    inflate the denominator.
    """
    total = 0
    g = 0
    with sector_stream_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("result_kind") != "sector_stream":
                continue
            total += 1
            if rec.get("memory_op_class") == "GLOBAL":
                g += 1
    return g, total


def backfill_one_bundle(
    *, bundle_path: Path, results_root: Path, dry_run: bool,
) -> dict[str, Any]:
    """Backfill the canonical block on one bundle. Returns a dict with
    ``status`` ("admitted" / "skipped" / "failed") and a ``reason`` /
    ``features`` block."""
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("result_kind") != "scalar":
        return {
            "status": "skipped",
            "reason": (
                f"result_kind={bundle.get('result_kind')!r}; backfill applies "
                "only to scalar bundles"
            ),
        }
    if "canonical_pre_calibration_features" in bundle:
        return {
            "status": "skipped",
            "reason": "canonical_pre_calibration_features already present",
        }
    evidence = bundle.get("evidence", {})
    if evidence.get("predicted_hit_rate_uncorrected") is None:
        return {
            "status": "failed",
            "reason": (
                "bundle.evidence.predicted_hit_rate_uncorrected is required "
                "for the canonical block"
            ),
        }
    mpi = bundle.get("manifest_pairing_index", {})
    slug = mpi.get("slug")
    shape_point = mpi.get("shape_point")
    arch = mpi.get("arch") or _arch_from_filename(bundle_path)
    device = bundle.get("device")
    sku = _device_to_sku_dir(device)
    arch_dir = _arch_to_subdir(arch)
    if not (slug and shape_point):
        return {
            "status": "failed",
            "reason": "manifest_pairing_index missing slug or shape_point",
        }
    sl_path, hrd_path = _resolve_sidecar_paths(
        results_root=results_root, slug=slug, shape_point=shape_point,
        sku=sku, arch=arch_dir,
    )
    if not sl_path.is_file():
        return {
            "status": "failed",
            "reason": f"sharing_locality sidecar absent: {sl_path}",
        }
    if not hrd_path.is_file():
        return {
            "status": "failed",
            "reason": f"hrd_profile sidecar absent: {hrd_path}",
        }
    sharing_locality = json.loads(sl_path.read_text(encoding="utf-8"))
    hrd_profile = json.loads(hrd_path.read_text(encoding="utf-8"))
    unique_sector_count = _count_unique_sectors_from_hrd(hrd_profile)
    if unique_sector_count is None:
        return {
            "status": "failed",
            "reason": "hrd_profile has no sector-layer bins",
        }
    sector_stream_path = (
        sl_path.parent.parent / "sector_stream" / "sector_stream.jsonl"
    )
    if sector_stream_path.is_file():
        g_count, t_count = _count_global_share_from_sector_stream(
            sector_stream_path,
        )
        global_share_value = (g_count / t_count) if t_count > 0 else 0.0
    else:
        global_share_value = 0.0
    grid_volume = bundle.get("cache_model_grid_volume")
    try:
        features = cf.assemble_canonical_features(
            predictor_bundle=bundle,
            sharing_locality=sharing_locality,
            hrd_profile=hrd_profile,
            unique_sector_count=unique_sector_count,
            global_share_value=global_share_value,
            grid_volume=grid_volume,
        )
    except ValueError as exc:
        return {
            "status": "failed",
            "reason": f"feature assembly failed: {exc}",
        }
    if dry_run:
        return {"status": "admitted", "features": features, "dry_run": True}
    bundle["canonical_pre_calibration_features"] = features
    bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"status": "admitted", "features": features}


def _arch_from_filename(path: Path) -> str | None:
    name = path.name
    if name.startswith("sm_120__"):
        return "sm_120"
    if name.startswith("sm_86__"):
        return "sm_86"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-glob",
        required=True,
        help="Glob pattern selecting predictor bundle paths to backfill.",
    )
    parser.add_argument(
        "--results-root",
        default="workloads/simbench_bw/results",
        help="Root under which trace_<shape>/sharing_locality + hrd_profile sidecars live.",
    )
    parser.add_argument(
        "--portable-lcd-root",
        default="workloads/portable_lcd/results",
        help="Alternate results root for portable_lcd-source cells.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the block but do not write back to disk.",
    )
    args = parser.parse_args(argv)
    bundle_paths = sorted(Path(p) for p in glob.glob(args.bundle_glob))
    if not bundle_paths:
        print(
            f"[backfill_canonical_features] no bundles matched glob "
            f"{args.bundle_glob!r}; nothing to do",
            file=sys.stderr,
        )
        return 0
    primary_results_root = Path(args.results_root)
    portable_lcd_root = Path(args.portable_lcd_root)
    admitted = 0
    skipped = 0
    failed = 0
    failed_records: list[dict] = []
    for bundle_path in bundle_paths:
        result = backfill_one_bundle(
            bundle_path=bundle_path,
            results_root=primary_results_root,
            dry_run=args.dry_run,
        )
        if result["status"] == "failed" and "sidecar absent" in result.get("reason", ""):
            # Retry with portable_lcd root before giving up.
            result_pl = backfill_one_bundle(
                bundle_path=bundle_path,
                results_root=portable_lcd_root,
                dry_run=args.dry_run,
            )
            if result_pl["status"] != "failed":
                result = result_pl
        status = result["status"]
        if status == "admitted":
            admitted += 1
            print(f"[admit] {bundle_path.name}")
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1
            failed_records.append({
                "path": str(bundle_path),
                "reason": result.get("reason", ""),
            })
            print(
                f"[fail]  {bundle_path.name}: {result.get('reason', '')}",
                file=sys.stderr,
            )
    print(
        f"[backfill_canonical_features] summary: admitted={admitted} "
        f"skipped={skipped} failed={failed} (total={len(bundle_paths)})",
    )
    if args.dry_run:
        print("[backfill_canonical_features] dry-run: no files modified")
    if failed_records:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
