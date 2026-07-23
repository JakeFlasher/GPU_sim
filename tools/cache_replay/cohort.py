"""Cohort loader for the sm_120 cache-only trace-replay predictor.

Enumerates the existing sm_120 NCU-labeled cells, resolves each cell's NVBit
trace directory, the ground-truth L2 (and diagnostic L1) sector hit-rate label,
and the family bucket, then decodes each cell's ordered access stream exactly
once (cached on disk so re-replays under many candidate configs stay cheap).

This module is sm_120-develop-only: it admits only ``arch == "sm_120"`` cells.
The historical sm_86 corpus is never consumed and is silently absent here.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sector_cache_replay as scr  # type: ignore[import-not-found]  # noqa: E402
import device_resolver  # type: ignore[import-not-found]  # noqa: E402

TARGET_ARCH = "sm_120"
SM120_SKU = "rtx_5060_laptop"
WORKLOADS = ("portable_lcd", "simbench_bw")
FAMILY_BUCKETS = ("tensor_compute", "activation_norm", "memory_stress", "other")

_LABEL_ADMITTED = _REPO_ROOT / "data/local_labels/label_admitted.json"
_CALIBRATION_HEAD_V5 = _REPO_ROOT / "data/local_labels/calibration_head_v5.json"
_STREAM_CACHE = _REPO_ROOT / ".cache_replay_streams"

# Knowable device L2 capacity for the RTX 5060 Laptop (sm_120, Blackwell):
# resolved through the shared device-property / cache-model surface
# (run_suite static card -> cache_model_spec.resolve_cache_model_spec), NOT a
# cache-replay-local literal. The resolver returns 33554432 bytes (32 MiB) with
# capacity_source="static_device_card"; the provenance is carried into every
# emitted artifact. The expected value is asserted below as a guard against
# silent resolver drift.
RTX_5060_LAPTOP_L2_BYTES, L2_CAPACITY_PROVENANCE = device_resolver.resolve_sm120_l2()
_EXPECTED_L2_BYTES = 32 * 1024 * 1024  # 33554432 (documented; cross-check only)
assert RTX_5060_LAPTOP_L2_BYTES == _EXPECTED_L2_BYTES, (
    f"resolver L2 capacity {RTX_5060_LAPTOP_L2_BYTES} != expected "
    f"{_EXPECTED_L2_BYTES}"
)


@dataclass(frozen=True)
class Cell:
    slug: str
    shape_point: str
    family_bucket: str
    label: float            # L2 sector hit ratio, fraction in [0, 1]
    l1_label: float | None  # L1TEX sector hit ratio, fraction or None
    workload: str
    trace_dir: Path

    @property
    def cell_id(self) -> str:
        return f"{self.slug}/{self.shape_point}"


def _resolve_trace_dir(slug: str, shape_point: str) -> tuple[str, Path] | None:
    for workload in WORKLOADS:
        d = (
            _REPO_ROOT
            / "workloads" / workload / "results" / slug / SM120_SKU
            / TARGET_ARCH / f"trace_{shape_point}"
        )
        tb = d / "threadblocks"
        if tb.is_dir() and any(tb.rglob("*.pb")):
            return workload, d
    return None


def _load_family_buckets() -> dict[tuple[str, str], str]:
    data = json.loads(_CALIBRATION_HEAD_V5.read_text())
    admitted = data.get("cohort_scope", {}).get("admitted", [])
    out: dict[tuple[str, str], str] = {}
    for row in admitted:
        if row.get("arch") == TARGET_ARCH:
            out[(row["slug"], row["shape_point"])] = row.get("family_bucket", "other")
    return out


def load_cohort() -> list[Cell]:
    """Return the admitted sm_120 cells with labels, family buckets, and
    resolved trace directories. Cells whose trace directory is absent on disk
    are silently skipped (predictor input absent)."""
    labels = json.loads(_LABEL_ADMITTED.read_text())
    cells_raw = labels["cells"]
    family = _load_family_buckets()
    cohort: list[Cell] = []
    for c in cells_raw:
        if c.get("arch") != TARGET_ARCH:
            continue
        slug = c["slug"]
        shape_point = c["shape_point"]
        resolved = _resolve_trace_dir(slug, shape_point)
        if resolved is None:
            continue
        workload, trace_dir = resolved
        l1 = None
        if c.get("l1tex_label_status") == "mean_present":
            l1 = float(c["cache_hit_rate_l1tex_sector_pct"]) / 100.0
        cohort.append(
            Cell(
                slug=slug,
                shape_point=shape_point,
                family_bucket=family.get((slug, shape_point), "other"),
                label=float(c["cache_hit_rate_l2_sector_pct"]) / 100.0,
                l1_label=l1,
                workload=workload,
                trace_dir=trace_dir,
            )
        )
    cohort.sort(key=lambda x: x.cell_id)
    return cohort


def _trace_fingerprint(trace_dir: Path) -> str:
    """Short content-identity digest of a cell's decode-input trace files.

    The stream cache MUST invalidate when the underlying NVBit trace is
    regenerated or swapped (same slug/shape, different bytes); keying the cache by
    cell id alone would silently replay stale ``line_ids``/``sector_offsets`` for
    the new trace. We hash a STRUCTURAL manifest -- the sorted
    ``(relative_path, size)`` of every ``*.pb`` file the decode consumes under
    ``trace_dir`` -- rather than the full ~35 MB x N file contents. This keeps the
    digest cheap (stat-only; no re-read of the trace on every cache probe, which
    would defeat the cache's purpose) and portable (no mtime, so it survives a git
    checkout / scp). A regenerated or swapped trace changes the ``*.pb`` set or
    their sizes and therefore the digest. Returns the empty-input sentinel
    ``"empty"`` when no ``*.pb`` files exist."""
    h = hashlib.sha256()
    pbs = sorted(trace_dir.rglob("*.pb"))
    if not pbs:
        return "empty"
    for p in pbs:
        rel = p.relative_to(trace_dir).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(str(p.stat().st_size).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def cohort_fingerprint(cohort: list[Cell]) -> str:
    """A single digest over the whole cohort's per-cell trace fingerprints.

    Used to namespace the higher-level prediction cache so that a change to ANY
    cell's trace invalidates the memoized config->prediction map (coarse but
    correct: a stale per-cell prediction under any cached config would otherwise
    survive a trace regeneration)."""
    h = hashlib.sha256()
    for cell in sorted(cohort, key=lambda c: c.cell_id):
        h.update(cell.cell_id.encode("utf-8"))
        h.update(b":")
        h.update(_trace_fingerprint(cell.trace_dir).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def _cache_path(cell: Cell) -> Path:
    safe = cell.cell_id.replace("/", "__")
    # Bind the cache file to the trace's content identity so a regenerated /
    # swapped trace (same slug/shape) is a cache MISS, not a stale hit.
    fp = _trace_fingerprint(cell.trace_dir)
    return _STREAM_CACHE / f"{safe}__{fp}.npz"


def decode_stream(cell: Cell, *, use_cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Decode (or load from cache) a cell's ordered access stream.

    Returns ``(line_ids: int64[N], sector_offsets: int8[N])``. The decode is the
    expensive part (proportional to trace size); the cached arrays let the grid
    search replay many configs over the same stream cheaply.
    """
    cache_path = _cache_path(cell)
    if use_cache and cache_path.exists():
        data = np.load(cache_path)
        return data["line_ids"], data["sector_offsets"]
    stream = scr.decode_cell_access_stream(cell.trace_dir)
    line_ids = np.asarray(stream.line_ids, dtype=np.int64)
    sector_offsets = np.asarray(stream.sector_offsets, dtype=np.int8)
    if use_cache:
        _STREAM_CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, line_ids=line_ids, sector_offsets=sector_offsets)
    return line_ids, sector_offsets


def iter_cohort_streams(
    cohort: list[Cell] | None = None, *, use_cache: bool = True,
) -> Iterator[tuple[Cell, np.ndarray, np.ndarray]]:
    """Yield ``(cell, line_ids, sector_offsets)`` for each admitted cell, one at
    a time (bounded memory: one cell's stream resident at a time)."""
    if cohort is None:
        cohort = load_cohort()
    for cell in cohort:
        line_ids, sector_offsets = decode_stream(cell, use_cache=use_cache)
        yield cell, line_ids, sector_offsets


def _main(argv: list[str] | None = None) -> int:
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Cohort staging audit + decode")
    parser.add_argument("--decode", action="store_true",
                        help="decode + cache every cell's access stream")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    cohort = load_cohort()
    print(f"admitted sm_120 cells: {len(cohort)}")
    from collections import Counter
    print("family buckets:", dict(Counter(c.family_bucket for c in cohort)))
    if args.decode:
        for cell in cohort:
            t0 = time.time()
            line_ids, _ = decode_stream(cell, use_cache=not args.no_cache)
            print(
                f"  {cell.cell_id:60s} {cell.family_bucket:14s} "
                f"label={cell.label:.4f} n={len(line_ids):>9d} "
                f"{time.time()-t0:6.2f}s"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
