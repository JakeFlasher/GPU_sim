"""L1-miss-filtered L2 sector-hit-rate evaluation over the committed sm_120 cohort.

This is the production caller that closes the loop the L1->miss-stream->L2 filter
(``l1_to_l2_miss_filter.filtered_replay_from_trace``) was built for: for every
admitted sm_120 cell that carries a committed NVBit trace, it

  1. resolves the cell's sm_120 binary and generates the per-PC ``.ca`` / ``.cg``
     cache-op eligibility manifest from LIVE cuobjdump SASS
     (``l1_cache_op_classifier.classify_binary``; cuobjdump is a CUDA binary
     utility, not an NCU collection surface);
  2. walks the per-threadblock protos into trace records carrying the real
     per-access ``sm_id`` (``l1_production_helper._walk_threadblock_protos``);
  3. derives the REAL per-CTA SM partition from that ``sm_id`` and replays ONLY
     the L1-miss stream through the L2 sector model (``real_sm_partition`` /
     ``l1_miss_filtered``); and
  4. compares the predicted L2 sector hit ratio to the admitted NCU
     ``lts__t_sector_hit_rate`` label.

It emits a per-cell comparison + cohort aggregates as a committed artifact
(``data/local_labels/sm120_l1_filtered_l2_comparison.json``). A cell whose binary
cannot be resolved, whose cuobjdump section does not match the traced kernel, or
which is multi-kernel (per-PC ambiguity across kernels) is recorded as a TYPED
SKIP -- never silently dropped -- so the evaluated set is auditable.

This is trace-accounting / input-normalization over already-collected traces, NOT
a GPU simulator: it reads committed traces + static SASS and models cache
residency to decide which accesses reach L2. It invokes no NCU, no privileged
CUPTI, no driver-root surface.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (
    _HERE,
    _REPO_ROOT / "tools" / "cache_predictor",
    _REPO_ROOT / "workloads" / "simbench_bw",
    _REPO_ROOT / "tools" / "nvbit_tracer" / "traces_enhanced",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cohort as cohort_mod  # noqa: E402
import l1_cache_op_classifier as classifier  # noqa: E402
import l1_production_helper as lph  # noqa: E402
import l1_to_l2_miss_filter as miss_filter  # noqa: E402
import sector_cache_replay as scr  # noqa: E402

ARCH = "sm_120"
ARCH_TOKEN = "sm_120"
# RTX 5060 Laptop L2 capacity (32 MiB), resolved through the shared device
# surface (NOT a replay-local literal). A fully-associative LRU L2 at this
# capacity is the principled upper-bound config: it gives the model every
# chance to hit, so a residual under/over-prediction isolates the
# trace-fidelity / cache-op-classification effect from cache-geometry effects.
L2_CAPACITY_BYTES = cohort_mod.RTX_5060_LAPTOP_L2_BYTES
L1_CAPACITY_BYTES = 128 * 1024  # per-SM L1 data cache (Blackwell sm_120)
L2_ASSOCIATIVITY = 16
COMPARISON_ARTIFACT = _REPO_ROOT / "data/local_labels/sm120_l1_filtered_l2_comparison.json"

SCHEMA_VERSION = 1
NAMED_TARGET = "lts_sector_hit_rate"
NAMED_NCU_TOKEN = "lts__t_sector_hit_rate.pct"

_KERNEL_INSTANCE_SUFFIX = re.compile(r"___\d+$")


@dataclass(frozen=True)
class CellEvaluation:
    slug: str
    shape_point: str
    family_bucket: str
    label: float
    predicted: float | None
    abs_error: float | None
    signed_error: float | None
    evaluated: bool
    skip_reason: str | None
    detail: dict


def _binary_for(slug: str, workload: str) -> Path | None:
    """Resolve a cell's sm_120 binary. Single-binary slugs only; the simbench
    L2 microbench slugs (perturbation_build variants) return None -> typed skip."""
    candidates = (
        _REPO_ROOT / "workloads" / workload / slug / f"{slug}_run",
        _REPO_ROOT / "workloads/simbench_bw/l1" / slug / f"{slug}_run",
        _REPO_ROOT / "workloads/simbench_bw/l2" / slug / f"{slug}_run",
    )
    for c in candidates:
        if c.is_file():
            return c
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _single_kernel_record(trace_pb2, trace_dir: Path):
    """Return the sole validated kernel, or None if absent/ambiguous."""
    trace = lph._load_validated_trace(
        trace_pb2,
        trace_dir / "dynamic_trace.pb",
    )
    kernels = []
    for did in sorted(trace.gpu_device.keys()):
        dev = trace.gpu_device[did]
        for sid in sorted(dev.streams.keys()):
            kernels.extend(dev.streams[sid].kernels)
    if len(kernels) != 1:
        return None
    return kernels[0]


def _skip(cell, reason: str, detail: dict) -> CellEvaluation:
    return CellEvaluation(
        slug=cell.slug, shape_point=cell.shape_point,
        family_bucket=cell.family_bucket, label=cell.label,
        predicted=None, abs_error=None, signed_error=None,
        evaluated=False, skip_reason=reason, detail=detail,
    )


def evaluate_cell(cell) -> CellEvaluation:
    """Run the L1-miss-filtered L2 replay for one cohort cell and compare to its
    admitted NCU L2 label. Returns a typed-skip evaluation on any fail-closed
    condition (binary unresolved, manifest unmatched, multi-kernel, no records)."""
    binary = _binary_for(cell.slug, cell.workload)
    if binary is None:
        return _skip(cell, "binary_unresolved", {"workload": cell.workload})

    # Multi-kernel cells produce per-kernel-relative PCs that collide in a single
    # merged manifest lookup; fail closed rather than mis-join.
    kdirs = [d for d in (cell.trace_dir / "threadblocks").rglob("kernel_*") if d.is_dir()]
    if len({d.name for d in kdirs}) > 1:
        return _skip(cell, "multi_kernel_pc_ambiguity",
                     {"kernel_dirs": sorted({d.name for d in kdirs})})

    trace_pb2, threadblock_pb2, _instr_pb2 = lph._load_proto_bindings()
    kernel = _single_kernel_record(trace_pb2, cell.trace_dir)
    if kernel is None:
        return _skip(cell, "not_single_kernel", {})

    grid_x, grid_y, grid_z = kernel.grid_dim.x, kernel.grid_dim.y, kernel.grid_dim.z
    block_x, block_y, block_z = kernel.block_dim.x, kernel.block_dim.y, kernel.block_dim.z
    grid_dim = grid_x * grid_y * grid_z
    block_dim = block_x * block_y * block_z
    if grid_dim <= 0 or block_dim <= 0:
        return _skip(cell, "degenerate_launch_dims",
                     {"grid_dim": grid_dim, "block_dim": block_dim})

    # The NVBit-recorded kernel name carries a ``___N`` instance suffix; the
    # cuobjdump SASS section uses the base symbol. Strip it for the join.
    kname = _KERNEL_INSTANCE_SUFFIX.sub("", kernel.name)
    manifest = classifier.classify_binary(
        binary, kernel_unmangled_name=kname, arch_token=ARCH_TOKEN,
    )
    status = manifest.get("sass_provenance_status")
    if status != "matched":
        return _skip(cell, f"manifest_{status}",
                     {"kernel_name": kname, "binary": str(binary.relative_to(_REPO_ROOT))})

    itc = lph._load_import_trace_contract()
    tb_paths = sorted((cell.trace_dir / "threadblocks").rglob("*.pb"))
    try:
        records = lph._walk_threadblock_protos(
            threadblock_pb2, tb_paths, itc,
            grid_x=grid_x, grid_y=grid_y, grid_dim=grid_dim,
        )
    except Exception as exc:  # noqa: BLE001  -- fail-closed typed skip
        return _skip(cell, "threadblock_walk_failed",
                     {"error": f"{type(exc).__name__}: {exc}"})

    sm_ids = {r.sm_id for r in records if r.sm_id is not None}
    if not sm_ids:
        return _skip(cell, "no_sm_id_metadata", {})
    sm_count = max(sm_ids) + 1

    l2_config = scr.CacheReplayConfig(
        L2_CAPACITY_BYTES, associativity=L2_ASSOCIATIVITY,
        replacement_policy="LRU", num_slices=1, slice_hash_id="line_mod",
    )
    try:
        result = miss_filter.filtered_replay_from_trace(
            records, manifest,
            sm_count=sm_count, grid_dim=grid_dim, block_dim=block_dim,
            l1_capacity_bytes=L1_CAPACITY_BYTES, l2_config=l2_config,
            headline_live=True, admit_round_robin_fallback=False,
            require_data_width=True,
        )
    except Exception as exc:  # noqa: BLE001  -- fail-closed typed skip
        return _skip(cell, "filtered_replay_failed",
                     {"error": f"{type(exc).__name__}: {exc}"})

    predicted = float(result.l2_predicted_hit_ratio)
    signed = predicted - cell.label

    # L1 sector-hit-rate leg (l1tex_sector_hit_rate): the per-SM L1's modeled hit
    # rate vs the live NCU L1TEX label (Cell.l1_label). When the classifier is
    # inert (zero L1-eligible accesses) the modeled hit_rate is a vacuous 0.0, so
    # we DO NOT treat it as a prediction -- we record a typed-unavailable reason +
    # the live label + the accounting, so the L1 target closes by an honest
    # classifier-inert falsification rather than a fabricated 0.0.
    l1_total_eligible = result.l1_result.total_eligible_accesses
    if cell.l1_label is None:
        l1_leg = {"l1_label": None, "l1_evaluable": False,
                  "l1_unavailable_reason": "no_l1_label",
                  "l1_predicted": None, "l1_abs_error": None,
                  "l1_total_eligible_accesses": l1_total_eligible}
    elif l1_total_eligible > 0:
        l1_pred = float(result.l1_result.hit_rate)
        l1_leg = {"l1_label": cell.l1_label, "l1_evaluable": True,
                  "l1_unavailable_reason": None,
                  "l1_predicted": l1_pred,
                  "l1_abs_error": abs(l1_pred - cell.l1_label),
                  "l1_total_eligible_accesses": l1_total_eligible}
    else:
        l1_leg = {"l1_label": cell.l1_label, "l1_evaluable": False,
                  "l1_unavailable_reason": "l1_classifier_inert_no_l1_eligible_denominator",
                  "l1_predicted": None, "l1_abs_error": None,
                  "l1_total_eligible_accesses": l1_total_eligible}

    return CellEvaluation(
        slug=cell.slug, shape_point=cell.shape_point,
        family_bucket=cell.family_bucket, label=cell.label,
        predicted=predicted, abs_error=abs(signed), signed_error=signed,
        evaluated=True, skip_reason=None,
        detail={
            "partition_provenance": result.partition_provenance,
            "l2_input_provenance": result.l2_input_provenance,
            "sm_count": sm_count,
            "n_records": len(records),
            "l1_hits_absorbed": result.l1_hits_absorbed,
            "l1_eligible_misses": result.l1_eligible_misses,
            "l1_bypass_accesses": result.l1_bypass_accesses,
            "l2_input_count": result.l2_input_count,
            "binary_sha256": _sha256(binary),
            "binary": str(binary.relative_to(_REPO_ROOT)),
            **l1_leg,
        },
    )


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _source_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return "0" * 40


def build_comparison() -> dict:
    cohort = cohort_mod.load_cohort()
    evals = [evaluate_cell(c) for c in cohort]
    evaluated = [e for e in evals if e.evaluated]
    skipped = [e for e in evals if not e.evaluated]

    abs_errors = [e.abs_error for e in evaluated if e.abs_error is not None]
    signed = [e.signed_error for e in evaluated if e.signed_error is not None]
    over = [e for e in evaluated if e.signed_error is not None and e.signed_error > 0.1]
    under = [e for e in evaluated if e.signed_error is not None and e.signed_error < -0.1]

    # L1-inertness accounting: the dominant sm_120 finding is that the
    # sm_86-derived cache-op classifier maps ZERO loads to L1-eligible (.ca) on
    # this cohort, so the per-SM L1 filter (the task that this evaluation
    # exercises) absorbs nothing -- it degenerates to either full-stream-to-L2
    # (bypass) or an empty stream (all PCs dropped as unmodeled).
    total_elig = sum(e.detail.get("l1_eligible_misses", 0) for e in evaluated)
    total_hits = sum(e.detail.get("l1_hits_absorbed", 0) for e in evaluated)
    total_bypass = sum(e.detail.get("l1_bypass_accesses", 0) for e in evaluated)
    zero_l2_input = [e for e in evaluated if e.detail.get("l2_input_count", 0) == 0]

    # L1 sector-hit-rate leg aggregates (the l1tex_sector_hit_rate target).
    l1_labelled = [e for e in evaluated if e.detail.get("l1_label") is not None]
    l1_evaluable = [e for e in l1_labelled if e.detail.get("l1_evaluable")]
    l1_abs_errors = [e.detail["l1_abs_error"] for e in l1_evaluable if e.detail.get("l1_abs_error") is not None]

    skip_reasons: dict[str, int] = {}
    for e in skipped:
        reason = e.skip_reason or "unknown"
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    cells_out = []
    for e in evals:
        cells_out.append({
            "slug": e.slug, "shape_point": e.shape_point,
            "family_bucket": e.family_bucket, "label": e.label,
            "predicted": e.predicted, "abs_error": e.abs_error,
            "signed_error": e.signed_error, "evaluated": e.evaluated,
            "skip_reason": e.skip_reason, "detail": e.detail,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "arch": ARCH,
        "named_target": NAMED_TARGET,
        "named_ncu_token": NAMED_NCU_TOKEN,
        "l2_input_provenance": miss_filter.L2_INPUT_L1_FILTERED,
        "replay_config": {
            "l2_capacity_bytes": L2_CAPACITY_BYTES,
            "l2_associativity": L2_ASSOCIATIVITY,
            "l2_replacement_policy": "LRU",
            "l1_capacity_bytes": L1_CAPACITY_BYTES,
            "warmup_launches": 0,
            "cache_op_eligibility_source": "live_cuobjdump_sm120",
        },
        "cohort_total": len(cohort),
        "n_evaluated": len(evaluated),
        "n_skipped": len(skipped),
        "skip_reasons": skip_reasons,
        "aggregates": {
            "median_abs_error": _median(abs_errors),
            "mean_signed_error": (sum(signed) / len(signed)) if signed else None,
            "cells_over_predicted_by_0p1": len(over),
            "cells_under_predicted_by_0p1": len(under),
            "max_abs_error": max(abs_errors) if abs_errors else None,
            "min_abs_error": min(abs_errors) if abs_errors else None,
            "total_l1_eligible_misses": total_elig,
            "total_l1_hits_absorbed": total_hits,
            "total_l1_bypass_accesses": total_bypass,
            "l1_filter_inert": (total_elig == 0 and total_hits == 0),
            "cells_with_zero_l2_input": len(zero_l2_input),
            "l1_label_coverage": len(l1_labelled),
            "l1_evaluated": len(l1_evaluable),
            "l1_unavailable": len(l1_labelled) - len(l1_evaluable),
            "l1_median_abs_error": _median(l1_abs_errors),
        },
        "cells": cells_out,
        "source_commit": _source_commit(),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=COMPARISON_ARTIFACT)
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args(argv)

    doc = build_comparison()
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    agg = doc["aggregates"]
    print(
        f"wrote {args.out} | evaluated={doc['n_evaluated']}/{doc['cohort_total']} "
        f"skipped={doc['n_skipped']} {doc['skip_reasons']}"
    )
    print(
        f"median_abs_error={agg['median_abs_error']} "
        f"mean_signed_error={agg['mean_signed_error']} "
        f"over0.1={agg['cells_over_predicted_by_0p1']} "
        f"under0.1={agg['cells_under_predicted_by_0p1']}"
    )
    if args.print:
        for c in doc["cells"]:
            if c["evaluated"]:
                print(f"  {c['slug'][:28]:28s} {c['shape_point'][:16]:16s} "
                      f"label={c['label']:.3f} pred={c['predicted']:.3f} "
                      f"abs={c['abs_error']:.3f}")
            else:
                print(f"  {c['slug'][:28]:28s} {c['shape_point'][:16]:16s} SKIP={c['skip_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
