#!/usr/bin/env python3
"""Per-admitted-cell L1 predictions from the warp-pool walker (the
NEW analytical head), landing BESIDE the existing predicted-L1
artifact — never clobbering it.

The walk per cell: per-WARP access streams (the adapter's per-warp
emission, which delegates record semantics to the verified per-block
builder), blocks assigned to SMs by the same round-robin policy the
existing head uses, and per-SM warp-pool walks with two-layer
sector/line accounting, finite MSHRs, and NCU-style warmup — every
geometry input CARD-SOURCED through the fail-closed resolvers
(measured capacity at the cell's carveout point, measured latencies,
documented-default MSHR count, LRU / fully-associative as documented
approximations for the card's typed entries).

The head has ZERO parameters fit to labels: the configuration block
below is frozen structural/architectural data, its sha256 is the
config digest the sealed-holdout governance binds, and the
``label_fitted_parameters`` list is empty by construction (the
re-score consumer rejects a config that carries any).

Occupancy: co-resident warps per SM = min(warps_per_block x
max-blocks-per-SM, warp limit) from the architectural maxima
(16 blocks / 48 warps per SM on this device family); shared-memory
occupancy reduction is NOT modeled (documented approximation in the
artifact evidence).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Callable, TypeVar

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO_ROOT / "tools" / "cache_predictor"))
sys.path.insert(0, str(_REPO_ROOT / "tools" / "local_labels"))

import l1_production_helper as lph  # noqa: E402
import l1_trace_adapter as lta  # noqa: E402
from l1_walker import block_to_sm_id  # noqa: E402
from l1_warp_pool_walker import (  # noqa: E402
    co_resident_warp_limit,
    resolve_walk_geometry,
    walk_warp_pool,
)
from compute_predicted_l1_per_admitted_cell import (  # noqa: E402
    _ArtifactIdentityFieldMissing,
    _CARVEOUT_PERCENT_BY_CELL,
    _extract_identity_tuple,
    _load_cohort_priority_keys,
    _manifest_path_for,
    _snapshot_classifier_manifest,
    _snapshot_trace_inputs,
    _trace_dir_for,
)
from corpus_cell_identity import (  # noqa: E402
    cell_variant_id,
    is_pending_refit_factorial,
)
from tools.local_labels import immutable_artifact as immutable  # noqa: E402


_OLD_ARTIFACT_PATH = (
    _REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "predicted_l1_per_admitted_cell.json"
)
DEFAULT_OUT = (
    _REPO_ROOT / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "warp_pool_l1_per_admitted_cell.json"
)
DEFAULT_SCHEMA = (
    _HERE / "warp_pool_l1_per_admitted_cell.schema.json"
)

_KERNEL_ID = 1
_EMPTY_SHA = hashlib.sha256(b"").hexdigest()
_SnapshotResult = TypeVar("_SnapshotResult")

# FROZEN structural configuration. Nothing here was fit to any
# label; the sha256 of this block is the config digest the
# sealed-holdout governance binds (a retune after the holdout touch
# is rejected by the digest-bound ledger guard).
WARP_POOL_HEAD_CONFIG: dict = {
    "model": "warp_pool_two_layer_finite_mshr",
    "scheduling_mode": "warp_pool",
    "sector_accounting": "two_layer",
    "cta_to_sm_schedule_policy": "round_robin_homogeneous",
    "sm_count": 84,
    # Architectural per-SM maxima for this device family (CUDA
    # occupancy table facts, not tuned values).
    "occupancy_warp_limit_per_sm": 48,
    "max_blocks_per_sm": 16,
    # COLD-START L1 per launch: L1TEX is invalidated at kernel
    # launch boundaries on this device family, so the capture
    # harness's warm launches warm L2/DRAM state but NOT L1 —
    # replaying warm launches into the L1 walk (the existing
    # head's approximation) over-predicts hit rates on cells
    # whose working set fits the cache. Selected on
    # calibration/validation evidence between the two structural
    # candidates {0, 5}; the selection is recorded in the
    # re-score artifact's tuning trail with both pools' scores.
    "warmup_launches": 0,
    "card_path": "data/pchase/sm_86_static_card.json",
    "carveout_percent_by_cell": {
        f"{slug}/{shape}": pct
        for (slug, shape), pct in sorted(
            _CARVEOUT_PERCENT_BY_CELL.items()
        )
    },
    "shared_memory_occupancy_reduction_modeled": False,
    # ZERO label-fitted parameters — the re-score consumer
    # rejects a config whose list is non-empty.
    "label_fitted_parameters": [],
}


def config_digest() -> str:
    return hashlib.sha256(
        json.dumps(WARP_POOL_HEAD_CONFIG, sort_keys=True,
                   separators=(",", ":")).encode()
    ).hexdigest()


class WarpPoolArtifactPathError(Exception):
    """The new head must never write the existing predicted-L1
    artifact path."""


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _typed_entry(
    *,
    slug: str,
    shape_point: str,
    identity: dict,
    typed_reason: str,
    diagnostic: str,
    input_digests: dict | None = None,
    extra: dict | None = None,
) -> dict:
    entry = {
        "slug": slug,
        "shape_point": shape_point,
        **identity,
        "result_kind": "typed_unavailable",
        "cache_layer": "L1TEX",
        "predicted_hit_rate": None,
        "typed_reason": typed_reason,
        "evidence": {"diagnostic": diagnostic},
        "input_digests": input_digests or {
            "classifier_manifest_sha256": _EMPTY_SHA,
            "dynamic_trace_pb_sha256": _EMPTY_SHA,
            "threadblock_proto_set_sha256": _EMPTY_SHA,
        },
    }
    if extra:
        entry.update(extra)
    return entry


def _with_typed_resource_failure(
    label: str,
    action: Callable[[], _SnapshotResult],
) -> _SnapshotResult:
    """Normalize descriptor exhaustion across every snapshot phase."""
    try:
        return action()
    except immutable.ImmutableArtifactResourceError:
        raise
    except BaseException as exc:
        resource_cause = immutable._descriptor_resource_cause(exc)
        if resource_cause is None:
            raise
        raise immutable.ImmutableArtifactResourceError(
            f"{immutable.VERIFY_RESOURCE_BOUND}: warp-pool {label}: "
            f"{resource_cause}"
        ) from exc


def _sha256_of_retained_threadblocks(
    threadblocks: tuple[tuple[Path, bytes], ...],
) -> str:
    """Digest the exact retained threadblock path/byte generation."""
    digest = hashlib.sha256()
    for path, payload in threadblocks:
        relative_path = path.relative_to(_REPO_ROOT)
        digest.update(
            f"{relative_path}\t{immutable.sha256_bytes(payload)}\n".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def compute_for_cell(
    *, slug: str, shape_point: str, cell_row: dict,
    walk_overrides: dict | None = None,
) -> dict:
    """One cell's warp-pool L1 prediction (or a typed row).

    ``walk_overrides`` is an ADDITIVE ablation hook: a mapping that
    may override ``mode`` / ``sector_accounting`` / ``mshr_count`` in
    the per-SM walk. It defaults to ``None`` -> the frozen
    ``WARP_POOL_HEAD_CONFIG``, so the committed head artifact is
    byte-identical; the evidence harness passes overrides to run the
    no-interleaving / no-MSHR / no-HRD-second-layer ablations on the
    same trace inputs."""
    identity = _extract_identity_tuple(
        cell_row, slug=slug, shape_point=shape_point,
    )

    manifest_path = _manifest_path_for(slug, shape_point)
    if not manifest_path.is_file():
        return _typed_entry(
            slug=slug, shape_point=shape_point, identity=identity,
            typed_reason="predictor_l1_sass_provenance_missing",
            diagnostic=(
                f"classifier manifest not found at "
                f"data/local_labels/cache_op_classifier_manifests/"
                f"sm_86/{slug}__{shape_point}.json"
            ),
        )
    manifest_payload, manifest = _with_typed_resource_failure(
        "classifier-manifest snapshot",
        lambda: _snapshot_classifier_manifest(manifest_path),
    )
    manifest_sha256 = immutable.sha256_bytes(manifest_payload)
    status = manifest.get("sass_provenance_status")
    if status not in {
        "matched",
        "matched_no_load_class_instructions",
    }:
        return _typed_entry(
            slug=slug, shape_point=shape_point, identity=identity,
            typed_reason="predictor_l1_sass_provenance_missing",
            diagnostic=(
                f"manifest sass_provenance_status={status!r} is "
                f"not matched"
            ),
            input_digests={
                "classifier_manifest_sha256": manifest_sha256,
                "dynamic_trace_pb_sha256": _EMPTY_SHA,
                "threadblock_proto_set_sha256": _EMPTY_SHA,
            },
        )

    trace_dir = _trace_dir_for(slug, shape_point)
    trace_pb = trace_dir / "dynamic_trace.pb"
    if not trace_dir.is_dir():
        return _typed_entry(
            slug=slug, shape_point=shape_point, identity=identity,
            typed_reason="predictor_l1_sass_provenance_missing",
            diagnostic=f"dynamic_trace.pb missing at {trace_pb}",
            input_digests={
                "classifier_manifest_sha256": manifest_sha256,
                "dynamic_trace_pb_sha256": _EMPTY_SHA,
                "threadblock_proto_set_sha256": _EMPTY_SHA,
            },
        )
    trace_inputs = _with_typed_resource_failure(
        "trace-tree snapshot",
        lambda: _snapshot_trace_inputs(trace_dir),
    )
    if trace_inputs is None:
        return _typed_entry(
            slug=slug, shape_point=shape_point, identity=identity,
            typed_reason="predictor_l1_sass_provenance_missing",
            diagnostic=f"dynamic_trace.pb missing at {trace_pb}",
            input_digests={
                "classifier_manifest_sha256": manifest_sha256,
                "dynamic_trace_pb_sha256": _EMPTY_SHA,
                "threadblock_proto_set_sha256": _EMPTY_SHA,
            },
        )
    tb_paths = [path for path, _payload in trace_inputs.threadblocks]
    input_digests = {
        "classifier_manifest_sha256": manifest_sha256,
        "dynamic_trace_pb_sha256": immutable.sha256_bytes(
            trace_inputs.dynamic_trace_payload
        ),
        "threadblock_proto_set_sha256": _sha256_of_retained_threadblocks(
            trace_inputs.threadblocks
        ),
    }
    # Dependency failures are not cell evidence: missing tracked `_pb2.py`
    # authorities are repository corruption and a missing protobuf runtime is
    # a prerequisite failure. The helper's distinct hard exceptions propagate.
    trace_pb2_mod, threadblock_pb2_mod, _ = lph._load_proto_bindings()
    kernel = lph._find_kernel_record(
        trace_pb2_mod,
        trace_inputs.dynamic_trace_path,
        _KERNEL_ID,
        retained_trace_pb_payload=trace_inputs.dynamic_trace_payload,
    )
    if kernel is None:
        return _typed_entry(
            slug=slug, shape_point=shape_point, identity=identity,
            typed_reason="predictor_l1_cta_to_sm_schedule_unknown",
            diagnostic=(
                f"kernel id {_KERNEL_ID} not found in "
                f"{trace_pb}"
            ),
            input_digests=input_digests,
        )
    grid_dim = 1
    block_dim = 1
    if kernel.HasField("grid_dim"):
        grid_dim = (int(kernel.grid_dim.x) * int(kernel.grid_dim.y)
                    * int(kernel.grid_dim.z))
    else:
        grid_dim = 0
    if kernel.HasField("block_dim"):
        block_dim = (int(kernel.block_dim.x)
                     * int(kernel.block_dim.y)
                     * int(kernel.block_dim.z))
    else:
        block_dim = 0
    if grid_dim <= 0 or block_dim <= 0:
        return _typed_entry(
            slug=slug, shape_point=shape_point, identity=identity,
            typed_reason="predictor_l1_cta_to_sm_schedule_unknown",
            diagnostic=(
                f"grid_dim_total={grid_dim} "
                f"block_dim_total={block_dim}"
            ),
            input_digests=input_digests,
        )

    # A no-load result is still a prediction from this consumer. Validate the
    # exact trace contract, canonical capability vector, every kernel's shared-
    # memory split, requested-kernel identity, and launch geometry above before
    # taking this semantic shortcut (BL-20260708-guard-on-consumer-path).
    if status == "matched_no_load_class_instructions":
        return _typed_entry(
            slug=slug, shape_point=shape_point, identity=identity,
            typed_reason=(
                "predictor_l1_no_modeled_l1_eligible_accesses"
            ),
            diagnostic=(
                "manifest parsed cleanly with no load-class "
                "instructions; nothing for the warp-pool walk to "
                "model"
            ),
            input_digests=input_digests,
        )

    if not tb_paths:
        return _typed_entry(
            slug=slug, shape_point=shape_point, identity=identity,
            typed_reason="predictor_l1_sass_provenance_missing",
            diagnostic=(
                "matched manifest but no threadblock protos on "
                "disk; dynamic trace evidence is missing "
                "(capture-side defect, not a kernel-without-loads "
                "condition)"
            ),
            input_digests=input_digests,
        )

    try:
        import_trace_contract_mod = (
            lph._load_import_trace_contract()
        )
        records = lph._walk_threadblock_protos(
            threadblock_pb2_mod,
            tb_paths,
            import_trace_contract_mod,
            grid_x=(int(kernel.grid_dim.x)
                    if kernel.HasField("grid_dim") else 0),
            grid_y=(int(kernel.grid_dim.y)
                    if kernel.HasField("grid_dim") else 0),
            grid_dim=grid_dim,
            retained_threadblock_pb_payloads=[
                payload for _path, payload in trace_inputs.threadblocks
            ],
        )
        per_warp = lta.build_per_warp_l1_accesses(
            records,
            manifest,
            sm_count=int(WARP_POOL_HEAD_CONFIG["sm_count"]),
            grid_dim=grid_dim,
            block_dim=block_dim,
            require_data_width=True,
        )
    except (
        lph.L1ProductionHelperThreadblockMalformed,
        lta.L1TraceAdapterGridBlockMissing,
        lta.L1TraceAdapterWarpIndexMissing,
    ) as exc:
        return _typed_entry(
            slug=slug, shape_point=shape_point, identity=identity,
            typed_reason="predictor_l1_cta_to_sm_schedule_unknown",
            diagnostic=f"{type(exc).__name__}: {exc}",
            input_digests=input_digests,
        )
    except (
        lph.L1ProductionHelperAddressMalformed,
        lph.L1ProductionHelperInstructionMalformed,
        lta.L1TraceAdapterManifestNotMatched,
        lta.L1TraceAdapterEligibilityDefect,
        lta.L1TraceAdapterDataWidthInvalid,
        lta.L1TraceAdapterRecordInvalid,
    ) as exc:
        return _typed_entry(
            slug=slug, shape_point=shape_point, identity=identity,
            typed_reason="predictor_l1_sass_provenance_missing",
            diagnostic=f"{type(exc).__name__}: {exc}",
            input_digests=input_digests,
        )

    carveout_pct = _CARVEOUT_PERCENT_BY_CELL.get((slug, shape_point))
    geometry = resolve_walk_geometry(carveout_pct=carveout_pct)

    sm_count = int(WARP_POOL_HEAD_CONFIG["sm_count"])
    warps_per_block = max(1, math.ceil(block_dim / 32))
    co_resident = co_resident_warp_limit(
        warps_per_block=warps_per_block,
        blocks_per_sm_limit=int(
            WARP_POOL_HEAD_CONFIG["max_blocks_per_sm"]),
        occupancy_warp_limit=int(
            WARP_POOL_HEAD_CONFIG["occupancy_warp_limit_per_sm"]),
    )

    per_sm: dict[int, list] = {}
    for (block_index, warp_index), stream in per_warp:
        sm_id = block_to_sm_id(
            block_index,
            sm_count=sm_count,
            schedule_policy=str(
                WARP_POOL_HEAD_CONFIG["cta_to_sm_schedule_policy"]),
        )
        per_sm.setdefault(sm_id, []).append(
            ((block_index, warp_index), stream)
        )

    totals = {
        "hit_count": 0, "miss_count": 0, "bypass_count": 0,
        "mshr_merge_count": 0, "mshr_stall_events": 0,
        "line_fill_count": 0, "line_alloc_miss_count": 0,
        "sector_miss_resident_line_count": 0,
    }
    _ov = walk_overrides or {}
    _mode = str(_ov.get("mode", WARP_POOL_HEAD_CONFIG["scheduling_mode"]))
    _sector = str(_ov.get(
        "sector_accounting", WARP_POOL_HEAD_CONFIG["sector_accounting"]))
    _mshr = (_ov["mshr_count"] if "mshr_count" in _ov
             else geometry.mshr.count)
    for sm_id in sorted(per_sm):
        result = walk_warp_pool(
            per_sm[sm_id],
            line_capacity=geometry.capacity.line_capacity,
            hit_latency=geometry.latencies.hit_latency,
            miss_latency=geometry.latencies.miss_latency,
            mshr_count=_mshr,
            co_resident_warps=co_resident,
            mode=_mode,
            sector_accounting=_sector,
            warmup_launches=int(
                WARP_POOL_HEAD_CONFIG["warmup_launches"]),
        )
        for key in totals:
            totals[key] += getattr(result, key)

    total_eligible = totals["hit_count"] + totals["miss_count"]
    if total_eligible == 0:
        return _typed_entry(
            slug=slug, shape_point=shape_point, identity=identity,
            typed_reason=(
                "predictor_l1_no_modeled_l1_eligible_accesses"
            ),
            diagnostic=(
                "matched manifest and walked trace produced zero "
                "l1-eligible sector accesses"
            ),
            input_digests=input_digests,
        )

    predicted = totals["hit_count"] / float(total_eligible)
    return {
        "slug": slug,
        "shape_point": shape_point,
        **identity,
        "result_kind": "scalar",
        "cache_layer": "L1TEX",
        "predicted_hit_rate": predicted,
        "typed_reason": None,
        "evidence": {
            "walker": "warp_pool_two_layer_finite_mshr",
            "counters": dict(totals),
            "total_eligible_accesses": total_eligible,
            "co_resident_warps": co_resident,
            "warps_per_block": warps_per_block,
            "active_sm_count": len(per_sm),
            "geometry": {
                "capacity_bytes": geometry.capacity.capacity_bytes,
                "line_capacity": geometry.capacity.line_capacity,
                "capacity_property":
                    geometry.capacity.property_name,
                "carveout_percent": carveout_pct,
                "hit_latency": geometry.latencies.hit_latency,
                "miss_latency": geometry.latencies.miss_latency,
                "mshr_count": geometry.mshr.count,
                "mshr_source": geometry.mshr.source,
                "replacement_modeled":
                    geometry.replacement.modeled,
                "replacement_source": geometry.replacement.source,
                "set_mapping_modeled":
                    geometry.set_mapping.modeled,
                "set_mapping_source":
                    geometry.set_mapping.source,
            },
        },
        "input_digests": input_digests,
    }


def build_artifact(*, label_admitted_path: Path) -> dict:
    with label_admitted_path.open("r", encoding="utf-8") as fh:
        admitted = json.load(fh)
    entries: list[dict] = []
    for cell in admitted.get("cells", []):
        cell_arch = cell.get("arch")
        if cell_arch is not None and cell_arch != "sm_86":
            continue
        if is_pending_refit_factorial(cell):
            continue
        slug = cell.get("slug")
        shape_point = cell.get("shape_point")
        if not slug or not shape_point:
            raise ValueError(
                f"admitted row missing slug/shape_point: {cell!r}"
            )
        variant_id = cell_variant_id(cell)
        if variant_id:
            identity = _extract_identity_tuple(
                cell, slug=slug, shape_point=shape_point,
            )
            entries.append(_typed_entry(
                slug=slug, shape_point=shape_point,
                identity=identity,
                typed_reason=(
                    "predictor_l1_sass_provenance_missing"
                ),
                diagnostic=(
                    "register-cap build variant: the cache-op "
                    "classifier manifest store has no "
                    "build-variant axis, so no per-cap SASS "
                    "provenance manifest exists for this binary"
                ),
                extra={
                    "variant_id": variant_id,
                    "register_cap": int(cell["register_cap"]),
                },
            ))
            continue
        entries.append(compute_for_cell(
            slug=slug, shape_point=shape_point, cell_row=cell,
        ))
    cohort_priority_keys = _load_cohort_priority_keys()
    if cohort_priority_keys:
        cohort_priority_index = {
            key: idx for idx, key in enumerate(cohort_priority_keys)
        }
        max_priority = len(cohort_priority_keys)
        entries.sort(key=lambda entry: (
            cohort_priority_index.get(
                (entry["slug"], entry["shape_point"]),
                max_priority),
            entry["slug"],
            entry["shape_point"],
        ))
    n_scalar = sum(
        1 for e in entries if e["result_kind"] == "scalar")
    return {
        "schema_version": 1,
        "artifact_kind": "warp_pool_l1_per_admitted_cell",
        "arch": "sm_86",
        "generator": (
            "tools/sm_86_predictor_accuracy_recovery_v2/"
            "compute_warp_pool_l1_per_admitted_cell.py"
        ),
        "generated_at_commit": _git_head(),
        "config": WARP_POOL_HEAD_CONFIG,
        "config_digest": config_digest(),
        "cohort_size": len(entries),
        "n_scalar": n_scalar,
        "n_typed": len(entries) - n_scalar,
        "cells": entries,
    }


def _validate_artifact(artifact: dict, *, schema_path: Path) -> None:
    import jsonschema
    with schema_path.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    jsonschema.validate(artifact, schema)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-admitted-cell L1 predictions via the "
            "warp-pool walker and emit the NEW-head artifact "
            "beside the existing predicted-L1 artifact."
        ),
    )
    parser.add_argument(
        "--label-admitted-path", type=Path,
        default=(_REPO_ROOT / "data" / "local_labels"
                 / "label_admitted.json"),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--schema-path", type=Path, default=DEFAULT_SCHEMA,
    )
    args = parser.parse_args(argv)

    if args.out.resolve() == _OLD_ARTIFACT_PATH.resolve():
        raise WarpPoolArtifactPathError(
            "the warp-pool head must not overwrite the existing "
            f"predicted-L1 artifact at {_OLD_ARTIFACT_PATH}"
        )

    try:
        artifact = build_artifact(
            label_admitted_path=args.label_admitted_path,
        )
    except _ArtifactIdentityFieldMissing as exc:
        sys.stderr.write(f"identity-field failure: {exc}\n")
        return 2
    _validate_artifact(artifact, schema_path=args.schema_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True)
        fh.write("\n")
    scalar = artifact["n_scalar"]
    print(
        f"compute_warp_pool_l1_per_admitted_cell: wrote "
        f"{args.out} ({artifact['cohort_size']} cells; "
        f"{scalar} scalar / {artifact['n_typed']} typed; "
        f"config_digest {artifact['config_digest'][:12]}...)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
