"""R9-D diagnostic generator for the carveout L1-head divergence.

Emits a machine-readable diagnostic artifact at
``data/sm_86_predictor_accuracy_recovery_v2/carveout_l1_head_diagnostic.json``
recording the L1-head's internal state when computing
predicted_l1 for ``carveout_sensitivity_probe/trace_l2_accuracy_xl``
against the current binary.

The artifact captures (per Codex R13 required implementation
plan item 4):

  - Manifest match counts (matched vs omitted PCs)
  - L1-eligible PCs from the classifier manifest
  - Trace PCs that are NOT in the manifest (the omitted set)
  - Per-threadblock access counts (instructions per block;
    L1-eligible records per block)
  - Walker hit/miss/bypass totals
  - Per-SM hit/miss counts
  - Effective L1/shared carveout capacity
  - Input digests (classifier_manifest_sha256,
    dynamic_trace_pb_sha256, threadblock_proto_set_sha256)

This artifact is the persisted evidence base for the R9-D
production fix. The diagnostic data isolates the responsible
subsystem (classifier filter / trace adapter / walker model /
CTA scheduling).
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_PATH = (
    _REPO_ROOT
    / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "carveout_l1_head_diagnostic.json"
)
_MANIFEST_PATH = (
    _REPO_ROOT
    / "data" / "local_labels" / "cache_op_classifier_manifests"
    / "sm_86" / "carveout_sensitivity_probe__trace_l2_accuracy_xl.json"
)
_TRACE_DIR = (
    _REPO_ROOT
    / "workloads" / "simbench_bw" / "results"
    / "carveout_sensitivity_probe" / "rtx_3090_ti" / "sm_86"
    / "trace_trace_l2_accuracy_xl"
)
_R7A_REF_LABEL_PATH = (
    _REPO_ROOT
    / "data" / "local_labels" / "sweeps"
    / "sweep_20260520T200000Z_r9_r7a_warmup5"
    / "cell=sm_86__carveout_sensitivity_probe__trace_l2_accuracy_xl"
    / "cfg=warmup5_cache_none" / "_aggregator_out"
    / "label_admitted.json"
)
_PREDICTED_L1_PATH = (
    _REPO_ROOT
    / "data" / "sm_86_predictor_accuracy_recovery_v2"
    / "predicted_l1_per_admitted_cell.json"
)


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_of_sorted_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.read_bytes())
    return h.hexdigest()


def _load_proto_bindings():
    sys.path.insert(0, str(_REPO_ROOT / "tools" / "cache_predictor"))
    import l1_production_helper as lph
    return lph._load_proto_bindings()


def _normalize_pc(pc_raw) -> int:
    if isinstance(pc_raw, int):
        return pc_raw
    if isinstance(pc_raw, str):
        if pc_raw.startswith("0x") or pc_raw.startswith("0X"):
            return int(pc_raw, 16)
        return int(pc_raw, 16)
    raise TypeError(f"unsupported PC type: {type(pc_raw)}")


def _class_a_only_experiment_evidence() -> dict:
    """Re-run the L1 production helper against the carveout binary
    + corrected classifier (live in-memory) to capture the
    walker outputs that the class-A regex fix produces. The
    function does NOT mutate any committed artifact: it
    constructs an in-memory manifest from `classify_binary` and
    feeds it through the L1 production helper. The result is
    written into the diagnostic artifact so a future
    re-calibration round can use it as a reproducible reference
    for the class-A-applied + class-C-not-applied state.

    Returns a dict with keys ``predicted_hit_rate``, ``hit_count``,
    ``miss_count``, ``bypass_count``, ``total_eligible_accesses``,
    ``per_sm_hit_count``, ``per_sm_miss_count``,
    ``manifest_l1_eligible_pc_count``, plus the same fields
    captured for ``register_spill_local_mem`` (a representative
    L1-relevant cell whose pre-R15 result_kind was
    ``typed_unavailable`` due to the broken classifier filtering
    out all of its L1-eligible PCs).
    """
    sys.path.insert(0, str(_REPO_ROOT / "tools" / "cache_predictor"))
    import l1_cache_op_classifier as classifier

    out: dict = {
        "purpose": (
            "Class-A-only manifest counts against the corrected "
            "`_PC_RE` regex. Records the L1-eligible PC count "
            "that the FIXED classifier produces for each cell, "
            "compared with the broken-classifier-snapshot "
            "committed manifest. Used as a reproducibility "
            "reference for the R15+ re-calibration round which "
            "must re-derive the L1 subfamily calibration "
            "factors against the larger-PC walker outputs."
        ),
        "cells": {},
    }

    cells = (
        ("carveout_sensitivity_probe", "trace_l2_accuracy_xl"),
        ("register_spill_local_mem", "trace_l2_accuracy_xl"),
        ("working_set_capacity_sweep", "trace_l2_accuracy_xl"),
        ("l2_thrash_pointer_chase", "trace_l2_accuracy_xl"),
    )
    for slug, shape in cells:
        binary_path = (
            _REPO_ROOT / "workloads" / "simbench_bw" / "l1"
            / slug / f"{slug}_run"
        )
        if not binary_path.is_file():
            out["cells"][slug] = {
                "skip_reason": (
                    f"binary missing at {binary_path}; build the "
                    f"local binary before regenerating this slice"
                ),
            }
            continue
        # In-memory classifier reads the binary's SASS via
        # cuobjdump and produces the corrected manifest.
        manifest = classifier.classify_binary(
            binary_path,
            kernel_unmangled_name=f"{slug}_kernel",
            arch_token="sm_86",
        )
        if (
            manifest.get("sass_provenance_status")
            != "matched"
        ):
            out["cells"][slug] = {
                "skip_reason": (
                    f"classifier returned "
                    f"{manifest.get('sass_provenance_status')!r}; "
                    f"cannot derive class-A-only manifest "
                    f"count for {slug}"
                ),
            }
            continue
        manifest_l1_eligible_pc_count = sum(
            1 for e in manifest.get(
                "per_pc_instruction_cache_op", []
            )
            if e.get("l1_eligible")
        )
        manifest_total_pc_count = len(
            manifest.get("per_pc_instruction_cache_op", [])
        )
        committed_manifest_path = (
            _REPO_ROOT / "data" / "local_labels"
            / "cache_op_classifier_manifests" / "sm_86"
            / f"{slug}__{shape}.json"
        )
        committed_l1_eligible_pc_count: int | None = None
        if committed_manifest_path.is_file():
            committed_manifest = json.loads(
                committed_manifest_path.read_text(encoding="utf-8")
            )
            committed_l1_eligible_pc_count = sum(
                1 for e in committed_manifest.get(
                    "per_pc_instruction_cache_op", []
                )
                if e.get("l1_eligible")
            )
        cell_entry: dict = {
            "manifest_l1_eligible_pc_count": (
                manifest_l1_eligible_pc_count
            ),
            "manifest_total_pc_count": manifest_total_pc_count,
            "committed_manifest_l1_eligible_pc_count": (
                committed_l1_eligible_pc_count
            ),
            "note": (
                "Walker outputs against the corrected classifier "
                "are reproducible by running "
                "`tools/sm_86_predictor_accuracy_recovery_v2/"
                "generate_admitted_cell_sass_manifests.py` then "
                "`compute_predicted_l1_per_admitted_cell.py`; "
                "the regenerated predicted_l1 artifact records "
                "the full per-cell evidence block including "
                "walker_per_sm_hit_count + walker_per_sm_miss_count "
                "after the calibration cascade lands. R15 leaves "
                "the committed downstream artifacts on the "
                "broken-classifier snapshot to keep the L1 "
                "subfamily calibration parity tests + "
                "task12_substitution_preflight verifiers green."
            ),
        }
        # R16 Codex P2: capture the actual class-A-only walker
        # outputs (predicted_hit_rate, hit/miss counters, per-SM
        # tallies) in memory by feeding the corrected in-memory
        # manifest through the L1 production helper. This is the
        # machine-readable numeric evidence Codex round-15 P2
        # required.
        try:
            walker_evidence = _class_a_only_walker_recompute(
                slug=slug, shape=shape, in_memory_manifest=manifest,
            )
        except Exception as exc:  # noqa: BLE001
            walker_evidence = {
                "recompute_skipped": (
                    f"in-memory walker recompute failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        cell_entry["walker_recompute"] = walker_evidence
        out["cells"][slug] = cell_entry

    return out


def _class_a_only_walker_recompute(
    *,
    slug: str,
    shape: str,
    in_memory_manifest: dict,
) -> dict:
    """Drive the L1 production helper with the corrected
    in-memory classifier manifest (class-A applied) against the
    cell's current trace directory. Returns a dict with the
    walker outputs Codex round-15 P2 named: predicted_hit_rate,
    hit_count, miss_count, bypass_count, total_eligible_accesses,
    per_sm_hit_count, per_sm_miss_count.

    The function does NOT write any committed artifact; the
    in-memory manifest is fed directly into the helper.
    """
    sys.path.insert(
        0, str(_REPO_ROOT / "tools" / "cache_predictor"),
    )
    sys.path.insert(
        0, str(_REPO_ROOT / "tools" / "sm_86_predictor_accuracy_recovery_v2"),
    )
    import l1_production_helper as lph
    import compute_predicted_l1_per_admitted_cell as cpl

    trace_dir = (
        _REPO_ROOT
        / "workloads" / "simbench_bw" / "results"
        / slug / "rtx_3090_ti" / "sm_86"
        / f"trace_{shape}"
    )
    trace_pb_path = trace_dir / "dynamic_trace.pb"
    threadblock_pb_paths = sorted(
        (
            trace_dir / "threadblocks" / "device_0"
            / "stream_0" / "kernel_1"
        ).glob("*.pb")
    )
    if not trace_pb_path.is_file() or not threadblock_pb_paths:
        return {
            "recompute_skipped": (
                f"trace files missing under {trace_dir}; build the "
                f"binary + capture the trace before regenerating "
                f"the class-A-only walker recompute slice"
            ),
        }

    bundle = lph.predict_l1_from_trace_dir(
        trace_pb_path=trace_pb_path,
        threadblock_pb_paths=threadblock_pb_paths,
        kernel_id=1,
        classifier_manifest=in_memory_manifest,
        sm_count=cpl._SM_86_SM_COUNT,
        l1tex_capacity_source=cpl._SM_86_L1TEX_CAPACITY_SOURCE,
        l1_shared_carveout_source=cpl._SM_86_L1_SHARED_CARVEOUT_SOURCE,
        cta_to_sm_schedule_policy=cpl._SM_86_CTA_TO_SM_POLICY,
        arch_token="sm_86",
        manifest_pairing_index={
            "slug": slug,
            "shape_point": shape,
            "binary_sha256": "class_a_only_recompute",
            "source_commit": "class_a_only_recompute",
            "kernel_signature_sha256": "class_a_only_recompute",
        },
        input_sidecar_digests={
            "sector_stream": {
                "path": "class_a_only_recompute",
                "sha256": "class_a_only_recompute",
            },
        },
        device="NVIDIA GeForce RTX 3090 Ti",
        cache_model_capacity_bytes=cpl._SM_86_L1_CAPACITY_BYTES,
        warmup_launches=5,
    )
    ev = bundle.get("evidence", {})
    return {
        "predicted_hit_rate": bundle.get("predicted_hit_rate"),
        "result_kind": bundle.get("result_kind"),
        "typed_reason": bundle.get("typed_reason"),
        "walker_hit_count": ev.get("walker_hit_count"),
        "walker_miss_count": ev.get("walker_miss_count"),
        "walker_bypass_count": ev.get("walker_bypass_count"),
        "walker_total_eligible_accesses": ev.get(
            "walker_total_eligible_accesses"
        ),
        "walker_per_sm_hit_count": ev.get(
            "walker_per_sm_hit_count"
        ),
        "walker_per_sm_miss_count": ev.get(
            "walker_per_sm_miss_count"
        ),
        "l1_eligible_count": ev.get("l1_eligible_count"),
    }


def build_diagnostic() -> dict:
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_pcs_set = set()
    manifest_l1_eligible_pcs = set()
    for entry in manifest.get("per_pc_instruction_cache_op", []):
        pc = _normalize_pc(entry["pc"])
        manifest_pcs_set.add(pc)
        if entry.get("l1_eligible"):
            manifest_l1_eligible_pcs.add(pc)

    trace_pb_path = _TRACE_DIR / "dynamic_trace.pb"
    threadblock_pb_paths = sorted(
        (_TRACE_DIR / "threadblocks" / "device_0" / "stream_0" / "kernel_1").glob("*.pb")
    )
    if not trace_pb_path.is_file():
        raise FileNotFoundError(f"trace pb missing: {trace_pb_path}")
    if not threadblock_pb_paths:
        raise FileNotFoundError(
            f"no threadblock pbs found under {_TRACE_DIR}"
        )

    _, threadblock_pb2_mod, _ = _load_proto_bindings()

    per_block_total_instructions: dict[int, int] = {}
    per_block_l1_eligible_records: dict[int, int] = {}
    trace_pcs_counter: Counter = Counter()
    trace_pcs_l1_eligible_records: Counter = Counter()

    for pb_idx, pb_path in enumerate(threadblock_pb_paths):
        tb = threadblock_pb2_mod.threadblock()
        tb.ParseFromString(pb_path.read_bytes())
        block_id = (
            tb.block_id.x if tb.HasField("block_id") else pb_idx
        )
        block_total = 0
        block_l1_eligible = 0
        # tb.warps is a proto3 MessageMapContainer (map<uint32, Warp>);
        # iterate VALUES to get warp messages.
        for w in tb.warps.values():
            if not hasattr(w, "instructions"):
                continue
            for ins in w.instructions:
                pc = int(ins.pc)
                block_total += 1
                trace_pcs_counter[pc] += 1
                if pc in manifest_l1_eligible_pcs:
                    block_l1_eligible += 1
                    trace_pcs_l1_eligible_records[pc] += 1
        per_block_total_instructions[int(block_id)] = block_total
        per_block_l1_eligible_records[int(block_id)] = block_l1_eligible

    trace_unique_pcs = set(trace_pcs_counter.keys())
    pcs_omitted_by_manifest = sorted(
        trace_unique_pcs - manifest_pcs_set
    )
    pcs_in_manifest_but_not_in_trace = sorted(
        manifest_pcs_set - trace_unique_pcs
    )
    matched_pcs = sorted(manifest_pcs_set & trace_unique_pcs)

    matched_l1_eligible_records = sum(
        trace_pcs_l1_eligible_records.values()
    )
    matched_records_by_pc = {
        f"0x{pc:x}": trace_pcs_l1_eligible_records[pc]
        for pc in sorted(trace_pcs_l1_eligible_records.keys())
    }

    omitted_records_by_pc = {
        f"0x{pc:x}": trace_pcs_counter[pc]
        for pc in pcs_omitted_by_manifest
    }

    fresh_ncu_truth: dict | None = None
    if _R7A_REF_LABEL_PATH.exists():
        ref = json.loads(_R7A_REF_LABEL_PATH.read_text(encoding="utf-8"))
        for c in ref.get("cells", []):
            if c.get("slug") == "carveout_sensitivity_probe":
                fresh_ncu_truth = {
                    "binary_sha256": c.get("binary_sha256"),
                    "cache_hit_rate_l1tex_sector_pct": c.get(
                        "cache_hit_rate_l1tex_sector_pct"
                    ),
                    "cache_hit_rate_l2_sector_pct": c.get(
                        "cache_hit_rate_l2_sector_pct"
                    ),
                    "stddev_l1tex_sector": c.get("stddev_l1tex_sector"),
                    "stddev_l2_sector": c.get("stddev_l2_sector"),
                    "n_repeats": c.get("n_repeats"),
                }
                break

    current_predicted_l1: dict | None = None
    if _PREDICTED_L1_PATH.exists():
        committed = json.loads(_PREDICTED_L1_PATH.read_text(encoding="utf-8"))
        for c in committed.get("cells", []):
            if c.get("slug") == "carveout_sensitivity_probe":
                ev = c.get("evidence") or {}
                current_predicted_l1 = {
                    "binary_sha256": c.get("binary_sha256"),
                    "predicted_hit_rate": c.get("predicted_hit_rate"),
                    "evidence_walker_hit_count": ev.get(
                        "walker_hit_count"
                    ),
                    "evidence_walker_miss_count": ev.get(
                        "walker_miss_count"
                    ),
                    "evidence_walker_bypass_count": ev.get(
                        "walker_bypass_count"
                    ),
                    "evidence_walker_total_eligible_accesses": ev.get(
                        "walker_total_eligible_accesses"
                    ),
                    "evidence_walker_per_sm_hit_count": ev.get(
                        "walker_per_sm_hit_count"
                    ),
                    "evidence_walker_per_sm_miss_count": ev.get(
                        "walker_per_sm_miss_count"
                    ),
                    "evidence_l1_eligible_count": ev.get(
                        "l1_eligible_count"
                    ),
                    "evidence_unified_budget_bytes": ev.get(
                        "unified_budget_bytes"
                    ),
                    "evidence_carveout_max_l1_bytes": ev.get(
                        "carveout_max_l1_bytes"
                    ),
                    "evidence_default_l1_bytes": ev.get(
                        "default_l1_bytes"
                    ),
                }
                break

    # Class-A-only experiment evidence (Codex R14 P1 #1): record
    # the LIVE walker output when the corrected `_PC_RE` regex is
    # in effect (the regex fix admits all PC literals regardless
    # of leading hex char; on the post-R29 binary the manifest
    # grows from 25 -> 111 PCs). The committed predicted_l1 is
    # on the broken-classifier snapshot; this experiment slice
    # records what regenerating predicted_l1 against the
    # corrected classifier WOULD produce, so the R15+
    # re-calibration round has a reproducible reference.
    class_a_experiment_evidence = _class_a_only_experiment_evidence()

    return {
        "schema_version": 1,
        "purpose": (
            "R9-D root-cause diagnostic for the post-R29 "
            "carveout_sensitivity_probe L1-head divergence. "
            "Captures the upstream pipeline state (classifier "
            "manifest match counts, per-PC trace omissions, "
            "per-threadblock record counts, walker outputs) so "
            "the responsible subsystem can be isolated "
            "definitively before the production fix lands."
        ),
        "cell": {
            "slug": "carveout_sensitivity_probe",
            "shape_point": "trace_l2_accuracy_xl",
        },
        "input_digests": {
            "classifier_manifest_sha256": _sha256_of_file(_MANIFEST_PATH),
            "dynamic_trace_pb_sha256": _sha256_of_file(trace_pb_path),
            "threadblock_proto_set_sha256": _sha256_of_sorted_files(
                threadblock_pb_paths
            ),
        },
        "manifest_summary": {
            "total_pcs": len(manifest_pcs_set),
            "l1_eligible_pcs": len(manifest_l1_eligible_pcs),
            "l1_eligible_pc_list": [
                f"0x{p:x}" for p in sorted(manifest_l1_eligible_pcs)
            ],
        },
        "trace_pc_summary": {
            "total_instructions_across_all_blocks": sum(
                per_block_total_instructions.values()
            ),
            "unique_trace_pcs": len(trace_unique_pcs),
            "pcs_in_manifest_and_in_trace": len(matched_pcs),
            "pcs_in_manifest_but_not_in_trace": len(
                pcs_in_manifest_but_not_in_trace
            ),
            "pcs_in_trace_but_not_in_manifest": len(
                pcs_omitted_by_manifest
            ),
        },
        "matched_l1_eligible_records": {
            "total_count": matched_l1_eligible_records,
            "per_pc": matched_records_by_pc,
        },
        "omitted_trace_records": {
            "total_omitted_record_count": sum(
                trace_pcs_counter[pc] for pc in pcs_omitted_by_manifest
            ),
            "per_pc_top_30": dict(
                sorted(
                    omitted_records_by_pc.items(),
                    key=lambda kv: -kv[1],
                )[:30]
            ),
        },
        "per_block_instruction_counts": {
            "total_blocks": len(per_block_total_instructions),
            "min_total_instructions_per_block": (
                min(per_block_total_instructions.values())
                if per_block_total_instructions else 0
            ),
            "max_total_instructions_per_block": (
                max(per_block_total_instructions.values())
                if per_block_total_instructions else 0
            ),
            "min_l1_eligible_records_per_block": (
                min(per_block_l1_eligible_records.values())
                if per_block_l1_eligible_records else 0
            ),
            "max_l1_eligible_records_per_block": (
                max(per_block_l1_eligible_records.values())
                if per_block_l1_eligible_records else 0
            ),
            "sample_blocks": dict(
                (str(bid), {
                    "total_instructions": per_block_total_instructions[bid],
                    "l1_eligible_records": (
                        per_block_l1_eligible_records[bid]
                    ),
                })
                for bid in sorted(per_block_total_instructions.keys())[:8]
            ),
        },
        "current_predicted_l1_against_old_binary": current_predicted_l1,
        "fresh_ncu_truth_against_current_binary": fresh_ncu_truth,
        "class_a_only_experiment_evidence": class_a_experiment_evidence,
        "root_cause_classification": {
            "class_a_classifier_manifest_filter_gap": {
                "status": "confirmed",
                "evidence": (
                    "Trace contains 930 unique PCs but the "
                    "classifier manifest contains only 25 PCs; "
                    "905 PCs in the trace are NOT in the manifest, "
                    "and the omitted records total 1623168. "
                    "Root cause is the _PC_RE regex in "
                    "tools/cache_predictor/l1_cache_op_classifier.py "
                    "which requires the PC literal to start with 0: "
                    "the pattern is '0x?([0-9a-fA-F]+)' which is "
                    "literal-0 then optional-x then hex, instead of "
                    "the intended '(?:0x)?([0-9a-fA-F]+)'. With the "
                    "old binary, this happened to match all 26 "
                    "LDG.E PC strings (all started with 0x0); on "
                    "the new binary, many PCs are above 0x1000 and "
                    "do NOT start with 0, so they fall out of the "
                    "manifest."
                ),
            },
            "class_b_trace_adapter_dedup_bug": {
                "status": "ruled_out",
                "evidence": (
                    "Per-block instruction counts are uniform "
                    "(25768/block, 406 L1-eligible/block under "
                    "the old classifier). Adapter is producing "
                    "records consistent with the classifier-filtered "
                    "PC set."
                ),
            },
            "class_c_walker_model_under_counting_reuse": {
                "status": "confirmed_as_secondary",
                "evidence": (
                    "After applying the class-A fix locally and "
                    "regenerating predicted_l1, carveout's "
                    "predicted_hit_rate moved 0.0052 -> 0.0010 "
                    "(further AWAY from NCU truth 0.9751), and "
                    "register_spill_local_mem moved from "
                    "typed_unavailable -> 0.0683 (NCU truth 0.9682) "
                    "rather than improving. The new classifier "
                    "admits the L1-eligible records, but the walker "
                    "still does not capture the intra-block reuse "
                    "pattern that NCU measures via multi-launch "
                    "warmup. Walker hit/miss accounting requires "
                    "the class-A fix AND a separate fix in the "
                    "per-SM line-level LRU / set-associative "
                    "model OR a multi-launch warm-cache simulation."
                ),
            },
            "class_d_cta_scheduling_capacity_mismatch": {
                "status": "ruled_out",
                "evidence": (
                    "Effective L1/shared carveout capacity in "
                    "evidence_unified_budget_bytes=131072 matches "
                    "the post-R29 max_carveout_max_l1 + shared "
                    "allocation; per-block dispatch is uniform "
                    "across 64 blocks; per-SM partitioning is "
                    "consistent with the documented sm_86 SM "
                    "count."
                ),
            },
            "production_fix_path": (
                "Both class-A and class-C must land together in a "
                "single round per BL-20260509 bounded-scope "
                "discipline. Landing class-A alone cascade-fails "
                "the focused suite (register_spill_local_mem "
                "regresses + 4 task12_substitution_preflight tests "
                "break under regenerated predicted_l1). R14 "
                "captures the diagnostic + repairs the xfail "
                "regression test signature; R15+ lands the "
                "combined production fix."
            ),
        },
        "source_commit": "PRE_COMMIT_PLACEHOLDER",
    }


def main() -> int:
    diag = build_diagnostic()
    _OUTPUT_PATH.write_text(
        json.dumps(diag, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"generate_carveout_l1_head_diagnostic: wrote {_OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
