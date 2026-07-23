"""Producer for the count-feasibility analysis artifact (the cache/memory count
falsification evidence).

For every `count` + `ratio` target in the typed metric registry it records whether
an NVBit-trace-derived count is defensibly comparable to the NCU counter, or
closes by `engineering_closure_refuted` due to an aggregation-domain mismatch.
Each entry documents the plan-mandated components: the NVBit-trace->NCU mapping,
the aggregation-domain mismatch, a coalescing / replay / write-allocation note,
the comparison geometry, and the terminal verdict.

The per-target verdicts integrate a `/humanize:ask-codex` count-feasibility
consult (the `analyze`-routed analysis); the consult output is committed at
`data/local_labels/count_feasibility_codex_consult.md` and referenced. The
`fraction` hit-rate family is NOT a count target and is excluded -- it is the
load-bearing predictor path handled separately.

This producer reads VALUES from the registry only; it invokes no NCU and writes
no `.ncu-rep`. The LIVE count comparison that would empirically confirm/deny each
verdict (real NVBit counts vs real NCU counters) is a separate live step.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import ncu_metric_registry as reg  # noqa: E402

SCHEMA_VERSION = 1
CODEX_CONSULT_REFERENCE = "data/local_labels/count_feasibility_codex_consult.md"
ANALYSIS_BASIS = (
    "Per-target comparability of an NVBit-trace-derived count (re-aggregated by "
    "dynamic warp instruction + 32B sector) to the NCU hardware counter, "
    "integrating a /humanize:ask-codex count-feasibility consult. A target is "
    "engineering_closure_refuted when its NCU counter lives in an aggregation "
    "domain the per-kernel global-LSU trace cannot defensibly reach."
)

VERDICT_COMPARABLE = "trace_count_comparable"
VERDICT_REFUTED = "engineering_closure_refuted"

# Per-registry-key analysis. Every count/ratio registry key MUST have an entry
# here or build() fails closed (the artifact cannot silently omit a target).
_ANALYSIS: dict[str, dict] = {
    "l1tex_global_op_ld_requests": {
        "verdict": VERDICT_COMPARABLE,
        "domain_mismatch_reason": "none",
        "trace_to_ncu_mapping": "one post-predicate warp global-LSU load instruction = one request; count instructions, not active lanes",
        "aggregation_domain_mismatch": "none for the global-LSU load request domain; the trace warp-instruction count matches the post-coalescing request count",
        "coalescing_replay_write_allocation_note": "replay only perturbs NCU if passes see different work/cache state; the request count itself is replay-stable",
        "comparison_geometry": "count_absolute_floor",
    },
    "l1tex_global_op_st_requests": {
        "verdict": VERDICT_COMPARABLE,
        "domain_mismatch_reason": "none",
        "trace_to_ncu_mapping": "one post-predicate warp global-LSU store instruction = one request; do not infer from store bytes/sectors",
        "aggregation_domain_mismatch": "none for the global-LSU store request domain",
        "coalescing_replay_write_allocation_note": "store write-allocation/fill behavior matters for sector counts later, not for this request count",
        "comparison_geometry": "count_absolute_floor",
    },
    "l1tex_global_op_atom_requests": {
        "verdict": VERDICT_COMPARABLE,
        "domain_mismatch_reason": "none",
        "trace_to_ncu_mapping": "one global atomic instruction = one request; the request count is comparable",
        "aggregation_domain_mismatch": "none for the request count; atomics route through L1 toward L2",
        "coalescing_replay_write_allocation_note": "lane-level atomic fanout/serialization is NOT the request count and must not be inferred from it",
        "comparison_geometry": "count_absolute_floor",
    },
    "l1tex_global_op_red_requests": {
        "verdict": VERDICT_COMPARABLE,
        "domain_mismatch_reason": "none",
        "trace_to_ncu_mapping": "one global reduction instruction = one request; the request count is comparable",
        "aggregation_domain_mismatch": "none for the request count; same atom/reduction routing as atomics",
        "coalescing_replay_write_allocation_note": "reduction lane fanout/serialization is NOT the request count",
        "comparison_geometry": "count_absolute_floor",
    },
    "l1tex_total_requests": {
        "verdict": VERDICT_REFUTED,
        "domain_mismatch_reason": "per_pipe_vs_total_aggregation",
        "trace_to_ncu_mapping": "no defensible mapping: the trace covers only the global-LSU pipe",
        "aggregation_domain_mismatch": "the unified L1/TEX total includes non-global-LSU domains (local/shared/texture/surface) and all pipes; the trace target is global-LSU only",
        "coalescing_replay_write_allocation_note": "not applicable; the mismatch is the aggregation scope, not coalescing",
        "comparison_geometry": "none (refuted)",
    },
    "lts_srcunit_tex_requests": {
        "verdict": VERDICT_REFUTED,
        "domain_mismatch_reason": "post_l1_filter_xbar_traffic",
        "trace_to_ncu_mapping": "no defensible mapping from per-instruction global memory accesses",
        "aggregation_domain_mismatch": "L2 request packets from source-unit TEX/L1TEX are post-L1, whole-srcunit, and packetized -- a different aggregation domain than per-instruction global-memory trace misses",
        "coalescing_replay_write_allocation_note": "L2 packetization + post-L1 filtering compound; not isomorphic to the trace",
        "comparison_geometry": "none (refuted)",
    },
    "l1tex_global_op_ld_sectors": {
        "verdict": VERDICT_COMPARABLE,
        "domain_mismatch_reason": "none",
        "trace_to_ncu_mapping": "coalesced unique 32B L1 sectors per global load request, summed; raw per-lane addresses re-aggregated to 32B sectors",
        "aggregation_domain_mismatch": "none for the L1 access-sector domain",
        "coalescing_replay_write_allocation_note": "compare coalesced unique sectors, not per-lane address counts",
        "comparison_geometry": "count_absolute_floor",
    },
    "l1tex_global_op_st_sectors": {
        "verdict": VERDICT_COMPARABLE,
        "domain_mismatch_reason": "none",
        "trace_to_ncu_mapping": "coalesced unique 32B L1 store sectors per global store request, summed",
        "aggregation_domain_mismatch": "none for the L1 store access-sector domain",
        "coalescing_replay_write_allocation_note": "write-allocation/fill behavior matters at the XBAR boundary, not for this L1 access-sector count",
        "comparison_geometry": "count_absolute_floor",
    },
    "l2_read_sectors_lg_op_ld": {
        "verdict": VERDICT_COMPARABLE,
        "domain_mismatch_reason": "post_l1_filter_xbar_traffic",
        "trace_to_ncu_mapping": "the L1-miss-filtered global load sector stream (xbar2l1tex read return/fill), NOT all load sectors",
        "aggregation_domain_mismatch": "comparable ONLY after the L1 miss-stream filter; this is post-L1 return/fill traffic",
        "coalescing_replay_write_allocation_note": "fidelity depends on the L1 filter model; the comparison is valid only on the filtered miss stream",
        "comparison_geometry": "count_absolute_floor",
    },
    "l2_write_sectors_lg_op_st": {
        "verdict": VERDICT_COMPARABLE,
        "domain_mismatch_reason": "write_allocation_sector_fill",
        "trace_to_ncu_mapping": "the L1->XBAR write sector stream for global stores",
        "aggregation_domain_mismatch": "post-L1 write traffic at the L1-to-XBAR boundary",
        "coalescing_replay_write_allocation_note": "comparable WITH CAUTION: must model .ca/.cg store forwarding/write policy; raw L1 store sectors may over/undercount if write allocation/coalescing differs",
        "comparison_geometry": "count_absolute_floor",
    },
    "dram_read_sectors": {
        "verdict": VERDICT_REFUTED,
        "domain_mismatch_reason": "post_l2_filter_dram_traffic",
        "trace_to_ncu_mapping": "no defensible mapping: per-kernel NVBit addresses cannot reach the chip-global DRAM domain",
        "aggregation_domain_mismatch": "DRAM is chip-global and after L2 fill/eviction/compression/system traffic",
        "coalescing_replay_write_allocation_note": "not applicable; L2 filtering + compression + system contamination dominate",
        "comparison_geometry": "none (refuted)",
    },
    "dram_write_sectors": {
        "verdict": VERDICT_REFUTED,
        "domain_mismatch_reason": "post_l2_filter_dram_traffic",
        "trace_to_ncu_mapping": "no defensible mapping from per-kernel store addresses",
        "aggregation_domain_mismatch": "chip-global DRAM after L2; writeback/write-combine/allocation timing makes store trace sectors non-isomorphic to DRAM writes",
        "coalescing_replay_write_allocation_note": "writeback/write-combine timing is not derivable from the trace",
        "comparison_geometry": "none (refuted)",
    },
    "gmem_ld_sectors_per_request": {
        "verdict": VERDICT_COMPARABLE,
        "domain_mismatch_reason": "none",
        "trace_to_ncu_mapping": "comparable L1 load sectors divided by comparable L1 load requests",
        "aggregation_domain_mismatch": "none; both numerator and denominator are L1 global-LSU load domains",
        "coalescing_replay_write_allocation_note": "replay mismatch can perturb NCU ratios; the trace-derived ratio is replay-stable",
        "comparison_geometry": "ratio_denominator_relative",
    },
    "gmem_st_sectors_per_request": {
        "verdict": VERDICT_COMPARABLE,
        "domain_mismatch_reason": "none",
        "trace_to_ncu_mapping": "comparable L1 store sectors divided by comparable L1 store requests",
        "aggregation_domain_mismatch": "none; do not substitute XBAR or DRAM sectors for the numerator",
        "coalescing_replay_write_allocation_note": "use the L1 store access-sector count, not the XBAR write count",
        "comparison_geometry": "ratio_denominator_relative",
    },
    "gmem_total_requests": {
        "verdict": VERDICT_COMPARABLE,
        "domain_mismatch_reason": "none",
        "trace_to_ncu_mapping": "sum of the comparable global-LSU load + store request counts",
        "aggregation_domain_mismatch": "none as a global-LSU ld+st total; would be refuted only if compared to the unified l1tex__t_requests.sum",
        "coalescing_replay_write_allocation_note": "inherits the component request counts' replay stability",
        "comparison_geometry": "count_absolute_floor",
    },
    "gmem_atom_red_requests": {
        "verdict": VERDICT_COMPARABLE,
        "domain_mismatch_reason": "none",
        "trace_to_ncu_mapping": "sum of the comparable global atomic + reduction request counts",
        "aggregation_domain_mismatch": "none for the global-LSU atom+red request domain",
        "coalescing_replay_write_allocation_note": "lane-level atomic/reduction fanout is not the request count",
        "comparison_geometry": "count_absolute_floor",
    },
    "gmem_total_sectors": {
        "verdict": VERDICT_COMPARABLE,
        "domain_mismatch_reason": "none",
        "trace_to_ncu_mapping": "sum of the comparable global-LSU load + store L1 access-sector counts",
        "aggregation_domain_mismatch": "none as a global-LSU ld+st L1 sector total; refuted if it claimed atom/red or unified L1 sectors without matching NCU domains",
        "coalescing_replay_write_allocation_note": "inherits the component sector counts",
        "comparison_geometry": "count_absolute_floor",
    },
    "l2_total_sectors": {
        "verdict": VERDICT_COMPARABLE,
        "domain_mismatch_reason": "post_l1_filter_xbar_traffic",
        "trace_to_ncu_mapping": "a modeled post-L1 miss-stream total for the tracked global traffic (xbar read + L1->xbar write sectors)",
        "aggregation_domain_mismatch": "comparable only as the modeled post-L1 miss-stream total; refuted against broad lts__*srcunit_tex* totals",
        "coalescing_replay_write_allocation_note": "fidelity depends on the L1 filter model + the write-policy caveat of its write component",
        "comparison_geometry": "count_absolute_floor",
    },
    "dram_total_sectors": {
        "verdict": VERDICT_REFUTED,
        "domain_mismatch_reason": "post_l2_filter_dram_traffic",
        "trace_to_ncu_mapping": "no defensible mapping; inherits the chip-global DRAM refutation of its read+write components",
        "aggregation_domain_mismatch": "chip-global DRAM after L2; the same contamination as its components",
        "coalescing_replay_write_allocation_note": "not applicable; refuted",
        "comparison_geometry": "none (refuted)",
    },
}


class CountFeasibilityBuildError(RuntimeError):
    pass


def _source_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return "0" * 40


def _count_ratio_specs() -> list[reg.MetricSpec]:
    """The registry's count + ratio targets (the fraction hit-rate family is
    excluded -- it is not a count target)."""
    return [s for s in reg.METRIC_SPECS if s.kind in (reg.KIND_COUNT, reg.KIND_RATIO)]


def _tokens_for(spec: reg.MetricSpec) -> list[str]:
    if spec.kind == reg.KIND_RATIO:
        return [spec.numerator_token, spec.denominator_token]
    if spec.ncu_token is not None:
        return [spec.ncu_token]
    # Derived count: name its component registry tokens.
    tokens: list[str] = []
    for component in spec.derived_from:
        comp = reg.REGISTRY[component]
        tokens.extend(_tokens_for(comp))
    return tokens


def build() -> dict:
    """Build the count-feasibility analysis from the registry + the analysis
    table. Fails closed if a registry count/ratio target has no analysis entry."""
    specs = _count_ratio_specs()
    targets = []
    for spec in specs:
        analysis = _ANALYSIS.get(spec.key)
        if analysis is None:
            raise CountFeasibilityBuildError(
                f"registry count/ratio target {spec.key!r} has no count-feasibility "
                f"analysis entry; refusing to emit an artifact that omits a target"
            )
        targets.append(
            {
                "key": spec.key,
                "kind": spec.kind,
                "ncu_tokens": _tokens_for(spec),
                "verdict": analysis["verdict"],
                "domain_mismatch_reason": analysis["domain_mismatch_reason"],
                "trace_to_ncu_mapping": analysis["trace_to_ncu_mapping"],
                "aggregation_domain_mismatch": analysis["aggregation_domain_mismatch"],
                "coalescing_replay_write_allocation_note": analysis[
                    "coalescing_replay_write_allocation_note"
                ],
                "comparison_geometry": analysis["comparison_geometry"],
            }
        )
    n_comparable = sum(1 for t in targets if t["verdict"] == VERDICT_COMPARABLE)
    n_refuted = sum(1 for t in targets if t["verdict"] == VERDICT_REFUTED)
    return {
        "schema_version": SCHEMA_VERSION,
        "registry_version": reg.REGISTRY_VERSION,
        "source_commit": _source_commit(),
        "codex_consult_reference": CODEX_CONSULT_REFERENCE,
        "analysis_basis": ANALYSIS_BASIS,
        "targets": targets,
        "summary": {
            "n_targets": len(targets),
            "n_comparable": n_comparable,
            "n_refuted": n_refuted,
        },
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "data" / "local_labels" / "count_feasibility_analysis.json",
    )
    args = parser.parse_args(argv)
    doc = build()
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({doc['summary']['n_comparable']} comparable / {doc['summary']['n_refuted']} refuted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
