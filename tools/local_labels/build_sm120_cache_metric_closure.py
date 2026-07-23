"""Producer for the sm_120 cache-metric prediction terminal closure.

This records ONE terminal verdict for every cache/memory metric in the typed
registry (``tools/local_labels/ncu_metric_registry.py``), closing the broader
counter-free cache-metric prediction effort by research-success OR honest
engineering-closure falsification:

  - The NAMED required-success family -- the L2 sector hit rate
    (``lts_sector_hit_rate`` / ``lts__t_sector_hit_rate.pct``) -- is evaluated on
    the L1-MISS-FILTERED replay (the normalization the per-SM L1 filter was built
    for) over the committed sm_120 cohort, scored against the live NCU labels.
    Its verdict is RECOMPUTED from the committed comparison artifact
    (``sm120_l1_filtered_l2_comparison.json``): research_success iff the
    full-cohort median absolute error clears the accuracy gate, else
    engineering_closure_refuted with the failure decomposition. On this cohort the
    median absolute error is ~0.28 -- far above the 0.055 gate -- and the dominant
    cause is that the sm_86-derived SASS cache-op classifier maps ZERO sm_120
    loads to L1-eligible ``.ca`` (Blackwell's descriptor-based / ``.CONSTANT`` /
    uniform-datapath load forms are outside its vocabulary), so the per-SM L1
    filter is INERT and the predictor cannot be exercised as designed. The named
    family therefore closes by engineering_closure_refuted.

  - The unified L1 sector hit rate (``l1tex_sector_hit_rate`` /
    ``l1tex__t_sector_hit_rate.pct``) HAS a committed live label cohort
    (``cache_hit_rate_l1tex_sector_pct`` on all 30 sm_120 cells), so it is
    evaluated against it on the L1-filtered replay's L1 leg. The same
    classification inertness applies (zero L1-eligible accesses -> no evaluable
    denominator), so it closes engineering_closure_refuted with the live labels
    acknowledged, NOT under a no-label basis. The global-load-specific L1 hit rate
    (``l1tex_global_op_ld_sector_hit_rate``) has NO committed label field, so its
    no-label closure is EARNED by proving that absence against
    ``label_admitted.json``.

  - Every count / ratio target inherits its terminal verdict from the committed
    count-feasibility analysis (``count_feasibility_analysis.json``): a
    domain-mismatch-refuted count is engineering_closure_refuted; a
    trace-count-comparable count still closes by engineering_closure_refuted with
    basis ``comparable_mapping_no_label_cohort`` because no NCU count label cohort
    was collected to score it against (counts may all close by falsification this
    loop; a count-appropriate label cohort + gate promotes them later).

The overall cache claim is load-bearing ONLY if the named family reaches a
load-bearing predictor; here it does not, so ``cache_claim_load_bearing`` is
false and the overall ``verdict`` is engineering_closure_refuted. This producer
RECOMPUTES every verdict from the cited committed evidence + the registry; it
reads VALUES only and invokes no NCU.
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

import ncu_metric_registry as registry  # noqa: E402

SCHEMA_VERSION = 1
CLOSURE_ID = "sm120_cache_metric_closure_v1"
ARCH = "sm_120"
REGISTRY_VERSION = registry.REGISTRY_VERSION

VERDICT_REFUTED = "engineering_closure_refuted"
VERDICT_SUCCESS = "research_success"

# The load-bearing accuracy-gate criterion for a hit-rate predictor: full-cohort
# median absolute error <= this. Mirrors
# tools/cache_replay/accuracy_gate.FULL_COHORT_MEDIAN_ABS_MAX (0.055); pinned here
# so the closure does not import the replay harness, and asserted equal below.
GATE_MEDIAN_ABS_THRESHOLD = 0.055

NAMED_TARGET = "lts_sector_hit_rate"
L2_INPUT_L1_FILTERED = "l1_miss_filtered"

# The L1 sector hit-rate target evaluated against the live committed L1TEX label
# cohort (the unified l1tex__t_sector_hit_rate.pct, stored as
# cache_hit_rate_l1tex_sector_pct in label_admitted.json) on the L1-filtered
# replay's L1 leg.
NAMED_L1_TARGET = "l1tex_sector_hit_rate"
L1TEX_UNIFIED_LABEL_FIELD = "cache_hit_rate_l1tex_sector_pct"
# The global-load-specific L1 hit-rate token (l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct)
# has NO committed label field; its no-label closure is EARNED by proving absence.
L1_GLOBAL_OP_LD_TARGET = "l1tex_global_op_ld_sector_hit_rate"
L1_GLOBAL_OP_LD_LABEL_FIELD = "cache_hit_rate_l1tex_global_op_ld_sector_pct"

# count-feasibility verdict -> (terminal verdict, basis)
_COUNT_FEASIBILITY_TO_TERMINAL = {
    "engineering_closure_refuted": (VERDICT_REFUTED, "aggregation_domain_mismatch"),
    "trace_count_comparable": (VERDICT_REFUTED, "comparable_mapping_no_label_cohort"),
}

COMPARISON_PATH = "data/local_labels/sm120_l1_filtered_l2_comparison.json"
COUNT_FEASIBILITY_PATH = "data/local_labels/count_feasibility_analysis.json"
L1L2_FROZEN_GRAIN_CLOSURE_PATH = "data/local_labels/sm120_l1l2_frozen_grain_engineering_closure.json"
LABEL_ADMITTED_PATH = "data/local_labels/label_admitted.json"

EVIDENCE_PATHS = {
    "l1_filtered_l2_comparison": COMPARISON_PATH,
    "count_feasibility_analysis": COUNT_FEASIBILITY_PATH,
    "l1l2_frozen_grain_closure": L1L2_FROZEN_GRAIN_CLOSURE_PATH,
    "label_admitted": LABEL_ADMITTED_PATH,
}

# evidence key each target kind cites for its terminal verdict.
EVIDENCE_NAMED = "l1_filtered_l2_comparison"
EVIDENCE_LABEL_ADMITTED = "label_admitted"
EVIDENCE_COUNT = "count_feasibility_analysis"

VERIFIED_LIVE_SM120_PROOF = {
    "gpu": "NVIDIA GeForce RTX 5060 Laptop GPU",
    "compute_capability": "sm_120",
    "driver": "595.71.05",
    "cuda": "13.2",
    "ncu": "2026.1.1.0",
    "evidence_kind": (
        "live NCU L2 sector hit-rate labels + live NVBit sm_120 traces; the "
        "predictor is a deterministic L1-miss-filtered replay over those committed "
        "live traces (no NCU invoked by this closure)"
    ),
}


class CacheMetricClosureBuildError(RuntimeError):
    pass


def _load(repo_root: Path, rel: str) -> dict:
    return json.loads((repo_root / rel).read_text(encoding="utf-8"))


def _source_commit(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return "0" * 40


def _named_target_gate(comparison: dict) -> dict:
    """Recompute the named L2 family's accuracy-gate outcome from the committed
    comparison artifact. The L2 leg MUST be the L1-miss-filtered stream."""
    if comparison.get("l2_input_provenance") != L2_INPUT_L1_FILTERED:
        raise CacheMetricClosureBuildError(
            f"comparison l2_input_provenance "
            f"{comparison.get('l2_input_provenance')!r} is not "
            f"{L2_INPUT_L1_FILTERED!r}; a full-stream L2 cannot back the named "
            f"family on the L1/L2 headline path"
        )
    agg = comparison.get("aggregates", {})
    median_abs = agg.get("median_abs_error")
    n_evaluated = comparison.get("n_evaluated", 0)
    if not isinstance(median_abs, (int, float)) or isinstance(median_abs, bool):
        raise CacheMetricClosureBuildError(
            f"comparison median_abs_error {median_abs!r} is not a number"
        )
    if not isinstance(n_evaluated, int) or n_evaluated <= 0:
        raise CacheMetricClosureBuildError(
            f"comparison n_evaluated {n_evaluated!r} is not a positive int"
        )
    passed = median_abs <= GATE_MEDIAN_ABS_THRESHOLD
    return {
        "median_abs_error": median_abs,
        "threshold_median_abs_error": GATE_MEDIAN_ABS_THRESHOLD,
        "passed": passed,
        "l2_input_provenance": L2_INPUT_L1_FILTERED,
        "n_evaluated": n_evaluated,
        "l1_filter_inert": bool(agg.get("l1_filter_inert")),
        "total_l1_eligible_misses": agg.get("total_l1_eligible_misses"),
        "total_l1_hits_absorbed": agg.get("total_l1_hits_absorbed"),
        "cells_with_zero_l2_input": agg.get("cells_with_zero_l2_input"),
    }


def _sm120_l1_label_fields(repo_root: Path) -> set:
    """The set of L1 hit-rate label field names actually present on sm_120 cells
    in the committed label cohort. Used to PROVE which L1 fraction targets have a
    live label cohort (so a no-label closure basis is earned, not assumed)."""
    fields: set = set()
    for c in _load(repo_root, LABEL_ADMITTED_PATH).get("cells", []):
        if c.get("arch") != ARCH:
            continue
        for k in c:
            if k.startswith("cache_hit_rate_l1tex"):
                fields.add(k)
    return fields


def _l1_target_gate(comparison: dict) -> dict:
    """Recompute the L1 sector-hit-rate (l1tex_sector_hit_rate) accuracy-gate
    outcome from the committed comparison's L1 leg. The L1 predictor is
    load-bearing only if some cells are EVALUABLE (non-zero L1-eligible denominator)
    and the median absolute error clears the gate; when the classifier is inert no
    cell is evaluable, so the target closes by classifier-inert falsification while
    still acknowledging the live label cohort."""
    agg = comparison.get("aggregates", {})
    evaluated = agg.get("l1_evaluated", 0)
    median_abs = agg.get("l1_median_abs_error")  # None when evaluated == 0
    passed = (
        isinstance(evaluated, int) and evaluated > 0
        and isinstance(median_abs, (int, float)) and not isinstance(median_abs, bool)
        and median_abs <= GATE_MEDIAN_ABS_THRESHOLD
    )
    return {
        "label_coverage": agg.get("l1_label_coverage", 0),
        "evaluated": evaluated,
        "unavailable": agg.get("l1_unavailable", 0),
        "median_abs_error": median_abs,
        "threshold_median_abs_error": GATE_MEDIAN_ABS_THRESHOLD,
        "passed": passed,
        "l1_filter_inert": bool(agg.get("l1_filter_inert")),
    }


def _count_feasibility_verdicts(count_feasibility: dict) -> dict:
    out = {}
    for t in count_feasibility.get("targets", []):
        out[t["key"]] = t["verdict"]
    return out


def _l1_sector_hit_rate_record(l1_gate: dict) -> tuple[str, str]:
    """Terminal (verdict, basis) for l1tex_sector_hit_rate from its L1-leg gate."""
    if l1_gate["passed"]:
        return VERDICT_SUCCESS, "l1_filtered_replay_l1_sector_hit_rate_load_bearing"
    if l1_gate.get("evaluated", 0) == 0 and l1_gate.get("l1_filter_inert"):
        return VERDICT_REFUTED, "l1_filtered_replay_l1_sector_hit_rate_refuted_l1_classification_inert"
    return VERDICT_REFUTED, "l1_filtered_replay_l1_sector_hit_rate_refuted_gate_failed"


def _target_records(gate: dict, l1_gate: dict, l1_label_fields: set, cf_verdicts: dict) -> list[dict]:
    """One terminal record per registry target, recomputed from the registry +
    cited evidence. Fails closed if a non-fraction target has no count-feasibility
    verdict (the registry and the count-feasibility analysis must agree on the
    count/ratio target set), or if an L1 fraction with a committed live label
    cohort is closed under a no-label basis."""
    records = []
    for key in sorted(registry.REGISTRY):
        spec = registry.REGISTRY[key]
        if key == NAMED_TARGET:
            verdict = VERDICT_SUCCESS if gate["passed"] else VERDICT_REFUTED
            basis = (
                "l1_filtered_replay_accuracy_gate_passed" if gate["passed"]
                else "l1_filtered_replay_accuracy_gate_failed_l1_classification_inert"
            )
            evidence_key = EVIDENCE_NAMED
        elif key == NAMED_L1_TARGET:
            # This target HAS a committed live L1TEX label cohort; it must be
            # evaluated/falsified against it, never closed under a no-label basis.
            if L1TEX_UNIFIED_LABEL_FIELD not in l1_label_fields:
                raise CacheMetricClosureBuildError(
                    f"{NAMED_L1_TARGET!r} has no committed {L1TEX_UNIFIED_LABEL_FIELD!r} "
                    f"label cohort but is being evaluated against it"
                )
            verdict, basis = _l1_sector_hit_rate_record(l1_gate)
            evidence_key = EVIDENCE_NAMED
        elif key == L1_GLOBAL_OP_LD_TARGET:
            # No-label closure is EARNED only by proving the global-op-ld-specific
            # token has no committed label field.
            if L1_GLOBAL_OP_LD_LABEL_FIELD in l1_label_fields:
                raise CacheMetricClosureBuildError(
                    f"{L1_GLOBAL_OP_LD_TARGET!r} DOES have a committed label cohort "
                    f"({L1_GLOBAL_OP_LD_LABEL_FIELD!r}); it cannot use a no-label basis"
                )
            verdict = VERDICT_REFUTED
            basis = "no_committed_global_op_ld_l1_label_cohort"
            evidence_key = EVIDENCE_LABEL_ADMITTED
        else:
            cf = cf_verdicts.get(key)
            if cf is None:
                raise CacheMetricClosureBuildError(
                    f"registry count/ratio target {key!r} has no count-feasibility "
                    f"verdict; registry and count_feasibility_analysis disagree"
                )
            if cf not in _COUNT_FEASIBILITY_TO_TERMINAL:
                raise CacheMetricClosureBuildError(
                    f"count-feasibility verdict {cf!r} for {key!r} is not a known "
                    f"terminal mapping"
                )
            verdict, basis = _COUNT_FEASIBILITY_TO_TERMINAL[cf]
            evidence_key = EVIDENCE_COUNT
        records.append({
            "target_id": key,
            "kind": spec.kind,
            "verdict": verdict,
            "basis": basis,
            "evidence_key": evidence_key,
            "cross_arch_precondition": False,
        })
    return records


def build(repo_root: Path = _REPO_ROOT) -> dict:
    if GATE_MEDIAN_ABS_THRESHOLD != 0.055:
        raise CacheMetricClosureBuildError("gate threshold drifted from 0.055")
    if registry.validate_registry():
        raise CacheMetricClosureBuildError("the metric registry is not internally consistent")

    comparison = _load(repo_root, COMPARISON_PATH)
    count_feasibility = _load(repo_root, COUNT_FEASIBILITY_PATH)
    if count_feasibility.get("registry_version") != REGISTRY_VERSION:
        raise CacheMetricClosureBuildError(
            "count_feasibility registry_version drift vs the registry"
        )

    gate = _named_target_gate(comparison)
    l1_gate = _l1_target_gate(comparison)
    l1_label_fields = _sm120_l1_label_fields(repo_root)
    cf_verdicts = _count_feasibility_verdicts(count_feasibility)
    targets = _target_records(gate, l1_gate, l1_label_fields, cf_verdicts)

    cache_claim_load_bearing = gate["passed"]
    overall_verdict = VERDICT_SUCCESS if cache_claim_load_bearing else VERDICT_REFUTED

    n_refuted = sum(1 for t in targets if t["verdict"] == VERDICT_REFUTED)
    n_success = sum(1 for t in targets if t["verdict"] == VERDICT_SUCCESS)

    decomposition = {
        "dominant_failure_mode": "sm120_l1_cache_op_classification_inert",
        "l1_filter_inert": gate["l1_filter_inert"],
        "total_l1_eligible_misses": gate["total_l1_eligible_misses"],
        "total_l1_hits_absorbed": gate["total_l1_hits_absorbed"],
        "cells_with_zero_l2_input": gate["cells_with_zero_l2_input"],
        "evidence": (
            "Across all evaluated sm_120 cohort cells the cache-op classifier maps "
            "ZERO loads to L1-eligible (.ca): total L1-eligible misses and L1 hits "
            "absorbed are both 0, so the per-SM L1 filter is inert. This is the "
            "DOCUMENTED conservative behavior of the arch-coupling discipline, not a "
            "defect: l1_cache_op_classifier._BARE_LDG_E_DEFAULTS_TO_CA admits only "
            "sm_86 (where bare LDG.E resolves to the ISA-default .CA), and every "
            "other arch -- sm_120 included -- keeps bare LDG.E -> OTHER (ineligible) "
            "'until their default modifier is explicitly documented and tested'. The "
            "Blackwell loads observed (LDG.E.U16.CONSTANT, descriptor-based "
            "desc[UR..]) match the bare-LDG.E rule and route to OTHER -> bypass, or "
            "to the unmodeled-drop path (4 cells reach L2 with zero accesses -> a "
            "vacuous 0.0). The L1-miss-filtered L2 predictor therefore cannot be "
            "exercised as designed on sm_120, and the median absolute error stays far "
            "above the gate."
        ),
        "promotion_trigger": (
            "Add sm_120 to l1_cache_op_classifier._BARE_LDG_E_DEFAULTS_TO_CA (and "
            "model the .CONSTANT / descriptor-based / uniform-datapath load forms) "
            "with documented + tested Blackwell cache-op semantics, so sm_120 loads "
            "acquire .ca/.cg eligibility and the per-SM L1 filter is exercised; then "
            "re-run the L1-miss-filtered evaluation. The named family becomes "
            "load-bearing only if the median absolute error then clears the gate."
        ),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "closure_id": CLOSURE_ID,
        "arch": ARCH,
        "registry_version": REGISTRY_VERSION,
        "verdict": overall_verdict,
        "cache_claim_load_bearing": cache_claim_load_bearing,
        "named_target": NAMED_TARGET,
        "named_target_gate": gate,
        "named_l1_target": NAMED_L1_TARGET,
        "named_l1_target_gate": l1_gate,
        "named_target_failure_decomposition": decomposition,
        "evidence_artifact_paths": dict(EVIDENCE_PATHS),
        "targets": targets,
        "target_counts": {
            "total": len(targets),
            "research_success": n_success,
            "engineering_closure_refuted": n_refuted,
        },
        "verified_live_sm120_proof": dict(VERIFIED_LIVE_SM120_PROOF),
        "source_commit": _source_commit(repo_root),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path,
        default=_REPO_ROOT / "data" / "local_labels" / "sm120_cache_metric_closure.json",
    )
    args = parser.parse_args(argv)
    doc = build()
    args.out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    c = doc["target_counts"]
    print(
        f"wrote {args.out} (verdict={doc['verdict']} "
        f"cache_claim_load_bearing={doc['cache_claim_load_bearing']} "
        f"targets: {c['research_success']} success / {c['engineering_closure_refuted']} refuted / {c['total']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
