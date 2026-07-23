#!/usr/bin/env python3
"""NCU lockstep validator: the per-(cell, metric) T1 count-replacement
verdict on sm_86 (quarantine).

Joins the committed replay-count emissions to the committed count
labels by the CLOSED identity tuple only (binary_sha256 +
kernel_signature_sha256 + slug + shape_point + variant), and emits one
closed-vocabulary verdict per (admitted cell, count metric):

    replacement_pass | replacement_miss | low_support |
    contract_mismatch | rollup_unsupported |
    trace_support_insufficient | label_count_unstable |
    label_unavailable

Producer-side typed reasons are RECONCILED into this vocabulary
without widening the join: a cell-level producer rejection
(trace_binary_stale / kernel_signature_mismatch / trace_absent ...)
maps to ``trace_support_insufficient`` with the producer reason
carried verbatim in ``detail`` and aggregated in the coverage report;
per-metric producer typing maps per ``_PRODUCER_METRIC_REASON_MAP``.
Label-side gaps map to ``label_count_unstable`` (the committed
capture-evidence exclusions) or ``label_unavailable``.

Statistics: relative error |produced - label| / max(label,
denominator_floor) per scored pair; p50/p90/max per metric and per
op-class; directional targets reported achieved-vs-target (a MISS is
an honest typed outcome).

HARD non-vacuity floor: >= 10 scored pairs spanning >= 5 distinct
slugs, with ld/st + atom-vs-red + vector-width family coverage and
>= 4 of the 8 confirmed metrics scored on >= 5 cells each. Scored
means a real label joined to a real producer count; typed exclusions
never count. A below-floor run cannot claim ANY replacement_pass
(would-be passes are demoted to ``low_support`` with a vacuity
detail).

Sealed-holdout governance (the AC the count-label capture deferred to
this validator): calibration/validation cells score from the admitted
store; holdout cells score ONLY from the sealed file in the single
final evaluation, recorded in the digest-keyed holdout-touch ledger
(touch count == 1). The validator has NO tunable parameters fit to
labels; the tuning trail records exactly that, and the
retune-after-touch guard binds the config digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import holdout_touch_ledger as htl  # noqa: E402
from capture_count_labels import (  # noqa: E402
    QuarantinePathError,
    assert_quarantine_path,
    count_metric_identifiers,
)

VERDICT_VOCABULARY = frozenset({
    "replacement_pass", "replacement_miss", "low_support",
    "contract_mismatch", "rollup_unsupported",
    "trace_support_insufficient", "label_count_unstable",
    "label_unavailable",
})

# The validator configuration is FROZEN data, not tuning: nothing here
# was fit to any label. Its sha256 is the config digest the
# retune-after-touch guard binds.
VALIDATOR_CONFIG = {
    "pair_pass_max_relative_error": 0.05,
    "directional_targets": {"p50": 0.01, "p90": 0.03, "max": 0.05},
    "relative_error_denominator_floor": 1.0,
    "low_support_label_floor": 10.0,
    "non_vacuity_floor": {
        "min_scored_pairs": 10,
        "min_distinct_slugs": 5,
        "min_confirmed_metrics_with_min_cells": 4,
        "min_cells_per_confirmed_metric": 5,
        "required_families": ["ld_st", "atom_vs_red", "vector_width"],
    },
    # The vector-width factorial family's slugs (32/64/128-bit coalesced
    # copies) — corpus facts, not tuning.
    "vector_width_family_slugs": [
        "global_scalar_32b_copy",
        "global_vectorized_64b_copy",
        "global_vectorized_128b_copy",
    ],
}

# The 8 pre-probe confirmed metrics (the registry's original available
# set; the floor's ">= 4 of the 8 confirmed metrics" clause).
CONFIRMED_8 = (
    "l1tex__t_requests.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "smsp__inst_executed.sum",
)

_M_LD = "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"
_M_ST = "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum"
_M_ATOM = "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum"
_M_RED = "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum"

# Producer per-metric typed reason -> closed verdict vocabulary.
_PRODUCER_METRIC_REASON_MAP = {
    "rollup_unsupported": "rollup_unsupported",
    "lane_reconstruction_failed": "trace_support_insufficient",
    "no_expressible_fixture": "contract_mismatch",
    "sector_scope_misaligned_footprint": "contract_mismatch",
    "async_copy_unfixtured": "contract_mismatch",
    "surface_kind_wire_unresolved": "contract_mismatch",
    "atom_red_sass_unresolved": "contract_mismatch",
    "contract_entry_missing": "contract_mismatch",
    "contract_mismatch": "contract_mismatch",
    "contract_underdetermined": "contract_mismatch",
}

# Op-class grouping for the per-op-class statistics.
_OP_CLASS_OF_METRIC = {
    "l1tex__t_requests.sum": "rollup",
    _M_ATOM: "global_atomic",
    _M_LD: "global_load",
    _M_RED: "global_reduction",
    _M_ST: "global_store",
    "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum": "local",
    "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum": "local",
    "l1tex__t_requests_pipe_tex_mem_surface_op_atom.sum": "surface",
    "l1tex__t_requests_pipe_tex_mem_surface_op_ld.sum": "surface",
    "l1tex__t_requests_pipe_tex_mem_surface_op_red.sum": "surface",
    "l1tex__t_requests_pipe_tex_mem_surface_op_st.sum": "surface",
    "l1tex__t_requests_pipe_tex_mem_texture.sum": "texture",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "global_sectors",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": "global_sectors",
    "smsp__inst_executed.sum": "instructions",
}

DEFAULT_VERDICT_OUT = "data/local_labels/t1_lockstep_verdict_sm_86.json"
DEFAULT_EMISSIONS = "data/local_labels/replay_count_emissions_sm_86.json"
DEFAULT_LABELS = "data/local_labels/label_admitted.json"
DEFAULT_COUNT_EXCLUSIONS = (
    "data/local_labels/count_label_exclusions_sm_86.json"
)
DEFAULT_SEALED = (
    "data/sm_86_predictor_accuracy_recovery_v2/sealed_holdout_labels.json"
)
DEFAULT_SPLIT = "data/sm_86_predictor_accuracy_recovery_v2/cohort_split.json"


class WeakIdentityJoinError(RuntimeError):
    typed_reason = "weak_identity_join_rejected"


def config_digest() -> str:
    return hashlib.sha256(
        json.dumps(VALIDATOR_CONFIG, sort_keys=True,
                   separators=(",", ":")).encode()
    ).hexdigest()


def _cell_key(c: dict) -> tuple:
    return (c["slug"], c["shape_point"], c.get("variant_id") or "")


def _identity_equal(a: dict, b: dict) -> bool:
    """Closed identity: every component must match; a comparison over
    anything weaker is a programming error, not a fallback."""
    fields = ("slug", "shape_point", "binary_sha256",
              "kernel_signature_sha256")
    if any(f not in a or f not in b for f in fields):
        raise WeakIdentityJoinError(
            "closed-identity comparison requires slug, shape_point, "
            "binary_sha256, and kernel_signature_sha256 on both sides"
        )
    return all(a[f] == b[f] for f in fields) and (
        (a.get("variant_id") or "") == (b.get("variant_id") or "")
    )


def relative_error(produced: float, label: float, floor: float) -> float:
    return abs(produced - label) / max(label, floor)


def _quantiles(errors: list[float]) -> dict:
    if not errors:
        return {"n": 0, "p50": None, "p90": None, "max": None}
    ordered = sorted(errors)
    return {
        "n": len(errors),
        "p50": statistics.quantiles(ordered, n=100)[49]
        if len(ordered) > 1 else ordered[0],
        "p90": statistics.quantiles(ordered, n=100)[89]
        if len(ordered) > 1 else ordered[0],
        "max": ordered[-1],
    }


def build_verdicts(
    emissions: dict,
    labels_by_key: dict[tuple, dict],
    holdout_keys: frozenset[tuple],
    sealed_by_key: dict[tuple, dict],
    count_excluded_keys: dict[tuple, dict],
    metrics: tuple[str, ...],
    config: dict | None = None,
) -> dict:
    """Pure verdict construction (no filesystem writes): returns the
    artifact body sans governance/identity stamps."""
    cfg = config or VALIDATOR_CONFIG
    floor_cfg = cfg["non_vacuity_floor"]
    denom_floor = cfg["relative_error_denominator_floor"]

    verdict_rows: list[dict] = []
    errors_by_metric: dict[str, list[float]] = defaultdict(list)
    errors_by_op_class: dict[str, list[float]] = defaultdict(list)
    scored_pairs: list[tuple] = []
    scored_slugs: set[str] = set()
    cells_scored_by_metric: dict[str, set[tuple]] = defaultdict(set)
    coverage_cell_reasons: dict[str, int] = defaultdict(int)
    family_ld_st_cells: set[tuple] = set()
    family_atom_red_cells: set[tuple] = set()
    family_vector_slugs: set[str] = set()

    for record in emissions["cells"]:
        ident = record["identity"]
        key = (ident["slug"], ident["shape_point"],
               ident.get("variant_id") or "")
        is_holdout = key in holdout_keys
        label_cell = (sealed_by_key if is_holdout else labels_by_key).get(
            key)
        cell_status = record["cell_status"]
        label_metrics = {}
        if label_cell is not None:
            if not _identity_equal(ident, label_cell):
                # The label store moved relative to the emissions —
                # fail-closed typed row, never a weak join.
                for metric in metrics:
                    verdict_rows.append({
                        "slug": key[0], "shape_point": key[1],
                        **({"variant_id": key[2]} if key[2] else {}),
                        "holdout": is_holdout,
                        "ncu_metric": metric,
                        "verdict": "trace_support_insufficient",
                        "detail": "emission identity != label identity "
                                  "(stale emissions artifact)",
                    })
                coverage_cell_reasons["emission_label_identity_drift"] += 1
                continue
            label_metrics = {
                m["ncu_metric"]: m["value"]
                for m in label_cell.get("unit_tagged_metrics") or []
                if m["ncu_metric"] in metrics
            }
        if cell_status.get("status") != "emitted":
            reason = cell_status.get("typed_reason", "unknown")
            coverage_cell_reasons[reason] += 1
            for metric in metrics:
                verdict_rows.append({
                    "slug": key[0], "shape_point": key[1],
                    **({"variant_id": key[2]} if key[2] else {}),
                    "holdout": is_holdout,
                    "ncu_metric": metric,
                    "verdict": "trace_support_insufficient",
                    "detail": f"producer cell-level typed: {reason}",
                })
            continue
        emitted = record["metrics"]
        for metric in metrics:
            em = emitted.get(metric)
            row = {
                "slug": key[0], "shape_point": key[1],
                **({"variant_id": key[2]} if key[2] else {}),
                "holdout": is_holdout,
                "ncu_metric": metric,
            }
            if em is None:
                row["verdict"] = "trace_support_insufficient"
                row["detail"] = "no producer emission for this metric"
                verdict_rows.append(row)
                continue
            if em["status"] == "typed":
                reason = em["typed_reason"]
                row["verdict"] = _PRODUCER_METRIC_REASON_MAP.get(
                    reason, "contract_mismatch")
                row["detail"] = f"producer typed: {reason}"
                verdict_rows.append(row)
                continue
            label_value = label_metrics.get(metric)
            if label_value is None:
                if key in count_excluded_keys:
                    row["verdict"] = "label_count_unstable"
                    row["detail"] = (
                        "cell is capture-evidence-typed in the count "
                        "exclusions artifact"
                    )
                else:
                    row["verdict"] = "label_unavailable"
                    row["detail"] = "no committed count label"
                verdict_rows.append(row)
                continue
            produced = em["value"]
            rel = relative_error(produced, label_value, denom_floor)
            row["produced"] = produced
            row["label"] = label_value
            row["relative_error"] = rel
            scored_pairs.append((key, metric))
            scored_slugs.add(key[0])
            cells_scored_by_metric[metric].add(key)
            errors_by_metric[metric].append(rel)
            errors_by_op_class[
                _OP_CLASS_OF_METRIC.get(metric, "other")].append(rel)
            if rel <= cfg["pair_pass_max_relative_error"]:
                row["verdict"] = "replacement_pass"
            elif label_value < cfg["low_support_label_floor"]:
                row["verdict"] = "low_support"
                row["detail"] = (
                    f"label {label_value} below the support floor "
                    f"{cfg['low_support_label_floor']}; the relative "
                    "error is not meaningful at this magnitude"
                )
            else:
                row["verdict"] = "replacement_miss"
            verdict_rows.append(row)
            if metric == _M_LD and _M_ST in label_metrics:
                family_ld_st_cells.add(key)
            if metric == _M_ATOM and _M_RED in label_metrics:
                family_atom_red_cells.add(key)
            if key[0] in cfg["vector_width_family_slugs"]:
                family_vector_slugs.add(key[0])

    # --- HARD non-vacuity floor ---
    confirmed_with_min_cells = [
        m for m in CONFIRMED_8
        if len(cells_scored_by_metric.get(m, ()))
        >= floor_cfg["min_cells_per_confirmed_metric"]
    ]
    families = {
        "ld_st": len(family_ld_st_cells) >= 1,
        "atom_vs_red": len(family_atom_red_cells) >= 1,
        "vector_width": len(family_vector_slugs) >= 2,
    }
    floor_checks = {
        "scored_pairs": len(scored_pairs),
        "scored_pairs_ok": len(scored_pairs)
        >= floor_cfg["min_scored_pairs"],
        "distinct_slugs": len(scored_slugs),
        "distinct_slugs_ok": len(scored_slugs)
        >= floor_cfg["min_distinct_slugs"],
        "families": families,
        "families_ok": all(
            families[f] for f in floor_cfg["required_families"]),
        "confirmed_metrics_with_min_cells": sorted(
            confirmed_with_min_cells),
        "confirmed_metrics_ok": len(confirmed_with_min_cells)
        >= floor_cfg["min_confirmed_metrics_with_min_cells"],
    }
    floor_met = all(v for k, v in floor_checks.items()
                    if k.endswith("_ok"))
    floor_checks["floor_met"] = floor_met

    if not floor_met:
        # Vacuity guard: a below-floor run cannot claim ANY
        # replacement_pass.
        for row in verdict_rows:
            if row["verdict"] == "replacement_pass":
                row["verdict"] = "low_support"
                row["detail"] = (
                    "vacuity_floor_unmet: the run is below the hard "
                    "non-vacuity floor; no replacement_pass may be "
                    "claimed"
                )

    # --- achieved vs target ---
    targets = cfg["directional_targets"]
    per_metric_stats = {
        m: _quantiles(errs) for m, errs in sorted(errors_by_metric.items())
    }
    per_op_class_stats = {
        c: _quantiles(errs)
        for c, errs in sorted(errors_by_op_class.items())
    }
    overall = _quantiles(
        [e for errs in errors_by_metric.values() for e in errs])
    achieved_vs_target = {
        "targets": targets,
        "achieved": overall,
        "met": {
            "p50": (overall["p50"] is not None
                    and overall["p50"] <= targets["p50"]),
            "p90": (overall["p90"] is not None
                    and overall["p90"] <= targets["p90"]),
            "max": (overall["max"] is not None
                    and overall["max"] <= targets["max"]),
        } if overall["n"] else {"p50": False, "p90": False, "max": False},
    }

    verdict_counts: dict[str, int] = defaultdict(int)
    for row in verdict_rows:
        verdict_counts[row["verdict"]] += 1

    return {
        "verdicts": verdict_rows,
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "statistics": {
            "per_metric": per_metric_stats,
            "per_op_class": per_op_class_stats,
            "overall": overall,
            "achieved_vs_target": achieved_vs_target,
        },
        "non_vacuity_floor": floor_checks,
        "coverage_report": {
            "cell_level_producer_reasons": dict(
                sorted(coverage_cell_reasons.items())),
            "scored_cells_per_metric": {
                m: len(cells)
                for m, cells in sorted(cells_scored_by_metric.items())
            },
            "family_evidence": {
                "ld_st_cells": len(family_ld_st_cells),
                "atom_vs_red_cells": len(family_atom_red_cells),
                "vector_width_slugs": sorted(family_vector_slugs),
            },
            "case_coverage": {
                "active_mask_and_predication": (
                    "scored GLOBAL/LOCAL pairs derive from "
                    "effective-mask (active & predicate) wire walks, "
                    "pinned by the lane-subset and inst-pred fixtures; "
                    "ATOMIC/TEXTURE/SURFACE counts derive from the "
                    "exclusion sidecar whose predication behavior is "
                    "unfixtured (independent-review scope note)"
                ),
                "vector_width": sorted(family_vector_slugs),
                "atom_vs_red": len(family_atom_red_cells),
                "local_memory": len(
                    cells_scored_by_metric.get(
                        "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum",
                        ())),
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emissions", default=DEFAULT_EMISSIONS)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--count-exclusions",
                        default=DEFAULT_COUNT_EXCLUSIONS)
    parser.add_argument("--sealed-holdout", default=DEFAULT_SEALED)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--out", default=DEFAULT_VERDICT_OUT)
    parser.add_argument("--ledger", default=None,
                        help="holdout-touch ledger path (default: the "
                             "committed ledger)")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    try:
        assert_quarantine_path(out_path)
    except QuarantinePathError as exc:
        print(json.dumps({"typed_reason": exc.typed_reason,
                          "detail": str(exc)}), file=sys.stderr)
        return 3

    emissions = json.loads(
        (REPO_ROOT / args.emissions).read_text(encoding="utf-8"))
    labels = json.loads(
        (REPO_ROOT / args.labels).read_text(encoding="utf-8"))
    count_excl = json.loads(
        (REPO_ROOT / args.count_exclusions).read_text(encoding="utf-8"))
    sealed = json.loads(
        (REPO_ROOT / args.sealed_holdout).read_text(encoding="utf-8"))
    split = json.loads(
        (REPO_ROOT / args.split).read_text(encoding="utf-8"))

    labels_by_key = {
        _cell_key(c): c for c in labels["cells"]
        if c.get("arch") == "sm_86"
    }
    sealed_by_key = {_cell_key(c): c for c in sealed["cells"]}
    holdout_keys = frozenset(
        (c["slug"], c["shape_point"],
         (c["identity_tuple"].get("variant_id") or "")
         if isinstance(c.get("identity_tuple"), dict) else "")
        for c in split["splits"]["holdout"]["cells"]
    )
    count_excluded_keys = {
        (x["slug"], x["shape_point"], x.get("variant_id") or ""): x
        for x in count_excl["exclusions"] if "ncu_metric" not in x
    }
    metrics = count_metric_identifiers()

    # --- sealed-holdout one-touch governance (BEFORE scoring) ---
    sealed_digest = sealed["cohort_manifest_digest"]
    scope = "t1_lockstep_count_replacement"
    cfg_digest = config_digest()
    ledger = htl.load_ledger(args.ledger)
    htl.assert_touch_legal(
        ledger, sealed_digest=sealed_digest, scope=scope)
    htl.assert_retune_legal(
        ledger, sealed_digest=sealed_digest, scope=scope,
        config_digest=cfg_digest)

    body = build_verdicts(
        emissions, labels_by_key, holdout_keys, sealed_by_key,
        count_excluded_keys, metrics,
    )

    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=60,
    ).stdout.strip()

    # Record the single holdout touch (the final evaluation above
    # scored the holdout cells from the sealed file exactly once).
    htl.record_touch(
        args.ledger,
        sealed_digest=sealed_digest,
        scope=scope,
        scored_artifact=str(out_path),
        config_digest=cfg_digest,
        source_commit=head,
    )
    ledger_after = htl.load_ledger(args.ledger)
    touches = [
        t for t in ledger_after.get("touches", [])
        if t.get("sealed_digest") == sealed_digest
        and t.get("scope") == scope
    ]

    artifact = {
        "schema_version": 1,
        "artifact_kind": "t1_lockstep_verdict",
        "arch": "sm_86",
        "source_commit": head,
        "validator_config": VALIDATOR_CONFIG,
        "validator_config_digest": cfg_digest,
        "emissions_artifact": args.emissions,
        "labels_artifact": args.labels,
        "holdout_governance": {
            "sealed_digest": sealed_digest,
            "scope": scope,
            "holdout_scoring_touches": len(touches),
            "tuning_trail": {
                "tunable_parameters": "none",
                "tuned_on": "none",
                "note": (
                    "The validator has no parameters fit to any label: "
                    "thresholds and floors are plan constants frozen "
                    "in VALIDATOR_CONFIG before the holdout touch; the "
                    "config digest binds them via the "
                    "retune-after-touch guard."
                ),
            },
        },
        **body,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    fl = artifact["non_vacuity_floor"]
    print(f"[lockstep] wrote {out_path}: "
          f"{fl['scored_pairs']} scored pairs, floor_met={fl['floor_met']}, "
          f"verdicts={artifact['verdict_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
