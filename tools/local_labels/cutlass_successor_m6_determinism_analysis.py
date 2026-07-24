#!/usr/bin/env python3
"""Pure diagnostic Q712 analysis for the CUTLASS successor M6 cohort."""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.local_labels import admit_cutlass_successor_capture as admission  # noqa: E402
from tools.local_labels import (  # noqa: E402
    build_cutlass_successor_m6_determinism_schedule as successor_schedule,
)
from tools.local_labels import cutlass_successor_m6_observation_journal as journal_contract  # noqa: E402
from tools.local_labels import m6_determinism_campaign as m6  # noqa: E402
from tools.local_labels import m6_determinism_stats as stats  # noqa: E402

MODULE_REL = Path("tools/local_labels/cutlass_successor_m6_determinism_analysis.py")
SCHEMA_REL = Path(
    "workloads/simbench_bw/schemas/"
    "cutlass_successor_m6_determinism_analysis.schema.json"
)
EXPECTED_CELLS = 3
EXPECTED_CONFIGS = 6
EXPECTED_METRICS = 55
EXPECTED_STRATA = EXPECTED_CELLS * EXPECTED_CONFIGS * EXPECTED_METRICS
EXPECTED_OOR_ROWS = EXPECTED_CONFIGS * EXPECTED_METRICS
EXPECTED_FLOOR_ROWS = EXPECTED_METRICS
EXPECTED_DENOMINATORS = EXPECTED_CELLS * EXPECTED_METRICS
MINIMUM_FLOOR_CELLS = 30
REFERENCE_CONFIG = successor_schedule.REFERENCE_CONFIG
TAXONOMY = tuple(stats.DETERMINISM_CLASSES)
FIXED_CLAIMS = {
    "artifact_scope": "diagnostic_only",
    "analysis_is_diagnostic_only": True,
    "publication_authorized": False,
    "publication_claim": False,
    "canonical_m6_completed": False,
    "m6_done_created": False,
    "m7_handoff_ready": False,
    "m7_consumable": False,
    "floor_fallback_authorized": False,
    "hardware_values_are_targets_only": True,
}


class AnalysisError(RuntimeError):
    """The successor Q712 analysis is malformed or not derivable."""


def _fail(code: str, detail: str) -> None:
    raise AnalysisError(f"{code}: {detail}")


def _without(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    omitted = set(keys)
    return {key: item for key, item in value.items() if key not in omitted}


def _finite(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("SUCCESSOR_M6_ANALYSIS_VALUE", "finite JSON number required")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        _fail("SUCCESSOR_M6_ANALYSIS_VALUE", "finite JSON number required")
    if not math.isfinite(number):
        _fail("SUCCESSOR_M6_ANALYSIS_VALUE", "finite JSON number required")
    return value


def _analysis_source_identity() -> dict[str, Any]:
    sources: dict[str, Any] = {}
    try:
        for name, relative in (("analyzer", MODULE_REL), ("schema", SCHEMA_REL)):
            sources[name] = {
                "path": relative.as_posix(),
                "sha256": admission.sha256_bytes((ROOT / relative).read_bytes()),
            }
    except (OSError, UnicodeError) as exc:
        _fail("SUCCESSOR_M6_ANALYSIS_SOURCE", str(exc))
    return sources


def _validated_replay(
    schedule: Mapping[str, Any],
    journal: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Validate both authorities, then replay the record stream again here."""

    try:
        successor_schedule.validate_schedule(schedule)
        journal_contract.validate_journal(schedule, journal)
        policy = journal_contract.derive_metric_policy(schedule)
        records, summary = journal_contract._replay(
            schedule,
            policy,
            journal.get("records", []),
            sealed=True,
        )
    except (successor_schedule.ScheduleError, journal_contract.JournalError,
            admission.AdmissionError, m6.M6Error, stats.DispersionError) as exc:
        _fail("SUCCESSOR_M6_ANALYSIS_INPUT", str(exc))
    if records != journal.get("records"):
        _fail("SUCCESSOR_M6_ANALYSIS_REPLAY", "normalized records differ")
    if summary != journal.get("replay_summary"):
        _fail("SUCCESSOR_M6_ANALYSIS_REPLAY", "derived summary differs")
    return policy, records, summary


def _metric_maps(
    policy: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    selected = [
        row for row in policy["metrics"] if row["capture_status"] == "selected"
    ]
    if len(selected) != EXPECTED_METRICS:
        _fail("SUCCESSOR_M6_ANALYSIS_METRICS", "selected metric count differs")
    if any(
        row["model_io_role"] != "target" or row["deployability"] != "label_only"
        for row in selected
    ):
        _fail("SUCCESSOR_M6_ANALYSIS_TARGET_ONLY", "selected metric role differs")
    selected.sort(key=lambda row: row["key"])
    return selected, {row["key"]: row for row in selected}


def _record_views(
    schedule: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> tuple[
    dict[int, Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
]:
    terminals: dict[int, Mapping[str, Any]] = {}
    decisions: list[Mapping[str, Any]] = []
    for record in records:
        if record["record_kind"] == "attempt_terminal":
            ordinal = record["ordinal"]
            if ordinal in terminals:
                _fail("SUCCESSOR_M6_ANALYSIS_REPLAY", f"duplicate terminal {ordinal}")
            terminals[ordinal] = record
        elif record["record_kind"] == "trigger_decision":
            decisions.append(record)
    activated = set(summary["activated_ordinals"])
    tier1 = sorted(schedule["slots"], key=lambda row: row["ordinal"])
    tier2 = sorted(
        (
            slot
            for slot in schedule["tier2_candidates"]
            if slot["ordinal"] in activated
        ),
        key=lambda row: row["ordinal"],
    )
    active = tier1 + tier2
    active_ordinals = {slot["ordinal"] for slot in active}
    if not set(terminals).issubset(active_ordinals):
        _fail("SUCCESSOR_M6_ANALYSIS_REPLAY", "terminal lies outside active slots")
    if summary["tier2_activation_emitted"]:
        if len(activated) != 10 * len(summary["triggered_cell_ids"]):
            _fail(
                "SUCCESSOR_M6_ANALYSIS_TIER2_ACCOUNTING",
                "activation is not 10 per triggered cell",
            )
    elif activated:
        _fail(
            "SUCCESSOR_M6_ANALYSIS_TIER2_ACCOUNTING",
            "inactive Tier-2 has activated ordinals",
        )
    return terminals, decisions, active


def _modality_payload(
    values: Sequence[float], metric: Mapping[str, Any], *, tier2: bool
) -> dict[str, Any]:
    result = stats.modality_result(
        values,
        discrete=metric["statistical_domain"] == "discrete",
        absolute_resolution=metric["absolute_resolution"],
        tier2=tier2,
    )
    return {
        "flagged": result.flagged,
        "outlier_flag": result.outlier_flag,
        "certified": result.certified,
        "suspected": result.suspected,
        "split_index": result.split_index,
        "gap": result.gap,
        "reason": result.reason,
    }


def _classification(
    *,
    cell: Mapping[str, Any],
    config: Mapping[str, Any],
    metric: Mapping[str, Any],
    slots: Sequence[Mapping[str, Any]],
    terminals: Mapping[int, Mapping[str, Any]],
    tier2_active: bool,
) -> dict[str, Any]:
    observed: list[dict[str, Any]] = []
    excluded = 0
    terminal_repeats = 0
    for slot in sorted(slots, key=lambda row: row["repeat_index"]):
        terminal = terminals.get(slot["ordinal"])
        if terminal is None:
            continue
        terminal_repeats += 1
        if terminal["outcome"] == "excluded_typed":
            excluded += 1
            continue
        by_key = {row["metric_key"]: row for row in terminal["observations"]}
        observation = by_key.get(metric["key"])
        if observation is None:
            _fail(
                "SUCCESSOR_M6_ANALYSIS_OBSERVATION_SET",
                f"{slot['ordinal']}:{metric['key']}",
            )
        observed.append(
            {
                "ordinal": slot["ordinal"],
                "value": _finite(observation["value"]),
                "validity": observation["validity"],
            }
        )
    in_range = [row["value"] for row in observed if row["validity"] == "in_range"]
    oor = [row for row in observed if row["validity"] != "in_range"]
    oor_low = [row for row in oor if row["validity"] == "out_of_range_low"]
    oor_high = [row for row in oor if row["validity"] == "out_of_range_high"]
    event_resolution = 1.0 / len(observed) if observed else None
    zero_event_upper_bound = (
        min(1.0, 3.0 / len(observed)) if observed and not oor else None
    )
    gate = stats.dispersion_gate(metric["repeatability_rule"], in_range)
    sigma = stats.sigma_rel(
        metric["repeatability_rule"],
        in_range,
        absolute_resolution=metric["absolute_resolution"],
    )
    sigma_upper = (
        sigma * stats.sigma_upper_factor(len(in_range))
        if sigma is not None and len(in_range) >= 2
        else None
    )
    dispersion_class = stats.classify_sigma(
        sigma_upper if sigma_upper is not None else sigma,
        in_range,
    )
    modality = _modality_payload(
        in_range,
        metric,
        tier2=tier2_active and len(in_range) >= 30,
    )
    if oor:
        determinism_class = "vendor_defect"
    elif len(in_range) < 5:
        determinism_class = "insufficient_evidence"
    elif modality["certified"]:
        determinism_class = "multimodal"
    else:
        determinism_class = dispersion_class
    annotations: list[str] = []
    if oor:
        annotations.append("oor_floor_lower_bound")
    if modality["suspected"]:
        annotations.append("suspected_multimodal")
    if modality["outlier_flag"]:
        annotations.append("outlier_flag")
    if stats.classify_sigma(sigma, in_range) != dispersion_class:
        annotations.append("uncertain_boundary")
    expected = len(slots)
    row = {
        "cell_id": cell["cell_id"],
        "cell_identity_sha256": cell["cell_identity_sha256"],
        "config_id": config["config_id"],
        "config_sha256": config["config_sha256"],
        "metric_key": metric["key"],
        "metric_sha256": metric["metric_sha256"],
        "repeatability_rule": metric["repeatability_rule"],
        "expected_repeats": expected,
        "terminal_repeats": terminal_repeats,
        "observed_repeats": len(observed),
        "terminal_typed_excluded_repeats": excluded,
        "pending_repeats": expected - terminal_repeats,
        "in_range_repeats": len(in_range),
        "out_of_range_repeats": len(oor),
        "out_of_range_low_repeats": len(oor_low),
        "out_of_range_high_repeats": len(oor_high),
        "observation_identity_holds": len(observed) == len(in_range) + len(oor_low) + len(oor_high),
        "event_resolution": event_resolution,
        "zero_event_upper_bound": zero_event_upper_bound,
        "out_of_range_values": oor,
        "values_sha256": m6.canonical_sha256(observed),
        "mean_in_range": statistics.fmean(in_range) if in_range else None,
        "dispersion_gate": gate,
        "admission_threshold": metric["admission_threshold"],
        "would_have_been_admitted": (
            None if gate is None else gate <= metric["admission_threshold"]
        ),
        "sigma_rel": sigma,
        "sigma_rel_upper_95": sigma_upper,
        "determinism_class": determinism_class,
        "dispersion_class": dispersion_class,
        "modality": modality,
        "annotations": sorted(annotations),
    }
    row["classification_sha256"] = m6.canonical_sha256(row)
    return row


def _completion_state(
    summary: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    *,
    pending_slots: int,
    excluded_slots: int,
) -> str:
    if summary["tier1_terminal_slots"] < successor_schedule.TIER1_SLOTS:
        return "incomplete_evidence"
    if not summary["tier1_closed"] or len(decisions) < EXPECTED_CELLS:
        return "trigger_decisions_required"
    if not summary["tier2_activation_emitted"]:
        return "tier2_activation_required"
    if pending_slots:
        return "tier2_required"
    if excluded_slots:
        return "incomplete_evidence"
    return "evidence_complete_diagnostic_only"


def _build_analysis(
    schedule: Mapping[str, Any],
    journal: Mapping[str, Any],
    policy: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    generated_at_utc: str,
) -> dict[str, Any]:
    source_identity = _analysis_source_identity()
    selected, metrics = _metric_maps(policy)
    terminals, decisions, active_slots = _record_views(schedule, records, summary)
    cells = sorted(schedule["cells"], key=lambda row: row["cell_id"])
    configs = sorted(schedule["configs"], key=lambda row: row["config_id"])
    active_by_stratum: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for slot in active_slots:
        active_by_stratum[(slot["cell_id"], slot["config_id"])].append(slot)
    activated_cells = set(summary["triggered_cell_ids"])

    classifications: list[dict[str, Any]] = []
    class_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for cell in cells:
        for config in configs:
            stratum_slots = active_by_stratum[(cell["cell_id"], config["config_id"])]
            for metric in selected:
                row = _classification(
                    cell=cell,
                    config=config,
                    metric=metric,
                    slots=stratum_slots,
                    terminals=terminals,
                    tier2_active=(
                        cell["cell_id"] in activated_cells
                        and config["config_id"] == REFERENCE_CONFIG
                    ),
                )
                classifications.append(row)
                class_by_key[(cell["cell_id"], config["config_id"], metric["key"])] = row
    if len(classifications) != EXPECTED_STRATA:
        _fail("SUCCESSOR_M6_ANALYSIS_STRATA", str(len(classifications)))

    oor_rates: list[dict[str, Any]] = []
    for config in configs:
        for metric in selected:
            rows = [
                class_by_key[(cell["cell_id"], config["config_id"], metric["key"])]
                for cell in cells
            ]
            observed = sum(row["observed_repeats"] for row in rows)
            oor = sum(row["out_of_range_repeats"] for row in rows)
            oor_rates.append(
                {
                    "config_id": config["config_id"],
                    "metric_key": metric["key"],
                    "observed_rows": observed,
                    "out_of_range_rows": oor,
                    "rate": oor / observed if observed else None,
                    "floor_is_lower_bound": oor > 0,
                    "event_resolution": 1.0 / observed if observed else None,
                    "zero_event_upper_bound": (
                        min(1.0, 3.0 / observed) if observed and oor == 0 else None
                    ),
                }
            )
    if len(oor_rates) != EXPECTED_OOR_ROWS:
        _fail("SUCCESSOR_M6_ANALYSIS_OOR_ROWS", str(len(oor_rates)))

    floors: list[dict[str, Any]] = []
    floor_by_metric: dict[str, Mapping[str, Any]] = {}
    for metric in selected:
        reference_rows = [
            class_by_key[(cell["cell_id"], REFERENCE_CONFIG, metric["key"])]
            for cell in cells
        ]
        valid_cells = sum(
            row["sigma_rel"] is not None
            and row["determinism_class"]
            not in {"vendor_defect", "insufficient_evidence", "multimodal"}
            for row in reference_rows
        )
        row = {
            "metric_key": metric["key"],
            "valid_cells": valid_cells,
            "required_valid_cells": MINIMUM_FLOOR_CELLS,
            "cross_cell_p90": None,
            "p90_upper_90": None,
            "absolute_resolution": metric["absolute_resolution"],
            "floor": None,
            "lower_bound": any(
                item["out_of_range_repeats"] > 0 for item in reference_rows
            ),
            "status": "insufficient_cells",
            "typed_reason": "successor_cohort_has_3_cells_below_required_30",
            "fallback_authorized": False,
        }
        floors.append(row)
        floor_by_metric[metric["key"]] = row
    if len(floors) != EXPECTED_FLOOR_ROWS:
        _fail("SUCCESSOR_M6_ANALYSIS_FLOORS", str(len(floors)))

    denominators: list[dict[str, Any]] = []
    for cell in cells:
        for metric in selected:
            classification = class_by_key[
                (cell["cell_id"], REFERENCE_CONFIG, metric["key"])
            ]
            cls = classification["determinism_class"]
            if cls == "deterministic":
                denominator = metric["absolute_resolution"]
                kind = "absolute"
                target = "exact_value"
                exact = True
                status = "diagnostic_only_exact_resolution"
            elif cls == "insufficient_evidence":
                denominator = None
                kind = "none"
                target = "none"
                exact = False
                status = "no_gate_insufficient_evidence"
            else:
                denominator = None
                kind = "none"
                target = "none"
                exact = False
                status = "pending_structural_floor"
            denominators.append(
                {
                    "cell_id": cell["cell_id"],
                    "metric_key": metric["key"],
                    "determinism_class": cls,
                    "absolute_resolution": metric["absolute_resolution"],
                    "pooled_floor": floor_by_metric[metric["key"]]["floor"],
                    "denominator": denominator,
                    "denominator_kind": kind,
                    "gate_target": target,
                    "gate_scope": "diagnostic_only",
                    "exact_match": exact,
                    "status": status,
                }
            )
    if len(denominators) != EXPECTED_DENOMINATORS:
        _fail("SUCCESSOR_M6_ANALYSIS_DENOMINATORS", str(len(denominators)))

    active_ordinals = {slot["ordinal"] for slot in active_slots}
    terminal_rows = [record for ordinal, record in terminals.items() if ordinal in active_ordinals]
    observed_slots = sum(row["outcome"] == "observed" for row in terminal_rows)
    excluded_slots = sum(row["outcome"] == "excluded_typed" for row in terminal_rows)
    terminal_slots = len(terminal_rows)
    pending_slots = len(active_slots) - terminal_slots
    in_range_rows = sum(
        observation["validity"] == "in_range"
        for row in terminal_rows
        if row["outcome"] == "observed"
        for observation in row["observations"]
    )
    out_of_range_low_rows = sum(
        observation["validity"] == "out_of_range_low"
        for row in terminal_rows
        if row["outcome"] == "observed"
        for observation in row["observations"]
    )
    out_of_range_high_rows = sum(
        observation["validity"] == "out_of_range_high"
        for row in terminal_rows
        if row["outcome"] == "observed"
        for observation in row["observations"]
    )
    observed_rows = in_range_rows + out_of_range_low_rows + out_of_range_high_rows
    terminal_typed_excluded_rows = excluded_slots * EXPECTED_METRICS
    expected_rows = len(active_slots) * EXPECTED_METRICS
    pending_rows = pending_slots * EXPECTED_METRICS

    bucket_rows: list[dict[str, Any]] = []
    for config in configs:
        config_slots = [slot for slot in active_slots if slot["config_id"] == config["config_id"]]
        for metric in selected:
            config_terminals = [
                terminals[slot["ordinal"]]
                for slot in config_slots
                if slot["ordinal"] in terminals
            ]
            observed_bucket = [
                next(obs for obs in terminal["observations"] if obs["metric_key"] == metric["key"])
                for terminal in config_terminals
                if terminal["outcome"] == "observed"
            ]
            bucket = {
                "config_id": config["config_id"],
                "metric_key": metric["key"],
                "expected_rows": len(config_slots),
                "observed_raw_rows": len(observed_bucket),
                "in_range_raw_rows": sum(
                    obs["validity"] == "in_range" for obs in observed_bucket
                ),
                "out_of_range_low_raw_rows": sum(
                    obs["validity"] == "out_of_range_low" for obs in observed_bucket
                ),
                "out_of_range_high_raw_rows": sum(
                    obs["validity"] == "out_of_range_high" for obs in observed_bucket
                ),
                "terminal_typed_excluded_rows": sum(
                    terminal["outcome"] == "excluded_typed"
                    for terminal in config_terminals
                ),
                "pending_raw_rows": len(config_slots) - len(config_terminals),
            }
            bucket["row_identity_holds"] = bucket["expected_rows"] == (
                bucket["observed_raw_rows"]
                + bucket["terminal_typed_excluded_rows"]
                + bucket["pending_raw_rows"]
            ) and bucket["observed_raw_rows"] == (
                bucket["in_range_raw_rows"]
                + bucket["out_of_range_low_raw_rows"]
                + bucket["out_of_range_high_raw_rows"]
            )
            bucket_rows.append(bucket)

    accounting = {
        "cells": EXPECTED_CELLS,
        "configs": EXPECTED_CONFIGS,
        "selected_metrics": EXPECTED_METRICS,
        "strata": len(classifications),
        "oor_rate_rows": len(oor_rates),
        "floor_rows": len(floors),
        "denominator_rows": len(denominators),
        "tier1_slots": successor_schedule.TIER1_SLOTS,
        "trigger_decisions_emitted": len(decisions),
        "triggered_cells": len(summary["triggered_cell_ids"]),
        "tier2_contract_slots": len(summary["triggered_cell_ids"]) * 10,
        "active_tier2_slots": len(summary["activated_ordinals"]),
        "active_slots": len(active_slots),
        "terminal_slots": terminal_slots,
        "pending_slots": pending_slots,
        "observed_slots": observed_slots,
        "excluded_typed_slots": excluded_slots,
        "expected_rows": expected_rows,
        "observed_raw_rows": observed_rows,
        "in_range_raw_rows": in_range_rows,
        "out_of_range_low_raw_rows": out_of_range_low_rows,
        "out_of_range_high_raw_rows": out_of_range_high_rows,
        "terminal_typed_excluded_rows": terminal_typed_excluded_rows,
        "pending_raw_rows": pending_rows,
        "observed_row_identity_holds": observed_rows
        == in_range_rows + out_of_range_low_rows + out_of_range_high_rows,
        "terminal_slot_identity_holds": terminal_slots == observed_slots + excluded_slots,
        "expected_row_identity_holds": expected_rows
        == observed_rows + terminal_typed_excluded_rows + pending_rows,
        "bucket_identities_hold": all(
            bucket["row_identity_holds"] for bucket in bucket_rows
        ),
        "tier2_accounting_holds": (
            len(summary["activated_ordinals"])
            == (
                len(summary["triggered_cell_ids"]) * 10
                if summary["tier2_activation_emitted"]
                else 0
            )
        ),
        "per_config_metric": bucket_rows,
    }
    if not all(
        accounting[key]
        for key in (
            "observed_row_identity_holds",
            "terminal_slot_identity_holds",
            "expected_row_identity_holds",
            "bucket_identities_hold",
            "tier2_accounting_holds",
        )
    ):
        _fail("SUCCESSOR_M6_ANALYSIS_ACCOUNTING", repr(accounting))
    if len(bucket_rows) != EXPECTED_OOR_ROWS:
        _fail("SUCCESSOR_M6_ANALYSIS_BUCKETS", str(len(bucket_rows)))

    discoveries = list(stats.bh_fdr([], q=0.05))
    multiplicity = {
        "status": "external_pending",
        "method": "BH-FDR",
        "q": 0.05,
        "family_definition": "metric_family_within_config_v1",
        "hypothesis_count": 0,
        "discoveries": discoveries,
        "discoveries_sha256": m6.canonical_sha256(discoveries),
    }
    multiplicity["policy_sha256"] = m6.canonical_sha256(multiplicity)
    local_history_projection_count = sum(
        row["would_have_been_admitted"] is not None for row in classifications
    )
    historical = {
        "status": "external_pending",
        "dispersion_gate_replayed": True,
        "local_projection_count": local_history_projection_count,
        "historical_admitted_set_sha256": None,
        "exact_set_match": None,
        "typed_reason": "historical_admitted_set_not_supplied",
    }
    completion = _completion_state(
        summary,
        decisions,
        pending_slots=pending_slots,
        excluded_slots=excluded_slots,
    )
    analysis = {
        "schema_version": 1,
        "artifact_kind": "cutlass_successor_m6_determinism_analysis",
        "analysis_id": "cutlass-successor-m6-analysis-"
        + m6.canonical_sha256(
            [
                schedule["schedule_sha256"],
                journal["journal_sha256"],
                generated_at_utc,
            ]
        )[:24],
        "generated_at_utc": generated_at_utc,
        "journal_created_at_utc": journal["created_at_utc"],
        "analysis_source_identity": source_identity,
        "analysis_source_identity_sha256": m6.canonical_sha256(source_identity),
        "bindings": {
            "schedule_id": schedule["schedule_id"],
            "schedule_sha256": schedule["schedule_sha256"],
            "journal_id": journal["journal_id"],
            "journal_sha256": journal["journal_sha256"],
            "records_sha256": journal["records_sha256"],
            "source_identity_sha256": schedule["source_identity_sha256"],
            "config_identity_sha256": schedule["config_identity_sha256"],
            "cell_identity_sha256": schedule["cell_identity_sha256"],
            "metric_policy_sha256": journal["metric_policy_sha256"],
            "statistical_contract_sha256": policy["source_identity"]["statistical_contract_sha256"],
        },
        "completion_state": completion,
        "taxonomy": list(TAXONOMY),
        "accounting": accounting,
        "classifications": classifications,
        "out_of_range_rates": oor_rates,
        "noise_floors": floors,
        "denominators": denominators,
        "analysis_policy": {
            "minimum_floor_cells": MINIMUM_FLOOR_CELLS,
            "floor_quantile": 0.9,
            "floor_upper_confidence": 0.9,
            "within_session_only": True,
            "raw_values_clamped": False,
            "finite_out_of_range_values_retained": True,
            "floor_fallback_policy": "forbidden_without_versioned_authority",
            "multiplicity_policy": multiplicity,
            "historical_admission_reconciliation": historical,
            "limitations": {
                "censoring": "finite out-of-range values remain retained and uncensored; affected floors are lower bounds",
                "finite_n": "the three-cell successor cohort is structurally below the 30-cell floor minimum; 1/N event resolution and zero-event rule-of-three 3/N bounds are diagnostic",
                "within_session": "repeatability classifications are limited to this predecessor-bound journal session",
                "multiplicity": "external_pending until a versioned hypothesis family and p-values are supplied",
                "historical_reconciliation": "external_pending until the exact historical admitted set identity is supplied",
                "fallback_authority": "no versioned floor fallback authority is supplied or accepted",
            },
        },
        "claims": dict(FIXED_CLAIMS),
    }
    analysis["analysis_sha256"] = m6.canonical_sha256(analysis)
    return analysis


def _derive_analysis(
    schedule: Mapping[str, Any],
    journal: Mapping[str, Any],
    policy: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    generated_at_utc: str,
) -> dict[str, Any]:
    try:
        return _build_analysis(
            schedule,
            journal,
            policy,
            records,
            summary,
            generated_at_utc=generated_at_utc,
        )
    except AnalysisError:
        raise
    except (ArithmeticError, ValueError, m6.M6Error, stats.DispersionError) as exc:
        _fail("SUCCESSOR_M6_ANALYSIS_STATISTICS", str(exc))


def _validate_schema(analysis: Mapping[str, Any]) -> None:
    try:
        schema = json.loads((ROOT / SCHEMA_REL).read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(
            schema,
            format_checker=jsonschema.FormatChecker(),
        ).validate(analysis)
    except (OSError, UnicodeError, json.JSONDecodeError, jsonschema.SchemaError) as exc:
        _fail("SUCCESSOR_M6_ANALYSIS_SCHEMA_UNAVAILABLE", str(exc))
    except jsonschema.ValidationError as exc:
        location = "/".join(str(item) for item in exc.absolute_path) or "<root>"
        _fail("SUCCESSOR_M6_ANALYSIS_SCHEMA", f"{location}: {exc.message}")


def analyze(
    schedule: Mapping[str, Any],
    journal: Mapping[str, Any],
    *,
    generated_at_utc: str,
) -> dict[str, Any]:
    """Return a deterministic diagnostic analysis without writing or publishing."""

    policy, records, summary = _validated_replay(schedule, journal)
    analysis = _derive_analysis(
        schedule,
        journal,
        policy,
        records,
        summary,
        generated_at_utc=generated_at_utc,
    )
    _validate_schema(analysis)
    return analysis


def validate_analysis(
    schedule: Mapping[str, Any],
    journal: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> None:
    """Fail closed unless analysis exactly rederives from schedule and records."""

    _validate_schema(analysis)
    try:
        observed_seal = m6.canonical_sha256(_without(analysis, "analysis_sha256"))
    except (TypeError, ValueError, OverflowError, m6.M6Error) as exc:
        _fail("SUCCESSOR_M6_ANALYSIS_SEAL", str(exc))
    if analysis.get("analysis_sha256") != observed_seal:
        _fail("SUCCESSOR_M6_ANALYSIS_SEAL", "analysis seal differs")
    policy, records, summary = _validated_replay(schedule, journal)
    expected = _derive_analysis(
        schedule,
        journal,
        policy,
        records,
        summary,
        generated_at_utc=analysis["generated_at_utc"],
    )
    if dict(analysis) != expected:
        _fail("SUCCESSOR_M6_ANALYSIS_REDERIVATION", "analysis differs")
