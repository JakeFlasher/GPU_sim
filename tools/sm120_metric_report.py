#!/usr/bin/env python3
"""Reduce replay metric predictions against independent NCU profiles.

This is a post-replay report path. It consumes predictions already emitted by
the simulator, never feeds NCU values back into replay, and writes a compact
compressed comparison stream plus a family-grouped summary.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import statistics
from typing import Any

from jsonl_io import jsonl_compression_description, open_jsonl_reader, open_jsonl_writer
from sm120_corpus_compare import predictions as timing_predictions
from sm120_replay_bench import join_ncu_rows, parse_ncu_raw


def distribution(values: list[float]) -> dict[str, Any]:
    values = sorted(values)
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": values[int((len(values) - 1) * 0.90)],
    }


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def family(metric: str) -> str:
    if metric.startswith("dram__"):
        return "DRAM"
    if metric.startswith("lts__"):
        return "L2"
    if metric.startswith("l1tex__"):
        return "L1/L1TEX"
    if metric.startswith(("sm__", "smsp__", "gpc__", "pmsampling:")):
        return "SM/compute"
    if metric.startswith("derived__"):
        return "derived"
    return metric.split("__", 1)[0]


def allowed_units(unit: str) -> set[str]:
    return {
        # ``metric_predictions`` use the compact internal spellings while
        # native launch/resource rows already carry NCU's presentation units.
        # Keep both spellings here; dimensional aliases that need scaling are
        # handled by ``target_scale_for_metric`` below.
        "pct": {"%", "pct"},
        "%": {"%", "pct"},
        # ``byte`` is intentionally excluded from the generic count alias;
        # byte-domain rows must use an explicitly named metric conversion.
        "count": {"inst", "sector", "request", "line", "wavefront", "warp"},
        "cycles": {"cycle"},
        "inst/cycle": {"inst/cycle"},
        "warp": {"warp"},
        "inst": {"inst"},
        "sector": {"sector"},
        "byte": {"byte"},
        "request": {"request", ""},
        "wavefront": {"wavefront"},
        "branches": {"branches"},
        "thread/block": {"thread/block", ""},
        "block": {"block"},
        "block/SM": {"block/SM", "block"},
        "register/thread": {"register/thread"},
    }.get(unit, set())


def target_scale_for_metric(metric: str, model_unit: str, target_unit: str) -> float | None:
    """Return a target-to-model scale for a name-qualified unit relation.

    NCU CSV exports mix SI-looking ``Kbyte``/``Mbyte`` labels with the
    simulator's byte-valued launch/resource rows.  Only the documented metric
    families below are admitted.  In particular, this is not a general
    unitless conversion and deliberately does not bridge GPU duration clocks.
    """
    if target_unit in allowed_units(model_unit):
        return 1.0
    if metric == "dram__bytes_read.sum" and model_unit == "byte":
        return {"Kbyte": 1_000.0, "Mbyte": 1_000_000.0}.get(target_unit)
    shared_bytes = {
        "launch__shared_mem_per_block_static",
        "launch__shared_mem_per_block_dynamic",
        "launch__shared_mem_per_block_allocated",
    }
    if metric in shared_bytes and model_unit in {"byte", "byte/block"}:
        # The model stores these launch quantities as bytes per block even
        # where the native metric carries an explicit ``/block`` suffix.
        if target_unit in {"byte", "byte/block"}:
            return 1.0
        if target_unit in {"Kbyte", "Kbyte/block"}:
            return 1_000.0
    if metric == "launch__shared_mem_config_size" and model_unit in {"byte", "byte/SM"}:
        return {"Kbyte": 1_000.0, "byte": 1.0}.get(target_unit)
    if metric == "launch__block_size" and model_unit == "thread/block":
        return 1.0 if target_unit == "" else None
    if metric == "launch__grid_size" and model_unit == "block":
        return 1.0 if target_unit == "" else None
    if metric.startswith("l1tex__t_requests_") and model_unit == "request":
        return 1.0 if target_unit == "" else None
    if metric == "smsp__branch_targets_threads_divergent" and model_unit == "branches":
        return 1.0 if target_unit == "branches" else None
    if metric.startswith("launch__occupancy_limit_") and model_unit == "block/SM":
        return 1.0 if target_unit == "block" else None
    return None


def metric_entries(launch: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize native metric and cache-observer predictions."""
    entries: list[dict[str, Any]] = []
    for prediction in launch.get("metric_predictions", []):
        if not isinstance(prediction, dict) or not prediction.get("ncu_metric"):
            continue
        # The generic L2 hit-rate alias has broader/version-dependent client
        # scope than the conserving TEX-sector observer. Keep its target in
        # the unavailable inventory, but never score it as if the endpoints
        # were identical. Source-specific hit/miss rows remain reportable via
        # cache_lookup_predictions.
        if prediction.get("ncu_metric") == "lts__t_sector_hit_rate.pct":
            continue
        outcome = prediction.get("outcome")
        if not isinstance(outcome, dict):
            continue
        value = outcome.get("value")
        estimate = value.get("value") if isinstance(value, dict) else None
        estimate = finite(estimate)
        if isinstance(value, dict) and value.get("kind") == "fraction" and estimate is not None:
            estimate *= 100.0
        entries.append({
            "metric": str(prediction["ncu_metric"]),
            "unit": str(prediction.get("unit", "")),
            "status": str(outcome.get("status", "unavailable")),
            "estimate": estimate,
            "source_key": f"metric_predictions.{prediction.get('key', prediction['ncu_metric'])}",
            "reason": (outcome.get("detail") or {}).get("reason")
            if isinstance(outcome.get("detail"), dict) else None,
        })
    for observer in launch.get("cache_lookup_predictions", []):
        if not isinstance(observer, dict) or observer.get("status") != "predicted":
            continue
        counts = observer.get("observer")
        if not isinstance(counts, dict):
            continue
        for field, metric_key in (("hits", "ncu_hit_metric"), ("misses", "ncu_miss_metric")):
            metric = observer.get(metric_key)
            estimate = finite(counts.get(field))
            if not metric:
                continue
            entries.append({
                "metric": str(metric),
                "unit": "count",
                "status": "predicted" if estimate is not None else "unavailable",
                "estimate": estimate,
                "reason": None if estimate is not None else "observer_count_unavailable",
                "source_key": f"cache_lookup_predictions.{metric_key}.{metric}",
            })
    native = launch.get("native_metrics", {})
    native_metrics = native.get("metrics", {}) if isinstance(native, dict) else {}
    for key, metric in native_metrics.items():
        if not isinstance(metric, dict) or not metric.get("ncu_metric"):
            continue
        estimate = finite(metric.get("value"))
        status = metric.get("status")
        entries.append({
            "metric": str(metric["ncu_metric"]),
            "unit": "cycles" if metric.get("unit") == "cycle" else str(metric.get("unit", "")),
            "status": "predicted" if status in {"modeled", "approximate"} and estimate is not None else str(status or "unavailable"),
            "estimate": estimate,
            "reason": metric.get("semantics") if estimate is None else None,
            "source_key": f"native_metrics.{key}",
            "semantics": metric.get("semantics"),
        })
    return entries


def selected_entries(launch: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    """One prediction per metric/launch, selected without inspecting NCU.

    Prefer a finite declared metric, then cache observer, then native metric,
    then the existing timing fallback. Preserve other representations in the
    row, but do not count aliases as independent launches. External measured
    values are not simulator predictions.
    """
    candidates = metric_entries(launch)
    for metric, estimate in timing_predictions(launch, mode).items():
        candidates.append({
            "metric": metric, "unit": "cycles",
            "status": "predicted" if estimate is not None else "unavailable",
            "estimate": estimate, "source_key": f"{mode}_timing.{metric}",
            "semantics": "model-relative timing endpoint; not an identified hardware clock",
        })
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for entry in candidates:
        groups[entry["metric"]].append(entry)
    result = []
    for entries in groups.values():
        def preference(entry: dict[str, Any]) -> tuple[int, int]:
            source = str(entry.get("source_key", ""))
            # Architecture-specific native metric projections carry the most
            # explicit endpoint/unit contract. Generic metric_predictions and
            # timing fallbacks remain alternatives, not reasons to suppress a
            # scoped native row from NCU comparison.
            rank = (0 if source.startswith("native_metrics.") else
                    1 if source.startswith("cache_lookup_predictions.") else
                    2 if source.startswith("metric_predictions.") else 3)
            finite = int(entry.get("status") == "predicted" and entry.get("estimate") is not None)
            return (rank, 0 if finite else 1)

        chosen = min(entries, key=preference)
        row = dict(chosen)
        alternatives = [entry for entry in entries if entry is not chosen]
        if alternatives:
            row["alternative_predictions"] = alternatives
        predicted = [entry for entry in entries
                     if entry["status"] == "predicted" and entry.get("estimate") is not None]
        # A post-replay reducer must not silently choose between incompatible
        # endpoints.  Equal numeric aliases remain compatible; differing
        # values or units are retained as evidence but excluded from scoring.
        if len(predicted) > 1:
            values = {float(entry["estimate"]) for entry in predicted}
            units = {entry.get("unit", "") for entry in predicted}
            # Older metric emitters used the generic unit ``count`` for exact
            # sector/request rows.  If the numeric estimate agrees, prefer a
            # source with the specific unit instead of manufacturing a
            # prediction conflict that suppresses an otherwise like-for-like
            # NCU comparison.
            count_alias = "count" in units and len(units) == 2
            compatible_alias = count_alias and len(values) == 1
            native_selected = str(chosen.get("source_key", "")).startswith("native_metrics.")
            if (len(values) > 1 or (len(units) > 1 and not compatible_alias)) and not native_selected:
                row["status"] = "conflict"
                row["estimate"] = None
                row["reason"] = "incompatible_prediction_alternatives"
                row["alternative_conflict"] = True
            elif native_selected and (len(values) > 1 or len(units) > 1):
                row["alternative_conflicts"] = True
            elif compatible_alias:
                specific = next(
                    (entry for entry in predicted if entry.get("unit") != "count"),
                    None,
                )
                if specific is not None:
                    row.update(specific)
                    row["alternative_predictions"] = alternatives
        result.append(row)
    return result


def scope_warnings(launch: dict[str, Any], target_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Diagnose gross opcode-scope disagreements without changing scoring.

    Some captured NCU rows expose only a narrow SASS/pipe scope while the
    trace-side logical memory ledger counts every admitted record.  A large
    disagreement is retained as a post-replay warning, never used to select a
    prediction or tune replay.
    """
    trace = launch.get("trace_metrics", {})
    memory = trace.get("memory", {}) if isinstance(trace, dict) else {}
    memory_scope = trace.get("memory_scope", {}) if isinstance(trace, dict) else {}
    warnings: list[dict[str, Any]] = []
    for kind, ncu_metric in (
        ("global_load", "smsp__sass_inst_executed_op_global_ld.sum"),
        ("global_store", "smsp__sass_inst_executed_op_global_st.sum"),
        ("local_load", "smsp__sass_inst_executed_op_local_ld.sum"),
        ("local_store", "smsp__sass_inst_executed_op_local_st.sum"),
    ):
        if kind == "global_load" and isinstance(memory_scope, dict):
            # The narrow SASS global-load endpoint excludes LDGSTS and
            # modifier forms such as CONSTANT/LTC128B. Prefer the plain-LDG
            # observer when the replay emitted the form crosswalk; retain the
            # legacy broad row for older artifacts.
            trace_row = (
                memory_scope.get("global_load_ldg")
                if memory_scope
                else memory.get(kind)
            )
            warning_kind = "global_load_opcode"
        else:
            trace_row = memory.get(kind)
            warning_kind = "opcode_scope_mismatch"
        target_row = target_metrics.get(ncu_metric)
        if not isinstance(trace_row, dict) or not isinstance(target_row, dict):
            continue
        trace_count = finite(trace_row.get("requests"))
        target_count = finite(target_row.get("value"))
        if trace_count is None or target_count is None or trace_count <= 0:
            continue
        ratio = None if target_count <= 0 else max(trace_count / target_count, target_count / trace_count)
        if ratio is not None and ratio < 4.0:
            continue
        warnings.append({
            "kind": warning_kind,
            "trace_kind": kind,
            "trace_requests": trace_count,
            "ncu_metric": ncu_metric,
            "ncu_value": target_count,
            "ratio": ratio,
            "detail": "post-replay scope diagnostic only; NCU target was not used as a replay input or prediction selector",
        })
    if isinstance(memory_scope, dict):
        trace_row = memory_scope.get("ldgsts_async_copy")
        target_row = target_metrics.get("smsp__inst_executed_op_ldgsts.sum")
        trace_count = finite(trace_row.get("requests")) if isinstance(trace_row, dict) else None
        target_count = finite(target_row.get("value")) if isinstance(target_row, dict) else None
        if trace_count is not None and target_count is not None and trace_count > 0:
            ratio = None if target_count <= 0 else max(trace_count / target_count, target_count / trace_count)
            if ratio is not None and ratio >= 4.0:
                warnings.append({
                    "kind": "ldgsts_opcode_scope_mismatch",
                    "trace_kind": "ldgsts_async_copy",
                    "trace_requests": trace_count,
                    "ncu_metric": "smsp__inst_executed_op_ldgsts.sum",
                    "ncu_value": target_count,
                    "ratio": ratio,
                    "detail": "post-replay scope diagnostic only; LDGSTS target was not used as a replay input or prediction selector",
                })
        # LTC128B is a supported SASS form, but GA102's NCU scope can book
        # LTC128B.CONSTANT traffic outside the ordinary TEX global-load
        # endpoint on some kernels. Keep this as a post-replay diagnostic;
        # never subtract the form from hierarchy traffic globally because
        # vendor/SOTA controls show LTC128B also contributing to TEX loads.
        trace_row = memory_scope.get("global_load_ltc128b")
        target_row = target_metrics.get("smsp__sass_inst_executed_op_global_ld.sum")
        trace_count = finite(trace_row.get("requests")) if isinstance(trace_row, dict) else None
        target_count = finite(target_row.get("value")) if isinstance(target_row, dict) else None
        if trace_count is not None and target_count is not None and trace_count > 0:
            ratio = None if target_count <= 0 else max(trace_count / target_count, target_count / trace_count)
            if ratio is not None and ratio >= 4.0:
                warnings.append({
                    "kind": "global_load_ltc128b_scope_mismatch",
                    "trace_kind": "global_load_ltc128b",
                    "trace_requests": trace_count,
                    "ncu_metric": "smsp__sass_inst_executed_op_global_ld.sum",
                    "ncu_value": target_count,
                    "ratio": ratio,
                    "detail": "post-replay LTC128B scope diagnostic only; broad hierarchy traffic remains unchanged",
                })
    counters = launch.get("counters", {})
    l2_by_source = counters.get("l2_by_source", {}) if isinstance(counters, dict) else {}
    l2_count = sum(
        float(value.get("sectors", 0))
        for value in l2_by_source.values()
        if isinstance(value, dict)
    )
    dram = counters.get("dram", {}) if isinstance(counters, dict) else {}
    dram_count = finite(dram.get("read_sectors")) if isinstance(dram, dict) else None
    for kind, trace_count, ncu_metric in (
        ("l2_source_sectors", l2_count, "lts__t_sectors_srcunit_tex.sum"),
        ("dram_read_sectors", dram_count, "dram__sectors_read.sum"),
    ):
        target_row = target_metrics.get(ncu_metric)
        if not isinstance(target_row, dict) or trace_count is None or trace_count <= 0:
            continue
        target_count = finite(target_row.get("value"))
        if target_count is None:
            continue
        ratio = None if target_count <= 0 else max(trace_count / target_count, target_count / trace_count)
        if ratio is not None and ratio < 4.0:
            continue
        warnings.append({
            "kind": "operation_scope_mismatch",
            "trace_kind": kind,
            "trace_requests": trace_count,
            "ncu_metric": ncu_metric,
            "ncu_value": target_count,
            "ratio": ratio,
            "detail": "post-replay scope diagnostic only; simulator counters and NCU operation domain may not be like-for-like",
        })
    native_metrics = launch.get("native_metrics", {}).get("metrics", {})
    if isinstance(native_metrics, dict):
        # H66/H73 GCC candidates are source-domain observers, not predictions
        # of a physical GCC request count. Retain a compact post-replay
        # crosswalk warning when an independent NCU GCC endpoint is grossly
        # different; never use the warning to alter DRAM scoring or timing.
        for candidate, ncu_metric, trace_kind in (
            (
                "dram_gcc_instruction_sector_candidate",
                "lts__t_sectors_srcunit_gcc.sum",
                "gcc_instruction_sector_candidate",
            ),
            (
                "dram_gcc_constant_sector_candidate",
                "lts__t_sectors_srcunit_gcc.sum",
                "gcc_constant_sector_candidate",
            ),
        ):
            candidate_row = native_metrics.get(candidate)
            target_row = target_metrics.get(ncu_metric)
            candidate_value = finite(candidate_row.get("value")) if isinstance(candidate_row, dict) else None
            target_value = finite(target_row.get("value")) if isinstance(target_row, dict) else None
            if candidate_value is None or target_value is None or candidate_value <= 0:
                continue
            ratio = None if target_value <= 0 else max(candidate_value / target_value, target_value / candidate_value)
            if ratio is None or ratio < 4.0:
                continue
            warnings.append({
                "kind": "gcc_scope_candidate_mismatch",
                "trace_kind": trace_kind,
                "trace_requests": candidate_value,
                "ncu_metric": ncu_metric,
                "ncu_value": target_value,
                "ratio": ratio,
                "detail": "observer-only GCC/TEX source crosswalk; no DRAM score, replay input, or timing selection",
            })
    return warnings


def scope_warning_metrics(warnings: list[dict[str, Any]]) -> set[str]:
    """Expand a scope warning to the metric endpoints it invalidates."""
    metrics: set[str] = set()
    for warning in warnings:
        kind = warning.get("trace_kind")
        if kind == "global_load":
            metrics.update({
                "smsp__sass_inst_executed_op_global_ld.sum",
            })
        elif kind == "global_load_opcode":
            # Opcode scope is independent of the L1TEX sector endpoint: the
            # latter may include the LDGSTS source traffic on its documented
            # op_ld/op_ldgsts surfaces.
            metrics.update({
                "smsp__sass_inst_executed_op_global_ld.sum",
            })
        elif kind == "ldgsts_opcode_scope_mismatch":
            metrics.update({
                "smsp__inst_executed_op_ldgsts.sum",
            })
        elif kind == "global_load_ltc128b":
            metrics.update({
                "smsp__sass_inst_executed_op_global_ld.sum",
                "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
                "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
                "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
            })
        elif kind == "global_load_legacy":
            metrics.update({
                "smsp__sass_inst_executed_op_global_ld.sum",
                "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
                "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
                "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
            })
        elif kind == "global_store":
            metrics.update({
                "smsp__sass_inst_executed_op_global_st.sum",
                "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
            })
        elif kind == "local_load":
            metrics.update({
                "smsp__sass_inst_executed_op_local_ld.sum",
                "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
            })
        elif kind == "local_store":
            metrics.update({
                "smsp__sass_inst_executed_op_local_st.sum",
                "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
            })
        elif kind == "l2_source_sectors":
            metrics.update({
                "lts__t_sectors_srcunit_tex.sum",
                "lts__t_sectors_srcunit_tex_lookup_hit.sum",
                "lts__t_sectors_srcunit_tex_lookup_miss.sum",
                "lts__t_sectors_srcunit_tex_op_read.sum",
                "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
                "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
            })
        elif kind == "dram_read_sectors":
            metrics.add("dram__sectors_read.sum")
    return metrics


def add_observation(
    table: dict[str, dict[str, Any]],
    entry: dict[str, Any],
    target_metrics: dict[str, Any],
    mode: str,
    capture: str,
    join_status: str,
    scope_mismatch_metrics: set[str] | None = None,
    allow_ordered_fallback: bool = False,
) -> dict[str, Any]:
    name = entry["metric"]
    workload_family = capture.split("_", 1)[0]
    row = table.setdefault(name, {
        # `family` is retained as a compatibility alias for the metric family.
        "family": family(name),
        "metric_family": family(name),
        "workload_families": set(),
        "modes": set(),
        "prediction_status": collections.Counter(),
        "reasons": collections.Counter(),
        "predicted": 0,
        "compared": 0,
        "target_available": 0,
        "scope_mismatch": 0,
        "unit_mismatch": 0,
        "target_missing": 0,
        "zero_opportunity": 0,
        "prediction_value_missing": 0,
        "predicted_zero_on_zero_target": 0,
        "predicted_nonzero_on_zero_target": 0,
        "errors": [],
        "relative_errors": [],
        "errors_by_capture": collections.defaultdict(list),
        "relative_by_family": collections.defaultdict(list),
        "relative_by_workload_family": collections.defaultdict(list),
        "relative_by_join": collections.defaultdict(list),
        "source_counts": collections.Counter(),
        "aliases_not_scored": 0,
        "different_numeric_alternatives": 0,
        "alternative_conflicts": 0,
        "zero_targets": 0,
        "exact": 0,
        "captures": set(),
    })
    row["modes"].add(mode)
    row["captures"].add(capture)
    row["workload_families"].add(workload_family)
    status = entry["status"]
    row["prediction_status"][status] += 1
    if entry.get("reason"):
        row["reasons"][entry["reason"]] += 1
    estimate = entry.get("estimate")
    row["source_counts"][entry["source_key"]] += 1
    alternatives = entry.get("alternative_predictions", [])
    row["aliases_not_scored"] += len(alternatives)
    row["different_numeric_alternatives"] += sum(
        candidate["status"] == "predicted"
        and candidate.get("estimate") is not None
        and estimate is not None
        and candidate["estimate"] != estimate
        for candidate in alternatives
    )
    if entry.get("alternative_conflict"):
        row["alternative_conflicts"] += 1
    result = {"metric": name, "source_key": entry["source_key"],
              "estimate": estimate, "model_status": status, "model_unit": entry["unit"]}
    if entry.get("semantics"):
        result["semantics"] = entry["semantics"]
    if alternatives:
        result["alternative_predictions"] = alternatives
    if entry.get("alternative_conflict"):
        result["alternative_conflict"] = True
    # A shape/order fallback is useful coverage metadata but is not an exact
    # launch identity. Never score it as NCU accuracy: the existing SM86
    # profiles contain one selected kernel row while many traces contain many
    # launches, and profiler-off names are often ambiguous.
    if join_status != "identity" and not (allow_ordered_fallback and join_status == "ordered_fallback"):
        return {**result, "status": "join_not_exact", "join_status": join_status}
    target = target_metrics.get(name, {})
    truth = finite(target.get("value")) if isinstance(target, dict) else None
    if status != "predicted" or estimate is None:
        if status == "predicted" and estimate is None:
            row["prediction_value_missing"] += 1
        return {**result, "status": "model_unavailable", "reason": entry.get("reason")}
    row["predicted"] += 1
    if truth is None:
        row["target_missing"] += 1
        return {**result, "status": "target_missing"}
    row["target_available"] += 1
    result.update(target=truth, target_unit=target.get("unit"))
    if scope_mismatch_metrics and name in scope_mismatch_metrics:
        row["scope_mismatch"] += 1
        result["scope_mismatch_reason"] = "independent NCU/trace operation scope warning"
        return {**result, "status": "scope_mismatch"}
    if truth == 0:
        row["zero_opportunity"] += 1
        if estimate == 0:
            row["predicted_zero_on_zero_target"] += 1
        else:
            row["predicted_nonzero_on_zero_target"] += 1
    # NCU exports these documented count metrics with an empty unit on SM120.
    # This is a name-qualified count relation, not permission to equate arbitrary
    # unitless metrics with wavefronts or to convert timing domains.
    shared_wavefront = name in {
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_atom.sum",
    } and entry["unit"] == "wavefront" and target.get("unit") == ""
    target_unit = str(target.get("unit", ""))
    target_scale = target_scale_for_metric(name, entry["unit"], target_unit)
    if target_scale is None and not shared_wavefront:
        row["unit_mismatch"] += 1
        return {**result, "status": "unit_mismatch"}
    # Preserve the raw NCU value/unit for auditability while comparing in the
    # simulator row's declared unit.  This is a presentation crosswalk only;
    # the converted value never enters replay or metric selection.
    compared_truth = truth * (target_scale if target_scale is not None else 1.0)
    result["comparison_target"] = compared_truth
    result["comparison_target_unit"] = entry["unit"]
    error = abs(estimate - compared_truth)
    row["compared"] += 1
    row["errors"].append(error)
    row["errors_by_capture"][capture].append(error)
    relative = error / abs(compared_truth) if compared_truth != 0 else None
    if relative is not None:
        row["relative_errors"].append(relative)
        row["relative_by_family"][workload_family].append(relative)
        row["relative_by_workload_family"][workload_family].append(relative)
        row["relative_by_join"][join_status].append(relative)
    else:
        row["zero_targets"] += 1
    row["exact"] += int(error == 0.0)
    return {**result, "status": "compared", "absolute_error": error,
            "relative_error_nonzero_target": relative}


def reduce_mode(
    path: Path,
    mode: str,
    profiles: Path,
    output,
    table: dict[str, dict[str, Any]],
    target_inventory: collections.Counter,
    allow_ordered_fallback: bool = False,
) -> dict[str, int]:
    key = "launches" if mode == "full" else "fast_frontier"
    counts = collections.Counter()
    with open_jsonl_reader(path) as stream:
        for line_number, line in enumerate(stream, 1):
            capture = json.loads(line)
            rel = str(capture.get("canonical_capture", ""))
            if not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
                raise ValueError(f"{path}:{line_number}: invalid capture identity")
            ncu = parse_ncu_raw(profiles / rel / "raw.csv")
            launches = capture.get(key, [])
            if not isinstance(launches, list):
                raise ValueError(f"{path}:{line_number}: {key} is not an array")
            counts["captures"] += 1
            counts[f"capture_status_{capture.get('status', 'unspecified')}"] += 1
            counts["launches"] += len(launches)
            for join in join_ncu_rows(launches, ncu):
                counts[f"join_{join['join_status']}"] += 1
                li, ni = join.get("launch_index"), join.get("ncu_row_index")
                if li is None or ni is None:
                    output.write(json.dumps({"capture": rel, "mode": mode, **join}) + "\n")
                    continue
                launch = launches[li]
                target_metrics = ncu[ni].get("metrics", {})
                target_inventory.update(target_metrics.keys())
                warnings = scope_warnings(launch, target_metrics)
                counts["scope_warnings"] += len(warnings)
                warning_metrics = scope_warning_metrics(warnings)
                results = []
                for entry in selected_entries(launch, mode):
                    results.append(add_observation(
                        table, entry, target_metrics, mode, rel, join["join_status"], warning_metrics
                    , allow_ordered_fallback=allow_ordered_fallback
                    ))
                output.write(json.dumps({"capture": rel, "mode": mode, **join, "scope_warnings": warnings, "results": results}, separators=(",", ":"), allow_nan=False) + "\n")
    return dict(counts)


def render_summary(table: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    unavailable: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for name, row in sorted(table.items()):
        compared = row["compared"]
        errors = row["errors"]
        workload_relative = {
            name: distribution(values)
            for name, values in sorted(row["relative_by_workload_family"].items())
        }
        item = {
            "family": row["family"],
            "metric_family": row["metric_family"],
            "workload_families": sorted(row["workload_families"]),
            "modes": sorted(row["modes"]),
            "predicted": row["predicted"],
            "compared": compared,
            "target_available": row["target_available"],
            "scope_mismatch": row["scope_mismatch"],
            "target_missing": row["target_missing"],
            "unit_mismatch": row["unit_mismatch"],
            "zero_opportunity": row["zero_opportunity"],
            "prediction_value_missing": row["prediction_value_missing"],
            "predicted_zero_on_zero_target": row["predicted_zero_on_zero_target"],
            "predicted_nonzero_on_zero_target": row["predicted_nonzero_on_zero_target"],
            "exact": row["exact"],
            "prediction_status": dict(row["prediction_status"]),
            "reasons": dict(row["reasons"]),
            "absolute_error": distribution(errors),
            "relative_error_nonzero_targets": distribution(row["relative_errors"]),
            "equal_capture_median_absolute_error": distribution([
                statistics.median(values) for values in row["errors_by_capture"].values()
            ]),
            "family_relative_error_nonzero_targets": {
                name: distribution(values) for name, values in sorted(row["relative_by_family"].items())
            },
            # The old key is metric-family stratification.  Keep it for
            # compatibility and expose the workload-family stratification
            # under an unambiguous name.
            "workload_family_relative_error_nonzero_targets": workload_relative,
            "join_relative_error_nonzero_targets": {
                name: distribution(values) for name, values in sorted(row["relative_by_join"].items())
            },
            "source_counts": dict(row["source_counts"]),
            "aliases_not_scored": row["aliases_not_scored"],
            "different_numeric_alternatives": row["different_numeric_alternatives"],
            "alternative_conflicts": row["alternative_conflicts"],
            "zero_targets": row["zero_targets"],
        }
        metrics[name] = item
        if row["predicted"] == 0:
            unavailable[row["metric_family"]].append({
                "metric": name,
                "rows": sum(row["prediction_status"].values()),
                "reasons": dict(row["reasons"]),
            })
    return {
        "metrics": metrics,
        "unavailable_by_family": {family_name: names for family_name, names in sorted(unavailable.items())},
        "conflicted_metrics": sorted(
            name for name, item in metrics.items() if item["alternative_conflicts"]
        ),
        "conflicted_metric_count": sum(
            bool(item["alternative_conflicts"]) for item in metrics.values()
        ),
        "available_metric_count": sum(item["predicted"] > 0 for item in metrics.values()),
        "compared_metric_count": sum(item["compared"] > 0 for item in metrics.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--fast", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--include-ordered-fallback",
        action="store_true",
        help="score ordered-fallback launch joins in addition to exact identity joins; report this expanded pairing explicitly",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tables: dict[str, dict[str, Any]] = {"full": {}, "fast": {}}
    target_inventory: collections.Counter = collections.Counter()
    with open_jsonl_writer(args.output / "metric_comparisons.jsonl") as output:
        full_counts = reduce_mode(args.full, "full", args.profiles, output, tables["full"], target_inventory, args.include_ordered_fallback)
        fast_counts = reduce_mode(args.fast, "fast", args.profiles, output, tables["fast"], target_inventory, args.include_ordered_fallback)
    full_counts["scope_mismatch_predictions"] = sum(
        row["scope_mismatch"] for row in tables["full"].values()
    )
    fast_counts["scope_mismatch_predictions"] = sum(
        row["scope_mismatch"] for row in tables["fast"].values()
    )
    by_mode = {mode: render_summary(table) for mode, table in tables.items()}
    available = {name for table in tables.values() for name, row in table.items() if row["predicted"]}
    declared = {name for table in tables.values() for name in table}
    unavailable: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for name in sorted((set(target_inventory) | declared) - available):
        unavailable[family(name)].append({"metric": name, "target_rows": target_inventory[name]})
    summary = {
        "schema_version": 4,
        "scope": "post-replay metric predictions; NCU values never replay/model inputs",
        "comparison_view": "scope_compatible_with_ordered_fallback" if args.include_ordered_fallback else "scope_compatible_secondary",
        "raw_all_row_ape_authoritative": True,
        "raw_all_row_ape_included": False,
        "scope_mismatch_policy": "retain_rows_exclude_from_raw_all_row_comparison_ape",
        "selection": "one prediction per metric/launch/mode; finite declared metric, cache observer, native metric, then timing fallback; alternatives retained but not scored; no NCU-based selection",
        "ordered_fallback_scored": bool(args.include_ordered_fallback),
        "join_pairing_policy": "exact identity plus user-authorized ordered fallback" if args.include_ordered_fallback else "exact identity only; ordered fallback retained as coverage metadata",
        "sources": {"full": str(args.full), "fast": str(args.fast), "profiles": str(args.profiles)},
        "jsonl_compression": jsonl_compression_description(),
        "full": full_counts,
        "fast": fast_counts,
        "by_mode": by_mode,
        "available_metric_count": len(available),
        "unavailable_metric_count": sum(map(len, unavailable.values())),
        "unavailable_by_family": dict(sorted(unavailable.items())),
        "conflicted_metric_count": sum(
            mode_summary["conflicted_metric_count"] for mode_summary in by_mode.values()
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "summary.md").open("w") as report:
        report.write("# SM86 metric coverage\n\n")
        report.write("Post-replay reduction of emitted predictions; NCU values were not replay inputs.\n\n")
        report.write("Each selected launch contributes at most once per metric and mode. Full and fast errors are not pooled; aliases and differing alternatives are retained in compressed rows, not counted as extra observations. In expanded mode, ordered-fallback joins are included by explicit user authorization; they remain a lower-confidence pairing than exact identity joins. Relative errors exclude zero targets.\n\n")
        for mode, mode_summary in by_mode.items():
            report.write(f"## {mode.capitalize()} scope-compatible compared metrics\n\n| metric | family | compared | median absolute error | P90 | mean APE | median APE | P90 APE | exact |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for name, item in mode_summary["metrics"].items():
                if item["compared"] == 0:
                    continue
                error = item["absolute_error"]
                relative = item["relative_error_nonzero_targets"]
                mean_ape = relative.get("mean")
                report.write(f"| `{name}` | {item['family']} | {item['compared']} | {error['median']} | {error['p90']} | {mean_ape} | {relative['median']} | {relative['p90']} | {item['exact']} |\n")
        report.write("\n## Currently unavailable model families\n\n")
        report.write("Metrics with no simulator prediction in either retained mode are listed once by family; raw NCU distributions are intentionally omitted.\n\n")
        for family_name, entries in summary["unavailable_by_family"].items():
            names = ", ".join(
                f"`{entry['metric']}` ({entry['target_rows']} target rows)" for entry in entries
            )
            report.write(f"- **{family_name}:** {names}\n")
        report.write("\n## Post-replay scope diagnostics\n\n")
        report.write("Opcode-scope warnings flag gross disagreement between trace-side logical memory counts and an independent NCU opcode metric. They never alter replay or prediction selection. Explicit scope-mismatch rows remain in compressed `metric_comparisons.jsonl.zst` and are excluded from raw/all-row comparison APE; the resulting scope-compatible table is a secondary like-for-like view.\n\n")
        report.write(f"- Full warnings: **{summary['full'].get('scope_warnings', 0)}**; fast warnings: **{summary['fast'].get('scope_warnings', 0)}**.\n")
        report.write(f"- Full predictions excluded from raw/all-row comparison APE for explicit scope mismatch: **{summary['full'].get('scope_mismatch_predictions', 0)}**; fast: **{summary['fast'].get('scope_mismatch_predictions', 0)}**.\n")
        report.write("\n## Conflicting prediction alternatives\n\n")
        report.write("Metrics with incompatible finite alternatives are retained in compressed rows but excluded from scoring until endpoint/scope provenance selects one.\n\n")
        for mode, mode_summary in by_mode.items():
            names = ", ".join(f"`{name}`" for name in mode_summary["conflicted_metrics"])
            if names:
                report.write(f"- **{mode}:** {names}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
