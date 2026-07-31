from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import pytest

from eval.contracts.validation import (
    CANONICAL_REGISTRY_SHA256,
    COHORT_ASSERTION_LAUNCH_FIELDS,
    METRIC_KEYS,
    REPLAY_AUTHENTICATED_LAUNCH_FIELDS,
    ContractValidationError,
    MatchedLabelAuthority,
    canonical_launch_key_token,
    canonical_sha256,
    load_json_file_strict,
    loads_json_strict,
    schema_document,
    validate_launch_key,
    validate_manifest,
    validate_matched_labels,
    validate_registry_projection,
    validate_report,
    validate_result,
    validate_schema_definitions,
    validate_sim_error,
)

ROOT = Path(__file__).resolve().parents[2]
PROJECTION_PATH = ROOT / "eval/contracts/metric_registry.7plus1.v1.json"
CANONICAL_REGISTRY_PATH = ROOT / "tools/local_labels/metric_registry.json"
CURRENT_V1_RESULT_PATH = (
    ROOT
    / "sim/run/tests/fixtures/addr_carry_canary_bundle/result.golden.json"
)
MANIFEST_V1_SCHEMA_PATH = ROOT / "sim/run/schema/sim_run_manifest.v1.schema.json"
RESULT_V1_SCHEMA_PATH = ROOT / "sim/run/schema/sim_result.v1.schema.json"

LOCAL_LABELS = ROOT / "tools/local_labels"
if str(LOCAL_LABELS) not in sys.path:
    sys.path.insert(0, str(LOCAL_LABELS))

import metric_registry as registry_consumer  # noqa: E402

COUNT_KEYS = frozenset(METRIC_KEYS[:4])
RATE_KEYS = frozenset(METRIC_KEYS[4:7])
NCU_METRICS = {
    "smsp_inst_executed_sum": "smsp__inst_executed.sum",
    "traffic_gmem_ld_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "traffic_gmem_st_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "traffic_dram_read_sectors": "dram__sectors_read.sum",
    "cache_hit_rate_l1tex_read_sector": (
        "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct"
    ),
    "cache_hit_rate_l1tex_sector": "l1tex__t_sector_hit_rate.pct",
    "cache_hit_rate_l2_sector": "lts__t_sector_hit_rate.pct",
    "exec_cycles_gpc_max": "gpc__cycles_elapsed.max",
}
PROJECTED_ROW_FIELDS = (
    "key",
    "ncu_metric",
    "unit",
    "family",
    "metric_class",
    "carrier",
    "valid_min",
    "valid_max",
    "parser_transform",
    "aggregation",
    "repeatability_rule",
    "nullable",
    "deployability",
    "availability",
    "per_arch_availability",
    "model_io_role",
)
HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64


def _load_json(path: Path) -> dict[str, Any]:
    value = load_json_file_strict(path)
    assert isinstance(value, dict)
    return value


def _seal(document: dict[str, Any], member: str) -> dict[str, Any]:
    document[member] = canonical_sha256(document, member)
    return document


def _assert_code(
    code: str, validator: Callable[[Any], Any], document: Any
) -> ContractValidationError:
    with pytest.raises(ContractValidationError) as caught:
        validator(document)
    assert caught.value.code == code
    return caught.value


def _walk(value: Any, path: str = "$"):
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, f"{path}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}/{index}")


def _convert_v1_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                child.replace("#/definitions/", "#/$defs/")
                if key == "$ref" and isinstance(child, str)
                else _convert_v1_refs(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_convert_v1_refs(child) for child in value]
    return value


def _valid_launch_key(ordinal: int = 0) -> dict[str, Any]:
    return {
        "workload_id": "fixture/vector-add:v1",
        "matched_execution_id": "matched-execution-0001",
        "process_launch_ordinal": ordinal,
        "kernel_mangled_name": "_Z10vector_addPKfS0_Pfi",
        "grid": {"x": 2, "y": 1, "z": 1},
        "block": {"x": 128, "y": 1, "z": 1},
        "device_id": 0,
        "stream_id": 7,
        "binary_architecture": "sm_86",
    }


def _valid_manifest() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "scope": "fixture/vector-add:v1",
        "architecture": {"arch_id": "sm_86", "sku": "rtx_3090_ti"},
        "configuration_evidence": {
            "arch_card_sha256": HEX_A,
            "pchase_card_sha256": HEX_B,
        },
        "launches": [
            {
                "launch_evidence_manifest_path": "traces/vector-add/launch.json",
                "expected_identity_sha256": HEX_C,
                "launch_key": _valid_launch_key(),
            }
        ],
        "replay_mode": {"mode": "l1_shared_l2", "dram_bandwidth": "carded"},
    }


def _predicted_exact(value: int) -> dict[str, Any]:
    return {
        "status": "predicted",
        "value": {"kind": "exact", "semantics": "simulator_scope", "value": value},
    }


def _predicted_fraction(numerator: int, denominator: int) -> dict[str, Any]:
    value = None if denominator == 0 else numerator / denominator
    return {
        "status": "predicted",
        "value": {
            "kind": "fraction",
            "semantics": "simulator_scope",
            "value": value,
            "numerator": numerator,
            "denominator": denominator,
        },
    }


def _required_cycles() -> dict[str, str]:
    return {
        "status": "required_unimplemented",
        "blocking": "aggregation_semantics_study",
    }


def _metric_map() -> dict[str, Any]:
    return {
        "smsp_inst_executed_sum": _predicted_exact(64),
        "traffic_gmem_ld_sectors": _predicted_exact(8),
        "traffic_gmem_st_sectors": _predicted_exact(4),
        "traffic_dram_read_sectors": _predicted_exact(2),
        "cache_hit_rate_l1tex_read_sector": _predicted_fraction(3, 4),
        "cache_hit_rate_l1tex_sector": _predicted_fraction(4, 8),
        "cache_hit_rate_l2_sector": _predicted_fraction(7, 8),
        "exec_cycles_gpc_max": _required_cycles(),
    }


def _shared_configuration() -> dict[str, Any]:
    return {
        "replay_mode": {"mode": "l1_shared_l2", "dram_bandwidth": "carded"},
        "co_residency": "modeled",
        "issue": {
            "arch": "sm_86",
            "l1_hit_latency_cycles": 34,
            "l1_miss_latency_cycles": 244,
            "global_flat_latency_cycles": 423,
            "shared_write_barrier_latency_cycles": 24,
            "non_memory_write_barrier_latency_cycles": 25,
            "read_barrier_latency_cycles": 8,
            "max_resident_warps_per_sm": 48,
            "latency_provenance": "architecture_card",
            "issue_hypothesis_provenance": (
                "architecture_local_reviewed_hypothesis"
            ),
        },
        "l1": {
            "status": "modeled",
            "value": {
                "arch": "sm_86",
                "line_capacity": 1024,
                "capacity_bytes": 131072,
                "mshr_count": 32,
                "hit_latency_cycles": 34,
                "miss_latency_cycles": 244,
                "capacity_provenance": "architecture_card",
                "mshr_provenance": "architecture_local_reviewed_hypothesis",
                "replacement_model": "lru",
                "replacement_provenance": "architecture_local_reviewed_default",
                "set_mapping_model": "line_mod",
                "set_mapping_provenance": "architecture_local_reviewed_default",
            },
        },
        "l2": {
            "status": "modeled",
            "value": {
                "arch": "sm_86",
                "capacity_bytes": 6291456,
                "line_bytes": 128,
                "sector_bytes": 32,
                "num_slices": 6,
                "associativity": 16,
                "replacement": "lru",
                "slice_hash": "line_mod",
                "geometry_provenance": "architecture_card",
                "hypothesis_provenance": "architecture_local_reviewed_default",
            },
        },
        "dram": {
            "status": "modeled",
            "value": {
                "mode": "carded",
                "bytes_per_sec": 1008000000000,
                "sm_clock_hz": 1860000000,
                "provenance": "architecture_card_locked_operating_point",
            },
        },
    }


def _flat_configuration() -> dict[str, Any]:
    config = _shared_configuration()
    config["replay_mode"] = {"mode": "flat"}
    config["l1"] = {"status": "unmodeled", "reason": "replay_mode_flat"}
    config["l2"] = {
        "status": "unmodeled",
        "reason": "replay_mode_without_shared_l2",
    }
    config["dram"] = {
        "status": "unmodeled",
        "reason": "replay_mode_without_shared_l2",
    }
    return config


def _trace_bundle() -> dict[str, Any]:
    return {
        "bundle_id": "trace-bundle-vector-add-v1",
        "bundle_path": "traces/vector-add/bundle.json",
        "bundle_sha256": HEX_B,
        "wire_version": "traces_enhanced.v1",
        "total_size_bytes": 380,
        "record_count": 64,
        "instruction_count": 64,
        "components": {
            "root_dynamic_trace": {"size_bytes": 100, "sha256": HEX_A},
            "threadblock_payloads": {
                "file_count": 1,
                "total_size_bytes": 200,
                "manifest_sha256": HEX_B,
                "content_sha256": HEX_C,
            },
            "execution_metadata": {"size_bytes": 30, "sha256": HEX_C},
            "control_bits_sidecar": {"size_bytes": 40, "sha256": HEX_D},
            "stats": {"size_bytes": 10, "sha256": HEX_A},
        },
    }


def _valid_result() -> dict[str, Any]:
    launch_key = _valid_launch_key()
    result = {
        "schema_version": "sim_result.v2",
        "result_sha256": "0" * 64,
        "build_identity": {
            "sim_git_commit": "1" * 40,
            "sim_git_dirty": False,
            "binary_sha256": HEX_A,
            "rustc_version": "rustc 1.90.0",
            "cargo_profile": "release",
            "claimability": {"status": "claimable", "basis": "clean_build"},
        },
        "trace_identity": _trace_bundle(),
        "arch_card_identity": {
            "card_id": "sm86-rtx3090ti-v1",
            "card_sha256": HEX_C,
            "sm_arch": "sm_86",
            "sku": "rtx_3090_ti",
        },
        "configuration_identity": {
            "configuration_id": "shared-sm86-v1",
            "run_manifest_sha256": HEX_D,
            "resolved_configuration_sha256": HEX_A,
            "arch_card_sha256": HEX_C,
            "pchase_card_sha256": HEX_B,
        },
        "resolved_configuration": _shared_configuration(),
        "launches": [
            {"launch_key": deepcopy(launch_key), "metrics": _metric_map()}
        ],
        "aggregate": {
            "scope": "all_launches",
            "launch_count": 1,
            "metrics": _metric_map(),
        },
        "diagnostics": {
            "admissibility": "non_admissible",
            "external_runtime_role": (
                "internal_timings_do_not_replace_spawn_to_exit_wall_time"
            ),
            "per_launch": [
                {
                    "launch_key": deepcopy(launch_key),
                    "diagnostic_last_issue_cycles": 123,
                    "terminal_modeled_event_time": {
                        "status": "modeled",
                        "value": 456,
                    },
                    "binding_cycles_role": (
                        "diagnostic_only_not_binding_prediction"
                    ),
                }
            ],
            "internal_phase_timings_ns": {
                "trace_open_ns": 1,
                "trace_read_decode_ns": 2,
                "initialization_ns": 3,
                "replay_event_memory_ns": 4,
                "emission_ns": 5,
                "result_serialization_ns": 6,
            },
        },
        "exit_status": {"status": "success", "exit_code": 0},
    }
    return _seal(result, "result_sha256")


def _label_metric(key: str) -> dict[str, Any]:
    if key in RATE_KEYS:
        unit, value, aggregation, rule = "pct", 0.5, "mean_fraction", "stddev_abs"
    elif key == "exec_cycles_gpc_max":
        unit, value, aggregation, rule = (
            "cycles",
            1000.0,
            "mean_nonneg",
            "coefficient_of_variation",
        )
    else:
        unit, value, aggregation, rule = (
            "count",
            64.0,
            "mean_nonneg",
            "relative_tolerance",
        )
    return {
        "metric_key": key,
        "ncu_metric": NCU_METRICS[key],
        "unit": unit,
        "outcome": {"status": "available", "value": value},
        "repeat_metadata": {
            "repeat_count": 3,
            "aggregation": aggregation,
            "dispersion": {"status": "available", "rule": rule, "scalar": 0.01},
        },
    }


def _valid_matched_labels() -> dict[str, Any]:
    labels = {
        "schema_version": "matched_labels.v1",
        "matched_labels_sha256": "0" * 64,
        "source_label_identity": {
            "label_set_id": "labels-vector-add-v5",
            "label_set_sha256": HEX_A,
            "label_schema_version": 5,
            "label_schema_sha256": HEX_B,
            "source_manifest_id": "measurement-vector-add-v2",
            "source_manifest_sha256": HEX_C,
            "authority_role": "independent_label_source_not_result_derived",
        },
        "crosswalk_identity": {
            "crosswalk_id": "crosswalk-vector-add-v1",
            "crosswalk_sha256": HEX_D,
            "crosswalk_version": 1,
            "authority_role": "independently_matched_not_result_authorized",
        },
        "architecture": {
            "binary_architecture": "sm_86",
            "gpu_sku": "rtx_3090_ti",
        },
        "launches": [
            {
                "launch_key": _valid_launch_key(),
                "kernel_family": "vector",
                "metrics": [_label_metric(key) for key in METRIC_KEYS],
            }
        ],
    }
    return _seal(labels, "matched_labels_sha256")


def _label_authority(labels: dict[str, Any] | None = None) -> MatchedLabelAuthority:
    artifact = labels or _valid_matched_labels()
    return MatchedLabelAuthority(
        matched_labels_sha256=artifact["matched_labels_sha256"],
        label_set_id=artifact["source_label_identity"]["label_set_id"],
        label_set_sha256=artifact["source_label_identity"]["label_set_sha256"],
        crosswalk_id=artifact["crosswalk_identity"]["crosswalk_id"],
        crosswalk_sha256=artifact["crosswalk_identity"]["crosswalk_sha256"],
    )


def _error_stats(n: int) -> dict[str, Any]:
    return {
        "n": n,
        "mean_absolute_error": 0.0 if n else None,
        "max_absolute_error": 0.0 if n else None,
        "signed_bias": 0.0 if n else None,
    }


def _noise(scored: int) -> dict[str, Any]:
    return {
        "comparison_semantics": (
            "absolute_error_native_unit_lte_declared_dispersion_threshold"
        ),
        "evaluated_count": scored,
        "within_count": scored,
        "not_applicable_count": 0,
        "not_applicable_reasons": [],
    }


def _metric_summary(key: str) -> dict[str, Any]:
    scored = 0 if key == "exec_cycles_gpc_max" else 1
    unavailable = 1 if key == "exec_cycles_gpc_max" else 0
    common = {
        "metric_key": key,
        "ncu_metric": NCU_METRICS[key],
        "kernel_family": "vector",
        "n_labeled": 1,
        "n_scored": scored,
        "n_unavailable": unavailable,
        "within_label_noise": _noise(scored),
    }
    if key in COUNT_KEYS:
        return {
            **common,
            "unit": "count",
            "unavailable_by_type": [],
            "summary": {
                "kind": "count",
                "exact_count": 1,
                "zero_label_nonzero_prediction_count": 0,
                "relative_error": _error_stats(1),
                "absolute_error": _error_stats(1),
            },
        }
    if key in RATE_KEYS:
        return {
            **common,
            "unit": "pct",
            "unavailable_by_type": [],
            "summary": {
                "kind": "rate",
                "exact_count": 1,
                "error_percentage_points": _error_stats(1),
            },
        }
    return {
        **common,
        "unit": "cycles",
        "unavailable_by_type": [
            {
                "status": "required_unimplemented",
                "blocking": "aggregation_semantics_study",
                "count": 1,
            }
        ],
        "summary": {
            "kind": "cycle",
            "status": "required_unimplemented",
            "blocking": "aggregation_semantics_study",
        },
    }


def _report_identities(labels: dict[str, Any]) -> dict[str, Any]:
    projection = _load_json(PROJECTION_PATH)
    return {
        "report_id": "accuracy-vector-add-v1",
        "result_schema_version": "sim_result.v2",
        "result_file_sha256": HEX_A,
        "result_canonical_sha256": HEX_B,
        "registry_projection_id": "metric_registry_projection.v1",
        "registry_projection_sha256": projection["projection_sha256"],
        "source_registry_sha256": CANONICAL_REGISTRY_SHA256,
        "arch_card_id": "sm86-rtx3090ti-v1",
        "arch_card_sha256": HEX_A,
        "binary_architecture": "sm_86",
        "gpu_sku": "rtx_3090_ti",
        "simulator_git_dirty": False,
        "simulator_build_claimability": {
            "status": "claimable",
            "basis": "clean_build",
        },
        "evaluator_git_commit": "2" * 40,
        "evaluator_git_dirty": False,
        "evaluator_build_claimability": {
            "status": "claimable",
            "basis": "clean_build",
        },
        "label_binding": {
            "status": "bound",
            "matched_labels_sha256": labels["matched_labels_sha256"],
            "label_set_id": "labels-vector-add-v5",
            "label_set_sha256": HEX_A,
            "crosswalk_id": "crosswalk-vector-add-v1",
            "crosswalk_sha256": HEX_D,
            "authority_role": "independently_admitted_label_and_crosswalk",
        },
    }


def _admitted_checks() -> dict[str, str]:
    return {
        "result_schema": "passed",
        "result_seal": "passed",
        "registry_projection": "passed",
        "architecture": "passed",
        "matched_label_schema": "passed",
        "matched_label_seal": "passed",
        "crosswalk_identity": "passed",
        "label_input_availability": "passed",
        "units": "passed",
    }


def _valid_report() -> tuple[dict[str, Any], MatchedLabelAuthority]:
    labels = _valid_matched_labels()
    authority = _label_authority(labels)
    report = {
        "schema_version": "accuracy_report.v1",
        "report_sha256": "0" * 64,
        "identities": _report_identities(labels),
        "outcome": {
            "status": "admitted",
            "admission": {
                "status": "admitted",
                "checks": _admitted_checks(),
                "failures": [],
            },
            "comparison_failures": [],
            "coverage": {
                "total_cells": 8,
                "labeled_cells": 8,
                "unlabeled_cells": 0,
                "scored_cells": 7,
                "unavailable_cells": 1,
                "comparison_failure_cells": 0,
                "silent_absence_count": 0,
                "unavailable_by_type": [
                    {
                        "status": "required_unimplemented",
                        "blocking": "aggregation_semantics_study",
                        "count": 1,
                    }
                ],
            },
            "per_metric_family": [_metric_summary(key) for key in METRIC_KEYS],
        },
    }
    return _seal(report, "report_sha256"), authority


def _rejected_report() -> dict[str, Any]:
    projection = _load_json(PROJECTION_PATH)
    identities = _report_identities(_valid_matched_labels())
    identities["registry_projection_sha256"] = projection["projection_sha256"]
    identities["label_binding"] = {
        "status": "unavailable",
        "requested_label_set_id": "labels-vector-add-v5",
        "reason": "no_independently_admitted_matched_labels",
    }
    checks = _admitted_checks()
    checks.update(
        {
            "matched_label_schema": "not_run",
            "matched_label_seal": "not_run",
            "crosswalk_identity": "not_run",
            "label_input_availability": "failed",
            "units": "not_run",
        }
    )
    report = {
        "schema_version": "accuracy_report.v1",
        "report_sha256": "0" * 64,
        "identities": identities,
        "outcome": {
            "status": "rejected",
            "admission": {
                "status": "rejected",
                "checks": checks,
                "failures": [
                    {
                        "failure_type": "LABEL_INPUT_UNAVAILABLE",
                        "detail": "independently admitted label input is unavailable",
                        "context": {
                            "requested_label_set_id": "labels-vector-add-v5",
                            "availability": "unavailable",
                        },
                    }
                ],
            },
            "candidate_status": {
                "status": "unknown",
                "reason": "label_input_unavailable",
            },
        },
    }
    return _seal(report, "report_sha256")


_REJECTED_FAILURE_CASES: dict[str, dict[str, Any]] = {
    "RESULT_SCHEMA_UNSUPPORTED": {
        "failed_check": "result_schema",
        "candidate_reason": "result_unavailable",
        "binding_status": "bound",
        "context": {"observed_schema_version": "sim_result.v1"},
    },
    "RESULT_SCHEMA_INVALID": {
        "failed_check": "result_schema",
        "candidate_reason": "result_unavailable",
        "binding_status": "bound",
        "context": {"json_path": "$.launches"},
    },
    "RESULT_SEAL_INVALID": {
        "failed_check": "result_seal",
        "candidate_reason": "result_unavailable",
        "binding_status": "bound",
        "context": {"expected_sha256": HEX_A, "observed_sha256": HEX_B},
    },
    "LABEL_SCHEMA_UNSUPPORTED": {
        "failed_check": "matched_label_schema",
        "candidate_reason": "label_input_unavailable",
        "binding_status": "unavailable",
        "context": {
            "requested_label_set_id": "labels-vector-add-v5",
            "observed_schema_version": "matched_labels.v0",
        },
    },
    "LABEL_INPUT_UNAVAILABLE": {
        "failed_check": "label_input_availability",
        "candidate_reason": "label_input_unavailable",
        "binding_status": "unavailable",
        "context": {
            "requested_label_set_id": "labels-vector-add-v5",
            "availability": "unavailable",
        },
    },
    "LABEL_SEAL_INVALID": {
        "failed_check": "matched_label_seal",
        "candidate_reason": "label_input_unavailable",
        "binding_status": "unavailable",
        "context": {
            "label_set_id": "labels-vector-add-v5",
            "expected_sha256": HEX_A,
            "observed_sha256": HEX_B,
        },
    },
    "CROSSWALK_IDENTITY_INVALID": {
        "failed_check": "crosswalk_identity",
        "candidate_reason": "label_input_unavailable",
        "binding_status": "unavailable",
        "context": {
            "label_set_id": "labels-vector-add-v5",
            "crosswalk_id": "crosswalk-vector-add-v1",
            "reason": "hash_mismatch",
        },
    },
    "REGISTRY_PROJECTION_MISMATCH": {
        "failed_check": "registry_projection",
        "candidate_reason": "admission_failed_before_enumeration",
        "binding_status": "bound",
        "context": {
            "projection_id": "metric_registry_projection.v1",
            "expected_sha256": HEX_A,
            "observed_sha256": HEX_B,
        },
    },
    "ARCH_MISMATCH": {
        "failed_check": "architecture",
        "candidate_reason": "admission_failed_before_enumeration",
        "binding_status": "bound",
        "context": {
            "result_architecture": "sm_86",
            "result_sku": "rtx_3090_ti",
            "label_architecture": "sm_120",
            "label_sku": "rtx_5060_laptop",
        },
    },
    "UNIT_MISMATCH": {
        "failed_check": "units",
        "candidate_reason": "admission_failed_before_enumeration",
        "binding_status": "bound",
        "context": {
            "metric_key": "smsp_inst_executed_sum",
            "expected_unit": "count",
            "observed_unit": "cycles",
        },
    },
}


def _rejected_report_for(
    failure_type: str,
) -> tuple[dict[str, Any], MatchedLabelAuthority | None]:
    case = _REJECTED_FAILURE_CASES[failure_type]
    labels = _valid_matched_labels()
    authority = _label_authority(labels)
    identities = _report_identities(labels)
    if case["binding_status"] == "unavailable":
        identities["label_binding"] = {
            "status": "unavailable",
            "requested_label_set_id": "labels-vector-add-v5",
            "reason": "no_independently_admitted_matched_labels",
        }
        authority = None

    checks = _admitted_checks()
    checks[case["failed_check"]] = "failed"
    if failure_type == "LABEL_INPUT_UNAVAILABLE":
        for check in (
            "matched_label_schema",
            "matched_label_seal",
            "crosswalk_identity",
            "units",
        ):
            checks[check] = "not_run"

    report = {
        "schema_version": "accuracy_report.v1",
        "report_sha256": "0" * 64,
        "identities": identities,
        "outcome": {
            "status": "rejected",
            "admission": {
                "status": "rejected",
                "checks": checks,
                "failures": [
                    {
                        "failure_type": failure_type,
                        "detail": f"typed {failure_type} fixture",
                        "context": deepcopy(case["context"]),
                    }
                ],
            },
            "candidate_status": {
                "status": "unknown",
                "reason": case["candidate_reason"],
            },
        },
    }
    return _seal(report, "report_sha256"), authority


def _valid_error() -> dict[str, Any]:
    document = {
        "schema_version": "sim_error.v2",
        "error_sha256": "0" * 64,
        "error": {
            "failure_type": "MANIFEST_ADMISSION_FAILED",
            "stage": "manifest_admission",
            "detail": "typed early failure",
        },
        "exit_status": {"status": "failure", "exit_code": 2},
    }
    return _seal(document, "error_sha256")


def _validate_admitted_report(report: dict[str, Any], authority: MatchedLabelAuthority):
    return validate_report(report, label_authority=authority)


def test_production_resolver_validates_every_p1_contract() -> None:
    validate_schema_definitions()
    validate_launch_key(_valid_launch_key())
    validate_manifest(_valid_manifest())
    validate_result(_valid_result())
    labels = _valid_matched_labels()
    validate_matched_labels(labels, authority=_label_authority(labels))
    report, authority = _valid_report()
    _validate_admitted_report(report, authority)
    validate_report(_rejected_report())
    validate_sim_error(_valid_error())


def test_p1_schemas_are_draft_202012_and_recursively_closed() -> None:
    for name in (
        "launch_key",
        "manifest",
        "result",
        "sim_error",
        "matched_labels",
        "projection",
        "report",
    ):
        schema = schema_document(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        for path, node in _walk(schema):
            if isinstance(node, dict) and node.get("type") == "object":
                assert node.get("additionalProperties") is False, (name, path)


def test_v2_simulator_schema_ids_are_frozen_to_consolidated_run_authority() -> None:
    assert schema_document("manifest")["$id"] == (
        "https://gpu-simulator.local/sim/run/schema/sim_run_manifest.v2.schema.json"
    )
    assert schema_document("result")["$id"] == (
        "https://gpu-simulator.local/sim/run/schema/sim_result.v2.schema.json"
    )
    assert schema_document("sim_error")["$id"] == (
        "https://gpu-simulator.local/sim/run/schema/sim_error.v2.schema.json"
    )


def test_launch_key_authority_split_and_public_token_are_frozen() -> None:
    assert REPLAY_AUTHENTICATED_LAUNCH_FIELDS == {
        "kernel_mangled_name",
        "grid",
        "block",
        "device_id",
        "stream_id",
        "binary_architecture",
    }
    assert COHORT_ASSERTION_LAUNCH_FIELDS == {
        "workload_id",
        "matched_execution_id",
        "process_launch_ordinal",
    }
    schema = schema_document("launch_key")
    assert "cohort assertions" in schema["description"]
    key = _valid_launch_key()
    reordered = dict(reversed(list(key.items())))
    assert canonical_launch_key_token(key) == canonical_launch_key_token(reordered)
    integral_float_key = deepcopy(key)
    integral_float_key["process_launch_ordinal"] = 0.0
    integral_float_key["grid"]["x"] = 2.0
    integral_float_key["stream_id"] = 7.0
    assert canonical_launch_key_token(key) == canonical_launch_key_token(
        integral_float_key
    )


def test_launch_key_is_reused_by_all_launch_consumers() -> None:
    launch_id = schema_document("launch_key")["$id"]
    for name in ("manifest", "result", "sim_error", "matched_labels", "report"):
        refs = {
            node["$ref"]
            for _, node in _walk(schema_document(name))
            if isinstance(node, dict) and isinstance(node.get("$ref"), str)
        }
        assert launch_id in refs


def test_manifest_v2_retains_v1_evidence_and_replay_contracts() -> None:
    old = _load_json(MANIFEST_V1_SCHEMA_PATH)
    new = schema_document("manifest")
    for property_name in ("scope", "configuration_evidence", "replay_mode"):
        assert new["properties"][property_name] == _convert_v1_refs(
            old["properties"][property_name]
        )


def test_manifest_describes_six_trace_checks_and_three_crosswalk_assertions() -> None:
    schema = schema_document("manifest")
    launch_description = schema["properties"]["launches"]["description"]
    assert "complete nine-field launch key" in launch_description
    assert "exactly the six replay-visible fields" in launch_description
    for field in REPLAY_AUTHENTICATED_LAUNCH_FIELDS:
        assert field in launch_description
    for field in COHORT_ASSERTION_LAUNCH_FIELDS:
        assert field in launch_description
    assert "cannot authenticate" in launch_description
    assert "independently sealed label/crosswalk evidence" in launch_description
    assert "complete nine-field launch key" in schema["description"]


def test_result_v2_preserves_v1_replay_and_outcome_definitions() -> None:
    old = _load_json(RESULT_V1_SCHEMA_PATH)
    new = schema_document("result")
    for name in (
        "resolvedConfiguration",
        "exactMetricValue",
        "fractionMetricValue",
        "predictedExact",
        "predictedFraction",
        "unavailableL1Tex",
        "unavailableLts",
        "unavailableDram",
        "requiredBindingCycles",
    ):
        assert new["$defs"][name] == _convert_v1_refs(old["definitions"][name])


def test_result_and_error_build_identity_shapes_are_identical() -> None:
    result_defs = schema_document("result")["$defs"]
    error_defs = schema_document("sim_error")["$defs"]
    for name in (
        "sha256",
        "gitCommit",
        "nonemptyString",
        "buildClaimability",
        "buildIdentity",
    ):
        assert error_defs[name] == result_defs[name]


@pytest.mark.parametrize(
    "forbidden_field,value",
    [
        ("labels", "labels.jsonl"),
        ("target_values", {"exec_cycles_gpc_max": 123}),
        ("ncu_metric_values", {"gpc__cycles_elapsed.max": 123}),
        ("baseline_runtime_ns", 123),
    ],
)
def test_manifest_rejects_label_target_and_runtime_inputs(
    forbidden_field: str, value: Any
) -> None:
    manifest = _valid_manifest()
    manifest[forbidden_field] = value
    _assert_code("SCHEMA_VALIDATION_FAILED", validate_manifest, manifest)


def test_manifest_and_result_reject_duplicate_launch_keys() -> None:
    manifest = _valid_manifest()
    manifest["launches"].append(deepcopy(manifest["launches"][0]))
    _assert_code("DUPLICATE_LAUNCH_KEY", validate_manifest, manifest)

    result = _valid_result()
    result["launches"].append(deepcopy(result["launches"][0]))
    result["aggregate"]["launch_count"] = 2
    _seal(result, "result_sha256")
    _assert_code("DUPLICATE_LAUNCH_KEY", validate_result, result)


def test_strict_json_rejects_duplicates_and_nonfinite_overflow_and_keeps_uint64() -> None:
    _assert_code(
        "DUPLICATE_JSON_KEY",
        lambda text: loads_json_strict(text),
        '{"outer":{"ordinal":1,"ordinal":2}}',
    )
    for token in ("1e400", "-1e400"):
        _assert_code(
            "JSON_NUMBER_NONFINITE",
            lambda text: loads_json_strict(text),
            f'{{"overflow":{token}}}',
        )
    key = _valid_launch_key(18446744073709551615)
    decoded = loads_json_strict(json.dumps(key))
    validate_launch_key(decoded)
    decoded["process_launch_ordinal"] += 1
    _assert_code("SCHEMA_VALIDATION_FAILED", validate_launch_key, decoded)


def test_matched_labels_require_external_authority_not_self_authorization() -> None:
    labels = _valid_matched_labels()
    _assert_code("MATCHED_LABEL_AUTHORITY_UNAVAILABLE", validate_matched_labels, labels)
    validate_matched_labels(labels, authority=_label_authority(labels))

    wrong = MatchedLabelAuthority(
        matched_labels_sha256=HEX_D,
        label_set_id="labels-vector-add-v5",
        label_set_sha256=HEX_A,
        crosswalk_id="crosswalk-vector-add-v1",
        crosswalk_sha256=HEX_D,
    )
    _assert_code(
        "MATCHED_LABEL_AUTHORITY_MISMATCH",
        lambda document: validate_matched_labels(document, authority=wrong),
        labels,
    )

    correct = _label_authority(labels)
    wrong_label_set_id = MatchedLabelAuthority(
        matched_labels_sha256=correct.matched_labels_sha256,
        label_set_id="different-label-set",
        label_set_sha256=correct.label_set_sha256,
        crosswalk_id=correct.crosswalk_id,
        crosswalk_sha256=correct.crosswalk_sha256,
    )
    _assert_code(
        "MATCHED_LABEL_AUTHORITY_MISMATCH",
        lambda document: validate_matched_labels(
            document, authority=wrong_label_set_id
        ),
        labels,
    )


def test_matched_labels_fail_closed_when_input_is_unavailable() -> None:
    error = _assert_code("LABEL_INPUT_UNAVAILABLE", validate_matched_labels, None)
    assert error.context["availability"] == "unavailable"


def test_matched_labels_allow_typed_nullable_metric_without_fabrication() -> None:
    labels = _valid_matched_labels()
    metric = labels["launches"][0]["metrics"][0]
    metric["outcome"] = {
        "status": "unavailable",
        "value": None,
        "reason": "source_null",
    }
    metric["repeat_metadata"]["dispersion"] = {
        "status": "unavailable",
        "rule": "relative_tolerance",
        "reason": "source_dispersion_null",
    }
    _seal(labels, "matched_labels_sha256")
    validate_matched_labels(labels, authority=_label_authority(labels))


def test_matched_labels_reject_duplicate_rows_and_wrong_registry_mapping() -> None:
    labels = _valid_matched_labels()
    labels["launches"][0]["metrics"][-1] = deepcopy(
        labels["launches"][0]["metrics"][0]
    )
    _seal(labels, "matched_labels_sha256")
    _assert_code(
        "DUPLICATE_LABEL_METRIC",
        lambda document: validate_matched_labels(
            document, authority=_label_authority(document)
        ),
        labels,
    )

    labels = _valid_matched_labels()
    labels["launches"][0]["metrics"][0]["ncu_metric"] = (
        "gpc__cycles_elapsed.max"
    )
    _seal(labels, "matched_labels_sha256")
    _assert_code(
        "SCHEMA_VALIDATION_FAILED",
        lambda document: validate_matched_labels(
            document, authority=_label_authority(document)
        ),
        labels,
    )


def test_projection_is_closed_sealed_and_exact_to_pinned_source_bytes() -> None:
    projection = validate_registry_projection()
    assert projection["source_registry_sha256"] == CANONICAL_REGISTRY_SHA256
    assert projection["projection_sha256"] == canonical_sha256(
        projection, "projection_sha256"
    )
    raw = registry_consumer.load_registry_raw(str(CANONICAL_REGISTRY_PATH))
    registry = registry_consumer.load_registry(str(CANONICAL_REGISTRY_PATH))
    by_key = {row["key"]: row for row in raw["metrics"]}
    expected = [
        {field: by_key[key][field] for field in PROJECTED_ROW_FIELDS}
        for key in METRIC_KEYS
    ]
    assert projection["metrics"] == expected
    assert projection["source_registry_version"] == registry.registry_version
    assert projection["source_label_schema_version"] == registry.label_schema_version


def test_current_v1_runner_output_is_explicitly_incompatible_with_v2() -> None:
    current = _load_json(CURRENT_V1_RESULT_PATH)
    assert current["schema_version"] == "sim_result.v1"
    _assert_code("SCHEMA_VALIDATION_FAILED", validate_result, current)


@pytest.mark.parametrize(
    "field", ["total_size_bytes", "record_count", "instruction_count"]
)
def test_success_result_rejects_vacuous_trace_bundle_counts(field: str) -> None:
    result = _valid_result()
    result["trace_identity"][field] = 0
    _seal(result, "result_sha256")
    _assert_code("SCHEMA_VALIDATION_FAILED", validate_result, result)


def test_trace_bundle_is_transitive_and_size_checked() -> None:
    schema = schema_document("result")["$defs"]["traceIdentity"]
    assert "transitive threadblock payloads" in schema["description"]
    assert "same authenticated bundle" in schema["description"]
    result = _valid_result()
    result["trace_identity"]["total_size_bytes"] += 1
    _seal(result, "result_sha256")
    _assert_code("TRACE_BUNDLE_SIZE_MISMATCH", validate_result, result)


def test_trace_bundle_components_are_nonempty_hashed_and_counted() -> None:
    result = _valid_result()
    del result["trace_identity"]["components"]["stats"]
    _seal(result, "result_sha256")
    _assert_code("SCHEMA_VALIDATION_FAILED", validate_result, result)

    result = _valid_result()
    result["trace_identity"]["components"]["root_dynamic_trace"][
        "size_bytes"
    ] = 0
    _seal(result, "result_sha256")
    _assert_code("SCHEMA_VALIDATION_FAILED", validate_result, result)

    result = _valid_result()
    result["trace_identity"]["components"]["threadblock_payloads"][
        "file_count"
    ] = 0
    _seal(result, "result_sha256")
    _assert_code("SCHEMA_VALIDATION_FAILED", validate_result, result)

    result = _valid_result()
    result["trace_identity"]["components"]["control_bits_sidecar"][
        "sha256"
    ] = "not-a-sha256"
    _seal(result, "result_sha256")
    _assert_code("SCHEMA_VALIDATION_FAILED", validate_result, result)


def test_result_rejects_diagnostic_set_and_aggregate_count_mismatches() -> None:
    result = _valid_result()
    result["diagnostics"]["per_launch"][0]["launch_key"] = _valid_launch_key(1)
    _seal(result, "result_sha256")
    _assert_code("DIAGNOSTIC_LAUNCH_SET_MISMATCH", validate_result, result)

    result = _valid_result()
    result["aggregate"]["launch_count"] = 2
    _seal(result, "result_sha256")
    _assert_code("AGGREGATE_LAUNCH_COUNT_MISMATCH", validate_result, result)


def test_result_rejects_aggregate_metric_and_replay_availability_contradictions() -> None:
    result = _valid_result()
    result["aggregate"]["metrics"]["smsp_inst_executed_sum"]["value"][
        "value"
    ] += 1
    _seal(result, "result_sha256")
    _assert_code("AGGREGATE_METRIC_INCONSISTENT", validate_result, result)

    result = _valid_result()
    result["resolved_configuration"] = _flat_configuration()
    _seal(result, "result_sha256")
    _assert_code("REPLAY_AVAILABILITY_MISMATCH", validate_result, result)


@pytest.mark.parametrize("contradiction", ["value", "numerator", "zero_denominator"])
def test_result_rejects_contradictory_fraction_values(contradiction: str) -> None:
    result = _valid_result()
    fraction = result["launches"][0]["metrics"][
        "cache_hit_rate_l1tex_read_sector"
    ]["value"]
    if contradiction == "value":
        fraction["value"] = 0.8
    elif contradiction == "numerator":
        fraction["numerator"] = 5
        fraction["denominator"] = 4
    else:
        fraction["numerator"] = 0
        fraction["denominator"] = 0
        fraction["value"] = 0.0
    _seal(result, "result_sha256")
    _assert_code("FRACTION_INCONSISTENT", validate_result, result)


def test_fraction_consistency_has_no_near_zero_tolerance_bypass() -> None:
    result = _valid_result()
    value = result["launches"][0]["metrics"]["cache_hit_rate_l1tex_sector"][
        "value"
    ]
    value.update(
        {
            "numerator": 1,
            "denominator": 18446744073709551615,
            "value": 0.0,
        }
    )
    _seal(result, "result_sha256")
    _assert_code("FRACTION_INCONSISTENT", validate_result, result)


def test_result_seal_success_only_cycles_and_dirty_claimability() -> None:
    result = _valid_result()
    result["aggregate"]["metrics"]["smsp_inst_executed_sum"]["value"][
        "value"
    ] += 1
    _assert_code("CANONICAL_SEAL_MISMATCH", validate_result, result)

    result = _valid_result()
    result["exit_status"] = {"status": "failure", "exit_code": 2}
    _seal(result, "result_sha256")
    _assert_code("SCHEMA_VALIDATION_FAILED", validate_result, result)

    result = _valid_result()
    result["launches"][0]["metrics"]["exec_cycles_gpc_max"] = (
        _predicted_exact(456)
    )
    _seal(result, "result_sha256")
    _assert_code("SCHEMA_VALIDATION_FAILED", validate_result, result)

    result = _valid_result()
    result["build_identity"]["sim_git_dirty"] = True
    _seal(result, "result_sha256")
    _assert_code("SCHEMA_VALIDATION_FAILED", validate_result, result)
    result["build_identity"]["claimability"] = {
        "status": "unclaimable",
        "reason": "dirty_build",
    }
    _seal(result, "result_sha256")
    validate_result(result)


def test_sim_error_has_explicit_cli_replay_emission_arms_and_no_catchall() -> None:
    schema = schema_document("sim_error")
    failure_types = {
        branch["properties"]["failure_type"]["const"]
        for branch in schema["$defs"]["typedError"]["oneOf"]
    }
    assert {"CLI_USAGE_FAILED", "REPLAY_FAILED", "EMISSION_FAILED"} <= failure_types
    assert "INTERNAL_ERROR" not in failure_types
    validate_sim_error(_valid_error())

    cli_error = _valid_error()
    cli_error["error"] = {
        "failure_type": "CLI_USAGE_FAILED",
        "stage": "cli_usage",
        "detail": "expected exactly one manifest argument",
        "evidence": {
            "provided_positional_argument_count": 0,
            "expected_positional_argument_count": 1,
        },
    }
    _seal(cli_error, "error_sha256")
    validate_sim_error(cli_error)

    cli_without_evidence = deepcopy(cli_error)
    del cli_without_evidence["error"]["evidence"]
    _seal(cli_without_evidence, "error_sha256")
    _assert_code(
        "SCHEMA_VALIDATION_FAILED", validate_sim_error, cli_without_evidence
    )

    catchall = _valid_error()
    catchall["error"] = {
        "failure_type": "INTERNAL_ERROR",
        "stage": "internal",
        "detail": "catch-all laundering",
    }
    _seal(catchall, "error_sha256")
    _assert_code("SCHEMA_VALIDATION_FAILED", validate_sim_error, catchall)


def test_sim_error_stage_evidence_is_optional_closed_and_sealed() -> None:
    error = _valid_error()
    error["error"]["stage"] = "trace_admission"
    _seal(error, "error_sha256")
    _assert_code("SCHEMA_VALIDATION_FAILED", validate_sim_error, error)

    error = _valid_error()
    error["error"]["evidence"] = {
        "manifest_path": "manifest.json",
        "manifest_sha256": HEX_A,
        "extra": "forbidden",
    }
    _seal(error, "error_sha256")
    _assert_code("SCHEMA_VALIDATION_FAILED", validate_sim_error, error)

    error = _valid_error()
    error["error"]["detail"] = "tampered"
    _assert_code("CANONICAL_SEAL_MISMATCH", validate_sim_error, error)


def test_admitted_report_is_sealed_and_binds_exact_label_crosswalk_authority() -> None:
    report, authority = _valid_report()
    _validate_admitted_report(report, authority)
    binding = report["identities"]["label_binding"]
    assert binding["matched_labels_sha256"] == authority.matched_labels_sha256
    assert binding["label_set_id"] == authority.label_set_id
    assert binding["crosswalk_id"] == authority.crosswalk_id
    assert binding["crosswalk_sha256"] == authority.crosswalk_sha256

    wrong = MatchedLabelAuthority(
        matched_labels_sha256=HEX_D,
        label_set_id=authority.label_set_id,
        label_set_sha256=authority.label_set_sha256,
        crosswalk_id=authority.crosswalk_id,
        crosswalk_sha256=authority.crosswalk_sha256,
    )
    _assert_code(
        "REPORT_LABEL_BINDING_MISMATCH",
        lambda document: validate_report(document, label_authority=wrong),
        report,
    )

    wrong_label_set_id = MatchedLabelAuthority(
        matched_labels_sha256=authority.matched_labels_sha256,
        label_set_id="different-label-set",
        label_set_sha256=authority.label_set_sha256,
        crosswalk_id=authority.crosswalk_id,
        crosswalk_sha256=authority.crosswalk_sha256,
    )
    _assert_code(
        "REPORT_LABEL_BINDING_MISMATCH",
        lambda document: validate_report(
            document, label_authority=wrong_label_set_id
        ),
        report,
    )

    mutated_report = deepcopy(report)
    mutated_report["identities"]["label_binding"]["label_set_id"] = (
        "different-label-set"
    )
    _seal(mutated_report, "report_sha256")
    _assert_code(
        "REPORT_LABEL_BINDING_MISMATCH",
        lambda document: validate_report(
            document, label_authority=authority
        ),
        mutated_report,
    )


def test_report_seal_mutation_fails_on_production_path() -> None:
    report, authority = _valid_report()
    report["outcome"]["coverage"]["scored_cells"] -= 1
    _assert_code(
        "CANONICAL_SEAL_MISMATCH",
        lambda document: validate_report(document, label_authority=authority),
        report,
    )


def test_rejected_failure_policy_covers_every_closed_schema_arm() -> None:
    schema_failure_types = {
        arm["properties"]["failure_type"]["const"]
        for arm in schema_document("report")["$defs"]["admissionFailure"][
            "oneOf"
        ]
    }
    assert schema_failure_types == set(_REJECTED_FAILURE_CASES)


@pytest.mark.parametrize("failure_type", tuple(_REJECTED_FAILURE_CASES))
def test_every_rejected_failure_arm_has_exact_lane_candidate_and_binding(
    failure_type: str,
) -> None:
    case = _REJECTED_FAILURE_CASES[failure_type]
    report, authority = _rejected_report_for(failure_type)
    validate_report(report, label_authority=authority)

    extra_lane = next(
        lane
        for lane in report["outcome"]["admission"]["checks"]
        if lane != case["failed_check"]
        and report["outcome"]["admission"]["checks"][lane] != "failed"
    )
    bad_lane = deepcopy(report)
    bad_lane["outcome"]["admission"]["checks"][extra_lane] = "failed"
    _seal(bad_lane, "report_sha256")
    _assert_code(
        "FAILURE_CONTEXT_MISMATCH",
        lambda document: validate_report(document, label_authority=authority),
        bad_lane,
    )

    bad_candidate = deepcopy(report)
    bad_candidate["outcome"]["candidate_status"]["reason"] = next(
        reason
        for reason in (
            "label_input_unavailable",
            "result_unavailable",
            "admission_failed_before_enumeration",
        )
        if reason != case["candidate_reason"]
    )
    _seal(bad_candidate, "report_sha256")
    _assert_code(
        "FAILURE_CONTEXT_MISMATCH",
        lambda document: validate_report(document, label_authority=authority),
        bad_candidate,
    )

    bad_binding = deepcopy(report)
    if case["binding_status"] == "bound":
        bad_binding["identities"]["label_binding"] = {
            "status": "unavailable",
            "requested_label_set_id": "labels-vector-add-v5",
            "reason": "no_independently_admitted_matched_labels",
        }
    else:
        bad_binding["identities"]["label_binding"] = _report_identities(
            _valid_matched_labels()
        )["label_binding"]
    _seal(bad_binding, "report_sha256")
    _assert_code(
        "FAILURE_CONTEXT_MISMATCH",
        lambda document: validate_report(document, label_authority=authority),
        bad_binding,
    )

    if case["binding_status"] == "bound":
        _assert_code(
            "MATCHED_LABEL_AUTHORITY_UNAVAILABLE", validate_report, report
        )


def test_rejected_bound_label_binding_is_checked_against_label_set_authority() -> None:
    report, authority = _rejected_report_for("RESULT_SCHEMA_INVALID")
    assert authority is not None
    wrong = MatchedLabelAuthority(
        matched_labels_sha256=authority.matched_labels_sha256,
        label_set_id="different-label-set",
        label_set_sha256=authority.label_set_sha256,
        crosswalk_id=authority.crosswalk_id,
        crosswalk_sha256=authority.crosswalk_sha256,
    )
    _assert_code(
        "REPORT_LABEL_BINDING_MISMATCH",
        lambda document: validate_report(document, label_authority=wrong),
        report,
    )


@pytest.mark.parametrize(
    "failure_type,label_set_field",
    [
        ("LABEL_SCHEMA_UNSUPPORTED", "requested_label_set_id"),
        ("LABEL_INPUT_UNAVAILABLE", "requested_label_set_id"),
        ("LABEL_SEAL_INVALID", "label_set_id"),
        ("CROSSWALK_IDENTITY_INVALID", "label_set_id"),
    ],
)
def test_rejected_unavailable_binding_matches_failure_label_set_context(
    failure_type: str, label_set_field: str
) -> None:
    report, authority = _rejected_report_for(failure_type)
    assert authority is None
    report["outcome"]["admission"]["failures"][0]["context"][
        label_set_field
    ] = "different-label-set"
    _seal(report, "report_sha256")
    _assert_code("REPORT_LABEL_BINDING_MISMATCH", validate_report, report)


def test_known_zero_candidate_requires_matching_authenticated_empty_evidence() -> None:
    report, authority = _rejected_report_for("RESULT_SCHEMA_INVALID")
    assert authority is not None
    authenticated = MatchedLabelAuthority(
        matched_labels_sha256=authority.matched_labels_sha256,
        label_set_id=authority.label_set_id,
        label_set_sha256=authority.label_set_sha256,
        crosswalk_id=authority.crosswalk_id,
        crosswalk_sha256=authority.crosswalk_sha256,
        authenticated_empty_cohort_sha256=HEX_C,
    )
    report["outcome"]["candidate_status"] = {
        "status": "known_zero",
        "total_cells": 0,
        "reason": "authenticated_empty_cohort",
        "authenticated_evidence": {
            "empty_cohort_sha256": HEX_C,
            "matched_labels_sha256": authority.matched_labels_sha256,
            "label_set_id": authority.label_set_id,
            "label_set_sha256": authority.label_set_sha256,
            "crosswalk_id": authority.crosswalk_id,
            "crosswalk_sha256": authority.crosswalk_sha256,
            "authority_role": "independently_authenticated_empty_cohort",
        },
    }
    _seal(report, "report_sha256")

    _assert_code(
        "AUTHENTICATED_EMPTY_COHORT_AUTHORITY_UNAVAILABLE",
        lambda document: validate_report(document, label_authority=authority),
        report,
    )
    validate_report(report, label_authority=authenticated)

    bad_evidence = deepcopy(report)
    bad_evidence["outcome"]["candidate_status"]["authenticated_evidence"][
        "empty_cohort_sha256"
    ] = HEX_D
    _seal(bad_evidence, "report_sha256")
    _assert_code(
        "AUTHENTICATED_EMPTY_COHORT_EVIDENCE_MISMATCH",
        lambda document: validate_report(
            document, label_authority=authenticated
        ),
        bad_evidence,
    )


def test_rejected_report_is_honest_without_fabricated_positive_universe() -> None:
    report = _rejected_report()
    validate_report(report)
    outcome = report["outcome"]
    assert outcome["status"] == "rejected"
    assert outcome["candidate_status"] == {
        "status": "unknown",
        "reason": "label_input_unavailable",
    }
    assert "coverage" not in outcome
    assert "per_metric_family" not in outcome


def test_label_unavailable_failure_uses_its_own_admission_lane() -> None:
    report = _rejected_report()
    checks = report["outcome"]["admission"]["checks"]
    checks["label_input_availability"] = "not_run"
    checks["units"] = "failed"
    _seal(report, "report_sha256")
    _assert_code("FAILURE_CONTEXT_MISMATCH", validate_report, report)


def test_rejected_admission_cannot_have_all_checks_passed() -> None:
    report = _rejected_report()
    report["outcome"]["admission"]["checks"] = {
        key: "passed" for key in report["outcome"]["admission"]["checks"]
    }
    _seal(report, "report_sha256")
    _assert_code("SCHEMA_VALIDATION_FAILED", validate_report, report)


def test_report_cycles_have_no_numeric_promotion_arm() -> None:
    report, authority = _valid_report()
    cycle = next(
        row
        for row in report["outcome"]["per_metric_family"]
        if row["metric_key"] == "exec_cycles_gpc_max"
    )
    cycle["n_scored"] = 1
    cycle["summary"] = {
        "kind": "cycle",
        "ratio_count": 1,
        "geometric_mean_ratio": 1.0,
    }
    _seal(report, "report_sha256")
    _assert_code(
        "SCHEMA_VALIDATION_FAILED",
        lambda document: validate_report(document, label_authority=authority),
        report,
    )


def test_zero_denominator_prediction_has_typed_report_unavailability() -> None:
    report, authority = _valid_report()
    outcome = report["outcome"]
    row = next(
        item
        for item in outcome["per_metric_family"]
        if item["metric_key"] == "cache_hit_rate_l1tex_sector"
    )
    row["n_scored"] = 0
    row["n_unavailable"] = 1
    row["unavailable_by_type"] = [
        {
            "status": "unavailable",
            "reason": "undefined_fraction",
            "detail": "zero_denominator",
            "count": 1,
        }
    ]
    row["within_label_noise"] = _noise(0)
    row["summary"]["exact_count"] = 0
    row["summary"]["error_percentage_points"] = _error_stats(0)
    outcome["coverage"]["scored_cells"] = 6
    outcome["coverage"]["unavailable_cells"] = 2
    outcome["coverage"]["unavailable_by_type"].append(
        deepcopy(row["unavailable_by_type"][0])
    )
    _seal(report, "report_sha256")
    _validate_admitted_report(report, authority)


def test_within_label_noise_rule_is_normative_and_not_applicable_is_typed() -> None:
    description = schema_document("report")["$defs"]["withinLabelNoise"][
        "description"
    ]
    assert "scalar for stddev_abs" in description
    assert "abs(label_value)*scalar" in description
    assert "never invent a standard deviation" in description

    report, authority = _valid_report()
    row = report["outcome"]["per_metric_family"][0]
    row["within_label_noise"] = {
        "comparison_semantics": (
            "absolute_error_native_unit_lte_declared_dispersion_threshold"
        ),
        "evaluated_count": 0,
        "within_count": 0,
        "not_applicable_count": 1,
        "not_applicable_reasons": ["dispersion_unavailable"],
    }
    _seal(report, "report_sha256")
    _validate_admitted_report(report, authority)


def test_report_rejects_duplicate_rows_zero_cells_and_bad_count_algebra() -> None:
    report, authority = _valid_report()
    report["outcome"]["per_metric_family"] = [
        deepcopy(report["outcome"]["per_metric_family"][0]) for _ in range(8)
    ]
    _seal(report, "report_sha256")
    _assert_code(
        "DUPLICATE_METRIC_FAMILY_ROW",
        lambda document: validate_report(document, label_authority=authority),
        report,
    )

    report, authority = _valid_report()
    report["outcome"]["coverage"]["total_cells"] = 0
    _seal(report, "report_sha256")
    _assert_code(
        "SCHEMA_VALIDATION_FAILED",
        lambda document: validate_report(document, label_authority=authority),
        report,
    )

    report, authority = _valid_report()
    report["outcome"]["coverage"]["scored_cells"] = 6
    _seal(report, "report_sha256")
    _assert_code(
        "COVERAGE_ALGEBRA_MISMATCH",
        lambda document: validate_report(document, label_authority=authority),
        report,
    )


def test_report_rejects_unavailable_discriminator_and_summary_mismatches() -> None:
    report, authority = _valid_report()
    unavailable = report["outcome"]["coverage"]["unavailable_by_type"][0]
    unavailable.clear()
    unavailable.update(
        {
            "status": "unavailable",
            "reason": "undefined_fraction",
            "detail": "zero_denominator",
            "count": 1,
        }
    )
    _seal(report, "report_sha256")
    _assert_code(
        "UNAVAILABLE_TYPE_ALGEBRA_MISMATCH",
        lambda document: validate_report(document, label_authority=authority),
        report,
    )

    report, authority = _valid_report()
    report["outcome"]["per_metric_family"][0]["summary"]["exact_count"] = 2
    _seal(report, "report_sha256")
    _assert_code(
        "SUMMARY_COUNT_ALGEBRA_MISMATCH",
        lambda document: validate_report(document, label_authority=authority),
        report,
    )


def test_every_comparison_failure_arm_carries_candidate_cell_identity() -> None:
    for arm in schema_document("report")["$defs"]["comparisonFailure"]["oneOf"]:
        required = set(arm["properties"]["context"]["required"])
        assert {"launch_key", "metric_key"} <= required


def test_join_ambiguous_match_count_rejects_uint64_overflow() -> None:
    report, authority = _valid_report()
    report["outcome"]["comparison_failures"] = [
        {
            "failure_type": "JOIN_AMBIGUOUS",
            "detail": "too many candidate matches",
            "context": {
                "launch_key": _valid_launch_key(),
                "metric_key": "smsp_inst_executed_sum",
                "match_count": 18446744073709551616,
            },
        }
    ]
    _seal(report, "report_sha256")
    _assert_code(
        "SCHEMA_VALIDATION_FAILED",
        lambda document: validate_report(document, label_authority=authority),
        report,
    )


def test_missing_label_failure_does_not_require_label_owned_family() -> None:
    report, authority = _valid_report()
    outcome = report["outcome"]
    row = outcome["per_metric_family"][0]
    row["n_labeled"] = 0
    row["n_scored"] = 0
    row["within_label_noise"] = _noise(0)
    row["summary"]["exact_count"] = 0
    row["summary"]["relative_error"] = _error_stats(0)
    row["summary"]["absolute_error"] = _error_stats(0)
    outcome["comparison_failures"] = [
        {
            "failure_type": "MISSING_LABEL",
            "detail": "label absent",
            "context": {
                "launch_key": _valid_launch_key(),
                "metric_key": row["metric_key"],
            },
        }
    ]
    outcome["coverage"].update(
        {
            "labeled_cells": 7,
            "unlabeled_cells": 1,
            "scored_cells": 6,
            "comparison_failure_cells": 1,
        }
    )
    _seal(report, "report_sha256")
    _validate_admitted_report(report, authority)


def test_duplicate_candidate_cell_cannot_hide_behind_cross_type_or_detail() -> None:
    report, authority = _valid_report()
    outcome = report["outcome"]
    context = {
        "launch_key": _valid_launch_key(),
        "metric_key": "smsp_inst_executed_sum",
    }
    same_cell_different_json_numbers = deepcopy(context)
    same_cell_different_json_numbers["launch_key"]["process_launch_ordinal"] = 0.0
    same_cell_different_json_numbers["launch_key"]["device_id"] = 0.0
    same_cell_different_json_numbers["launch_key"]["stream_id"] = 7.0
    for geometry in ("grid", "block"):
        for dimension in ("x", "y", "z"):
            value = same_cell_different_json_numbers["launch_key"][geometry][
                dimension
            ]
            same_cell_different_json_numbers["launch_key"][geometry][dimension] = float(
                value
            )
    outcome["comparison_failures"] = [
        {
            "failure_type": "MISSING_LABEL",
            "detail": "first typed failure",
            "context": context,
        },
        {
            "failure_type": "UNSUPPORTED_OUTCOME",
            "detail": "different type and detail",
            "context": {
                **same_cell_different_json_numbers,
                "observed_status": "future_status",
            },
        },
    ]
    outcome["coverage"].update(
        {
            "total_cells": 9,
            "labeled_cells": 7,
            "unlabeled_cells": 2,
            "scored_cells": 6,
            "comparison_failure_cells": 2,
        }
    )
    row = outcome["per_metric_family"][0]
    row["n_labeled"] = 0
    row["n_scored"] = 0
    row["within_label_noise"] = _noise(0)
    row["summary"]["exact_count"] = 0
    row["summary"]["relative_error"] = _error_stats(0)
    row["summary"]["absolute_error"] = _error_stats(0)
    _seal(report, "report_sha256")
    _assert_code(
        "DUPLICATE_COMPARISON_FAILURE",
        lambda document: validate_report(document, label_authority=authority),
        report,
    )


def test_report_failure_arms_and_architecture_are_fail_closed() -> None:
    report = _rejected_report()
    report["outcome"]["admission"]["failures"][0]["failure_type"] = "SILENT_PASS"
    _seal(report, "report_sha256")
    _assert_code("SCHEMA_VALIDATION_FAILED", validate_report, report)

    report, authority = _valid_report()
    report["identities"]["binary_architecture"] = "sm_120"
    _seal(report, "report_sha256")
    _assert_code(
        "SCHEMA_VALIDATION_FAILED",
        lambda document: validate_report(document, label_authority=authority),
        report,
    )


def test_dirty_report_identity_is_explicitly_unclaimable() -> None:
    report, authority = _valid_report()
    report["identities"]["evaluator_git_dirty"] = True
    _seal(report, "report_sha256")
    _assert_code(
        "SCHEMA_VALIDATION_FAILED",
        lambda document: validate_report(document, label_authority=authority),
        report,
    )
    report["identities"]["evaluator_build_claimability"] = {
        "status": "unclaimable",
        "reason": "dirty_build",
    }
    _seal(report, "report_sha256")
    _validate_admitted_report(report, authority)


def test_all_metric_contracts_pin_the_same_exact_7plus1_keys() -> None:
    projection = validate_registry_projection()
    assert tuple(row["key"] for row in projection["metrics"]) == METRIC_KEYS
    assert tuple(schema_document("result")["$defs"]["metricMap"]["properties"]) == (
        METRIC_KEYS
    )
    assert tuple(schema_document("report")["$defs"]["metricKey"]["enum"]) == (
        METRIC_KEYS
    )
    assert tuple(
        schema_document("matched_labels")["$defs"]["metricKey"]["enum"]
    ) == METRIC_KEYS
