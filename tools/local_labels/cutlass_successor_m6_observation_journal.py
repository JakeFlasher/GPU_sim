#!/usr/bin/env python3
"""Offline causal journal contract for the CUTLASS successor M6 cohort."""
from __future__ import annotations

import copy
import json
import math
import sys
from dataclasses import dataclass, field
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
from tools.local_labels import m6_determinism_campaign as m6  # noqa: E402
from tools.local_labels import m6_determinism_stats as stats  # noqa: E402

SCHEMA_REL = Path(
    "workloads/simbench_bw/schemas/"
    "cutlass_successor_m6_observation_journal.schema.json"
)
MODULE_REL = Path("tools/local_labels/cutlass_successor_m6_observation_journal.py")
EXPECTED_METRICS = 61
EXPECTED_TRIGGER_METRICS = 55
EXPECTED_TIER1 = 210
EXPECTED_TIER2 = 30
EXPECTED_MAXIMUM = 240
MAX_ATTEMPTS_PER_ORDINAL = 2
MAX_FRESH_PROCESSES = EXPECTED_MAXIMUM * MAX_ATTEMPTS_PER_ORDINAL
MAX_INTERRUPTED_ATTEMPTS = EXPECTED_MAXIMUM + 1
MAX_EXCLUSION_DETAIL_BYTES = 4096
MAX_RECORDS = 4096
REFERENCE_CONFIG = "cfg=clock_none__cache_all"
ZERO_SHA256 = "0" * 64
TRIGGER_CODES = (
    "boundary_proximity",
    "discrete_multiple_values",
    "modality_flag",
    "out_of_range",
    "single_point_outlier",
)
PROCESS_STARTED_EXCLUSIONS = (
    "identity_mismatch",
    "metric_missing",
    "profile_failed",
    "sidecar_missing",
)
EXPECTED_CLAIMS = {
    "artifact_scope": "offline_contract_only",
    "live_dispatch_performed": False,
    "publication_authorized": False,
    "m6_done_created": False,
    "m7_handoff_ready": False,
    "hardware_values_are_targets_only": True,
}


class JournalError(RuntimeError):
    """The successor causal journal is malformed or not derivable."""


def _fail(code: str, detail: str) -> None:
    raise JournalError(f"{code}: {detail}")


def _without(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    omitted = set(keys)
    return {key: item for key, item in value.items() if key not in omitted}


def _file_ref(path: Path, *, role: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail("SUCCESSOR_M6_JOURNAL_SOURCE", f"{role} is unavailable")
    if path.stat().st_nlink != 1:
        _fail("SUCCESSOR_M6_JOURNAL_SOURCE", f"{role} link count differs")
    payload = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": admission.sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _require_schedule(schedule: Mapping[str, Any]) -> None:
    try:
        successor_schedule.validate_schedule(schedule)
    except (successor_schedule.ScheduleError, admission.AdmissionError) as exc:
        _fail("SUCCESSOR_M6_JOURNAL_SCHEDULE", str(exc))
    if schedule.get("artifact_kind") != "cutlass_successor_m6_determinism_schedule":
        _fail("SUCCESSOR_M6_JOURNAL_SCHEDULE", "artifact kind differs")
    if schedule.get("counts") != {
        "cells": 3,
        "configs": 6,
        "tier1_slots": EXPECTED_TIER1,
        "tier2_candidate_slots": EXPECTED_TIER2,
        "maximum_slots": EXPECTED_MAXIMUM,
        "tier1_fresh_processes": EXPECTED_TIER1,
        "maximum_fresh_processes": EXPECTED_MAXIMUM,
    }:
        _fail("SUCCESSOR_M6_JOURNAL_SCHEDULE", "schedule counts differ")
    expected_dispatch = {
        "status": "required_unimplemented",
        "typed_reason": successor_schedule.EXECUTION_BLOCK_REASON,
        "schedule_is_executable": False,
        "tier1_dispatch_allowed": False,
        "tier2_activation_allowed": False,
        "tier2_candidates_are_contract_only": True,
    }
    if schedule.get("execution_dispatch") != expected_dispatch:
        _fail("SUCCESSOR_M6_JOURNAL_DISPATCH", "static dispatch boundary changed")


def derive_metric_policy(
    schedule: Mapping[str, Any],
    *,
    input_cache: m6.InputSnapshotCache | None = None,
) -> dict[str, Any]:
    """Derive the closed target-only trigger policy from pinned authorities."""

    _require_schedule(schedule)
    cache = input_cache or m6.InputSnapshotCache()
    matrix, matrix_bytes, _ = m6._load_canonical_matrix(
        m6.MATRIX_PATH,
        repo_root=ROOT,
        require_semantic_rederivation=True,
        cache=cache,
    )
    registry, registry_bytes = m6.load_json(
        m6.REGISTRY_PATH,
        role="metric registry",
        cache=cache,
    )
    if not isinstance(registry, Mapping):
        _fail("SUCCESSOR_M6_JOURNAL_METRICS", "registry is not an object")
    probe_keys, probe_identity, _ = m6._load_sm86_probe(
        repo_root=ROOT,
        cache=cache,
    )
    try:
        metrics = m6._registry_metrics(
            registry,
            matrix,
            probe_keys=probe_keys,
        )
    except m6.M6Error as exc:
        _fail("SUCCESSOR_M6_JOURNAL_METRICS", str(exc))
    selected = [row for row in metrics if row["capture_status"] == "selected"]
    if len(metrics) != EXPECTED_METRICS or len(selected) != EXPECTED_TRIGGER_METRICS:
        _fail("SUCCESSOR_M6_JOURNAL_METRICS", "61/55 metric closure differs")
    if any(
        row["model_io_role"] != "target" or row["deployability"] != "label_only"
        for row in selected
    ):
        _fail("SUCCESSOR_M6_JOURNAL_TARGET_ONLY", "selected metric role differs")
    source_identity = {
        "journal_module": _file_ref(ROOT / MODULE_REL, role="journal module"),
        "journal_schema": _file_ref(ROOT / SCHEMA_REL, role="journal schema"),
        "canonical_matrix": {
            "path": m6.MATRIX_PATH.relative_to(ROOT).as_posix(),
            "sha256": admission.sha256_bytes(matrix_bytes),
            "size_bytes": len(matrix_bytes),
        },
        "metric_registry": {
            "path": m6.REGISTRY_PATH.relative_to(ROOT).as_posix(),
            "sha256": admission.sha256_bytes(registry_bytes),
            "size_bytes": len(registry_bytes),
        },
        "availability_probe": probe_identity,
        "canonical_metric_controller": _file_ref(
            Path(m6.__file__).resolve(), role="canonical metric controller"
        ),
        "statistics_module": _file_ref(
            Path(stats.__file__).resolve(), role="statistics module"
        ),
        "statistical_contract_sha256": m6.STATISTICAL_CONTRACT_SHA256,
    }
    policy = {
        "policy_version": 1,
        "reference_config_id": REFERENCE_CONFIG,
        "required_reference_repeats": 20,
        "trigger_codes": list(TRIGGER_CODES),
        "registry_metric_count": EXPECTED_METRICS,
        "trigger_eligible_count": EXPECTED_TRIGGER_METRICS,
        "metrics": metrics,
        "metric_identity_sha256": m6.canonical_sha256(metrics),
        "source_identity": source_identity,
        "source_identity_sha256": m6.canonical_sha256(source_identity),
    }
    return policy


def _metric_maps(
    policy: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    all_metrics = {row["key"]: row for row in policy["metrics"]}
    selected = {
        key: row
        for key, row in all_metrics.items()
        if row["capture_status"] == "selected"
    }
    if len(all_metrics) != EXPECTED_METRICS or len(selected) != EXPECTED_TRIGGER_METRICS:
        _fail("SUCCESSOR_M6_JOURNAL_METRICS", "metric map counts differ")
    return all_metrics, selected


def _validity(value: float, metric: Mapping[str, Any]) -> str:
    if value < float(metric["valid_min"]):
        return "out_of_range_low"
    valid_max = metric["valid_max"]
    if valid_max is not None and value > float(valid_max):
        return "out_of_range_high"
    return "in_range"


def _normalize_observations(
    raw: Any,
    selected: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        _fail("SUCCESSOR_M6_JOURNAL_OBSERVATIONS", "array required")
    by_key: dict[str, Mapping[str, Any]] = {}
    for row in raw:
        if not isinstance(row, Mapping):
            _fail("SUCCESSOR_M6_JOURNAL_OBSERVATIONS", "row must be an object")
        key = row.get("metric_key")
        metric = selected.get(key)
        if metric is None or key in by_key:
            _fail("SUCCESSOR_M6_JOURNAL_OBSERVATIONS", repr(key))
        value = row.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail("SUCCESSOR_M6_JOURNAL_OBSERVATIONS", f"{key}: numeric value required")
        try:
            numeric = float(value)
        except (OverflowError, ValueError):
            _fail(
                "SUCCESSOR_M6_JOURNAL_OBSERVATIONS",
                f"{key}: finite value required",
            )
        if not math.isfinite(numeric):
            _fail("SUCCESSOR_M6_JOURNAL_OBSERVATIONS", f"{key}: finite value required")
        if row.get("metric_sha256") != metric["metric_sha256"]:
            _fail("SUCCESSOR_M6_JOURNAL_METRIC_HASH", str(key))
        expected_validity = _validity(numeric, metric)
        if row.get("validity") != expected_validity:
            _fail("SUCCESSOR_M6_JOURNAL_VALIDITY", str(key))
        by_key[str(key)] = row
    if set(by_key) != set(selected):
        _fail("SUCCESSOR_M6_JOURNAL_OBSERVATIONS", "selected metric set differs")
    return [
        {
            "metric_key": key,
            "metric_sha256": selected[key]["metric_sha256"],
            "value": by_key[key]["value"],
            "validity": by_key[key]["validity"],
        }
        for key in sorted(selected)
    ]


@dataclass
class _ReplayState:
    schedule: Mapping[str, Any]
    selected_metrics: Mapping[str, Mapping[str, Any]]
    epoch: int | None = None
    last_closed_epoch: int = -1
    checkpoint_seen: bool = False
    epoch_attempts: int = 0
    open_attempt: dict[str, Any] | None = None
    interrupted_attempt: dict[str, Any] | None = None
    attempts_per_ordinal: dict[int, int] = field(default_factory=dict)
    terminals: dict[int, dict[str, Any]] = field(default_factory=dict)
    tier1_closed: bool = False
    decisions: list[dict[str, Any]] = field(default_factory=list)
    activation_emitted: bool = False
    activated_ordinals: list[int] = field(default_factory=list)
    epoch_cost: dict[str, int] = field(
        default_factory=lambda: {"fresh_processes": 0, "ncu_replay_passes": 0}
    )
    cumulative_cost: dict[str, int] = field(
        default_factory=lambda: {"fresh_processes": 0, "ncu_replay_passes": 0}
    )
    interrupted_count: int = 0

    @property
    def slots_by_ordinal(self) -> dict[int, Mapping[str, Any]]:
        return {
            slot["ordinal"]: slot
            for slot in self.schedule["slots"] + self.schedule["tier2_candidates"]
        }

    def active_slots(self) -> list[Mapping[str, Any]]:
        tier1 = sorted(self.schedule["slots"], key=lambda row: row["ordinal"])
        if not self.activation_emitted:
            return tier1
        active = set(self.activated_ordinals)
        tier2 = sorted(
            (
                slot
                for slot in self.schedule["tier2_candidates"]
                if slot["ordinal"] in active
            ),
            key=lambda row: row["ordinal"],
        )
        return tier1 + tier2

    def next_slot(self) -> Mapping[str, Any] | None:
        if self.interrupted_attempt is not None:
            return self.slots_by_ordinal[self.interrupted_attempt["ordinal"]]
        for slot in self.active_slots():
            if slot["ordinal"] not in self.terminals:
                return slot
        return None


def _record_request(record: Mapping[str, Any]) -> dict[str, Any]:
    kind = record.get("record_kind")
    if kind in {"epoch_open", "checkpoint", "epoch_closed", "tier1_closed", "tier2_activated"}:
        return {"record_kind": kind}
    if kind == "attempt_started":
        return {
            "record_kind": kind,
            "ordinal": record.get("ordinal"),
            "retry_of_attempt_id": record.get("retry_of_attempt_id"),
        }
    if kind == "attempt_terminal":
        request = {
            "record_kind": kind,
            "outcome": record.get("outcome"),
            "ncu_replay_passes": (record.get("cost") or {}).get(
                "ncu_replay_passes"
            ),
        }
        if record.get("outcome") == "observed":
            request["observations"] = record.get("observations")
        else:
            request["typed_exclusion"] = record.get("typed_exclusion")
        return request
    if kind == "trigger_decision":
        return {"record_kind": kind, "cell_id": record.get("cell_id")}
    return {"record_kind": kind}


def _derive_trigger_decisions(state: _ReplayState) -> list[dict[str, Any]]:
    reference: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    expected_counts: dict[str, int] = {
        cell["cell_id"]: 0 for cell in state.schedule["cells"]
    }
    for ordinal, terminal in state.terminals.items():
        slot = state.slots_by_ordinal[ordinal]
        if (
            slot["tier"] != 1
            or slot["config_id"] != REFERENCE_CONFIG
            or slot["repeat_index"] >= 20
        ):
            continue
        if terminal["outcome"] != "observed":
            _fail(
                "SUCCESSOR_M6_JOURNAL_TRIGGER_INCOMPLETE",
                f"reference slot {ordinal} is censored",
            )
        expected_counts[slot["cell_id"]] += 1
        for row in terminal["observations"]:
            reference.setdefault((slot["cell_id"], row["metric_key"]), []).append(
                row
            )
    if any(count != 20 for count in expected_counts.values()):
        _fail("SUCCESSOR_M6_JOURNAL_TRIGGER_INCOMPLETE", repr(expected_counts))

    decisions: list[dict[str, Any]] = []
    for cell in state.schedule["cells"]:
        reason_codes: set[str] = set()
        evidence: set[str] = set()
        for key, metric in state.selected_metrics.items():
            rows = reference.get((cell["cell_id"], key), [])
            if len(rows) != 20:
                _fail(
                    "SUCCESSOR_M6_JOURNAL_TRIGGER_INCOMPLETE",
                    f"{cell['cell_id']}:{key}",
                )
            in_range = [
                float(row["value"])
                for row in rows
                if row["validity"] == "in_range"
            ]
            any_out_of_range = any(
                row["validity"] != "in_range" for row in rows
            )
            codes = stats.trigger_codes(
                in_range,
                rule=metric["repeatability_rule"],
                absolute_resolution=metric["absolute_resolution"],
                discrete=metric["statistical_domain"] == "discrete",
                any_out_of_range=any_out_of_range,
            )
            if codes:
                evidence.add(key)
                reason_codes.update(codes)
        decision = {
            "cell_id": cell["cell_id"],
            "cell_identity_sha256": cell["cell_identity_sha256"],
            "triggered": bool(reason_codes),
            "reason_codes": sorted(reason_codes),
            "evidence_metric_keys": sorted(evidence),
        }
        decision["decision_sha256"] = m6.canonical_sha256(decision)
        decisions.append(decision)
    return decisions


def _require_open_epoch(state: _ReplayState, kind: str) -> int:
    if state.epoch is None or state.checkpoint_seen:
        _fail("SUCCESSOR_M6_JOURNAL_EPOCH", f"{kind} outside active epoch")
    return state.epoch


def _apply_request(
    state: _ReplayState,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    kind = request.get("record_kind")
    if kind == "epoch_open":
        if state.epoch is not None:
            _fail("SUCCESSOR_M6_JOURNAL_EPOCH", "epoch already open")
        next_epoch = state.last_closed_epoch + 1
        if next_epoch >= MAX_FRESH_PROCESSES:
            _fail("SUCCESSOR_M6_JOURNAL_RESOURCE_BOUND", "epoch count exceeded")
        state.epoch = next_epoch
        state.checkpoint_seen = False
        state.epoch_attempts = 0
        state.epoch_cost = {"fresh_processes": 0, "ncu_replay_passes": 0}
        return {"record_kind": kind, "epoch": state.epoch}

    if kind == "attempt_started":
        epoch = _require_open_epoch(state, kind)
        if state.open_attempt is not None:
            _fail("SUCCESSOR_M6_JOURNAL_ATTEMPT", "another attempt is open")
        slot = state.next_slot()
        if slot is None:
            _fail("SUCCESSOR_M6_JOURNAL_ORDINAL", "no active ordinal is eligible")
        ordinal = request.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            _fail("SUCCESSOR_M6_JOURNAL_ORDINAL", repr(ordinal))
        if ordinal != slot["ordinal"]:
            _fail(
                "SUCCESSOR_M6_JOURNAL_ORDINAL",
                f"expected {slot['ordinal']}, got {ordinal}",
            )
        retry_of = request.get("retry_of_attempt_id")
        expected_retry = (
            state.interrupted_attempt["attempt_id"]
            if state.interrupted_attempt is not None
            else None
        )
        if retry_of != expected_retry:
            _fail("SUCCESSOR_M6_JOURNAL_RETRY", "retry predecessor differs")
        attempt_index = state.attempts_per_ordinal.get(ordinal, 0)
        if attempt_index >= MAX_ATTEMPTS_PER_ORDINAL:
            _fail(
                "SUCCESSOR_M6_JOURNAL_RETRY",
                "per-ordinal attempt bound exceeded",
            )
        if expected_retry is not None and state.interrupted_attempt["epoch"] == epoch:
            _fail("SUCCESSOR_M6_JOURNAL_RETRY", "same-epoch retry is forbidden")
        attempt_id = m6.canonical_sha256(
            {
                "schedule_sha256": state.schedule["schedule_sha256"],
                "epoch": epoch,
                "ordinal": ordinal,
                "attempt_index": attempt_index,
                "slot_sha256": slot["slot_sha256"],
                "retry_of_attempt_id": retry_of,
            }
        )
        attempt = {
            "epoch": epoch,
            "ordinal": ordinal,
            "slot_id": slot["slot_id"],
            "slot_sha256": slot["slot_sha256"],
            "attempt_index": attempt_index,
            "attempt_id": attempt_id,
            "retry_of_attempt_id": retry_of,
        }
        state.open_attempt = attempt
        state.attempts_per_ordinal[ordinal] = attempt_index + 1
        state.epoch_attempts += 1
        state.epoch_cost["fresh_processes"] += 1
        state.cumulative_cost["fresh_processes"] += 1
        return {
            "record_kind": kind,
            **attempt,
            "fresh_process_started": True,
        }

    if kind == "attempt_terminal":
        epoch = _require_open_epoch(state, kind)
        attempt = state.open_attempt
        if attempt is None or attempt["epoch"] != epoch:
            _fail("SUCCESSOR_M6_JOURNAL_ATTEMPT", "terminal lacks open attempt")
        replay_passes = request.get("ncu_replay_passes")
        if (
            isinstance(replay_passes, bool)
            or not isinstance(replay_passes, int)
            or replay_passes < 1
        ):
            _fail("SUCCESSOR_M6_JOURNAL_COST", "positive replay count required")
        outcome = request.get("outcome")
        terminal: dict[str, Any] = {
            "record_kind": kind,
            **attempt,
            "outcome": outcome,
            "cost": {"fresh_processes": 1, "ncu_replay_passes": replay_passes},
        }
        if outcome == "observed":
            terminal["observations"] = _normalize_observations(
                request.get("observations"),
                state.selected_metrics,
            )
            terminal["typed_exclusion"] = None
        elif outcome == "excluded_typed":
            exclusion = request.get("typed_exclusion")
            if not isinstance(exclusion, Mapping):
                _fail("SUCCESSOR_M6_JOURNAL_EXCLUSION", "object required")
            reason = exclusion.get("reason")
            detail = exclusion.get("detail")
            if reason not in PROCESS_STARTED_EXCLUSIONS:
                _fail("SUCCESSOR_M6_JOURNAL_EXCLUSION", repr(reason))
            if (
                not isinstance(detail, str)
                or not detail
                or len(detail) > MAX_EXCLUSION_DETAIL_BYTES
            ):
                _fail(
                    "SUCCESSOR_M6_JOURNAL_EXCLUSION",
                    "detail must be nonempty and at most 4096 bytes",
                )
            try:
                detail_size = len(detail.encode("utf-8"))
            except UnicodeEncodeError:
                _fail(
                    "SUCCESSOR_M6_JOURNAL_EXCLUSION",
                    "detail is not valid UTF-8",
                )
            if detail_size > MAX_EXCLUSION_DETAIL_BYTES:
                _fail(
                    "SUCCESSOR_M6_JOURNAL_EXCLUSION",
                    "detail must be nonempty and at most 4096 bytes",
                )
            terminal["observations"] = []
            terminal["typed_exclusion"] = {"reason": reason, "detail": detail}
        else:
            _fail("SUCCESSOR_M6_JOURNAL_OUTCOME", repr(outcome))
        state.epoch_cost["ncu_replay_passes"] += replay_passes
        state.cumulative_cost["ncu_replay_passes"] += replay_passes
        state.terminals[attempt["ordinal"]] = terminal
        if (
            state.interrupted_attempt is not None
            and state.interrupted_attempt["ordinal"] == attempt["ordinal"]
        ):
            state.interrupted_attempt = None
        state.open_attempt = None
        return terminal

    if kind == "checkpoint":
        epoch = _require_open_epoch(state, kind)
        if state.interrupted_attempt is not None and state.epoch_attempts == 0:
            _fail("SUCCESSOR_M6_JOURNAL_RETRY", "resume epoch did not retry")
        interrupted_attempt_id = None
        if state.open_attempt is not None:
            state.interrupted_attempt = dict(state.open_attempt)
            interrupted_attempt_id = state.open_attempt["attempt_id"]
            state.interrupted_count += 1
            state.open_attempt = None
        state.checkpoint_seen = True
        next_slot = state.next_slot()
        return {
            "record_kind": kind,
            "epoch": epoch,
            "epoch_cost": dict(state.epoch_cost),
            "cumulative_cost": dict(state.cumulative_cost),
            "terminal_count": len(state.terminals),
            "next_eligible_ordinal": (
                next_slot["ordinal"] if next_slot is not None else None
            ),
            "interrupted_attempt_id": interrupted_attempt_id,
        }

    if kind == "epoch_closed":
        if state.epoch is None or not state.checkpoint_seen:
            _fail("SUCCESSOR_M6_JOURNAL_EPOCH", "close requires checkpoint")
        epoch = state.epoch
        state.last_closed_epoch = epoch
        state.epoch = None
        return {"record_kind": kind, "epoch": epoch}

    epoch = _require_open_epoch(state, str(kind))
    if state.open_attempt is not None or state.interrupted_attempt is not None:
        _fail("SUCCESSOR_M6_JOURNAL_ATTEMPT", f"{kind} with unresolved attempt")

    if kind == "tier1_closed":
        if state.tier1_closed or any(
            slot["ordinal"] not in state.terminals for slot in state.schedule["slots"]
        ):
            _fail("SUCCESSOR_M6_JOURNAL_TIER1", "Tier-1 is not terminal")
        state.tier1_closed = True
        return {"record_kind": kind, "epoch": epoch}

    if kind == "trigger_decision":
        if not state.tier1_closed or state.activation_emitted:
            _fail("SUCCESSOR_M6_JOURNAL_TRIGGER", "decision is not eligible")
        expected = _derive_trigger_decisions(state)
        index = len(state.decisions)
        if index >= len(expected):
            _fail("SUCCESSOR_M6_JOURNAL_TRIGGER", "too many decisions")
        decision = expected[index]
        if request.get("cell_id") != decision["cell_id"]:
            _fail("SUCCESSOR_M6_JOURNAL_TRIGGER", "cell order differs")
        state.decisions.append(decision)
        return {"record_kind": kind, "epoch": epoch, **decision}

    if kind == "tier2_activated":
        if not state.tier1_closed or len(state.decisions) != len(state.schedule["cells"]):
            _fail("SUCCESSOR_M6_JOURNAL_ACTIVATION", "decisions are incomplete")
        if state.activation_emitted:
            _fail("SUCCESSOR_M6_JOURNAL_ACTIVATION", "activation is duplicated")
        triggered = {
            decision["cell_id"]
            for decision in state.decisions
            if decision["triggered"]
        }
        activated_cells = [
            cell["cell_id"]
            for cell in state.schedule["cells"]
            if cell["cell_id"] in triggered
        ]
        ordinals = [
            slot["ordinal"]
            for slot in state.schedule["tier2_candidates"]
            if slot["cell_id"] in triggered
        ]
        if len(ordinals) != 10 * len(activated_cells):
            _fail("SUCCESSOR_M6_JOURNAL_ACTIVATION", "candidate count differs")
        state.activation_emitted = True
        state.activated_ordinals = ordinals
        return {
            "record_kind": kind,
            "epoch": epoch,
            "activated_cell_ids": activated_cells,
            "activated_ordinals": ordinals,
        }

    _fail("SUCCESSOR_M6_JOURNAL_RECORD_KIND", repr(kind))


def _summary(state: _ReplayState, record_count: int) -> dict[str, Any]:
    next_slot = state.next_slot()
    tier1_ordinals = {slot["ordinal"] for slot in state.schedule["slots"]}
    tier1_terminal = len(tier1_ordinals & state.terminals.keys())
    tier2_terminal = len(state.terminals) - tier1_terminal
    return {
        "record_count": record_count,
        "epochs_opened": state.last_closed_epoch + 1 + (state.epoch is not None),
        "epochs_closed": state.last_closed_epoch + 1,
        "current_epoch": state.epoch,
        "open_attempt_id": (
            state.open_attempt["attempt_id"]
            if state.open_attempt is not None
            else None
        ),
        "interrupted_attempt_id": (
            state.interrupted_attempt["attempt_id"]
            if state.interrupted_attempt is not None
            else None
        ),
        "interrupted_attempt_count": state.interrupted_count,
        "terminal_slots": len(state.terminals),
        "tier1_terminal_slots": tier1_terminal,
        "tier2_terminal_slots": tier2_terminal,
        "next_eligible_ordinal": (
            next_slot["ordinal"] if next_slot is not None else None
        ),
        "tier1_closed": state.tier1_closed,
        "trigger_decisions_emitted": len(state.decisions),
        "triggered_cell_ids": [
            decision["cell_id"]
            for decision in state.decisions
            if decision["triggered"]
        ],
        "tier2_activation_emitted": state.activation_emitted,
        "activated_ordinals": list(state.activated_ordinals),
        "cumulative_cost": dict(state.cumulative_cost),
        "active_prefix_complete": (
            state.activation_emitted
            and next_slot is None
            and state.open_attempt is None
            and state.interrupted_attempt is None
        ),
    }


def _replay(
    schedule: Mapping[str, Any],
    policy: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    sealed: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(records) > MAX_RECORDS:
        _fail("SUCCESSOR_M6_JOURNAL_RESOURCE_BOUND", "record count exceeded")
    _, selected = _metric_maps(policy)
    state = _ReplayState(schedule=schedule, selected_metrics=selected)
    normalized: list[dict[str, Any]] = []
    previous = ZERO_SHA256
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            _fail("SUCCESSOR_M6_JOURNAL_RECORD", f"record {index} is not an object")
        if sealed:
            if (
                record.get("sequence") != index
                or record.get("previous_record_sha256") != previous
                or record.get("schedule_sha256") != schedule["schedule_sha256"]
                or record.get("record_sha256")
                != m6.canonical_sha256(_without(record, "record_sha256"))
            ):
                _fail("SUCCESSOR_M6_JOURNAL_CHAIN", str(index))
            request = _record_request(record)
        else:
            request = copy.deepcopy(dict(record))
        body = _apply_request(state, request)
        expected = m6._seal_causal_record(
            {"schedule_sha256": schedule["schedule_sha256"], **body},
            previous,
            index,
        )
        if sealed and dict(record) != expected:
            _fail("SUCCESSOR_M6_JOURNAL_REPLAY", str(index))
        normalized.append(expected)
        previous = expected["record_sha256"]
    return normalized, _summary(state, len(normalized))


def _journal_id(
    schedule_sha256: str,
    source_sha256: str,
    policy_sha256: str,
    created_at_utc: str,
) -> str:
    identity = m6.canonical_sha256(
        {
            "schedule_sha256": schedule_sha256,
            "source_identity_sha256": source_sha256,
            "metric_policy_sha256": policy_sha256,
            "created_at_utc": created_at_utc,
        }
    )
    return f"cutlass-successor-m6-journal-{identity[:24]}"


def _assemble(
    schedule: Mapping[str, Any],
    policy: Mapping[str, Any],
    records: list[dict[str, Any]],
    summary: Mapping[str, Any],
    *,
    created_at_utc: str,
) -> dict[str, Any]:
    source_identity = policy["source_identity"]
    policy_body = _without(policy, "source_identity", "source_identity_sha256")
    source_sha = m6.canonical_sha256(source_identity)
    policy_sha = m6.canonical_sha256(policy_body)
    journal = {
        "schema_version": 1,
        "artifact_kind": "cutlass_successor_m6_observation_journal",
        "journal_id": _journal_id(
            schedule["schedule_sha256"], source_sha, policy_sha, created_at_utc
        ),
        "created_at_utc": created_at_utc,
        "schedule_binding": {
            "schedule_id": schedule["schedule_id"],
            "schedule_sha256": schedule["schedule_sha256"],
            "source_identity_sha256": schedule["source_identity_sha256"],
            "config_identity_sha256": schedule["config_identity_sha256"],
            "cell_identity_sha256": schedule["cell_identity_sha256"],
            "cells": 3,
            "tier1_slots": EXPECTED_TIER1,
            "tier2_candidate_slots": EXPECTED_TIER2,
            "maximum_slots": EXPECTED_MAXIMUM,
        },
        "source_identity": source_identity,
        "source_identity_sha256": source_sha,
        "metric_policy": policy_body,
        "metric_policy_sha256": policy_sha,
        "resource_policy": {
            "max_attempts_per_ordinal": MAX_ATTEMPTS_PER_ORDINAL,
            "max_fresh_processes": MAX_FRESH_PROCESSES,
            "max_interrupted_attempts": MAX_INTERRUPTED_ATTEMPTS,
            "max_exclusion_detail_bytes": MAX_EXCLUSION_DETAIL_BYTES,
            "max_records": MAX_RECORDS,
        },
        "records": records,
        "records_sha256": m6.canonical_sha256(records),
        "replay_summary": dict(summary),
        "claims": dict(EXPECTED_CLAIMS),
    }
    journal["journal_sha256"] = m6.canonical_sha256(journal)
    return journal


def _validate_schema(journal: Mapping[str, Any]) -> None:
    try:
        schema = json.loads((ROOT / SCHEMA_REL).read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(
            schema,
            format_checker=jsonschema.FormatChecker(),
        ).validate(journal)
    except (OSError, UnicodeError, json.JSONDecodeError, jsonschema.SchemaError) as exc:
        _fail("SUCCESSOR_M6_JOURNAL_SCHEMA_UNAVAILABLE", str(exc))
    except jsonschema.ValidationError as exc:
        location = "/".join(str(item) for item in exc.absolute_path) or "<root>"
        _fail("SUCCESSOR_M6_JOURNAL_SCHEMA", f"{location}: {exc.message}")


def build_journal(
    schedule: Mapping[str, Any],
    record_bodies: Sequence[Mapping[str, Any]],
    *,
    created_at_utc: str,
) -> dict[str, Any]:
    _require_schedule(schedule)
    policy = derive_metric_policy(schedule)
    records, summary = _replay(schedule, policy, record_bodies, sealed=False)
    journal = _assemble(
        schedule,
        policy,
        records,
        summary,
        created_at_utc=created_at_utc,
    )
    validate_journal(schedule, journal)
    return journal


def extend_journal(
    schedule: Mapping[str, Any],
    journal: Mapping[str, Any],
    record_bodies: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_journal(schedule, journal)
    requests = [_record_request(record) for record in journal["records"]]
    requests.extend(copy.deepcopy([dict(record) for record in record_bodies]))
    policy = derive_metric_policy(schedule)
    records, summary = _replay(schedule, policy, requests, sealed=False)
    extended = _assemble(
        schedule,
        policy,
        records,
        summary,
        created_at_utc=journal["created_at_utc"],
    )
    validate_journal(schedule, extended)
    return extended


def validate_journal(
    schedule: Mapping[str, Any],
    journal: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_schema(journal)
    _require_schedule(schedule)
    body = _without(journal, "journal_sha256")
    if journal.get("journal_sha256") != m6.canonical_sha256(body):
        _fail("SUCCESSOR_M6_JOURNAL_SEAL", "journal seal differs")
    expected_binding = {
        "schedule_id": schedule["schedule_id"],
        "schedule_sha256": schedule["schedule_sha256"],
        "source_identity_sha256": schedule["source_identity_sha256"],
        "config_identity_sha256": schedule["config_identity_sha256"],
        "cell_identity_sha256": schedule["cell_identity_sha256"],
        "cells": 3,
        "tier1_slots": EXPECTED_TIER1,
        "tier2_candidate_slots": EXPECTED_TIER2,
        "maximum_slots": EXPECTED_MAXIMUM,
    }
    if journal.get("schedule_binding") != expected_binding:
        _fail("SUCCESSOR_M6_JOURNAL_SCHEDULE", "schedule binding differs")
    policy = derive_metric_policy(schedule)
    expected_source = policy["source_identity"]
    expected_policy = _without(policy, "source_identity", "source_identity_sha256")
    if journal.get("source_identity") != expected_source:
        _fail("SUCCESSOR_M6_JOURNAL_SOURCE", "source identity differs")
    if journal.get("source_identity_sha256") != m6.canonical_sha256(expected_source):
        _fail("SUCCESSOR_M6_JOURNAL_SOURCE", "source seal differs")
    if journal.get("metric_policy") != expected_policy:
        _fail("SUCCESSOR_M6_JOURNAL_METRICS", "metric policy differs")
    if journal.get("metric_policy_sha256") != m6.canonical_sha256(expected_policy):
        _fail("SUCCESSOR_M6_JOURNAL_METRICS", "metric policy seal differs")
    if journal.get("resource_policy") != {
        "max_attempts_per_ordinal": MAX_ATTEMPTS_PER_ORDINAL,
        "max_fresh_processes": MAX_FRESH_PROCESSES,
        "max_interrupted_attempts": MAX_INTERRUPTED_ATTEMPTS,
        "max_exclusion_detail_bytes": MAX_EXCLUSION_DETAIL_BYTES,
        "max_records": MAX_RECORDS,
    }:
        _fail("SUCCESSOR_M6_JOURNAL_RESOURCE_BOUND", "policy differs")
    records, summary = _replay(
        schedule,
        policy,
        journal.get("records", []),
        sealed=True,
    )
    if journal.get("records") != records:
        _fail("SUCCESSOR_M6_JOURNAL_REPLAY", "record stream differs")
    if journal.get("records_sha256") != m6.canonical_sha256(records):
        _fail("SUCCESSOR_M6_JOURNAL_CHAIN", "record-array seal differs")
    if journal.get("replay_summary") != summary:
        _fail("SUCCESSOR_M6_JOURNAL_SUMMARY", "replay summary differs")
    if journal.get("claims") != EXPECTED_CLAIMS:
        _fail("SUCCESSOR_M6_JOURNAL_CLAIMS", "claims differ")
    expected_id = _journal_id(
        schedule["schedule_sha256"],
        journal["source_identity_sha256"],
        journal["metric_policy_sha256"],
        journal["created_at_utc"],
    )
    if journal.get("journal_id") != expected_id:
        _fail("SUCCESSOR_M6_JOURNAL_ID", "journal identity differs")
    return summary
