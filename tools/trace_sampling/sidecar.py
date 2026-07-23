"""Closed-schema `trace_sampling.json` writer + validator.

Validates a sidecar against `schema/trace_sampling.schema.json` — closed
(`additionalProperties:false`), so any stray/renamed field is rejected, and the
`method` enum structurally forbids the inadmissible sampling methods (per-record
reservoir, deterministic every-k). `build_sidecar` assembles a schema-valid dict
from the gate + estimate + compression + wave model; segment->block_ids
membership is recorded explicitly so the estimator + reader guards are
re-derivable from the dumped trace (the simulator has no start-order).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import jsonschema

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "trace_sampling.schema.json"
_SCHEMA = json.loads(_SCHEMA_PATH.read_text())
_VALIDATOR = jsonschema.Draft7Validator(_SCHEMA)


class SidecarSemanticError(ValueError):
    """A schema-valid sidecar contains contradictory sampling evidence."""


def _fail(message: str) -> None:
    raise SidecarSemanticError(message)


def _validate_gate_budget(gates: dict) -> None:
    certainty = gates["certainty_bytes"]
    maximum = gates["max_trace_bytes"]
    exceeded = gates["certainty_budget_exceeded"]
    if certainty is not None and maximum is None:
        _fail("gates.certainty_bytes requires gates.max_trace_bytes")
    if certainty is None:
        if exceeded:
            _fail("gates.certainty_budget_exceeded cannot be true without byte evidence")
        return
    expected = 5 * certainty > 2 * maximum
    if exceeded != expected:
        _fail(
            "gates.certainty_budget_exceeded disagrees with the exact 2/5 budget rule")


def _validate_subset_semantics(sidecar: dict) -> None:
    segments = sidecar["segments"]
    selected_ids = sidecar["selected_segment_ids"]
    certainty_ids = sidecar["certainty_segments"]
    force_blocks = sidecar["force_include_blocks"]
    strata = sidecar["strata"]
    gates = sidecar["gates"]

    if gates["verdict"] != "subset_admissible":
        _fail("a sampled subset requires gates.verdict=subset_admissible")
    if gates["coop"]:
        _fail("a cooperative launch cannot emit a sampled subset")
    if gates["skew_heavy_block_count"] != len(force_blocks):
        _fail("gates.skew_heavy_block_count disagrees with force_include_blocks")

    segment_by_id: dict[int, dict] = {}
    all_blocks: set[int] = set()
    for segment in segments:
        segment_id = segment["id"]
        if segment_id in segment_by_id:
            _fail(f"segments contains duplicate id {segment_id}")
        segment_by_id[segment_id] = segment
        for block_id in segment["block_ids"]:
            if block_id in all_blocks:
                _fail(f"block {block_id} belongs to more than one selected segment")
            all_blocks.add(block_id)

    selected_set = set(selected_ids)
    certainty_set = set(certainty_ids)
    if set(segment_by_id) != selected_set:
        _fail("segments ids must exactly equal selected_segment_ids")
    if not certainty_set.issubset(selected_set):
        _fail("certainty_segments must be a subset of selected_segment_ids")

    roster_blocks: set[int] = set()
    roster_sms: set[int] = set()
    roster_by_sm: dict[int, list[int]] = {}
    for slot in sidecar["per_sm_slot_map"]:
        sm_id = slot["sm_id"]
        if sm_id in roster_sms:
            _fail(f"per_sm_slot_map contains duplicate sm_id {sm_id}")
        roster_sms.add(sm_id)
        if slot["block_ids"] != sorted(slot["block_ids"]):
            _fail(f"per_sm_slot_map block_ids are not in launch-index order for sm {sm_id}")
        roster_by_sm[sm_id] = slot["block_ids"]
        for block_id in slot["block_ids"]:
            if block_id in roster_blocks:
                _fail(f"block {block_id} occurs more than once in per_sm_slot_map")
            roster_blocks.add(block_id)

    original_grid = sidecar["shape"]["original_grid"]
    captured_grid = sidecar["shape"]["captured_grid"]
    if captured_grid != original_grid:
        _fail("a sampled subset cannot rewrite captured_grid")
    grid_total = original_grid["x"] * original_grid["y"] * original_grid["z"]
    expected_roster = set(range(grid_total))
    if roster_blocks != expected_roster:
        missing = sorted(expected_roster.difference(roster_blocks))[:8]
        extra = sorted(roster_blocks.difference(expected_roster))[:8]
        _fail(
            "per_sm_slot_map must exhaust the original grid exactly: "
            f"missing={missing}, extra={extra}")

    wave_model = sidecar["wave_model"]
    resident_by_sm: dict[int, int] = {}
    for row in wave_model["per_sm_resident_blocks"]:
        sm_id = row["sm_id"]
        if sm_id in resident_by_sm:
            _fail(f"per_sm_resident_blocks contains duplicate sm_id {sm_id}")
        resident_by_sm[sm_id] = row["resident_blocks"]
    if set(resident_by_sm) != roster_sms:
        _fail("per_sm_resident_blocks keys must equal per_sm_slot_map keys")

    expected_n_waves = max(
        ((len(roster_by_sm[sm]) + resident_by_sm[sm] - 1)
         // resident_by_sm[sm] for sm in roster_sms),
        default=0)
    if wave_model["n_waves"] != expected_n_waves:
        _fail("wave_model.n_waves disagrees with the per-SM launch-index cohorts")
    if gates["wave_count"] != wave_model["n_waves"]:
        _fail("gates.wave_count disagrees with wave_model.n_waves")

    segment_len = wave_model["segment_len"]
    complete_waves = min(
        (len(roster_by_sm[sm]) // resident_by_sm[sm] for sm in roster_sms),
        default=0)
    complete_wave_cutoff = (complete_waves // segment_len) * segment_len
    residual_tail = expected_n_waves > complete_wave_cutoff
    tail_wave_count = (
        expected_n_waves - complete_wave_cutoff
        if residual_tail else min(segment_len, complete_wave_cutoff))
    expected_tail_measured = max(
        0, tail_wave_count - wave_model["ghost_depth"])
    segment_count = (complete_wave_cutoff // segment_len
                     + (1 if residual_tail else 0))
    expected_measured = max(
        0, expected_n_waves - wave_model["ghost_depth"] * segment_count)
    if gates["tail_measured_waves"] != expected_tail_measured:
        _fail("gates.tail_measured_waves disagrees with the reconstructed tail")
    if gates["n_measured_waves"] != expected_measured:
        _fail("gates.n_measured_waves disagrees with the reconstructed cohorts")
    expected_segments: dict[int, set[int]] = {}
    for sm_id, block_ids in roster_by_sm.items():
        resident = resident_by_sm[sm_id]
        for sequence, block_id in enumerate(block_ids):
            wave = sequence // resident
            segment_id = (
                wave // segment_len
                if wave < complete_wave_cutoff
                else complete_wave_cutoff // segment_len)
            expected_segments.setdefault(segment_id, set()).add(block_id)
    for segment_id, segment in segment_by_id.items():
        expected_blocks = expected_segments.get(segment_id)
        if expected_blocks is None or set(segment["block_ids"]) != expected_blocks:
            _fail(
                f"segment {segment_id} is not a complete reconstructed wave cohort")
    cold_segment_id = min(expected_segments)
    tail_segment_id = complete_wave_cutoff // segment_len
    if tail_segment_id not in expected_segments:
        tail_segment_id = max(expected_segments)
    required_certainty = {cold_segment_id, tail_segment_id}
    if not required_certainty.issubset(certainty_set):
        _fail(
            "certainty_segments must include reconstructed cold/tail segments: "
            f"{sorted(required_certainty)}")

    stratum_ids: set[int] = set()
    stratum_keys: set[tuple[int, int]] = set()
    sampled_ordinary: set[int] = set()
    for stratum in strata:
        stratum_id = stratum["id"]
        for field in ("pi", "sector_stddev", "neyman_weight"):
            if not math.isfinite(stratum[field]):
                _fail(f"stratum {stratum_id} {field} must be finite")
        if stratum_id in stratum_ids:
            _fail(f"strata contains duplicate id {stratum_id}")
        stratum_ids.add(stratum_id)
        key = (stratum["phase_id"], stratum["launch_tercile"])
        expected_stratum_id = stratum["phase_id"] * 3 + stratum["launch_tercile"]
        if stratum_id != expected_stratum_id:
            _fail(
                f"stratum {stratum_id} id does not encode its phase/launch tercile")
        if key in stratum_keys:
            _fail(f"strata contains duplicate phase/tercile key {key}")
        stratum_keys.add(key)

        ids = stratum["segment_ids"]
        if stratum["n_h"] != len(ids):
            _fail(f"stratum {stratum_id} n_h does not match sampled segment_ids")
        if stratum["n_h"] > stratum["N_h"]:
            _fail(f"stratum {stratum_id} has n_h greater than N_h")
        expected_pi = stratum["n_h"] / stratum["N_h"]
        if not math.isclose(stratum["pi"], expected_pi, rel_tol=0.0, abs_tol=1e-12):
            _fail(f"stratum {stratum_id} pi does not equal n_h/N_h")
        expected_weight = stratum["N_h"] * stratum["sector_stddev"]
        if not math.isclose(
                stratum["neyman_weight"], expected_weight,
                rel_tol=1e-12, abs_tol=1e-12):
            _fail(f"stratum {stratum_id} neyman_weight does not equal N_h*sector_stddev")
        overlap = sampled_ordinary.intersection(ids)
        if overlap:
            _fail(f"ordinary segments occur in multiple strata: {sorted(overlap)}")
        sampled_ordinary.update(ids)

    if sampled_ordinary.intersection(certainty_set):
        _fail("certainty segments cannot also be sampled ordinary segments")
    if sampled_ordinary.union(certainty_set) != selected_set:
        _fail(
            "selected_segment_ids must equal certainty segments plus sampled strata")

    certainty_blocks = {
        block_id
        for segment_id in certainty_set
        for block_id in segment_by_id[segment_id]["block_ids"]
    }
    missing_force = set(force_blocks).difference(certainty_blocks)
    if missing_force:
        _fail(
            "force_include_blocks are not contained in certainty segments: "
            f"{sorted(missing_force)}")

    expected_block_hash = hashlib.sha256(
        json.dumps(sorted(all_blocks), separators=(",", ":")).encode()).hexdigest()
    if sidecar["hashes"]["selected_block_ids_sha256"] != expected_block_hash:
        _fail("selected_block_ids_sha256 does not match selected segment membership")

    reconciliation = sidecar["selection_reconciliation"]
    if reconciliation["attempts"] > reconciliation["max_attempts"]:
        _fail("selection reconciliation attempts exceeds max_attempts")

    budget = sidecar["selection_budget"]
    if 5 * budget["certainty_bytes"] > 2 * budget["max_trace_bytes"]:
        _fail("a sampled subset cannot exceed the exact 2/5 certainty-byte budget")
    if gates["certainty_bytes"] != budget["certainty_bytes"]:
        _fail("selection_budget.certainty_bytes disagrees with gates")
    if gates["max_trace_bytes"] != budget["max_trace_bytes"]:
        _fail("selection_budget.max_trace_bytes disagrees with gates")
    if gates["certainty_budget_exceeded"]:
        _fail("a certainty-budget-exceeded gate cannot emit a sampled subset")
    if sidecar["bytes"]["raw"] > budget["max_trace_bytes"]:
        _fail("selected raw payload exceeds selection_budget.max_trace_bytes")

    auxiliary = sidecar["selection_auxiliary"]
    if auxiliary["selected"] > auxiliary["expected_full"]:
        _fail("selection auxiliary selected exceeds expected_full")


def _validate_semantics(sidecar: dict) -> None:
    _validate_gate_budget(sidecar["gates"])
    if sidecar["method"] == "wave_segment_stratified_subset":
        _validate_subset_semantics(sidecar)
    claimed = sidecar["hashes"].get("sidecar_sha256")
    if claimed is not None:
        unsigned = json.loads(json.dumps(sidecar))
        unsigned["hashes"].pop("sidecar_sha256")
        if claimed != sidecar_sha256(unsigned):
            _fail("hashes.sidecar_sha256 does not match the sealed sidecar")


def validate_sidecar(sidecar: dict) -> None:
    """Raise on the first structural or cross-field contract violation."""
    _VALIDATOR.validate(sidecar)
    _validate_semantics(sidecar)


def is_valid(sidecar: dict) -> bool:
    try:
        validate_sidecar(sidecar)
    except (jsonschema.ValidationError, SidecarSemanticError):
        return False
    return True


def sidecar_sha256(sidecar: dict) -> str:
    canonical = json.dumps(sidecar, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _dim3(t: tuple[int, int, int]) -> dict:
    return {"x": t[0], "y": t[1], "z": t[2]}


_DEFAULT_LABEL_PAIRING = {
    "wave_segment_stratified_subset": "full_shape_label",
    "lossless_compressed_full": "full_shape_label",
    "reduced_shape_matched_label": "matched_reduced_shape_label",
    "unavailable": "no_comparable_label",
    "none": "full_shape_label",
    "temporal_window": "full_shape_label",
}


def build_sidecar(*, cell_id: str, kernel_name: str, args_hash: str, stats,
                  gate, compression, records_captured: int, tracer_version: int,
                  stats_pass_version: int, method: str, reason: str,
                  segments: list[dict] | None = None,
                  strata: list[dict] | None = None,
                  estimator: dict | None = None,
                  selected_segment_ids: list[int] | None = None,
                  certainty_segments: list[int] | None = None,
                  force_include_blocks: list[int] | None = None,
                  selection_reconciliation: dict | None = None,
                  selection_budget: dict | None = None,
                  selection_auxiliary: dict | None = None,
                  label_pairing: str | None = None,
                  reduced_shape_variant: str | None = None,
                  decompressed_payload_sha256: str | None = None,
                  seed: int = 0, rng_algo: str = "pcg64") -> dict:
    """Assemble a schema-valid sidecar. `segments`/`strata`/`estimator` are only
    required for the subset method; compression-only cells omit them.
    `label_pairing` defaults per method; a reduced-shape cell must supply
    `reduced_shape_variant`. Compressed/subset methods carry the contract hash
    over the DECOMPRESSED payload bytes (`decompressed_payload_sha256`)."""
    per_sm_resident_blocks = [
        {"sm_id": sm, "resident_blocks": resident}
        for sm, resident in sorted(stats.per_sm_resident.items())
    ]
    per_sm_slot_map = [{"sm_id": sm, "block_ids": ids}
                       for sm, ids in sorted(stats.per_sm_block_ids.items())]
    if segments is not None:
        selected_blocks = sorted(
            block_id for segment in segments for block_id in segment["block_ids"])
    elif method in {"none", "lossless_compressed_full", "temporal_window"}:
        selected_blocks = sorted(block.block_id for block in stats.blocks)
    else:
        selected_blocks = []
    sel_hash = hashlib.sha256(
        json.dumps(selected_blocks, separators=(",", ":")).encode()).hexdigest()

    sidecar: dict[str, Any] = {
        "schema_version": 2,
        "cell_id": cell_id,
        "kernel": {"mangled_name": kernel_name, "args_hash": args_hash},
        "method": method,
        "reason": reason,
        "shape": {
            "original_grid": _dim3(stats.grid),
            "original_block": _dim3(stats.block),
            "captured_grid": _dim3(stats.grid),
            "captured_block": _dim3(stats.block),
            "reduced_shape_variant": None,
        },
        "gates": gate.to_sidecar_gates(),
        "wave_model": {
            "reconstruction": "launch_index",
            "per_sm_resident_blocks": per_sm_resident_blocks,
            "n_waves": stats.n_waves,
            "segment_len": 3,
            "ghost_depth": 1,
        },
        "per_sm_slot_map": per_sm_slot_map,
        "rng": {"algo": rng_algo, "seed": seed},
        "tracer": {
            "version": tracer_version,
            "features": {"atomic_addr": True, "ldgsts_addr": True},
            "stats_pass_version": stats_pass_version,
        },
        "bytes": {
            "raw": compression.raw_bytes if compression else 0,
            "compressed": compression.compressed_bytes if compression else 0,
            "codec": "zstd", "level": 19, "long": 27, "dict_id": None,
        },
        "records": {"expected_full": stats.total_records, "captured": records_captured},
        "hashes": {"selected_block_ids_sha256": sel_hash},
    }
    payload_hash = decompressed_payload_sha256 or getattr(
        compression, "decompressed_sha256", None)
    if payload_hash is not None:
        sidecar["hashes"]["decompressed_payload_sha256"] = payload_hash
    sidecar["label_pairing"] = label_pairing or _DEFAULT_LABEL_PAIRING.get(
        method, "full_shape_label")
    if reduced_shape_variant is not None:
        sidecar["shape"]["reduced_shape_variant"] = reduced_shape_variant
    if segments is not None:
        sidecar["segments"] = segments
    if selected_segment_ids is not None:
        sidecar["selected_segment_ids"] = selected_segment_ids
    if certainty_segments is not None:
        sidecar["certainty_segments"] = certainty_segments
    if force_include_blocks is not None:
        sidecar["force_include_blocks"] = force_include_blocks
    if strata is not None:
        sidecar["strata"] = strata
    if selection_reconciliation is not None:
        sidecar["selection_reconciliation"] = selection_reconciliation
    if selection_budget is not None:
        sidecar["selection_budget"] = selection_budget
    if selection_auxiliary is not None:
        sidecar["selection_auxiliary"] = selection_auxiliary
    if estimator is not None:
        sidecar["estimator"] = estimator
    sidecar["hashes"]["sidecar_sha256"] = sidecar_sha256(sidecar)
    validate_sidecar(sidecar)
    return sidecar


def write_sidecar(sidecar: dict, path: Path) -> None:
    validate_sidecar(sidecar)
    # Exclusive creation preserves pre-existing evidence and closes the race
    # between an orchestration freshness check and the final write.
    with path.open("x") as stream:
        stream.write(json.dumps(sidecar, indent=2) + "\n")
