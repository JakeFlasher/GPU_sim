#!/usr/bin/env python3
"""Build a fail-closed M6 schedule for the admitted CUTLASS successor cohort."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.local_labels import admit_cutlass_successor_capture as admission  # noqa: E402
from tools.local_labels import m6_determinism_campaign as m6  # noqa: E402

SCHEMA_REL = Path(
    "workloads/simbench_bw/schemas/"
    "cutlass_successor_m6_determinism_schedule.schema.json"
)
BUILDER_REL = Path("tools/local_labels/build_cutlass_successor_m6_determinism_schedule.py")
ADMISSION_PRODUCER_REL = Path("tools/local_labels/admit_cutlass_successor_capture.py")
M6_CONTROLLER_REL = Path("tools/local_labels/m6_determinism_campaign.py")
COHORT_BUILDER_REL = Path("tools/local_labels/build_cutlass_successor_cohort.py")
CAPTURE_PRODUCER_REL = Path("tools/local_labels/capture_cutlass_successor_receipts.py")
PARSE_NCU_REL = Path("tools/local_labels/parse_ncu_metrics.py")
METRIC_REGISTRY_MODULE_REL = Path("tools/local_labels/metric_registry.py")
METRIC_REGISTRY_REL = Path("tools/local_labels/metric_registry.json")
IMPORT_LABEL_REL = Path("tools/local_labels/import_label.py")
IMPORT_TRACE_CONTRACT_REL = Path("tools/nvbit_tracer/import_trace_contract.py")
TRACE_ADMISSION_REL = Path("tools/nvbit_tracer/trace_admission.py")
LABEL_DRIVER_REL = Path("tools/local_labels/label_driver.sh")
TRACER_TOOL_REL = Path("tools/nvbit_tracer/tracer_tool/tracer_tool.so")
TRACKED_PUBLICATION_SOURCES = (
    BUILDER_REL,
    SCHEMA_REL,
    ADMISSION_PRODUCER_REL,
    admission.OVERLAY_SCHEMA_REL,
    admission.RECEIPT_SCHEMA_REL,
    M6_CONTROLLER_REL,
    COHORT_BUILDER_REL,
    CAPTURE_PRODUCER_REL,
    PARSE_NCU_REL,
    METRIC_REGISTRY_MODULE_REL,
    METRIC_REGISTRY_REL,
    IMPORT_LABEL_REL,
    admission.LOCAL_LABEL_SIDECAR_SCHEMA_REL,
    IMPORT_TRACE_CONTRACT_REL,
    TRACE_ADMISSION_REL,
    LABEL_DRIVER_REL,
)
EXPECTED_COUNTS = {"total": 3, "pending": 0, "capture_ready": 3, "chunks": 1}
TIER1_SLOTS = 210
TIER2_CANDIDATES = 30
MAXIMUM_SLOTS = TIER1_SLOTS + TIER2_CANDIDATES
REFERENCE_CONFIG = "cfg=clock_none__cache_all"
EXECUTION_BLOCK_REASON = (
    "successor_m6_collector_and_tier2_trigger_authority_required"
)


class ScheduleError(RuntimeError):
    """The successor cohort cannot produce an M6 schedule."""


def _fail(code: str, detail: str) -> None:
    raise ScheduleError(f"{code}: {detail}")


def _sha256_file(path: Path, *, role: str) -> str:
    if path.is_symlink() or not path.is_file():
        _fail("SUCCESSOR_M6_SOURCE_UNAVAILABLE", role)
    if path.stat().st_nlink != 1:
        _fail("SUCCESSOR_M6_SOURCE_LINK_COUNT", role)
    return admission.sha256_bytes(path.read_bytes())


def _file_ref(relative: Path, *, role: str) -> dict[str, Any]:
    path = ROOT / relative
    return {
        "path": relative.as_posix(),
        "sha256": _sha256_file(path, role=role),
        "size_bytes": path.stat().st_size,
    }


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    value = result.stdout.decode("ascii", errors="replace").strip()
    if result.returncode != 0 or len(value) != 40:
        _fail("SUCCESSOR_M6_SOURCE_COMMIT", "cannot resolve repository HEAD")
    return value


def _git_blob_bytes(commit: str, relative: Path) -> bytes:
    result = subprocess.run(
        ["git", "cat-file", "blob", f"{commit}:{relative.as_posix()}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        _fail("SUCCESSOR_M6_SOURCE_UNTRACKED", relative.as_posix())
    return result.stdout


def _require_publication_source(schedule: Mapping[str, Any]) -> None:
    commit = schedule["source_identity"]["producer_commit"]
    if commit != _git_head():
        _fail("SUCCESSOR_M6_SOURCE_COMMIT", "producer commit is not current HEAD")
    for relative in TRACKED_PUBLICATION_SOURCES:
        current = (ROOT / relative).read_bytes()
        if current != _git_blob_bytes(commit, relative):
            _fail("SUCCESSOR_M6_SOURCE_DIRTY", relative.as_posix())


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("matrix_key")), str(row.get("variant_id"))


def _require_unique(values: Sequence[Any], *, code: str, role: str) -> None:
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            _fail(code, f"duplicate {role}: {value!r}")
        seen.add(value)


def _require_capture_ready(
    overlay: Mapping[str, Any], matrix: Mapping[str, Any]
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if overlay.get("dispatch") != {
        "status": "capture_ready",
        "m6_schedule_created": False,
        "typed_reason": "all_successor_receipt_pairs_valid",
    }:
        _fail("SUCCESSOR_M6_CAPTURE_NOT_READY", repr(overlay.get("dispatch")))
    if overlay.get("counts") != EXPECTED_COUNTS:
        _fail("SUCCESSOR_M6_CAPTURE_COUNTS", repr(overlay.get("counts")))
    chunks = overlay.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != 1:
        _fail("SUCCESSOR_M6_CAPTURE_CHUNKS", repr(chunks))

    matrix_rows = matrix.get("rows")
    overlay_rows = overlay.get("rows")
    if not isinstance(matrix_rows, list) or not isinstance(overlay_rows, list):
        _fail("SUCCESSOR_M6_CAPTURE_ROWS", "row arrays are absent")
    expected_order = [
        {"matrix_key": row["matrix_key"], "variant_id": row["variant_id"]}
        for row in matrix_rows
    ]
    chunk = chunks[0]
    if chunk.get("rows") != expected_order:
        _fail("SUCCESSOR_M6_CAPTURE_ORDER", "chunk order differs from immutable matrix")
    if chunk.get("membership_sha256") != admission.canonical_sha(expected_order):
        _fail("SUCCESSOR_M6_CAPTURE_MEMBERSHIP", "chunk seal differs")
    chunking = overlay.get("chunking") or {}
    if chunking.get("runnable_order") != expected_order:
        _fail("SUCCESSOR_M6_CAPTURE_ORDER", "runnable order differs")
    if chunking.get("runnable_membership_sha256") != admission.canonical_sha(
        expected_order
    ):
        _fail("SUCCESSOR_M6_CAPTURE_MEMBERSHIP", "runnable seal differs")

    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in overlay_rows:
        key = _key(row)
        if key in by_key:
            _fail("SUCCESSOR_M6_CAPTURE_DUPLICATE", repr(key))
        by_key[key] = row
    expected_keys = [_key(row) for row in matrix_rows]
    if set(by_key) != set(expected_keys):
        _fail("SUCCESSOR_M6_CAPTURE_IDENTITY", "overlay row set differs")

    ncu_paths: set[str] = set()
    nvbit_paths: set[str] = set()
    joined: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for matrix_row in matrix_rows:
        ready = by_key[_key(matrix_row)]
        if ready.get("status") != "capture_ready" or ready.get(
            "typed_reason"
        ) != "valid_joined_receipt_pair":
            _fail("SUCCESSOR_M6_CAPTURE_NOT_READY", repr(_key(matrix_row)))
        if (ready.get("process_separation") or {}).get("separate") is not True:
            _fail("SUCCESSOR_M6_PROCESS_SEPARATION", repr(_key(matrix_row)))
        ncu_path = str((ready.get("ncu_receipt") or {}).get("path"))
        nvbit_path = str((ready.get("nvbit_receipt") or {}).get("path"))
        if not ncu_path or ncu_path == "None" or ncu_path in ncu_paths:
            _fail("SUCCESSOR_M6_RECEIPT_IDENTITY", f"NCU {ncu_path!r}")
        if not nvbit_path or nvbit_path == "None" or nvbit_path in nvbit_paths:
            _fail("SUCCESSOR_M6_RECEIPT_IDENTITY", f"NVBit {nvbit_path!r}")
        ncu_paths.add(ncu_path)
        nvbit_paths.add(nvbit_path)
        joined.append((matrix_row, ready))
    if ncu_paths & nvbit_paths:
        _fail("SUCCESSOR_M6_RECEIPT_IDENTITY", "NCU/NVBit paths overlap")
    return joined


def _cell(matrix_row: Mapping[str, Any], ready: Mapping[str, Any]) -> dict[str, Any]:
    identity = ready["identity"]
    launch = ready["launch_identity"]
    static = matrix_row["static_launch_inventory"]
    shape = matrix_row["shape"]
    launch_config = shape["trace_shape"]["launch_config"]
    if identity["slug"] != matrix_row["successor_slug"]:
        _fail("SUCCESSOR_M6_CELL_IDENTITY", str(matrix_row["matrix_key"]))
    if identity["shape_point"] != shape["shape_point"]:
        _fail("SUCCESSOR_M6_SHAPE_IDENTITY", str(matrix_row["matrix_key"]))
    if identity["binary_sha256"] != static["binary_sha256"]:
        _fail("SUCCESSOR_M6_BINARY_IDENTITY", str(matrix_row["matrix_key"]))
    if launch["kernel_signature_sha256"] != identity["kernel_signature_sha256"]:
        _fail("SUCCESSOR_M6_LAUNCH_IDENTITY", str(matrix_row["matrix_key"]))
    if launch["grid_dim"] != launch_config["grid"] or launch[
        "block_dim"
    ] != launch_config["block"]:
        _fail("SUCCESSOR_M6_LAUNCH_GEOMETRY", str(matrix_row["matrix_key"]))
    cell = {
        "cell_id": matrix_row["matrix_key"],
        "matrix_key": matrix_row["matrix_key"],
        "variant_id": matrix_row["variant_id"],
        "successor_slug": matrix_row["successor_slug"],
        "base_slug": matrix_row["base_slug"],
        "shape_point": shape["shape_point"],
        "authority_row_sha256": matrix_row["authority_row_sha256"],
        "binary_sha256": static["binary_sha256"],
        "actual_args": shape["actual_args"],
        "launch_identity": launch,
        "ncu_receipt": ready["ncu_receipt"],
        "nvbit_receipt": ready["nvbit_receipt"],
        "process_separation": ready["process_separation"],
        "trace_budget": {
            key: matrix_row["trace_budget"][key]
            for key in ("max_trace_bytes", "max_wall_time_s", "sampling_policy")
        },
    }
    cell["cell_identity_sha256"] = m6.canonical_sha256(cell)
    return cell


def _build_slots(
    cells: Sequence[Mapping[str, Any]],
    configs: Sequence[Mapping[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference = next(config for config in configs if config["reference"])
    cells_by_id = {cell["cell_id"]: cell for cell in cells}
    configs_by_id = {config["config_id"]: config for config in configs}
    tier1: list[dict[str, Any]] = []
    ordinal = 0
    for sweep in range(20):
        sweep_configs = configs if sweep < 10 else [reference]
        pending = [
            m6._make_slot(ordinal=0, tier=1, sweep=sweep, cell=cell, config=config)
            for cell in cells
            for config in sweep_configs
        ]
        pending.sort(key=lambda slot: m6._shuffle_key(seed, sweep, 1, slot))
        for pending_slot in pending:
            tier1.append(
                m6._make_slot(
                    ordinal=ordinal,
                    tier=1,
                    sweep=sweep,
                    cell=cells_by_id[pending_slot["cell_id"]],
                    config=configs_by_id[pending_slot["config_id"]],
                )
            )
            ordinal += 1
    tier2: list[dict[str, Any]] = []
    for sweep in range(20, 30):
        pending = [
            m6._make_slot(ordinal=0, tier=2, sweep=sweep, cell=cell, config=reference)
            for cell in cells
        ]
        pending.sort(key=lambda slot: m6._shuffle_key(seed, sweep, 2, slot))
        for pending_slot in pending:
            tier2.append(
                m6._make_slot(
                    ordinal=ordinal,
                    tier=2,
                    sweep=sweep,
                    cell=cells_by_id[pending_slot["cell_id"]],
                    config=reference,
                )
            )
            ordinal += 1
    if len(tier1) != TIER1_SLOTS or len(tier2) != TIER2_CANDIDATES:
        _fail("SUCCESSOR_M6_SLOT_COUNT", f"{len(tier1)}/{len(tier2)}")
    if [slot["ordinal"] for slot in tier1 + tier2] != list(range(MAXIMUM_SLOTS)):
        _fail("SUCCESSOR_M6_SLOT_COORDINATE", "ordinals are not contiguous")
    return tier1, tier2


def _source_identity(
    overlay: Mapping[str, Any], *, producer_commit: str | None = None
) -> dict[str, Any]:
    if producer_commit is None:
        producer_commit = _git_head()
    return {
        "pinned_bundle_commit": admission.PINNED_BUNDLE_COMMIT,
        "producer_commit": producer_commit,
        "base_bundle": overlay["base_bundle"],
        "admission_overlay": {
            "overlay_id": overlay["overlay_id"],
            "generated_at_utc": overlay["generated_at_utc"],
            "overlay_seal_sha256": overlay["overlay_seal_sha256"],
            "receipt_root": overlay["receipt_root"],
            "runnable_membership_sha256": overlay["chunking"][
                "runnable_membership_sha256"
            ],
        },
        "producer_files": {
            "schedule_builder": _file_ref(BUILDER_REL, role="schedule builder"),
            "schedule_schema": _file_ref(SCHEMA_REL, role="schedule schema"),
            "admission_producer": _file_ref(
                ADMISSION_PRODUCER_REL, role="admission producer"
            ),
            "admission_schema": _file_ref(
                admission.OVERLAY_SCHEMA_REL, role="admission schema"
            ),
            "receipt_schema": _file_ref(
                admission.RECEIPT_SCHEMA_REL, role="receipt schema"
            ),
            "canonical_m6_helpers": _file_ref(
                M6_CONTROLLER_REL, role="canonical M6 helper source"
            ),
            "cohort_validator": _file_ref(
                COHORT_BUILDER_REL, role="successor cohort validator"
            ),
            "capture_producer": _file_ref(
                CAPTURE_PRODUCER_REL, role="successor capture producer"
            ),
            "ncu_parser": _file_ref(PARSE_NCU_REL, role="NCU report parser"),
            "metric_registry_module": _file_ref(
                METRIC_REGISTRY_MODULE_REL, role="metric registry module"
            ),
            "metric_registry": _file_ref(
                METRIC_REGISTRY_REL, role="metric registry"
            ),
            "label_validator": _file_ref(
                IMPORT_LABEL_REL, role="label validator"
            ),
            "label_schema": _file_ref(
                admission.LOCAL_LABEL_SIDECAR_SCHEMA_REL, role="label schema"
            ),
            "trace_contract_validator": _file_ref(
                IMPORT_TRACE_CONTRACT_REL, role="trace contract validator"
            ),
            "trace_admission": _file_ref(
                TRACE_ADMISSION_REL, role="trace admission validator"
            ),
            "label_driver": _file_ref(LABEL_DRIVER_REL, role="NCU phase tool"),
            "nvbit_tracer": _file_ref(TRACER_TOOL_REL, role="NVBit phase tool"),
        },
    }


def _schedule_id(source_sha: str, config_sha: str, cell_sha: str, seed: int) -> str:
    return "cutlass-successor-m6-" + m6.canonical_sha256(
        {
            "source_identity_sha256": source_sha,
            "config_identity_sha256": config_sha,
            "cell_identity_sha256": cell_sha,
            "seed": seed,
        }
    )[:24]


def build_schedule(
    receipt_root: Path,
    *,
    seed: int,
    generated_at_utc: str,
) -> dict[str, Any]:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 9007199254740991
    ):
        _fail("SUCCESSOR_M6_SEED", repr(seed))
    if not receipt_root.is_absolute():
        _fail("SUCCESSOR_M6_RECEIPT_ROOT", "receipt root must be absolute")
    _, _, _, matrix = admission.validate_immutable_bundle()
    overlay = admission.build_overlay(
        receipt_root, generated_at_utc=generated_at_utc
    )
    joined = _require_capture_ready(overlay, matrix)
    cells = [_cell(matrix_row, ready) for matrix_row, ready in joined]
    configs = m6._configs()
    slots, tier2 = _build_slots(cells, configs, seed)
    source_identity = _source_identity(overlay)
    source_sha = m6.canonical_sha256(source_identity)
    config_sha = m6.canonical_sha256(configs)
    cell_sha = m6.canonical_sha256(cells)
    schedule_id = _schedule_id(source_sha, config_sha, cell_sha, seed)
    schedule = {
        "schema_version": 1,
        "artifact_kind": "cutlass_successor_m6_determinism_schedule",
        "schedule_id": schedule_id,
        "generated_at_utc": generated_at_utc,
        "seed": seed,
        "shuffle_algorithm": "sha256_seeded_sweep_v1",
        "source_identity": source_identity,
        "source_identity_sha256": source_sha,
        "admission": {
            "status": overlay["dispatch"]["status"],
            "typed_reason": overlay["dispatch"]["typed_reason"],
            "admission_m6_schedule_created": overlay["dispatch"][
                "m6_schedule_created"
            ],
            "overlay_seal_sha256": overlay["overlay_seal_sha256"],
            "counts": overlay["counts"],
        },
        "execution_dispatch": {
            "status": "required_unimplemented",
            "typed_reason": EXECUTION_BLOCK_REASON,
            "schedule_is_executable": False,
            "tier1_dispatch_allowed": False,
            "tier2_activation_allowed": False,
            "tier2_candidates_are_contract_only": True,
        },
        "configs": configs,
        "config_identity_sha256": config_sha,
        "cells": cells,
        "cell_identity_sha256": cell_sha,
        "policy": {
            "reference_config_id": REFERENCE_CONFIG,
            "reference_tier1_repeats": 20,
            "other_tier1_repeats": 10,
            "tier2_additional_reference_repeats": 10,
            "fresh_process_per_profile": True,
            "adaptive_dispersion_stopping": False,
        },
        "slots": slots,
        "tier2_candidates": tier2,
        "slots_sha256": m6.canonical_sha256(slots),
        "tier2_candidates_sha256": m6.canonical_sha256(tier2),
        "counts": {
            "cells": 3,
            "configs": 6,
            "tier1_slots": TIER1_SLOTS,
            "tier2_candidate_slots": TIER2_CANDIDATES,
            "maximum_slots": MAXIMUM_SLOTS,
            "tier1_fresh_processes": TIER1_SLOTS,
            "maximum_fresh_processes": MAXIMUM_SLOTS,
        },
        "claims": {
            "canonical_v1_impact": "none_additive_supplement_only",
            "canonical_m6_completed": False,
            "m6_done_created": False,
            "m7_handoff_ready": False,
            "publication_claim": False,
            "hardware_values_are_targets_only": True,
        },
    }
    schedule["schedule_sha256"] = m6.canonical_sha256(schedule)
    validate_schedule(schedule)
    return schedule


def _validate_schema(schedule: Mapping[str, Any]) -> None:
    try:
        schema = json.loads((ROOT / SCHEMA_REL).read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(
            schema, format_checker=jsonschema.FormatChecker()
        ).validate(schedule)
    except (OSError, UnicodeError, json.JSONDecodeError, jsonschema.SchemaError) as exc:
        _fail("SUCCESSOR_M6_SCHEMA_UNAVAILABLE", str(exc))
    except jsonschema.ValidationError as exc:
        location = "/".join(str(item) for item in exc.absolute_path) or "<root>"
        _fail("SUCCESSOR_M6_SCHEMA", f"{location}: {exc.message}")


def validate_schedule(schedule: Mapping[str, Any]) -> None:
    _validate_schema(schedule)
    body = {key: value for key, value in schedule.items() if key != "schedule_sha256"}
    if schedule.get("schedule_sha256") != m6.canonical_sha256(body):
        _fail("SUCCESSOR_M6_SCHEDULE_SEAL", "schedule seal differs")
    if schedule.get("source_identity_sha256") != m6.canonical_sha256(
        schedule["source_identity"]
    ):
        _fail("SUCCESSOR_M6_SOURCE_SEAL", "source identity differs")
    if schedule.get("config_identity_sha256") != m6.canonical_sha256(
        schedule["configs"]
    ):
        _fail("SUCCESSOR_M6_CONFIG_SEAL", "config identity differs")
    if schedule.get("cell_identity_sha256") != m6.canonical_sha256(
        schedule["cells"]
    ):
        _fail("SUCCESSOR_M6_CELL_SEAL", "cell identity differs")
    if schedule.get("schedule_id") != _schedule_id(
        schedule["source_identity_sha256"],
        schedule["config_identity_sha256"],
        schedule["cell_identity_sha256"],
        schedule["seed"],
    ):
        _fail("SUCCESSOR_M6_SCHEDULE_ID", "schedule identity does not rederive")
    if schedule.get("slots_sha256") != m6.canonical_sha256(schedule["slots"]):
        _fail("SUCCESSOR_M6_SLOT_SEAL", "Tier-1 slots differ")
    if schedule.get("tier2_candidates_sha256") != m6.canonical_sha256(
        schedule["tier2_candidates"]
    ):
        _fail("SUCCESSOR_M6_SLOT_SEAL", "Tier-2 slots differ")
    configs = m6._configs()
    if schedule.get("configs") != configs:
        _fail("SUCCESSOR_M6_CONFIG_CONTRACT", "canonical M6 configs differ")
    _require_unique(
        [config["config_id"] for config in configs],
        code="SUCCESSOR_M6_CONFIG_DUPLICATE",
        role="config identity",
    )

    cells = schedule["cells"]
    _require_unique(
        [cell["cell_id"] for cell in cells],
        code="SUCCESSOR_M6_CELL_DUPLICATE",
        role="cell identity",
    )
    _require_unique(
        [(cell["matrix_key"], cell["variant_id"]) for cell in cells],
        code="SUCCESSOR_M6_CELL_DUPLICATE",
        role="matrix/variant identity",
    )
    _require_unique(
        [
            cell[receipt_kind]["path"]
            for cell in cells
            for receipt_kind in ("ncu_receipt", "nvbit_receipt")
        ],
        code="SUCCESSOR_M6_RECEIPT_DUPLICATE",
        role="receipt path",
    )
    for cell in cells:
        body = {
            key: value for key, value in cell.items() if key != "cell_identity_sha256"
        }
        if cell.get("cell_identity_sha256") != m6.canonical_sha256(body):
            _fail("SUCCESSOR_M6_CELL_SEAL", str(cell.get("cell_id")))

    all_slots = schedule["slots"] + schedule["tier2_candidates"]
    _require_unique(
        [slot["ordinal"] for slot in all_slots],
        code="SUCCESSOR_M6_SLOT_DUPLICATE",
        role="ordinal",
    )
    _require_unique(
        [slot["slot_id"] for slot in all_slots],
        code="SUCCESSOR_M6_SLOT_DUPLICATE",
        role="slot identity",
    )
    _require_unique(
        [
            (
                slot["tier"],
                slot["sweep_index"],
                slot["cell_id"],
                slot["config_id"],
            )
            for slot in all_slots
        ],
        code="SUCCESSOR_M6_SLOT_DUPLICATE",
        role="tier/sweep/cell/config coordinate",
    )
    if [slot["ordinal"] for slot in all_slots] != list(range(MAXIMUM_SLOTS)):
        _fail("SUCCESSOR_M6_SLOT_COORDINATE", "ordinals are not contiguous")
    expected_slots, expected_tier2 = _build_slots(cells, configs, schedule["seed"])
    if schedule.get("slots") != expected_slots or schedule.get(
        "tier2_candidates"
    ) != expected_tier2:
        _fail("SUCCESSOR_M6_SLOT_DERIVATION", "slot arrays do not rederive")

    receipt_root = Path(
        schedule["source_identity"]["admission_overlay"]["receipt_root"]["path"]
    )
    _, _, _, matrix = admission.validate_immutable_bundle()
    overlay = admission.build_overlay(
        receipt_root, generated_at_utc=schedule["generated_at_utc"]
    )
    joined = _require_capture_ready(overlay, matrix)
    if schedule["source_identity"] != _source_identity(
        overlay,
        producer_commit=schedule["source_identity"]["producer_commit"],
    ):
        _fail("SUCCESSOR_M6_SOURCE_DRIFT", "source identity changed")
    expected_cells = [_cell(matrix_row, ready) for matrix_row, ready in joined]
    if schedule["cells"] != expected_cells:
        _fail("SUCCESSOR_M6_CELL_DRIFT", "admitted cell identity changed")
    expected_admission = {
        "status": overlay["dispatch"]["status"],
        "typed_reason": overlay["dispatch"]["typed_reason"],
        "admission_m6_schedule_created": overlay["dispatch"]["m6_schedule_created"],
        "overlay_seal_sha256": overlay["overlay_seal_sha256"],
        "counts": overlay["counts"],
    }
    if schedule["admission"] != expected_admission:
        _fail("SUCCESSOR_M6_ADMISSION_DRIFT", "admission overlay changed")


def _stage_payload(output: Path, payload: bytes) -> Path:
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
    except OSError as exc:
        _fail("SUCCESSOR_M6_OUTPUT", str(exc))
    stage = Path(name)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("staged schedule write made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
    except OSError as exc:
        try:
            os.close(descriptor)
        except OSError:
            pass
        stage.unlink(missing_ok=True)
        _fail("SUCCESSOR_M6_OUTPUT", str(exc))
    return stage


def _install_staged(stage: Path, output: Path) -> None:
    try:
        os.link(stage, output)
    except FileExistsError:
        _fail("SUCCESSOR_M6_NO_OVERWRITE", str(output))
    except OSError as exc:
        _fail("SUCCESSOR_M6_OUTPUT", str(exc))
    finally:
        try:
            stage.unlink(missing_ok=True)
        except OSError:
            # Once the final hard link exists, stale-stage cleanup cannot turn a
            # complete atomic publication into a reported refusal.
            pass


def write_schedule(
    schedule: Mapping[str, Any],
    output: Path,
    receipt_root: Path,
) -> None:
    if not output.is_absolute():
        _fail("SUCCESSOR_M6_OUTPUT", "output path must be absolute")
    if not receipt_root.is_absolute():
        _fail("SUCCESSOR_M6_RECEIPT_ROOT", "receipt root must be absolute")
    resolved_receipts = receipt_root.resolve()
    resolved_output = output.resolve(strict=False)
    try:
        resolved_output.relative_to(resolved_receipts)
    except ValueError:
        pass
    else:
        _fail("SUCCESSOR_M6_OUTPUT_OVERLAP", "output is inside receipt root")
    if output.exists() or output.is_symlink():
        _fail("SUCCESSOR_M6_NO_OVERWRITE", str(output))
    if not output.parent.is_dir() or output.parent.is_symlink():
        _fail("SUCCESSOR_M6_OUTPUT", "output parent is unavailable")

    try:
        payload = (json.dumps(schedule, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        snapshot = json.loads(payload)
    except (TypeError, ValueError, RuntimeError) as exc:
        _fail("SUCCESSOR_M6_OUTPUT", f"cannot snapshot schedule: {exc}")
    stage = _stage_payload(output, payload)
    try:
        # Validate the exact semantic JSON staged for publication. Inputs are
        # content-addressed and are revalidated again whenever the artifact is
        # consumed; the final path itself is installed atomically.
        validate_schedule(snapshot)
        _require_publication_source(snapshot)
        _install_staged(stage, output)
    except Exception:
        stage.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--generated-at-utc", required=True)
    args = parser.parse_args(argv)
    try:
        schedule = build_schedule(
            args.receipt_root,
            seed=args.seed,
            generated_at_utc=args.generated_at_utc,
        )
        write_schedule(schedule, args.output, args.receipt_root)
    except (ScheduleError, admission.AdmissionError, OSError, ValueError) as exc:
        print(f"CUTLASS successor M6 schedule refused: {exc}", file=sys.stderr)
        return 1
    print(f"CUTLASS successor M6 schedule: pass=True -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
