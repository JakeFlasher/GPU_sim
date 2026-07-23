"""Host-side aggregator + emitter for `phase_cycle_ledger_<repeat>_<launch>.json`
(non-root measurement NVBit phase-injection contract, producer half).

The aggregator takes `phase_clock_record_t`-shaped dicts (the on-the-
wire payload the device-side `record_phase_clock` injected function
in `tools/nvbit_tracer/tracer_tool/inject_funcs.cu` pushes through
the phase channel), pairs phase_open / phase_close records by
`(launch_index, cta_id_x, cta_id_y, cta_id_z, sm_id, warpid_tb,
phase_id)`, computes non-negative cycle deltas via
`close.clock64 - open.clock64`, and produces a schema-valid
`phase_cycle_ledger_<repeat>_<launch>.json` after self-validating
against `phase_cycle_ledger.schema.json`.

The receive-thread phase-record path in `tracer_tool.cu`'s
`phase_recv_thread_fun` decodes the channel bytes into instances of
`PhaseClockRecord`; this aggregator's contract is locked in here so
the producer-side test fixtures match the real receive-thread output.

Seven typed diagnostics required by the non-root measurement
NVBit phase-injection contract:

  * `PhaseRecordPairingFailed` — a phase_open record without a
    matching phase_close record (or vice versa).
  * `PhaseRecordCycleDeltaNegative` — close.clock64 < open.clock64
    (clock counter went backward).
  * `PhaseLedgerSchemaInvalid` — the assembled ledger fails
    `phase_cycle_ledger.schema.json` validation.
  * `PhaseRecordDuplicateOpen` — two phase_open records share the
    same per-warp-phase pairing key. The receive thread saw
    duplicate opens; the aggregator refuses to silently use only
    the later open.
  * `PhaseRecordDuplicateClose` — two phase_close records share the
    same per-warp-phase pairing key. Same as duplicate-open but
    for the closing edge.
  * `PhaseRecordLaunchIndexMismatch` — `build_phase_cycle_ledger`
    was given a top-level `launch_index` that disagrees with at
    least one input record's `launch_index`. Per-launch sidecars
    must contain only that launch's records.
  * `PhaseDefinitionInconsistent` — two records with the same
    `phase_id` carry different `phase_kind` integers. A first-
    record-wins fallback would silently hide such a conflict;
    surfacing it as a typed diagnostic lets the producer (or the
    SASS / basic-block discovery pass) be debugged loudly.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import validate as _validate  # type: ignore[import-not-found]


_SCHEMAS_DIR = Path(__file__).resolve().parent / "schemas"


# --- phase enums (mirror the non-root measurement-named field encodings) ------------


class PhaseKind(enum.Enum):
    """`phase_kind` enum encoding (matches the
    `phase_cycle_ledger.schema.json` `phase_kind` enum)."""

    function_entry = "function_entry"
    basic_block_entry = "basic_block_entry"
    bar_sync = "bar_sync"
    function_exit = "function_exit"


class ClockRole(enum.IntEnum):
    """`clock_role` encoding for paired %clock64 reads. Matches the
    `phase_clock_record_t` struct layout in
    `tools/nvbit_tracer/tracer_tool/common.h`."""

    phase_open = 0
    phase_close = 1


# Map phase_kind integer encoding (matches common.h's clock_role
# encoding for phase_kind: 0=function_entry, 1=function_exit,
# 2=basic_block_entry, 3=bar_sync) to the schema's enum value.
_PHASE_KIND_INT_TO_NAME: dict[int, str] = {
    0: PhaseKind.function_entry.value,
    1: PhaseKind.function_exit.value,
    2: PhaseKind.basic_block_entry.value,
    3: PhaseKind.bar_sync.value,
}


# --- phase-clock record dataclass -------------------------------------------


@dataclass(frozen=True)
class PhaseClockRecord:
    """Host-side dataclass mirroring `phase_clock_record_t` from
    `tools/nvbit_tracer/tracer_tool/common.h`. The receive-thread
    decode path in `tracer_tool.cu`'s `phase_recv_thread_fun`
    constructs instances of this class from the channel bytes."""

    launch_index: int
    cta_id_x: int
    cta_id_y: int
    cta_id_z: int
    sm_id: int
    warpid_tb: int
    phase_id: int
    phase_kind: int
    clock_role: int
    clock64: int
    active_mask: int


# --- Typed diagnostics ------------------------------------------------------


class PhaseRecordPairingFailed(Exception):
    TOKEN = "phase_record_pairing_failed"

    def __init__(
        self,
        unpaired_open: tuple[tuple, ...] = (),
        unpaired_close: tuple[tuple, ...] = (),
    ):
        super().__init__(
            f"{self.TOKEN}: unpaired phase_open records: "
            f"{[list(t) for t in unpaired_open]}; unpaired "
            f"phase_close records: {[list(t) for t in unpaired_close]}. "
            "The non-root measurement phase-aggregation rule requires every "
            "phase to be a paired-clock interval."
        )
        self.unpaired_open = unpaired_open
        self.unpaired_close = unpaired_close

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "unpaired_open": [list(t) for t in self.unpaired_open],
            "unpaired_close": [list(t) for t in self.unpaired_close],
        }


class PhaseRecordCycleDeltaNegative(Exception):
    TOKEN = "phase_record_cycle_delta_negative"

    def __init__(
        self,
        warp_phase_key: tuple,
        open_clock64: int,
        close_clock64: int,
    ):
        super().__init__(
            f"{self.TOKEN}: warp_phase_key={list(warp_phase_key)!r} "
            f"open.clock64={open_clock64} close.clock64={close_clock64}; "
            "the close clock is smaller than the open clock (clock "
            "wraparound or record reordering). The non-root measurement stat "
            "tuple requires non-negative cycle deltas."
        )
        self.warp_phase_key = warp_phase_key
        self.open_clock64 = open_clock64
        self.close_clock64 = close_clock64

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "warp_phase_key": list(self.warp_phase_key),
            "open_clock64": self.open_clock64,
            "close_clock64": self.close_clock64,
        }


class PhaseLedgerSchemaInvalid(Exception):
    TOKEN = "phase_ledger_schema_invalid"

    def __init__(self, schema_path: str, validation_error: str):
        super().__init__(
            f"{self.TOKEN}: assembled ledger failed schema-check "
            f"against {schema_path!r}: {validation_error}. The "
            "aggregator refuses to write a schema-invalid ledger."
        )
        self.schema_path = schema_path
        self.validation_error = validation_error

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "schema_path": self.schema_path,
            "validation_error": self.validation_error,
        }


class PhaseRecordDuplicateOpen(Exception):
    """Two phase_open records share the same per-warp-phase
    pairing key. The receive thread saw duplicate opens; the
    aggregator refuses to silently use only the later open."""

    TOKEN = "phase_record_duplicate_open"

    def __init__(
        self,
        warp_phase_key: tuple,
        first_clock64: int,
        second_clock64: int,
    ):
        super().__init__(
            f"{self.TOKEN}: warp_phase_key={list(warp_phase_key)!r} "
            f"first_open.clock64={first_clock64} "
            f"second_open.clock64={second_clock64}; the receive thread "
            "saw two phase_open records for the same warp-phase tuple. "
            "Real producer behavior must emit exactly one open per phase."
        )
        self.warp_phase_key = warp_phase_key
        self.first_clock64 = first_clock64
        self.second_clock64 = second_clock64

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "warp_phase_key": list(self.warp_phase_key),
            "first_clock64": self.first_clock64,
            "second_clock64": self.second_clock64,
        }


class PhaseRecordDuplicateClose(Exception):
    """Two phase_close records share the same per-warp-phase
    pairing key."""

    TOKEN = "phase_record_duplicate_close"

    def __init__(
        self,
        warp_phase_key: tuple,
        first_clock64: int,
        second_clock64: int,
    ):
        super().__init__(
            f"{self.TOKEN}: warp_phase_key={list(warp_phase_key)!r} "
            f"first_close.clock64={first_clock64} "
            f"second_close.clock64={second_clock64}; the receive thread "
            "saw two phase_close records for the same warp-phase tuple. "
            "Real producer behavior must emit exactly one close per phase."
        )
        self.warp_phase_key = warp_phase_key
        self.first_clock64 = first_clock64
        self.second_clock64 = second_clock64

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "warp_phase_key": list(self.warp_phase_key),
            "first_clock64": self.first_clock64,
            "second_clock64": self.second_clock64,
        }


class PhaseRecordLaunchIndexMismatch(Exception):
    """`build_phase_cycle_ledger` was given a top-level
    `launch_index` that disagrees with at least one input record's
    `launch_index`. Per-launch sidecars must contain only that
    launch's records."""

    TOKEN = "phase_record_launch_index_mismatch"

    def __init__(
        self,
        ledger_launch_index: int,
        offending_launch_indices: tuple[int, ...],
    ):
        super().__init__(
            f"{self.TOKEN}: ledger top-level launch_index="
            f"{ledger_launch_index} but input records carry "
            f"launch_index values {sorted(offending_launch_indices)!r}. "
            "Per-launch ledger sidecars must contain only that "
            "launch's phase records; mixed-launch input would fold "
            "two launches into one ledger and corrupt the per-launch "
            "contract."
        )
        self.ledger_launch_index = ledger_launch_index
        self.offending_launch_indices = offending_launch_indices

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "ledger_launch_index": self.ledger_launch_index,
            "offending_launch_indices": sorted(
                self.offending_launch_indices
            ),
        }


class PhaseDefinitionInconsistent(Exception):
    """Two records with the same `phase_id` carry different
    `phase_kind` integers. The first-record-wins policy of an
    earlier round hid this conflict; surfacing it lets the producer
    (or the SASS / basic-block discovery pass) be debugged."""

    TOKEN = "phase_definition_inconsistent"

    def __init__(
        self,
        phase_id: int,
        phase_kinds: tuple[int, ...],
    ):
        super().__init__(
            f"{self.TOKEN}: phase_id={phase_id} has conflicting "
            f"phase_kind integers {sorted(phase_kinds)!r}; one phase_id "
            "must map to exactly one phase_kind across all records."
        )
        self.phase_id = phase_id
        self.phase_kinds = phase_kinds

    def to_json(self) -> dict[str, Any]:
        return {
            "diagnostic": self.TOKEN,
            "phase_id": self.phase_id,
            "phase_kinds": sorted(self.phase_kinds),
        }


# --- Aggregation -----------------------------------------------------------


def _warp_phase_key(rec: PhaseClockRecord) -> tuple:
    return (
        rec.launch_index,
        rec.cta_id_x,
        rec.cta_id_y,
        rec.cta_id_z,
        rec.sm_id,
        rec.warpid_tb,
        rec.phase_id,
    )


def aggregate_phase_records(
    records: list[PhaseClockRecord],
) -> list[dict[str, Any]]:
    """Pair phase_open / phase_close records by warp+phase key,
    compute non-negative cycle deltas, and return a list of
    schema-shape `per_phase_cycle_record` dicts.

    Raises:
      * `PhaseRecordPairingFailed` if any record is unpaired.
      * `PhaseRecordCycleDeltaNegative` if `close.clock64 <
        open.clock64` for any pair.
      * `PhaseRecordDuplicateOpen` / `PhaseRecordDuplicateClose`
        if two records of the same role share the same
        per-warp-phase pairing key.
    """
    opens: dict[tuple, PhaseClockRecord] = {}
    closes: dict[tuple, PhaseClockRecord] = {}
    for rec in records:
        key = _warp_phase_key(rec)
        if rec.clock_role == ClockRole.phase_open.value:
            if key in opens:
                raise PhaseRecordDuplicateOpen(
                    warp_phase_key=key,
                    first_clock64=opens[key].clock64,
                    second_clock64=rec.clock64,
                )
            opens[key] = rec
        elif rec.clock_role == ClockRole.phase_close.value:
            if key in closes:
                raise PhaseRecordDuplicateClose(
                    warp_phase_key=key,
                    first_clock64=closes[key].clock64,
                    second_clock64=rec.clock64,
                )
            closes[key] = rec
        else:
            raise ValueError(
                f"unknown clock_role={rec.clock_role!r} in "
                f"phase_clock_record_t at key={list(key)!r}; expected "
                "0 (phase_open) or 1 (phase_close)."
            )

    open_keys = set(opens.keys())
    close_keys = set(closes.keys())
    unpaired_open = open_keys - close_keys
    unpaired_close = close_keys - open_keys
    if unpaired_open or unpaired_close:
        raise PhaseRecordPairingFailed(
            unpaired_open=tuple(sorted(unpaired_open)),
            unpaired_close=tuple(sorted(unpaired_close)),
        )

    out: list[dict[str, Any]] = []
    for key in sorted(open_keys):
        open_rec = opens[key]
        close_rec = closes[key]
        delta = close_rec.clock64 - open_rec.clock64
        if delta < 0:
            raise PhaseRecordCycleDeltaNegative(
                warp_phase_key=key,
                open_clock64=open_rec.clock64,
                close_clock64=close_rec.clock64,
            )
        out.append(
            {
                "phase_id": open_rec.phase_id,
                "cta_id_x": open_rec.cta_id_x,
                "cta_id_y": open_rec.cta_id_y,
                "cta_id_z": open_rec.cta_id_z,
                "sm_id": open_rec.sm_id,
                "warpid_tb": open_rec.warpid_tb,
                "cycles": int(delta),
            }
        )
    return out


def derive_phase_definitions(
    records: list[PhaseClockRecord],
    *,
    source_location_hints: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Derive a `phase_definitions` list from the unique
    (phase_id, phase_kind) pairs observed in the records. The
    optional `source_location_hints` map provides per-phase-id
    diagnostic strings; missing entries default to a placeholder.

    Raises `PhaseDefinitionInconsistent` if the same phase_id
    is observed with two different phase_kind integers."""
    by_phase: dict[int, set[int]] = {}
    for rec in records:
        by_phase.setdefault(rec.phase_id, set()).add(rec.phase_kind)
    out: list[dict[str, Any]] = []
    hints = source_location_hints or {}
    for phase_id in sorted(by_phase.keys()):
        kinds = by_phase[phase_id]
        if len(kinds) > 1:
            raise PhaseDefinitionInconsistent(
                phase_id=phase_id,
                phase_kinds=tuple(sorted(kinds)),
            )
        (phase_kind_int,) = kinds
        phase_kind_name = _PHASE_KIND_INT_TO_NAME.get(phase_kind_int)
        if phase_kind_name is None:
            raise ValueError(
                f"phase_id={phase_id} has unknown phase_kind integer "
                f"{phase_kind_int}; expected one of "
                f"{sorted(_PHASE_KIND_INT_TO_NAME.keys())} matching "
                "the non-root measurement-named common.h encoding."
            )
        out.append(
            {
                "phase_id": phase_id,
                "phase_kind": phase_kind_name,
                "source_location_hint": hints.get(
                    phase_id, f"phase_id={phase_id}"
                ),
            }
        )
    return out


# --- Ledger assembly + emission ---------------------------------------------


def build_phase_cycle_ledger(
    records: list[PhaseClockRecord],
    *,
    slug: str,
    shape_point: str,
    repeat_index: int,
    launch_index: int,
    kernel_signature_sha256: str,
    binary_sha256: str,
    nvbit_tracer_build_hash: str,
    claim_scope: str,
    ab_overhead_record: dict[str, Any],
    dual_run_manifest_ref: str,
    source_location_hints: dict[int, str] | None = None,
    phase_boundary_status: str = "resolved",
) -> dict[str, Any]:
    """Assemble a schema-valid `phase_cycle_ledger.schema.json`
    object from a list of `PhaseClockRecord`s + the per-launch
    metadata. Does NOT write to disk; the caller invokes
    `emit_phase_cycle_ledger` for that step.

    Per-launch sidecar contract: every input record's `launch_index`
    must equal the top-level `launch_index` argument. Mixed-launch
    input raises `PhaseRecordLaunchIndexMismatch` so two launches
    do not silently fold into one ledger."""
    offending = {
        rec.launch_index for rec in records if rec.launch_index != launch_index
    }
    if offending:
        raise PhaseRecordLaunchIndexMismatch(
            ledger_launch_index=launch_index,
            offending_launch_indices=tuple(sorted(offending)),
        )
    phase_definitions = derive_phase_definitions(
        records, source_location_hints=source_location_hints
    )
    per_phase_cycle_records = aggregate_phase_records(records)
    return {
        "schema_version": 1,
        "slug": slug,
        "shape_point": shape_point,
        "repeat_index": repeat_index,
        "launch_index": launch_index,
        "kernel_signature_sha256": kernel_signature_sha256,
        "binary_sha256": binary_sha256,
        "nvbit_tracer_build_hash": nvbit_tracer_build_hash,
        "claim_scope": claim_scope,
        "phase_definitions": phase_definitions,
        "per_phase_cycle_records": per_phase_cycle_records,
        "ab_overhead_record": ab_overhead_record,
        "dual_run_manifest_ref": dual_run_manifest_ref,
        "phase_boundary_status": phase_boundary_status,
    }


def _schema_check_ledger(ledger: dict[str, Any]) -> None:
    """Validate `ledger` against `phase_cycle_ledger.schema.json`
    using the existing schema-check route. Raises
    `PhaseLedgerSchemaInvalid` on failure."""
    schema_path = _SCHEMAS_DIR / "phase_cycle_ledger.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        _validate.validate(ledger, schema, root_path=schema_path)
    except _validate.ValidationError as e:
        raise PhaseLedgerSchemaInvalid(
            schema_path=str(schema_path),
            validation_error=str(e),
        ) from e


def emit_phase_cycle_ledger(
    ledger: dict[str, Any],
    output_path: Path,
) -> None:
    """Schema-validate `ledger` against
    `phase_cycle_ledger.schema.json` and atomically write it to
    `output_path` via tmp+rename. Raises `PhaseLedgerSchemaInvalid`
    on schema failure; no `<output>.tmp` lands on disk in that
    case."""
    _schema_check_ledger(ledger)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(ledger, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(output_path)


# --- Receive-thread phase-record decoding ----------------------------------


# `phase_clock_record_t` byte layout (little-endian, packed without
# explicit padding pragmas — matches the C struct in
# `tools/nvbit_tracer/tracer_tool/common.h`):
#
#     uint32_t launch_index;       // bytes 0..3
#     int32_t  cta_id_x;           // bytes 4..7
#     int32_t  cta_id_y;           // bytes 8..11
#     int32_t  cta_id_z;           // bytes 12..15
#     int32_t  sm_id;              // bytes 16..19
#     int32_t  warpid_tb;          // bytes 20..23
#     uint32_t phase_id;           // bytes 24..27
#     uint32_t phase_kind;         // bytes 28..31
#     uint32_t clock_role;         // bytes 32..35
#     uint64_t clock64;            // bytes 40..47 (8-byte aligned by C ABI)
#     uint32_t active_mask;        // bytes 48..51
#
# Total size with natural alignment: 56 bytes (4-byte trailing
# padding to align next struct boundary).
PHASE_CLOCK_RECORD_STRUCT_FORMAT = "<IiiiiiIIIxxxxQIxxxx"
PHASE_CLOCK_RECORD_SIZE = 56


def decode_phase_clock_records(buffer: bytes) -> list[PhaseClockRecord]:
    """Decode a contiguous byte buffer of `phase_clock_record_t`
    values into a list of `PhaseClockRecord` dataclasses. The
    receive-thread phase decode path in `tracer_tool.cu`'s
    `phase_recv_thread_fun` writes records to the host channel in
    this exact byte layout; the host-side aggregator decodes via
    this helper before feeding records into
    `build_phase_cycle_ledger`."""
    import struct

    if len(buffer) % PHASE_CLOCK_RECORD_SIZE != 0:
        raise ValueError(
            f"phase_clock_record_t buffer size {len(buffer)} is not a "
            f"multiple of {PHASE_CLOCK_RECORD_SIZE}; the channel byte "
            "stream is corrupted or the C struct layout has drifted."
        )
    out: list[PhaseClockRecord] = []
    for offset in range(0, len(buffer), PHASE_CLOCK_RECORD_SIZE):
        chunk = buffer[offset : offset + PHASE_CLOCK_RECORD_SIZE]
        (
            launch_index,
            cta_id_x,
            cta_id_y,
            cta_id_z,
            sm_id,
            warpid_tb,
            phase_id,
            phase_kind,
            clock_role,
            clock64,
            active_mask,
        ) = struct.unpack(PHASE_CLOCK_RECORD_STRUCT_FORMAT, chunk)
        out.append(
            PhaseClockRecord(
                launch_index=launch_index,
                cta_id_x=cta_id_x,
                cta_id_y=cta_id_y,
                cta_id_z=cta_id_z,
                sm_id=sm_id,
                warpid_tb=warpid_tb,
                phase_id=phase_id,
                phase_kind=phase_kind,
                clock_role=clock_role,
                clock64=clock64,
                active_mask=active_mask,
            )
        )
    return out
