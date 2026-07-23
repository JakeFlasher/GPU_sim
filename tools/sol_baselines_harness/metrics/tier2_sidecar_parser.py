"""Parser for the Tier-2 NVBit address-trace sidecar binary.

Reads ``nvbit_metrics/tier2_address_trace.t2a`` and yields one
parsed launch frame per ``LaunchHeaderV1`` / ``LaunchTrailerV1``
delimiter pair. The producer side is implemented in
``nvbit_tracer/metric_extractor/metric_extractor.cu`` (search for
``LaunchHeaderV1`` / ``LaunchTrailerV1`` to find the writer).

Per-frame the parser:

* validates the 160-byte ``LaunchHeaderV1`` (``'TI2A'`` magic,
  ``schema_major == 1``, ``endian_marker == 0x0102``);
* iterates length-prefixed records (kind ``0x01`` compact /
  ``0x02`` per-lane / ``0x7e`` truncation);
* validates the 80-byte ``LaunchTrailerV1`` (``'T2AT'`` magic, mirror
  fields matching the header's launch_uid / access_record_count /
  budget_bytes_used);
* recomputes SHA-256 over the canonical bytes (every byte after the
  160-byte header through the final trailer byte) and verifies
  against ``LaunchHeaderV1.sidecar_sha256``;
* checks per-record invariants: monotonically increasing
  ``sequence_id`` within the frame, ``sector_count <= 128``,
  ``record_len`` 8-byte aligned, ``record_len >= header_len``,
  ``(active_mask & ~lane_mask) == 0``.

On error the parser raises ``Tier2ParserError`` with a typed reason
matching the schema's closed enum.
"""

from __future__ import annotations

import dataclasses
import hashlib
import struct
from pathlib import Path
from typing import Dict, Iterator, List, Optional


HEADER_MAGIC = b"TI2A"
TRAILER_MAGIC = b"T2AT"
HEADER_LEN = 160
TRAILER_LEN = 80
RECORD_KIND_COMPACT = 0x01
RECORD_KIND_PER_LANE = 0x02
RECORD_KIND_TRUNCATION = 0x7E
SECTOR_COUNT_CAP = 128


class Tier2ParserError(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclasses.dataclass
class CompactAccessRecord:
    sequence_id: int
    pc_or_instruction_id: int
    cta_x: int
    cta_y: int
    cta_z: int
    warp_id: int
    bytes_per_access: int
    lane_mask: int
    active_mask: int
    op_kind: int
    cache_modifier: int
    address_space: int
    record_flags: int
    sector_ids: List[int]
    # L1 access policy byte at wire offset 50 (low half of the
    # legacy reserved-pad u16). Defaults to ``l1_unknown`` (0)
    # so existing fixtures and callers that build a
    # ``CompactAccessRecord`` without the field keep working.
    # Values: 0=l1_unknown, 1=l1_admitted, 2=l1_bypassed.
    l1_access_policy: int = 0


@dataclasses.dataclass
class TruncationRecord:
    truncation_reason: int
    next_sequence_id_not_emitted: int
    budget_bytes_limit: int
    budget_bytes_used_before_marker: int
    dropped_record_count_estimate: int
    # Per-launch deterministic seed derived in the producer as
    # ``first64_le(SHA256("solbench-tier2-sampling-v1\x00" || uint64_le(launch_uid)
    # || binary_sha256))``. Downstream consumers (predictor abstention,
    # later-task sampling decisions) read this for reproducibility
    # provenance. Truncated frames written before this field landed
    # in the producer carry 0 here; old sidecars therefore have
    # ``deterministic_sampling_seed == 0`` which the parser still
    # accepts (parser does not impose a non-zero invariant, since the
    # schema documents 0 as the v1.0 default when neither truncation
    # nor sampling occurred).
    deterministic_sampling_seed: int


@dataclasses.dataclass
class PerLaneAddressRecord:
    """Decoded `PerLaneAddressRecordV1` (record_kind=0x02) payload
    exposed by the parser for downstream fidelity checks. The
    fields mirror the schema's per-lane record body; the parser
    fills this in after `_parse_per_lane_record` validates the
    mirror invariants + canonicalization against the preceding
    compact record.
    """
    sequence_id: int
    lane_mask: int
    active_mask: int
    present_mask: int
    lane_addresses: List[int]


@dataclasses.dataclass
class LaunchFrame:
    launch_uid: int
    stream_ordinal: int
    grid: tuple
    access_record_count: int
    budget_bytes_used: int
    trailer_file_offset: int
    schema_major: int
    schema_minor: int
    sidecar_sha256: bytes
    digest_verified: bool
    records: List[CompactAccessRecord]
    truncation: Optional[TruncationRecord]
    # Surface the LaunchHeaderV1 metadata downstream consumers need
    # for the sampled/compressed-trace abstention path.
    # ``header_flags`` is the full 32-bit trailer-mirror word (bit 0
    # per_lane_records_present, bit 1 trace_truncated, bit 2
    # sampled_or_compressed_trace). ``binary_sha256`` is the 32-byte
    # image-identity digest the producer sourced via
    # METRIC_EXTRACTOR_TIER2_BINARY_SHA256_HEX.
    # ``sampled_or_compressed_trace`` mirrors bit 2 as a bool so
    # ``cache_predictor.predict_from_launch_frame`` can branch on it
    # without re-decoding the flag word.
    header_flags: int = 0
    binary_sha256: bytes = b"\x00" * 32
    sampled_or_compressed_trace: bool = False
    # Decoded per-lane records keyed by the matching compact's
    # sequence_id. Empty when no `record_kind=0x02` records appear
    # in the frame. The parser populates this in `parse_launch_frames`
    # after `_parse_per_lane_record` validates each per-lane record
    # against the preceding compact's mirrors + sector canonicalization.
    per_lane_records_by_sequence: Dict[int, PerLaneAddressRecord] = (
        dataclasses.field(default_factory=dict)
    )


def _read_exact(data: bytes, offset: int, length: int, label: str) -> bytes:
    if offset + length > len(data):
        raise Tier2ParserError(
            "tier2_truncated_record",
            f"need {length} bytes at offset {offset} for {label}; "
            f"file size {len(data)}",
        )
    return data[offset:offset + length]


def _parse_compact_record(data: bytes, offset: int) -> tuple:
    if offset + 4 > len(data):
        raise Tier2ParserError(
            "tier2_truncated_record",
            f"compact record header begins beyond EOF at offset {offset}",
        )
    record_len = struct.unpack_from("<H", data, offset)[0]
    record_kind = data[offset + 2]
    header_len = data[offset + 3]
    if record_kind != RECORD_KIND_COMPACT:
        raise Tier2ParserError(
            "tier2_unknown_record_kind",
            f"expected compact (0x01); got 0x{record_kind:02x} at offset {offset}",
        )
    if header_len != 0x38:
        raise Tier2ParserError(
            "tier2_record_length_invalid",
            f"compact header_len must be 56; got {header_len}",
        )
    if record_len < 56 or record_len % 8 != 0:
        raise Tier2ParserError(
            "tier2_record_length_invalid",
            f"compact record_len {record_len} not 8-byte aligned or < 56",
        )
    if offset + record_len > len(data):
        raise Tier2ParserError(
            "tier2_truncated_record",
            f"compact record at offset {offset} length {record_len} exceeds EOF",
        )
    op_kind = data[offset + 4]
    cache_modifier = data[offset + 5]
    address_space = data[offset + 6]
    record_flags = data[offset + 7]
    # L1 access policy occupies the low byte of what was
    # previously the u16 reserved-pad at wire offset 50.
    # Values:
    #   0 = l1_unknown   (legacy / atomic / cp.async / TMA / spill)
    #   1 = l1_admitted  (ordinary L1+L2 global load)
    #   2 = l1_bypassed  (`.cg` SASS suffix on sm_90+ OR Ampere
    #                     `LDG.E.STRONG.GPU` fold on sm_86)
    # Legacy sidecars carry zero at offset 50 (l1_unknown) and
    # continue to parse cleanly. The high byte (offset 51) plus
    # the u32 at offsets 52..55 stay reserved-zero.
    l1_access_policy = data[offset + 50]
    # record_flags in v1.0 defines only bit 0 (per_lane_extension_
    # follows). Bits 1..7 are reserved-zero per schema Section 2;
    # reject any frame with reserved bits set.
    if (record_flags & ~0x1) != 0:
        raise Tier2ParserError(
            "tier2_sidecar_schema_violation",
            f"compact record_flags=0x{record_flags:02x} has reserved "
            f"bits set; only bit 0 (per_lane_extension_follows) is "
            f"defined in v1.0",
        )
    # Enum-range validation. v1.0 schema admits the listed values
    # plus the explicit ``unknown=255`` sentinel for cache_modifier
    # only. Other enums use 0=unknown and have no out-of-band 255
    # sentinel, so any value outside the closed range is a violation.
    _OP_KIND_VALID = {0, 1, 2, 3, 4, 5, 6, 7}
    _CACHE_MODIFIER_VALID = {0, 1, 2, 3, 4, 5, 255}
    # Address-space enum: 0=unknown, 1=global, 2=shared,
    # 3=local, 4=dsmem, 5=constant. Constant memory loads
    # were added to disambiguate the kernel-parameter-
    # passing path (LDC reads from c[0x0] bank) from the
    # global-memory path; constant cache misses contribute
    # to the L2 read denominator.
    _ADDRESS_SPACE_VALID = {0, 1, 2, 3, 4, 5}
    _L1_ACCESS_POLICY_VALID = {0, 1, 2}
    if op_kind not in _OP_KIND_VALID:
        raise Tier2ParserError(
            "tier2_sidecar_schema_violation",
            f"op_kind {op_kind} not in admitted range {sorted(_OP_KIND_VALID)}",
        )
    if cache_modifier not in _CACHE_MODIFIER_VALID:
        raise Tier2ParserError(
            "tier2_sidecar_schema_violation",
            f"cache_modifier {cache_modifier} not in admitted range "
            f"{sorted(_CACHE_MODIFIER_VALID)}",
        )
    if address_space not in _ADDRESS_SPACE_VALID:
        raise Tier2ParserError(
            "tier2_sidecar_schema_violation",
            f"address_space {address_space} not in admitted range "
            f"{sorted(_ADDRESS_SPACE_VALID)}",
        )
    if l1_access_policy not in _L1_ACCESS_POLICY_VALID:
        raise Tier2ParserError(
            "tier2_sidecar_schema_violation",
            f"l1_access_policy {l1_access_policy} not in admitted "
            f"range {sorted(_L1_ACCESS_POLICY_VALID)}",
        )
    (sequence_id, pc_or_instruction_id) = struct.unpack_from(
        "<QQ", data, offset + 8
    )
    (cta_x, cta_y, cta_z) = struct.unpack_from("<III", data, offset + 24)
    (warp_id, bytes_per_access) = struct.unpack_from("<HH", data, offset + 36)
    (lane_mask, active_mask) = struct.unpack_from("<II", data, offset + 40)
    sector_count = struct.unpack_from("<H", data, offset + 48)[0]
    if sector_count > SECTOR_COUNT_CAP:
        raise Tier2ParserError(
            "tier2_sector_count_overflow",
            f"sector_count {sector_count} > cap {SECTOR_COUNT_CAP}",
        )
    expected_body = 56 + sector_count * 8
    expected_len = expected_body + ((-expected_body) & 7)
    if expected_len != record_len:
        raise Tier2ParserError(
            "tier2_record_length_invalid",
            f"compact record_len {record_len} mismatches "
            f"align8(56 + 8 * {sector_count}) = {expected_len}",
        )
    if (active_mask & (~lane_mask & 0xffffffff)) != 0:
        raise Tier2ParserError(
            "tier2_active_mask_inconsistent",
            f"active_mask=0x{active_mask:08x} not subset of "
            f"lane_mask=0x{lane_mask:08x} at offset {offset}",
        )
    # Active lanes with positive bytes_per_access must produce a
    # non-empty sector set; the device-side capture path's sector-
    # derivation expands [addr, addr+bytes_per_access) into at least
    # one sector per predicate-true lane.
    if (
        active_mask != 0
        and bytes_per_access > 0
        and sector_count == 0
    ):
        raise Tier2ParserError(
            "tier2_active_mask_inconsistent",
            f"active_mask=0x{active_mask:08x} with bytes_per_access "
            f"{bytes_per_access} but sector_count=0 at offset {offset}",
        )
    # Reserved-zero discipline. The schema previously defined a u16
    # reserved-pad at offsets 50..51 plus a u32 at offsets 52..55.
    # The LOW byte (offset 50) is carved out as
    # ``l1_access_policy``; the HIGH byte (offset 51) plus the u32
    # at offsets 52..55 remain reserved-zero. Nonzero bytes at
    # offset 51 or offsets 52..55 are still a schema violation.
    reserved_pad_hi = data[offset + 51]
    reserved_u32 = struct.unpack_from("<I", data, offset + 52)[0]
    if reserved_pad_hi != 0 or reserved_u32 != 0:
        raise Tier2ParserError(
            "tier2_sidecar_schema_violation",
            f"compact reserved-zero fields nonzero "
            f"(pad_hi=0x{reserved_pad_hi:02x} "
            f"u32=0x{reserved_u32:08x}) "
            f"at offset {offset}",
        )
    # Padding bytes after the sector_ids tail must be zero (the host
    # serializer fills them with zeros to reach the 8-byte alignment).
    body_end = 56 + sector_count * 8
    if body_end < record_len:
        padding = data[offset + body_end:offset + record_len]
        if any(b != 0 for b in padding):
            raise Tier2ParserError(
                "tier2_sidecar_schema_violation",
                f"compact padding bytes nonzero "
                f"({padding.hex()}) at offset {offset + body_end}",
            )
    sector_ids = list(
        struct.unpack_from(f"<{sector_count}Q", data, offset + 56)
    )
    rec = CompactAccessRecord(
        sequence_id=sequence_id,
        pc_or_instruction_id=pc_or_instruction_id,
        cta_x=cta_x,
        cta_y=cta_y,
        cta_z=cta_z,
        warp_id=warp_id,
        bytes_per_access=bytes_per_access,
        lane_mask=lane_mask,
        active_mask=active_mask,
        op_kind=op_kind,
        cache_modifier=cache_modifier,
        address_space=address_space,
        record_flags=record_flags,
        sector_ids=sector_ids,
        l1_access_policy=l1_access_policy,
    )
    return rec, record_len


def _parse_per_lane_record(
    data: bytes, offset: int, preceding_compact: CompactAccessRecord
) -> tuple:
    """Validate a PerLaneAddressRecordV1 (record_kind=0x02) per schema
    Section 7. Length is fixed at 288 bytes; the record carries 32
    per-lane base addresses immediately after a 32-byte header. The
    helper enforces mirror invariants against the preceding compact
    record (same sequence_id, lane_mask, active_mask;
    present_mask == active_mask), reserved-zero bytes at offsets
    24-31, and the inactive-lane sentinel 0xffffffffffffffff for
    lanes not in the active mask.

    Returns (offset_advance, per_lane_record) where per_lane_record
    is a fully-populated `PerLaneAddressRecord` carrying sequence_id,
    lane_mask, active_mask, present_mask, and the 32-entry
    `lane_addresses` list. Callers (currently `parse_launch_frames`)
    key the record by sequence_id into the matching `LaunchFrame`'s
    `per_lane_records_by_sequence` dict so downstream consumers can
    inspect decoded per-lane payloads.
    """
    if offset + 288 > len(data):
        raise Tier2ParserError(
            "tier2_truncated_record",
            f"per-lane record at offset {offset} truncated by EOF",
        )
    (record_len,) = struct.unpack_from("<H", data, offset)
    record_kind = data[offset + 2]
    header_len = data[offset + 3]
    if record_kind != RECORD_KIND_PER_LANE:
        raise Tier2ParserError(
            "tier2_unknown_record_kind",
            f"expected per-lane (0x02); got 0x{record_kind:02x} at "
            f"offset {offset}",
        )
    if record_len != 288:
        raise Tier2ParserError(
            "tier2_record_length_invalid",
            f"per-lane record_len {record_len} must be 288",
        )
    if header_len != 0x20:  # 32
        raise Tier2ParserError(
            "tier2_record_length_invalid",
            f"per-lane header_len {header_len} must be 32",
        )
    (present_mask,) = struct.unpack_from("<I", data, offset + 4)
    (sequence_id,) = struct.unpack_from("<Q", data, offset + 8)
    (lane_mask, active_mask) = struct.unpack_from(
        "<II", data, offset + 16
    )
    # Reserved-zero bytes at offsets 24-31 (8 bytes).
    reserved_block = data[offset + 24:offset + 32]
    if any(b != 0 for b in reserved_block):
        raise Tier2ParserError(
            "tier2_sidecar_schema_violation",
            f"per-lane reserved-zero region nonzero: "
            f"{reserved_block.hex()}",
        )
    # Mirror invariants against the preceding compact record.
    if sequence_id != preceding_compact.sequence_id:
        raise Tier2ParserError(
            "tier2_sidecar_schema_violation",
            f"per-lane sequence_id {sequence_id} != preceding compact "
            f"sequence_id {preceding_compact.sequence_id}",
        )
    if lane_mask != preceding_compact.lane_mask:
        raise Tier2ParserError(
            "tier2_sidecar_schema_violation",
            f"per-lane lane_mask 0x{lane_mask:08x} != preceding "
            f"compact lane_mask 0x{preceding_compact.lane_mask:08x}",
        )
    if active_mask != preceding_compact.active_mask:
        raise Tier2ParserError(
            "tier2_sidecar_schema_violation",
            f"per-lane active_mask 0x{active_mask:08x} != preceding "
            f"compact active_mask 0x{preceding_compact.active_mask:08x}",
        )
    if present_mask != active_mask:
        raise Tier2ParserError(
            "tier2_sidecar_schema_violation",
            f"per-lane present_mask 0x{present_mask:08x} != "
            f"active_mask 0x{active_mask:08x}",
        )
    lane_addresses = list(struct.unpack_from("<32Q", data, offset + 32))
    # Inactive lanes must carry the sentinel 0xffffffffffffffff.
    inactive_sentinel = 0xffffffffffffffff
    for lane_idx in range(32):
        bit = (active_mask >> lane_idx) & 1
        if bit == 0 and lane_addresses[lane_idx] != inactive_sentinel:
            raise Tier2ParserError(
                "tier2_sidecar_schema_violation",
                f"per-lane lane {lane_idx} is inactive "
                f"(active_mask bit clear) but base address "
                f"0x{lane_addresses[lane_idx]:016x} is not the "
                f"sentinel 0xffffffffffffffff",
            )
    # Sector canonicalization: reconstruct the ordered 32 B sector
    # set from active-lane base addresses and the preceding compact's
    # bytes_per_access. Each active lane's [base, base + bytes) byte
    # range expands to one or more 32 B sectors (base >> 5 through
    # (base + bytes - 1) >> 5). Sectors are accumulated in ascending
    # lane order with first-observed de-duplication and must equal
    # the preceding compact record's sector_ids exactly.
    bpa = preceding_compact.bytes_per_access
    seen = set()
    reconstructed: List[int] = []
    for lane_idx in range(32):
        if not ((active_mask >> lane_idx) & 1):
            continue
        base = lane_addresses[lane_idx]
        first_sec = base >> 5
        last_sec = (
            (base + bpa - 1) >> 5 if bpa > 0 else first_sec
        )
        for sec in range(first_sec, last_sec + 1):
            if sec not in seen:
                seen.add(sec)
                reconstructed.append(sec)
    if reconstructed != preceding_compact.sector_ids:
        raise Tier2ParserError(
            "tier2_active_mask_inconsistent",
            f"per-lane reconstructed sector set {reconstructed} != "
            f"preceding compact sector_ids "
            f"{preceding_compact.sector_ids}",
        )
    per_lane_record = PerLaneAddressRecord(
        sequence_id=sequence_id,
        lane_mask=lane_mask,
        active_mask=active_mask,
        present_mask=present_mask,
        lane_addresses=lane_addresses,
    )
    return 288, per_lane_record


def _parse_truncation_record(data: bytes, offset: int) -> tuple:
    if offset + 56 > len(data):
        raise Tier2ParserError(
            "tier2_truncated_record",
            f"truncation record at offset {offset} truncated by EOF",
        )
    (record_len,) = struct.unpack_from("<H", data, offset)
    record_kind = data[offset + 2]
    header_len = data[offset + 3]
    if record_kind != RECORD_KIND_TRUNCATION:
        raise Tier2ParserError(
            "tier2_unknown_record_kind",
            f"expected truncation (0x7e); got 0x{record_kind:02x}",
        )
    if header_len != 0x38 or record_len != 56:
        raise Tier2ParserError(
            "tier2_record_length_invalid",
            f"truncation record_len {record_len} header_len {header_len} "
            f"must both be 56",
        )
    (
        truncation_reason,
        _reserved,
        next_seq,
        budget_limit,
        budget_used_before,
        dropped_estimate,
        sampling_seed,
        _reserved2,
    ) = struct.unpack_from("<HHQQQQQQ", data, offset + 4)
    trunc = TruncationRecord(
        truncation_reason=truncation_reason,
        next_sequence_id_not_emitted=next_seq,
        budget_bytes_limit=budget_limit,
        budget_bytes_used_before_marker=budget_used_before,
        dropped_record_count_estimate=dropped_estimate,
        deterministic_sampling_seed=sampling_seed,
    )
    return trunc, record_len


def parse_launch_frames(path: Path) -> List[LaunchFrame]:
    """Parse every launch frame in the sidecar binary. Returns the
    decoded frames; raises Tier2ParserError on the first violation.
    """
    raw = Path(path).read_bytes()
    frames: List[LaunchFrame] = []
    offset = 0
    while offset < len(raw):
        if offset + HEADER_LEN > len(raw):
            raise Tier2ParserError(
                "tier2_truncated_record",
                f"header at offset {offset} truncated by EOF",
            )
        magic = raw[offset:offset + 4]
        if magic != HEADER_MAGIC:
            raise Tier2ParserError(
                "tier2_sidecar_schema_violation",
                f"expected header magic 'TI2A' at offset {offset}; "
                f"got {magic!r}",
            )
        schema_major = struct.unpack_from("<H", raw, offset + 4)[0]
        schema_minor = struct.unpack_from("<H", raw, offset + 6)[0]
        header_len = struct.unpack_from("<H", raw, offset + 8)[0]
        endian_marker = struct.unpack_from("<H", raw, offset + 10)[0]
        if schema_major != 1:
            raise Tier2ParserError(
                "tier2_unknown_version",
                f"schema_major {schema_major} not supported",
            )
        if header_len != HEADER_LEN:
            raise Tier2ParserError(
                "tier2_sidecar_schema_violation",
                f"header_len {header_len} != {HEADER_LEN}",
            )
        if endian_marker != 0x0102:
            raise Tier2ParserError(
                "tier2_sidecar_schema_violation",
                f"endian_marker 0x{endian_marker:04x} != 0x0102",
            )
        header_flags = struct.unpack_from("<I", raw, offset + 12)[0]
        # binary_sha256 lives at header offset 16, 32 bytes. The
        # schema requires it non-zero whenever header_flags bit 2
        # (sampled_or_compressed_trace) is set; a zero digest in a
        # sampled/compressed frame is a schema violation because the
        # downstream predictor abstention path cannot honestly derive
        # its provenance seed without a real binary identity.
        binary_sha256 = bytes(raw[offset + 16:offset + 48])
        sampled_or_compressed_trace = bool(header_flags & 0x4)
        if sampled_or_compressed_trace and binary_sha256 == b"\x00" * 32:
            raise Tier2ParserError(
                "tier2_sidecar_schema_violation",
                "header_flags bit 2 (sampled_or_compressed_trace) is "
                "set but binary_sha256 is all zeros; the schema "
                "requires a real binary identity digest whenever a "
                "frame is sampled or compressed (predictor abstention "
                "provenance depends on it).",
            )
        launch_uid = struct.unpack_from("<Q", raw, offset + 48)[0]
        stream_ordinal = struct.unpack_from("<I", raw, offset + 56)[0]
        grid = struct.unpack_from("<III", raw, offset + 60)
        budget_bytes_used = struct.unpack_from("<Q", raw, offset + 80)[0]
        budget_bytes_limit = struct.unpack_from(
            "<Q", raw, offset + 88
        )[0]
        access_record_count = struct.unpack_from(
            "<Q", raw, offset + 96
        )[0]
        trailer_offset_decl = struct.unpack_from(
            "<Q", raw, offset + 104
        )[0]
        # Reserved-zero region at header offsets 112..127 (16 bytes).
        header_reserved = raw[offset + 112:offset + 128]
        if any(b != 0 for b in header_reserved):
            raise Tier2ParserError(
                "tier2_sidecar_schema_violation",
                f"header reserved-zero region at offset {offset + 112} "
                f"nonzero: {header_reserved.hex()}",
            )
        declared_digest = bytes(raw[offset + 128:offset + 160])
        records_start = offset + HEADER_LEN

        # Walk records up to the trailer.
        records: List[CompactAccessRecord] = []
        per_lane_records_by_sequence: Dict[int, PerLaneAddressRecord] = {}
        truncation: Optional[TruncationRecord] = None
        cur = records_start
        last_seq = -1
        canonical_end = trailer_offset_decl + TRAILER_LEN
        if canonical_end > len(raw):
            raise Tier2ParserError(
                "tier2_truncated_record",
                f"declared trailer end {canonical_end} > file size {len(raw)}",
            )
        # The per_lane_extension_follows flag on a compact record
        # requires the next record to be a kind=0x02 per-lane record
        # carrying the same sequence_id. Track that across iterations.
        expect_per_lane_for_seq = None
        while cur < trailer_offset_decl:
            kind = raw[cur + 2]
            if kind == RECORD_KIND_COMPACT:
                if expect_per_lane_for_seq is not None:
                    raise Tier2ParserError(
                        "tier2_sidecar_schema_violation",
                        f"compact record at offset {cur} expected a "
                        f"preceding per-lane (0x02) follow-up for "
                        f"sequence_id {expect_per_lane_for_seq}",
                    )
                rec, rec_len = _parse_compact_record(raw, cur)
                if rec.sequence_id <= last_seq:
                    raise Tier2ParserError(
                        "tier2_sequence_order_violation",
                        f"sequence_id {rec.sequence_id} not > prev {last_seq}",
                    )
                last_seq = rec.sequence_id
                records.append(rec)
                cur += rec_len
                if (rec.record_flags & 0x1) != 0:
                    expect_per_lane_for_seq = rec.sequence_id
            elif kind == RECORD_KIND_TRUNCATION:
                if expect_per_lane_for_seq is not None:
                    raise Tier2ParserError(
                        "tier2_sidecar_schema_violation",
                        f"truncation at offset {cur} where a per-lane "
                        f"follow-up was expected for sequence_id "
                        f"{expect_per_lane_for_seq}",
                    )
                trunc, rec_len = _parse_truncation_record(raw, cur)
                truncation = trunc
                cur += rec_len
                # No records may follow the marker; the next bytes
                # MUST be the trailer.
                if cur != trailer_offset_decl:
                    raise Tier2ParserError(
                        "tier2_trace_budget_marker_invalid",
                        f"records present after truncation marker: "
                        f"offset {cur} != trailer_file_offset "
                        f"{trailer_offset_decl}",
                    )
                break
            elif kind == RECORD_KIND_PER_LANE:
                if expect_per_lane_for_seq is None:
                    raise Tier2ParserError(
                        "tier2_sidecar_schema_violation",
                        f"per-lane record at offset {cur} not flagged "
                        f"by the preceding compact record's "
                        f"per_lane_extension_follows bit",
                    )
                # Validate the per-lane record against the
                # preceding compact (last entry in records[]).
                if not records:
                    raise Tier2ParserError(
                        "tier2_sidecar_schema_violation",
                        f"per-lane record at offset {cur} with no "
                        f"preceding compact record on this launch frame",
                    )
                advance, per_lane_rec = _parse_per_lane_record(
                    raw, cur, records[-1]
                )
                per_lane_records_by_sequence[per_lane_rec.sequence_id] = (
                    per_lane_rec
                )
                cur += advance
                expect_per_lane_for_seq = None
            else:
                raise Tier2ParserError(
                    "tier2_unknown_record_kind",
                    f"unknown record kind 0x{kind:02x} at offset {cur}",
                )
        if expect_per_lane_for_seq is not None:
            raise Tier2ParserError(
                "tier2_sidecar_schema_violation",
                f"per_lane_extension_follows flag set on last compact "
                f"record but no per-lane record before trailer "
                f"(sequence_id {expect_per_lane_for_seq})",
            )
        if cur != trailer_offset_decl:
            raise Tier2ParserError(
                "tier2_sidecar_schema_violation",
                f"records ended at {cur}, expected trailer_file_offset "
                f"{trailer_offset_decl}",
            )
        # Truncation marker presence iff header_flags trace_truncated
        # bit (bit 1) is set. Either both are present or neither.
        flag_trace_truncated = (header_flags & 0x2) != 0
        if flag_trace_truncated and truncation is None:
            raise Tier2ParserError(
                "tier2_trace_budget_marker_invalid",
                "header_flags trace_truncated bit set but no "
                "TruncationRecordV1 emitted",
            )
        if (not flag_trace_truncated) and truncation is not None:
            raise Tier2ParserError(
                "tier2_trace_budget_marker_invalid",
                "TruncationRecordV1 emitted but header_flags "
                "trace_truncated bit not set",
            )
        # Per-lane record presence iff header_flags
        # per_lane_records_present bit (bit 0) is set. The schema's
        # mirror invariant ties the header bit to whether ANY 0x02
        # record appears in the frame; the trailer mirror is checked
        # separately below.
        flag_per_lane_present = (header_flags & 0x1) != 0
        per_lane_count = len(per_lane_records_by_sequence)
        if flag_per_lane_present and per_lane_count == 0:
            raise Tier2ParserError(
                "tier2_sidecar_schema_violation",
                "header_flags per_lane_records_present bit set but "
                "no PerLaneAddressRecordV1 records emitted",
            )
        if (not flag_per_lane_present) and per_lane_count > 0:
            raise Tier2ParserError(
                "tier2_sidecar_schema_violation",
                f"PerLaneAddressRecordV1 records emitted "
                f"({per_lane_count}) but header_flags "
                f"per_lane_records_present bit not set",
            )

        # Trailer.
        trailer = raw[cur:cur + TRAILER_LEN]
        if trailer[:4] != TRAILER_MAGIC:
            raise Tier2ParserError(
                "tier2_trailer_missing",
                f"expected trailer magic 'T2AT' at offset {cur}; "
                f"got {trailer[:4]!r}",
            )
        trailer_len = struct.unpack_from("<H", trailer, 4)[0]
        trailer_schema_major = struct.unpack_from("<H", trailer, 6)[0]
        if trailer_len != TRAILER_LEN:
            raise Tier2ParserError(
                "tier2_sidecar_schema_violation",
                f"trailer_len {trailer_len} != {TRAILER_LEN}",
            )
        if trailer_schema_major != 1:
            raise Tier2ParserError(
                "tier2_unknown_version",
                f"trailer schema_major {trailer_schema_major} != 1",
            )
        trailer_launch_uid = struct.unpack_from("<Q", trailer, 8)[0]
        trailer_record_count = struct.unpack_from("<Q", trailer, 16)[0]
        trailer_budget_used = struct.unpack_from("<Q", trailer, 24)[0]
        trailer_budget_limit = struct.unpack_from("<Q", trailer, 32)[0]
        trailer_flags = struct.unpack_from("<I", trailer, 40)[0]
        trailer_reserved = trailer[56:80]
        if any(b != 0 for b in trailer_reserved):
            raise Tier2ParserError(
                "tier2_sidecar_schema_violation",
                f"trailer reserved-zero region nonzero: "
                f"{trailer_reserved.hex()}",
            )
        if trailer_launch_uid != launch_uid:
            raise Tier2ParserError(
                "tier2_sidecar_schema_violation",
                f"trailer launch_uid {trailer_launch_uid} != "
                f"header launch_uid {launch_uid}",
            )
        if trailer_record_count != access_record_count:
            raise Tier2ParserError(
                "tier2_sidecar_schema_violation",
                f"trailer record_count {trailer_record_count} != "
                f"header record_count {access_record_count}",
            )
        if trailer_budget_used != budget_bytes_used:
            raise Tier2ParserError(
                "tier2_sidecar_schema_violation",
                f"trailer budget_bytes_used {trailer_budget_used} != "
                f"header budget_bytes_used {budget_bytes_used}",
            )
        if trailer_budget_limit != budget_bytes_limit:
            raise Tier2ParserError(
                "tier2_sidecar_schema_violation",
                f"trailer budget_bytes_limit {trailer_budget_limit} != "
                f"header budget_bytes_limit {budget_bytes_limit}",
            )
        # Flag-word discipline. The schema defines bits 0/1/2
        # (per_lane_records_present, trace_truncated,
        # sampled_or_compressed_trace) in v1.0; bits 3..31 are
        # reserved zero. Reject any frame with reserved bits set on
        # EITHER side, AND require the FULL 32-bit flag word to
        # mirror (not just the low 3 bits); schema Section 2's
        # mirror invariant is over the whole word.
        if (header_flags & ~0x7) != 0:
            raise Tier2ParserError(
                "tier2_sidecar_schema_violation",
                f"header_flags 0x{header_flags:08x} has reserved bits "
                f"set; only bits 0..2 are defined in v1.0",
            )
        if (trailer_flags & ~0x7) != 0:
            raise Tier2ParserError(
                "tier2_sidecar_schema_violation",
                f"trailer_flags 0x{trailer_flags:08x} has reserved "
                f"bits set; only bits 0..2 are defined in v1.0",
            )
        if header_flags != trailer_flags:
            raise Tier2ParserError(
                "tier2_sidecar_schema_violation",
                f"header_flags 0x{header_flags:08x} does not mirror "
                f"trailer_flags 0x{trailer_flags:08x}",
            )

        # Recompute digest over canonical bytes (records + trailer).
        canonical = raw[records_start:canonical_end]
        computed = hashlib.sha256(canonical).digest()
        digest_verified = (computed == declared_digest)
        if not digest_verified:
            raise Tier2ParserError(
                "tier2_sidecar_digest_mismatch",
                f"computed digest {computed.hex()} != declared "
                f"{declared_digest.hex()}",
            )
        frames.append(
            LaunchFrame(
                launch_uid=launch_uid,
                stream_ordinal=stream_ordinal,
                grid=grid,
                access_record_count=access_record_count,
                budget_bytes_used=budget_bytes_used,
                trailer_file_offset=trailer_offset_decl,
                schema_major=schema_major,
                schema_minor=schema_minor,
                sidecar_sha256=declared_digest,
                digest_verified=True,
                records=records,
                truncation=truncation,
                header_flags=header_flags,
                binary_sha256=binary_sha256,
                sampled_or_compressed_trace=sampled_or_compressed_trace,
                per_lane_records_by_sequence=per_lane_records_by_sequence,
            )
        )
        offset = canonical_end
    return frames


__all__ = [
    "CompactAccessRecord",
    "HEADER_LEN",
    "HEADER_MAGIC",
    "LaunchFrame",
    "PerLaneAddressRecord",
    "RECORD_KIND_COMPACT",
    "RECORD_KIND_PER_LANE",
    "RECORD_KIND_TRUNCATION",
    "SECTOR_COUNT_CAP",
    "TRAILER_LEN",
    "TRAILER_MAGIC",
    "Tier2ParserError",
    "TruncationRecord",
    "parse_launch_frames",
]
