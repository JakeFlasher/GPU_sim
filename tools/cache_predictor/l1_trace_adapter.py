"""Trace -> per-block L1 access-stream adapter.

The per-SM L1 walker (`l1_walker.walk_per_sm_l1`) consumes a list
of per-block access sequences, each sequence holding
``(line_id, l1_eligible)`` tuples. This module provides the
production-side adapter that builds those sequences from parsed
dynamic-trace records + the cuobjdump-backed cache-op classifier
manifest produced by ``l1_cache_op_classifier``.

The adapter is a pure-function helper: it takes already-parsed
record dicts (block_index, pc, byte addresses) rather than a
``dynamic_trace.pb`` file. The proto-loading bridge that
materializes those record dicts from a real trace can be a thin
wrapper added in a follow-up slice (likely paired with the
layer-aware comparison path slice that gives the adapter a real
production caller). Building the adapter as a pure function lets
the L1-walker verification tests exercise it with synthetic
records under host control.

Surfaces:

  - ``build_per_block_l1_accesses(trace_records, manifest,
    sm_count, grid_dim, block_dim)``: the main entry. Joins
    records to the manifest's per-PC mapping, computes
    ``line_id = byte_address // 128`` per access, preserves
    ``l1_eligible`` from the manifest, and groups by
    ``block_index``. Returns ``per_block_accesses`` ready for
    ``walk_per_sm_l1``.

  - ``L1TraceAdapterGridBlockMissing``: typed exception raised
    when ``grid_dim`` is None / 0 or ``block_dim`` is None / 0.
    The L1 head catches this to emit
    ``predictor_l1_cta_to_sm_schedule_unknown`` because CTAs on
    different SMs do not evict each other's L1 lines; without
    grid_dim / block_dim the walker has no defensible CTA-to-SM
    mapping.

  - ``L1TraceAdapterManifestNotMatched``: typed exception
    raised when the supplied manifest's ``sass_provenance_status``
    is not ``matched``. The L1 head catches this to emit the
    matching typed_unavailable arm (the upstream layer should
    have already short-circuited; this is a defensive guard).

NOTE: manifests from multi-arch fatbins are cross-arch
contaminated — see the KNOWN DEFECT (2026-07-06) note in
``l1_cache_op_classifier``'s module docstring.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


from l1_walker import (  # noqa: E402
    LINE_SIZE_BYTES,
    SECTOR_SIZE_BYTES,
)
from l1_cache_op_classifier import (  # noqa: E402
    INSTRUCTION_CACHE_OP_TOKENS,
    derive_l1_eligible,
)


class L1TraceAdapterGridBlockMissing(Exception):
    """Raised when the trace lacks ``grid_dim`` / ``block_dim``
    metadata. CTAs on different SMs do not evict each other's
    L1 lines on the baseline Ampere / Blackwell paths, so
    without the grid / block parameters the walker has no
    defensible CTA-to-SM mapping. The L1 head catches this to
    emit ``predictor_l1_cta_to_sm_schedule_unknown``."""

    TOKEN = "predictor_l1_cta_to_sm_schedule_unknown"


class L1TraceAdapterManifestNotMatched(Exception):
    """Raised when the supplied manifest's
    ``sass_provenance_status`` is not ``matched``. The L1
    head's upstream gate should have already routed this case
    to ``predictor_l1_sass_provenance_missing`` /
    ``predictor_l1_no_modeled_l1_eligible_accesses``; this is a
    defensive guard."""

    TOKEN = "predictor_l1_sass_provenance_missing"


class L1TraceAdapterWarpIndexMissing(Exception):
    """Raised by ``build_per_warp_l1_accesses`` when a trace
    record lacks ``warp_index``. The warp-pool walk schedules
    PER-WARP access streams; silently folding warp-less records
    into one stream would reintroduce the block-sequential
    behavior the warp-pool model exists to correct. Callers that
    only have per-block identity must use
    ``build_per_block_l1_accesses``."""

    TOKEN = "predictor_l1_cta_to_sm_schedule_unknown"


class L1TraceAdapterDataWidthInvalid(Exception):
    """Raised when a trace record's ``data_width`` is missing or
    non-positive while the adapter is in strict mode (required
    for sector-with-line-allocation hit decisions).

    Per the plan's L1 model, every load reaches the L1 walker
    with a known byte width so the adapter can expand the base
    byte address into every touched 32B sector
    (``first_sector = address // 32``,
    ``last_sector = (address + data_width - 1) // 32``).
    Silently assuming a single sector would underreport
    sector-crossing loads in the walker's denominator and bias
    the prediction; the adapter fails-close instead.

    The L1 head catches this to emit
    ``predictor_l1_sass_provenance_missing`` with evidence
    naming the offending pc."""

    TOKEN = "predictor_l1_sass_provenance_missing"

    def __init__(
        self,
        *,
        pc: str,
        block_index: int,
        data_width: object,
        detail: str = "",
    ):
        message = (
            f"{self.TOKEN}: adapter trace record has invalid "
            f"data_width at pc={pc!r}, block_index="
            f"{block_index!r}, data_width={data_width!r}"
        )
        if detail:
            message = f"{message} ({detail})"
        super().__init__(message)
        self.pc = pc
        self.block_index = block_index
        self.data_width = data_width
        self.detail = detail

    def to_evidence(self) -> dict:
        return {
            "pc": self.pc,
            "block_index": self.block_index,
            "data_width": self.data_width,
            "detail": self.detail,
        }


class L1TraceAdapterEligibilityDefect(Exception):
    """Raised when a manifest entry's ``l1_eligible`` field
    fails strict admission: the value is not a real ``bool``,
    the ``instruction_cache_op`` token is outside the closed
    12-token vocabulary, or the serialized boolean disagrees
    with the eligibility derived from the token.

    The production helper catches this to emit
    ``predictor_l1_sass_provenance_missing`` with evidence
    naming the offending pc / token / serialized / expected.
    Schema-valid manifests cannot trigger this exception (the
    schema's per-item ``oneOf`` arm rejects token-eligibility
    mismatches at the wire boundary); the exception is the
    second line of defense for programmatic manifest producers
    that bypass schema validation."""

    TOKEN = "predictor_l1_sass_provenance_missing"

    def __init__(
        self,
        *,
        pc: str,
        instruction_cache_op: object,
        serialized_l1_eligible: object,
        expected_l1_eligible: object,
        detail: str = "",
    ):
        message = (
            f"{self.TOKEN}: adapter manifest eligibility defect "
            f"at pc={pc!r}, instruction_cache_op="
            f"{instruction_cache_op!r}, serialized_l1_eligible="
            f"{serialized_l1_eligible!r}, expected_l1_eligible="
            f"{expected_l1_eligible!r}"
        )
        if detail:
            message = f"{message} ({detail})"
        super().__init__(message)
        self.pc = pc
        self.instruction_cache_op = instruction_cache_op
        self.serialized_l1_eligible = serialized_l1_eligible
        self.expected_l1_eligible = expected_l1_eligible
        self.detail = detail

    def to_evidence(self) -> dict:
        """Serialize the defect for inclusion in a typed
        unavailable bundle's evidence."""
        return {
            "pc": self.pc,
            "instruction_cache_op": self.instruction_cache_op,
            "serialized_l1_eligible": self.serialized_l1_eligible,
            "expected_l1_eligible": self.expected_l1_eligible,
            "detail": self.detail,
        }


class TraceRecord(NamedTuple):
    """One parsed access record from a dynamic trace.

    Fields:
      - ``block_index``: 0-indexed CTA index in launch order.
      - ``pc``: the PC of the issuing instruction; matched
        against the manifest's per-PC mapping.
      - ``byte_addresses``: the per-lane byte addresses for
        this instruction. Each address contributes one or more
        walker entries depending on ``data_width``.
      - ``data_width``: byte width of the load (e.g., 4 for a
        32-bit load, 16 for a 128-bit load). When ``data_width``
        is supplied (a positive integer), the adapter expands
        each active lane address into one walker entry per
        touched 32B sector
        (``first_sector = address // SECTOR_SIZE_BYTES``,
        ``last_sector = (address + data_width - 1) //
        SECTOR_SIZE_BYTES``); the walker entries are 3-tuples
        ``(line_id, sector_id, l1_eligible)``. When
        ``data_width`` is ``None`` (legacy / synthetic
        fixtures), the adapter emits a single 2-tuple
        ``(line_id, l1_eligible)`` per active lane address;
        the walker degrades to line-level LRU for those
        records. Production callers must enable strict mode
        (``require_data_width=True``) so missing data_width
        fails-close at the adapter boundary.
      - ``warp_index``: the warp's index within its threadblock
        (the trace pb tree's per-warp key). ``None`` for legacy
        callers that only need per-block grouping; REQUIRED
        (fail-closed) by ``build_per_warp_l1_accesses``.
      - ``sm_id``: the SM the issuing warp executed on, from the
        trace's optional per-access ``sm_id`` field (proto3
        ``optional``; legitimately absent on older NVBit
        releases). ``None`` means the trace did not expose it for
        this record. Used by ``derive_cta_to_sm_assignment`` to
        build a REAL per-CTA SM partition; ``warpid_sm`` (the warp
        SLOT on the SM) does not by itself identify the SM, so
        ``sm_id`` is the authoritative SM-identity field.
    """

    block_index: int
    pc: str
    byte_addresses: tuple[int, ...]
    data_width: int | None = None
    warp_index: int | None = None
    sm_id: int | None = None


def _normalize_pc(pc: object) -> str:
    """Mirror the manifest's PC normalization: lowercase 0x-hex."""
    if isinstance(pc, int):
        return f"0x{pc:x}"
    if isinstance(pc, str):
        s = pc.lower().strip()
        if s.startswith("0x"):
            hex_part = s[2:].lstrip("0") or "0"
            return f"0x{hex_part}"
        hex_part = s.lstrip("0") or "0"
        return f"0x{hex_part}"
    raise TypeError(
        f"unsupported pc value type {type(pc).__name__}: {pc!r}"
    )


def _materialize_record(raw: "TraceRecord | dict") -> TraceRecord:
    """Coerce a raw trace record (TraceRecord or dict) to a
    TraceRecord. Shared by the per-block and per-warp builders so
    both materialize identically."""
    if isinstance(raw, TraceRecord):
        return raw
    if isinstance(raw, dict):
        # PC normalization: when the caller passes an
        # integer-typed PC (proto `uint32 pc` field),
        # format it as a hex string so the downstream
        # `_normalize_pc` reads it as hex rather than as
        # a decimal-coerced string. The bug this fix
        # closes: `str(16)` would be `"16"` which
        # `_normalize_pc` then interprets as hex
        # `0x16` instead of the correct `0x10`,
        # silently dropping the access on the manifest
        # join. Strings pass through unchanged so the
        # already-hex string path is preserved.
        raw_pc = raw["pc"]
        if isinstance(raw_pc, int) and not isinstance(
            raw_pc, bool,
        ):
            pc_str = f"0x{raw_pc:x}"
        else:
            pc_str = str(raw_pc)
        return TraceRecord(
            block_index=int(raw["block_index"]),
            pc=pc_str,
            byte_addresses=tuple(int(a) for a in raw["byte_addresses"]),
            data_width=(
                int(raw["data_width"])
                if "data_width" in raw and raw["data_width"] is not None
                else None
            ),
            warp_index=(
                int(raw["warp_index"])
                if "warp_index" in raw and raw["warp_index"] is not None
                else None
            ),
            sm_id=(
                int(raw["sm_id"])
                if "sm_id" in raw and raw["sm_id"] is not None
                else None
            ),
        )
    raise TypeError(
        f"trace_records must yield TraceRecord or dict; "
        f"got {type(raw).__name__}"
    )


def build_per_block_l1_accesses(
    trace_records: Iterable[TraceRecord | dict],
    classifier_manifest: dict,
    *,
    sm_count: int,
    grid_dim: int | None,
    block_dim: int | None,
    require_data_width: bool = False,
) -> list[list[tuple]]:
    """Join trace records to the classifier manifest and build
    the per-block access sequences the walker consumes.

    Arguments:
      trace_records: iterable of ``TraceRecord`` or dict (with
        ``block_index`` / ``pc`` / ``byte_addresses``).
      classifier_manifest: a matched cache-op manifest from
        ``l1_cache_op_classifier.classify_sass_text`` /
        ``classify_binary``.
      sm_count: per-arch SM count (sourced from device
        properties / manifest). Used by the caller for the
        walker; the adapter only checks it is positive.
      grid_dim: total number of CTAs in the launch. None / 0
        raises ``L1TraceAdapterGridBlockMissing``.
      block_dim: threads per CTA. None / 0 raises.

    Returns:
      ``per_block_accesses`` — a list of length ``grid_dim``
      (zero-padded for blocks that emit no accesses); each
      entry is a list of ``(line_id, l1_eligible)`` tuples in
      walk order. Trace records whose PC is not in the
      manifest are dropped silently (the walker's
      ``l1_eligible`` filter handles bypass cases; an
      unrecognized PC is not a fail-close — it's just a
      non-load instruction the classifier didn't tag).
    """
    if not isinstance(sm_count, int) or sm_count <= 0:
        raise L1TraceAdapterGridBlockMissing(
            f"sm_count must be a positive int, got "
            f"{sm_count!r}"
        )
    if not grid_dim:
        raise L1TraceAdapterGridBlockMissing(
            "trace lacks grid_dim metadata; CTAs cannot be "
            "mapped to SMs without it"
        )
    if not block_dim:
        raise L1TraceAdapterGridBlockMissing(
            "trace lacks block_dim metadata; the walker uses "
            "block_dim to estimate per-SM occupancy when "
            "scheduling policy is occupancy_derived"
        )
    if (
        classifier_manifest.get("sass_provenance_status")
        != "matched"
    ):
        raise L1TraceAdapterManifestNotMatched(
            f"manifest sass_provenance_status="
            f"{classifier_manifest.get('sass_provenance_status')!r} "
            f"is not 'matched'; the L1 head's upstream gate "
            f"should have routed this case before reaching the "
            f"adapter"
        )

    # Build the PC -> (instruction_cache_op, l1_eligible) lookup
    # with strict admission. Three defects fail-close at this
    # boundary so a programmatic manifest cannot bias the L1
    # prediction by smuggling a non-bool serialized eligibility
    # or a token-eligibility mismatch past the schema:
    #   (a) instruction_cache_op outside the closed 12-token
    #       vocabulary,
    #   (b) l1_eligible is not a real ``bool`` (no truthy
    #       coercion -- the schema is the wire-format source of
    #       truth and the schema requires a JSON boolean),
    #   (c) serialized l1_eligible disagrees with the
    #       eligibility derived from instruction_cache_op
    #       (the same partition the schema enforces via its
    #       per-item ``oneOf`` arm).
    pc_lookup: dict[str, tuple[str, bool]] = {}
    for entry in classifier_manifest.get(
        "per_pc_instruction_cache_op", []
    ):
        pc = _normalize_pc(entry["pc"])
        token = entry.get("instruction_cache_op")
        serialized = entry.get("l1_eligible")
        if token not in INSTRUCTION_CACHE_OP_TOKENS:
            raise L1TraceAdapterEligibilityDefect(
                pc=pc,
                instruction_cache_op=token,
                serialized_l1_eligible=serialized,
                expected_l1_eligible=(
                    "<token_not_in_vocabulary>"
                ),
                detail=(
                    f"instruction_cache_op {token!r} is not in "
                    f"the closed 12-token vocabulary"
                ),
            )
        if not isinstance(serialized, bool):
            raise L1TraceAdapterEligibilityDefect(
                pc=pc,
                instruction_cache_op=token,
                serialized_l1_eligible=serialized,
                expected_l1_eligible=derive_l1_eligible(token),
                detail=(
                    f"l1_eligible must be a strict bool, got "
                    f"{type(serialized).__name__}"
                ),
            )
        expected = derive_l1_eligible(token)
        if serialized is not expected:
            raise L1TraceAdapterEligibilityDefect(
                pc=pc,
                instruction_cache_op=token,
                serialized_l1_eligible=serialized,
                expected_l1_eligible=expected,
                detail=(
                    f"serialized l1_eligible disagrees with the "
                    f"token-derived eligibility"
                ),
            )
        pc_lookup[pc] = (token, serialized)

    per_block_accesses: list[list[tuple]] = [
        [] for _ in range(grid_dim)
    ]

    for raw in trace_records:
        record = _materialize_record(raw)
        if record.block_index < 0 or record.block_index >= grid_dim:
            raise L1TraceAdapterGridBlockMissing(
                f"trace record block_index "
                f"{record.block_index} out of range for "
                f"grid_dim {grid_dim}"
            )
        pc_normalized = _normalize_pc(record.pc)
        lookup = pc_lookup.get(pc_normalized)
        if lookup is None:
            # PC not in manifest -> non-load instruction not
            # tagged by the classifier. Skip silently; the
            # manifest is the source of truth for what the L1
            # walker sees.
            continue
        _instruction_cache_op, l1_eligible = lookup

        data_width = record.data_width
        if data_width is None:
            if require_data_width:
                raise L1TraceAdapterDataWidthInvalid(
                    pc=pc_normalized,
                    block_index=record.block_index,
                    data_width=None,
                    detail=(
                        "production trace records must carry a "
                        "positive data_width to expand into "
                        "per-sector walker entries; the adapter "
                        "refuses to silently assume one sector "
                        "(would underreport the denominator on "
                        "sector-crossing loads)"
                    ),
                )
            # Legacy line-level 2-tuple emission for synthetic
            # fixtures that pre-date the data_width contract.
            for byte_address in record.byte_addresses:
                line_id = int(byte_address) // LINE_SIZE_BYTES
                per_block_accesses[record.block_index].append(
                    (line_id, l1_eligible)
                )
            continue

        if not isinstance(data_width, int) or data_width <= 0:
            raise L1TraceAdapterDataWidthInvalid(
                pc=pc_normalized,
                block_index=record.block_index,
                data_width=data_width,
                detail=(
                    "data_width must be a positive integer "
                    "(byte width of the load)"
                ),
            )

        # Sector-with-line-allocation expansion: each active
        # lane address becomes ceil(data_width/32) +
        # cross-sector adjustment walker entries. Sector IDs are
        # DEDUPLICATED within a single trace record so a
        # coalesced 32-lane load touching only N unique sectors
        # produces N walker accesses, not 32. NCU's sector
        # metric is request-based over UNIQUE 32B sectors per
        # warp instruction; if we emit 32 events for a 32-lane
        # 4B load that touches 4 sectors, later lanes in each
        # sector are observed as hits by the walker (the
        # earlier lane installed the sector) and the modeled
        # L1 hit rate inflates well above NCU truth. Dedup
        # collapses the per-lane events to per-sector events,
        # which is what NCU measures.
        seen_sector_ids: set[int] = set()
        for byte_address in record.byte_addresses:
            base = int(byte_address)
            first_sector = base // SECTOR_SIZE_BYTES
            last_sector = (
                (base + data_width - 1) // SECTOR_SIZE_BYTES
            )
            for sector_id in range(first_sector, last_sector + 1):
                if sector_id in seen_sector_ids:
                    continue
                seen_sector_ids.add(sector_id)
                line_id = (
                    sector_id * SECTOR_SIZE_BYTES // LINE_SIZE_BYTES
                )
                per_block_accesses[record.block_index].append(
                    (line_id, sector_id, l1_eligible)
                )

    return per_block_accesses


def build_per_warp_l1_accesses(
    trace_records: Iterable[TraceRecord | dict],
    classifier_manifest: dict,
    *,
    sm_count: int,
    grid_dim: int | None,
    block_dim: int | None,
    require_data_width: bool = False,
) -> list[tuple[tuple[int, int], list[tuple]]]:
    """Per-WARP grouping of the SAME per-record emission the
    verified per-block builder produces.

    Implementation guarantee: this function does NOT reimplement
    the record->walker-entry semantics. It materializes the
    records, validates per-warp identity (fail-closed), relabels
    each record's grouping key to a dense ordinal, and delegates
    to ``build_per_block_l1_accesses`` — so the manifest join,
    eligibility audit, data-width expansion, and per-record
    sector dedup are the per-block code path BY CONSTRUCTION,
    and concatenating a block's warp groups (warp-sorted)
    reproduces the per-block list exactly.

    Returns an ordered list of ``((block_index, warp_index),
    accesses)`` pairs sorted by (block, warp) — the warp-pool
    walk's block-then-warp admission order. Warps that emit no
    walker entries are omitted (a memory-silent warp contributes
    nothing to the pool).
    """
    records = [_materialize_record(raw) for raw in trace_records]
    for record in records:
        if record.warp_index is None:
            raise L1TraceAdapterWarpIndexMissing(
                f"trace record at block_index="
                f"{record.block_index} pc={record.pc} lacks "
                f"warp_index; the warp-pool walk needs per-warp "
                f"identity (use build_per_block_l1_accesses for "
                f"block-level callers)"
            )
    if not grid_dim:
        raise L1TraceAdapterGridBlockMissing(
            "trace lacks grid_dim metadata; CTAs cannot be "
            "mapped to SMs without it"
        )
    group_keys = sorted({
        (record.block_index, int(record.warp_index))
        for record in records
    })
    ordinal_by_key = {key: i for i, key in enumerate(group_keys)}
    relabeled = [
        record._replace(
            block_index=ordinal_by_key[
                (record.block_index, int(record.warp_index))
            ]
        )
        for record in records
    ]
    # Validate the ORIGINAL block range exactly as the per-block
    # builder would (the relabeled ordinals are dense, so the
    # delegate cannot see an out-of-range original index).
    for record in records:
        if record.block_index < 0 or record.block_index >= grid_dim:
            raise L1TraceAdapterGridBlockMissing(
                f"trace record block_index "
                f"{record.block_index} out of range for "
                f"grid_dim {grid_dim}"
            )
    grouped = build_per_block_l1_accesses(
        relabeled,
        classifier_manifest,
        sm_count=sm_count,
        grid_dim=max(len(group_keys), 1),
        block_dim=block_dim,
        require_data_width=require_data_width,
    )
    return [
        (key, grouped[ordinal])
        for key, ordinal in ordinal_by_key.items()
        if grouped[ordinal]
    ]


class L1CtaToSmConflictingObservation(Exception):
    """Raised when a single CTA (``block_index``) is observed on MORE THAN ONE
    distinct ``sm_id`` in the trace (or on an out-of-range ``sm_id``). A CTA runs
    on exactly one SM for its lifetime in this cache-accounting model, so
    conflicting per-CTA ``sm_id`` observations are an untrustworthy-metadata
    fail-closed condition -- the partition cannot pick an SM arbitrarily."""

    TOKEN = "l1_cta_to_sm_conflicting_observation"


class L1CtaToSmPartialMetadata(Exception):
    """Raised when SOME CTAs carry a real ``sm_id`` but at least one CTA that
    issued trace records carries NONE. A REAL per-CTA partition must be EXACT
    over every block that did memory work; partial ``sm_id`` coverage is
    fail-closed rather than silently degraded to "no metadata" (the trace exposed
    ``sm_id`` for some accesses but not others -- a data-quality defect)."""

    TOKEN = "l1_cta_to_sm_partial_metadata"


class CtaToSmDerivation(NamedTuple):
    """Result of deriving a per-CTA SM partition from trace ``sm_id`` metadata.

    - ``assignment``: per-block ``sm_id`` for ``block_index`` 0..grid_dim-1 when
      ``metadata_present`` is True (blocks that issued no trace records are
      filled with the round-robin placeholder ``block_index % sm_count`` -- they
      contribute no accesses, so the value is immaterial to the L1 walk); ``None``
      when no ``sm_id`` metadata is present anywhere.
    - ``metadata_present``: True iff every block that issued trace records carries
      a consistent real ``sm_id`` (so ``assignment`` is a REAL per-CTA partition).
    - ``observed_block_count``: number of distinct blocks that carried a real
      ``sm_id``.
    """

    assignment: tuple[int, ...] | None
    metadata_present: bool
    observed_block_count: int


def _record_block_and_sm(raw: TraceRecord | dict) -> tuple[int, int | None]:
    """Extract ``(block_index, sm_id_or_None)`` from a TraceRecord or dict."""
    if isinstance(raw, TraceRecord):
        return int(raw.block_index), (None if raw.sm_id is None else int(raw.sm_id))
    if isinstance(raw, dict):
        sm = raw.get("sm_id")
        return int(raw["block_index"]), (None if sm is None else int(sm))
    raise TypeError(
        f"trace record must be a TraceRecord or dict, got {type(raw).__name__}"
    )


def derive_cta_to_sm_assignment(
    trace_records: Iterable[TraceRecord | dict],
    *,
    grid_dim: int,
    sm_count: int,
) -> CtaToSmDerivation:
    """Build an EXACT per-CTA SM assignment from the trace's real ``sm_id`` field.

    A CTA runs on exactly one SM; this groups each ``block_index``'s observed
    ``sm_id`` values and:
      - RAISES ``L1CtaToSmConflictingObservation`` if a block is seen on more
        than one ``sm_id``, or any ``sm_id`` is out of range ``[0, sm_count)``;
      - returns ``(None, metadata_present=False, 0)`` if NO record carries
        ``sm_id`` (the caller then falls back / fails closed);
      - RAISES ``L1CtaToSmPartialMetadata`` if SOME block carries ``sm_id`` but a
        record-bearing block carries none (partial coverage is untrustworthy);
      - otherwise returns the full REAL assignment (``metadata_present=True``),
        with record-less blocks filled by the round-robin placeholder.

    Reads only ``block_index`` + ``sm_id``; ``pc`` / ``byte_addresses`` are
    ignored here (the access decode is ``build_per_block_l1_accesses``)."""
    if not isinstance(sm_count, int) or sm_count <= 0:
        raise ValueError(f"sm_count must be a positive int, got {sm_count!r}")
    if not isinstance(grid_dim, int) or grid_dim <= 0:
        raise L1TraceAdapterGridBlockMissing(
            f"grid_dim must be a positive int, got {grid_dim!r}"
        )
    per_block_sm: dict[int, set[int]] = {}
    record_blocks: set[int] = set()
    for raw in trace_records:
        block_index, sm_id = _record_block_and_sm(raw)
        if not (0 <= block_index < grid_dim):
            raise L1TraceAdapterGridBlockMissing(
                f"trace record block_index {block_index} out of range "
                f"[0, {grid_dim})"
            )
        record_blocks.add(block_index)
        if sm_id is None:
            continue
        if not (0 <= sm_id < sm_count):
            raise L1CtaToSmConflictingObservation(
                f"block_index {block_index} carries sm_id {sm_id} out of range "
                f"[0, {sm_count})"
            )
        observed = per_block_sm.setdefault(block_index, set())
        observed.add(sm_id)
        if len(observed) > 1:
            raise L1CtaToSmConflictingObservation(
                f"block_index {block_index} observed on multiple SMs "
                f"{sorted(observed)}; a CTA runs on exactly one SM"
            )
    observed_blocks = set(per_block_sm)
    if not observed_blocks:
        return CtaToSmDerivation(
            assignment=None, metadata_present=False, observed_block_count=0
        )
    unobserved_record_blocks = record_blocks - observed_blocks
    if unobserved_record_blocks:
        raise L1CtaToSmPartialMetadata(
            f"sm_id present for blocks {sorted(observed_blocks)} but absent for "
            f"record-bearing blocks {sorted(unobserved_record_blocks)}; a real "
            f"per-CTA partition must be exact over every block that issued accesses"
        )
    assignment = tuple(
        next(iter(per_block_sm[b])) if b in per_block_sm else (b % sm_count)
        for b in range(grid_dim)
    )
    return CtaToSmDerivation(
        assignment=assignment,
        metadata_present=True,
        observed_block_count=len(observed_blocks),
    )


__all__ = [
    "CtaToSmDerivation",
    "L1CtaToSmConflictingObservation",
    "L1CtaToSmPartialMetadata",
    "L1TraceAdapterDataWidthInvalid",
    "L1TraceAdapterEligibilityDefect",
    "L1TraceAdapterGridBlockMissing",
    "L1TraceAdapterManifestNotMatched",
    "L1TraceAdapterWarpIndexMissing",
    "TraceRecord",
    "build_per_block_l1_accesses",
    "build_per_warp_l1_accesses",
    "derive_cta_to_sm_assignment",
]
