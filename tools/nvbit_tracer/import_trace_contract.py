"""Offline trace-contract validator for the NVBit tracer.

Walks a serialized ``dynamic_trace.threadblock`` protobuf (or a tree
of them under a real trace's ``threadblocks/`` directory) and
verifies the AC4 producer-side per-instruction contract that
R8 added to the schema and R9 wired up in
``recv_thread_fun``:

- Every memory instruction (``addresses`` non-empty) carries a
  ``memory_op_class`` whose value is present and != UNSPECIFIED.
- Every memory instruction carries a ``data_width`` that is present
  and > 0. Width is meaningful only for memory ops (it describes the
  per-lane access width); for non-memory instructions the producer
  intentionally leaves it absent on the wire because the device-side
  channel reports a meaningless 0 for those records. The validator
  recognizes "looks like memory" via "addresses non-empty OR
  memory_op_class present" — either signal is sufficient to require
  data_width.
- Every instruction carries a ``warpid_tb`` (warp index inside the
  threadblock). Marked ``optional`` in the proto so absence is
  observationally distinct from value 0; the validator rejects
  absence.

Other fields (``warpid_sm``, ``sm_id``) are also ``optional`` per
the schema but legitimately absent on traces from older NVBit
releases that do not expose those values; the validator does not
require them.

Storage layout note. The tracer writes the top-level
``Trace.pb`` alongside a separate ``threadblocks/device_<id>/
stream_<id>/kernel_<id>/<tb_x>_<tb_y>_<tb_z>.pb`` tree (one
``threadblock`` message per file). The instruction records the
contract guards live inside those per-threadblock files; the
top-level ``Trace`` only carries the device/stream/kernel chain
and has no instruction records to walk. This module therefore
takes a ``threadblock`` proto (or a tree of them) as its input,
not a top-level ``Trace``. Callers that want a whole-trace sweep
use ``validate_trace_dir`` which walks the conventional layout.

Public entry points:

- ``validate_threadblock(tb)`` walks one ``threadblock`` proto and
  returns a ``ValidationResult``.
- ``validate_threadblock_path(path)`` parses ``path`` as a
  ``threadblock`` and forwards to ``validate_threadblock``.
- ``validate_trace_dir(traces_dir)`` walks every
  ``*.pb`` under ``<traces_dir>/threadblocks/`` and returns the
  first failing record (or ok if all pass).
- ``main(argv)`` is the CLI entry point.

Typed reasons returned in ``ValidationResult.typed_reason`` on
failure:

- ``trace_unreadable``: file missing or bytes do not parse as a
  ``threadblock``.
Protobuf dependencies fail before validation: a missing/incompatible runtime
raises ``TraceContractRuntimePrerequisiteMissing`` and a missing, damaged, or
misresolved tracked binding raises ``TraceContractTrackedBindingsCorrupt``.
Neither condition is encoded as trace evidence or repaired by generation.
- ``memory_op_class_missing_on_memory_instruction``: the offending
  instruction's ``addresses`` list is non-empty but
  ``memory_op_class`` is absent on the wire.
- ``memory_op_class_unspecified``: ``memory_op_class`` is present
  but its value is ``MEMORY_OP_CLASS_UNSPECIFIED``. Per the proto
  enum comment, this is a tracer contract violation: a memory
  instruction whose op-class the tracer could not classify should
  have been emitted as a typed ``unsupported_memory_path``
  exclusion sidecar instead.
- ``memory_op_class_address_space_conflict``: an address record carries
  an explicit ``address_space`` incompatible with the instruction's
  operation-aware class. GLOBAL accepts GLOBAL/GENERIC, SHARED accepts
  SHARED, LOCAL accepts LOCAL, ATOMIC accepts any concrete effective
  space, and ASYNC_COPY accepts only its GLOBAL/GENERIC source. An
  absent ``address_space`` retains the legacy class fallback.
- ``unknown_enum_value``: ``memory_op_class`` or an explicit
  ``address_space`` carries a number outside the committed protobuf enum;
  this is schema skew, not an unsupported known operation.
- ``data_width_missing``: ``data_width`` is absent on the wire on
  an instruction that looks like memory (addresses non-empty or
  memory_op_class present).
- ``data_width_invalid``: ``data_width`` is present but <= 0.
- ``warpid_tb_missing``: ``warpid_tb`` is absent on the wire.
- ``function_unique_id_invalid``: ``function_unique_id <= 0`` on
  any instruction. The tracer's
  ``instrument_function_if_needed`` increments
  ``next_candidate_unique_function_id`` (initial value 0) before
  assigning, so the first instrumented function gets id 1; the
  proto3 default 0 is therefore the "tracer-did-not-emit"
  sentinel.
- ``active_mask_missing``: ``active_mask`` is absent on the wire
  (``HasField('active_mask')`` returns false). Marked
  ``optional`` in the proto for explicit-presence so a regression
  that dropped the producer's ``inst->set_active_mask(...)`` call
  surfaces as a typed contract failure rather than silently
  admitting a default-zero record (which is a valid lane-mask
  value for the predicated-off all-lanes-inactive case).
- ``predicate_mask_missing``: ``predicate_mask`` is absent on the
  wire. Same explicit-presence reasoning as ``active_mask``.
- ``vpc_missing``: ``vpc`` is absent on the wire. Marked
  ``optional`` in the proto so the validator can distinguish
  "tracer did not emit vpc" from "vpc=0" (which legitimately
  equals ``pc=0`` for the first instruction in a kernel).
- ``vpc_inconsistent_with_pc``: ``(vpc & 0xFFFFFFFF) != pc``.
  The producer sets both ``pc`` and ``vpc`` from the same
  per-instruction PC offset; a wire record where the low 32
  bits diverge means the tracer either forgot to populate
  ``vpc`` (proto3 zero default) or corrupted the record. The
  bitmask preserves forward compatibility with the queued
  channel-side widening of ``vpc`` to a true 64-bit virtual PC.
- ``lane_mask_inconsistent``: a memory instruction whose
  ``(active_mask & predicate_mask) == 0`` AND any inner
  ``address.addrs`` list contains at least one non-sentinel
  entry (a "real" address). The producer creates the outer
  ``address`` submessage unconditionally for every STANDARD_MEM
  operand. Under the AC4 32-entry sentinel format, a
  ``list_all`` record with AND mask zero carries 32
  ``LANE_INACTIVE_SENTINEL`` (-1) entries — that admits. Under
  the compressed formats (``base_stride`` / ``base_delta``)
  with AND mask zero, the addrs list is empty — also admits.
  A ``list_all`` record with at least one non-sentinel
  ``addrs[i]`` (real address) and AND mask zero, OR a
  compressed-format record with non-empty addrs and AND mask
  zero, would be an internal inconsistency.
- ``lane_vector_length_invalid``: a memory instruction's inner
  ``address`` record uses ``compression_format ==
  address_format::list_all`` (proto value 0) AND its ``addrs``
  list does not have exactly 32 entries. AC4 requires the
  ``list_all`` form to carry one slot per lane (32 lanes per
  warp); inactive lanes use ``LANE_INACTIVE_SENTINEL`` (-1)
  rather than being omitted. Compressed formats
  (``base_stride`` / ``base_delta``) use a different on-wire
  contract and are not subject to this clause.
- ``compressed_format_malformed``: a compressed-format address
  record (``base_stride`` or ``base_delta``) has structural
  damage that prevents reconstruction into 32 lane slots. For
  ``base_stride``: the active lanes implied by
  ``(active_mask & predicate_mask)`` are not contiguous in
  lane index, OR popcount==1 (the producer's
  ``base_stride_compress`` requires at least two adjacent
  active lanes to compute stride; a single-active-lane case
  falls through to ``base_delta``), OR the addrs vector is
  non-empty (the ``base_stride`` format encodes lanes via
  base+stride and must leave addrs empty). For
  ``base_delta``: the addrs delta count does not equal
  ``popcount(and_mask) - 1`` (or zero when popcount is zero
  or one). For unknown compression_format values: the
  validator cannot reconstruct the lane vector.
- ``lane_position_mismatch``: after reconstruction, an
  address record's per-lane vector violates positional
  identity with ``(active_mask & predicate_mask)``. For each
  lane index ``i`` (0..31): bit ``i`` of the AND mask is set
  iff the reconstructed slot ``i`` carries a real
  (non-sentinel) address. For ``list_all`` records this
  catches a malformed wire shape where the producer (or a
  bug-introduced rewrite) put the real address in the wrong
  slot. For compressed formats this fires only on a
  reconstruction-internal bug (the reconstruction places real
  addresses at active-lane positions by construction).
- ``allocation_id_missing``: a supported GLOBAL memory address
  record has at least one active+predicated lane
  (``(active_mask & predicate_mask) != 0``) but
  ``addr.allocation_id`` is absent or 0. The producer's per-lane
  resolution gate guarantees that every active GLOBAL record
  either carries a non-zero ``allocation_id`` (all lanes
  resolved to the same id) OR routes through one of the typed
  exclusion sidecars (``unmapped_memory_range`` for any-zero
  lanes, ``cross_allocation_lane_span`` for multi-non-zero
  lanes); a wire record that has active lanes AND no
  ``allocation_id`` is a producer-side regression that would
  silently lose allocation identity downstream. Predicated-off
  GLOBAL records (``and_mask == 0``) admit without an
  ``allocation_id`` because no lane address requires identity.
- ``allocations_sidecar_missing``: the trace carries at least
  one memory record with ``address.allocation_id != 0`` but
  ``<traces_dir>/extra_info/allocations.json`` does not exist
  (or was not provided to the validator). Required only when
  memory records carry an allocation_id; a trace whose
  records all have ``allocation_id`` absent / 0 admits
  without the sidecar only when ``and_mask == 0`` on every
  GLOBAL record (otherwise ``allocation_id_missing`` fires
  first).
- ``allocations_sidecar_malformed``: the allocations sidecar
  exists but at least one entry is missing a required field,
  has the wrong type, or has a ``base_address`` string that
  does not parse as ``0x``-prefixed hex.
- ``allocation_id_duplicate``: the allocations sidecar
  contains two or more entries with the same ``allocation_id``.
  Per the producer-side registry (monotonic counter starting
  at 1), allocation_ids are globally unique across contexts;
  a duplicate means the sidecar was corrupted or merged
  incorrectly.
- ``allocation_id_unknown``: a memory record carries
  ``address.allocation_id`` whose value is not present in the
  allocations sidecar table. The producer only stamps an id
  it just looked up against its own registry, so a wire id
  the consumer cannot resolve indicates the trace and the
  sidecar are out of sync (e.g., partial copy) or the
  registry was reset between sidecar emission and the trace.
- ``lane_address_outside_allocation_range``: a GLOBAL memory
  record carries ``address.allocation_id`` resolving to a
  known table entry, but at least one non-sentinel
  active+predicated lane address is NOT within
  ``[base_address, base_address + size_bytes)``. The
  producer-side per-lane resolution gate
  (``cross_allocation_lane_span`` typed exclusion) guarantees
  every active lane resolves to the SAME id when an
  allocation_id is stamped; a wire record where lanes fall
  outside the resolved range is a tracer regression in the
  per-lane resolution loop.

The first failure encountered short-circuits the walk; the
``detail`` string carries the offending record path
(``warp/instruction-index``, plus the threadblock file path when
walking a directory) so the operator can locate the bad record
with a focused inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PB_GENERATED_DIR = (
    _REPO_ROOT / "workloads" / "simbench_bw" / "_pb_generated"
)


# AC4 32-entry lane-address sentinel format: the producer's
# `list_all` compression format writes this value at every
# inactive-lane position so each `address.addrs` vector on the
# wire carries exactly 32 entries. Real GPU virtual addresses
# are non-negative `uint64` values; -1 cast through the
# `repeated int64 addrs` proto field reinterprets as
# 0xFFFFFFFFFFFFFFFF, which never collides with a real address.
# Mirrors `LANE_INACTIVE_SENTINEL` in
# `tools/nvbit_tracer/tracer_tool/tracer_tool.cu`.
LANE_INACTIVE_SENTINEL = -1

# `address_format` enum values. Mirror the C++ enum in
# `tools/nvbit_tracer/tracer_tool/tracer_tool.cu` (defined as
# `enum class address_format { list_all = 0, base_stride = 1,
# base_delta = 2 };`). Each format encodes the per-lane address
# vector differently:
#   - list_all: 32 explicit slots; sentinel at inactive lanes.
#   - base_stride: contiguous active lanes; address[i] =
#       base_address + (i - i_first) * stride for i in
#       [i_first, i_first + popcount(and_mask)). addrs empty.
#   - base_delta: walking active lanes in lane-index order;
#       first active lane = base_address, subsequent active
#       lane k = base_address + sum(addrs[0..k-1]). addrs has
#       popcount(and_mask) - 1 deltas (or 0 when popcount<=1).
_ADDRESS_FORMAT_LIST_ALL = 0
_ADDRESS_FORMAT_BASE_STRIDE = 1
_ADDRESS_FORMAT_BASE_DELTA = 2

# Number of lanes per warp on every supported GPU architecture.
# The 32-entry sentinel format pins this exactly; any deviation
# would be a contract violation worth catching at the validator.
_LANES_PER_WARP = 32


@dataclass
class AllocationEntry:
    """One row of the allocations sidecar.

    Mirrors the C++ ``AllocationRecord`` struct in
    ``tools/nvbit_tracer/tracer_tool/allocation_registry.h``
    after parsing ``base_address`` from its hex string form.
    Per-record range identity uses ``base_address`` and
    ``size_bytes`` to verify each non-sentinel lane address is in
    ``[base_address, base_address + size_bytes)``.
    """

    allocation_id: int
    context_id: int
    device_id: int
    base_address: int
    size_bytes: int
    alive: bool


def _active_lane_indices(and_mask: int) -> list[int]:
    """Return the active lane indices implied by ``and_mask``
    (``active_mask & predicate_mask``), in lane-index order."""
    return [i for i in range(_LANES_PER_WARP) if (and_mask >> i) & 1]


# LOCKSTEP: sim/trace/src/lanes.rs (Rust replay decoder) is an exact port of
# this function + the per-slot positional check below. Shared golden fixture:
# tools/nvbit_tracer/lane_reconstruction_golden.json (extend it in the same
# commit as any semantic change here).
def _reconstruct_lane_vector(
    addr: Any,
    and_mask: int,
) -> tuple[list[int] | None, str | None]:
    """Reconstruct the 32-entry per-lane address vector from any
    compression format. Returns ``(vector, None)`` on success or
    ``(None, reason)`` on a structural failure.

    On success, the vector has length 32. Active lanes (bits set in
    ``and_mask``) carry the real address the producer encoded;
    inactive lanes carry ``LANE_INACTIVE_SENTINEL``.

    On failure, ``reason`` is a human-readable diagnostic that the
    validator promotes into a ``compressed_format_malformed`` typed
    rejection. Failures cover: ``list_all`` records whose addrs
    length is not 32 (already caught upstream by
    ``lane_vector_length_invalid`` but defensively re-checked
    here); ``base_stride`` records whose active lanes are
    non-contiguous or whose addrs is non-empty; ``base_delta``
    records whose delta count does not equal popcount-1; unknown
    compression formats.
    """
    cf = addr.compression_format
    active_lanes = _active_lane_indices(and_mask)
    n_active = len(active_lanes)

    if cf == _ADDRESS_FORMAT_LIST_ALL:
        if len(addr.addrs) != _LANES_PER_WARP:
            return None, (
                f"list_all record has {len(addr.addrs)} entr"
                f"{'y' if len(addr.addrs) == 1 else 'ies'}; "
                f"expected exactly {_LANES_PER_WARP}"
            )
        return list(addr.addrs), None

    if cf == _ADDRESS_FORMAT_BASE_STRIDE:
        if len(addr.addrs) != 0:
            return None, (
                f"base_stride record has non-empty addrs "
                f"(len={len(addr.addrs)}); the base_stride format "
                f"encodes lanes via base_address+stride, addrs "
                f"must be empty"
            )
        vector: list[int] = [LANE_INACTIVE_SENTINEL] * _LANES_PER_WARP
        if n_active == 0:
            return vector, None
        if n_active == 1:
            # Producer's base_stride_compress requires the first
            # active lane to be immediately followed by another
            # active lane before it computes stride; a single-
            # active-lane case sets const_stride=false and falls
            # through to base_delta. So a base_stride record with
            # popcount==1 is structurally impossible from the
            # conforming producer.
            return None, (
                f"base_stride record with popcount(and_mask)=1 "
                f"is structurally impossible: the producer's "
                f"base_stride_compress requires at least two "
                f"adjacent active lanes to compute stride; a "
                f"single-active-lane case falls through to "
                f"base_delta"
            )
        i_first = active_lanes[0]
        if active_lanes != list(range(i_first, i_first + n_active)):
            return None, (
                f"base_stride record but active lanes "
                f"{active_lanes} are not contiguous starting at "
                f"{i_first}; the producer's base_stride_compress "
                f"only succeeds when active lanes form a "
                f"contiguous run"
            )
        for k, j in enumerate(active_lanes):
            vector[j] = addr.base_address + k * addr.stride
        return vector, None

    if cf == _ADDRESS_FORMAT_BASE_DELTA:
        expected_deltas = max(0, n_active - 1)
        if len(addr.addrs) != expected_deltas:
            return None, (
                f"base_delta record has {len(addr.addrs)} delta "
                f"entr{'y' if len(addr.addrs) == 1 else 'ies'}; "
                f"expected {expected_deltas} (popcount(and_mask)="
                f"{n_active} minus 1, or 0 when popcount<=1)"
            )
        vector = [LANE_INACTIVE_SENTINEL] * _LANES_PER_WARP
        if n_active == 0:
            return vector, None
        current = addr.base_address
        vector[active_lanes[0]] = current
        for k in range(1, n_active):
            current = current + addr.addrs[k - 1]
            vector[active_lanes[k]] = current
        return vector, None

    return None, (
        f"unknown compression_format={cf}; the validator only "
        f"supports list_all (0), base_stride (1), and base_delta "
        f"(2)"
    )


def _reconstruct_instruction_addresses(
    inst: Any,
    and_mask: int,
    record_path: str,
    address_pb2_mod: Any,
) -> tuple[list[list[int]] | None, ValidationResult | None]:
    """Structurally validate every address before semantic diagnostics."""
    known_spaces = (
        frozenset(address_pb2_mod.MemorySpace.values())
        if address_pb2_mod is not None
        else frozenset()
    )
    vectors: list[list[int]] = []
    for addr_index, addr in enumerate(inst.addresses):
        if (
            addr.compression_format == _ADDRESS_FORMAT_LIST_ALL
            and len(addr.addrs) != _LANES_PER_WARP
        ):
            return None, ValidationResult(
                ok=False,
                typed_reason="lane_vector_length_invalid",
                detail=(
                    f"{record_path}: address[{addr_index}] uses "
                    f"compression_format=list_all but its addrs "
                    f"list has {len(addr.addrs)} entr"
                    f"{'y' if len(addr.addrs) == 1 else 'ies'}; "
                    f"AC4 requires exactly {_LANES_PER_WARP} "
                    f"entries (one slot per lane, with "
                    f"LANE_INACTIVE_SENTINEL={LANE_INACTIVE_SENTINEL} "
                    f"at inactive-lane positions)"
                ),
            )

        if addr.compression_format == _ADDRESS_FORMAT_LIST_ALL:
            real_count = sum(
                1
                for entry in addr.addrs
                if entry != LANE_INACTIVE_SENTINEL
            )
        else:
            real_count = len(addr.addrs)
        if and_mask == 0 and real_count > 0:
            return None, ValidationResult(
                ok=False,
                typed_reason="lane_mask_inconsistent",
                detail=(
                    f"{record_path}: "
                    f"(active_mask=0x{inst.active_mask:x}) & "
                    f"(predicate_mask=0x{inst.predicate_mask:x}) "
                    f"is zero but an inner address record carries "
                    f"{real_count} non-sentinel addrs entr"
                    f"{'y' if real_count == 1 else 'ies'}; "
                    f"the producer only emits per-lane real addresses "
                    f"for lanes the AND mask gates true"
                ),
            )
        if (
            and_mask != 0
            and addr.compression_format == _ADDRESS_FORMAT_LIST_ALL
            and real_count == 0
        ):
            return None, ValidationResult(
                ok=False,
                typed_reason="lane_mask_inconsistent",
                detail=(
                    f"{record_path}: "
                    f"(active_mask=0x{inst.active_mask:x}) & "
                    f"(predicate_mask=0x{inst.predicate_mask:x}) "
                    f"is 0x{and_mask:x} (non-zero) but a list_all "
                    f"address record carries 32 sentinel entries with "
                    f"zero real addresses; the producer emits a real "
                    f"address at every lane the AND mask gates true"
                ),
            )

        vector, malformed = _reconstruct_lane_vector(addr, and_mask)
        if malformed is not None:
            return None, ValidationResult(
                ok=False,
                typed_reason="compressed_format_malformed",
                detail=(
                    f"{record_path}: address[{addr_index}] "
                    f"(compression_format="
                    f"{addr.compression_format}): {malformed}"
                ),
            )
        assert vector is not None  # invariant: malformed is None
        for lane_index in range(_LANES_PER_WARP):
            bit_set = bool((and_mask >> lane_index) & 1)
            slot = vector[lane_index]
            slot_is_sentinel = slot == LANE_INACTIVE_SENTINEL
            if bit_set and slot_is_sentinel:
                return None, ValidationResult(
                    ok=False,
                    typed_reason="lane_position_mismatch",
                    detail=(
                        f"{record_path}: address[{addr_index}] "
                        f"lane {lane_index} reconstructs to "
                        f"LANE_INACTIVE_SENTINEL but bit "
                        f"{lane_index} of "
                        f"(active_mask & predicate_mask)="
                        f"0x{and_mask:x} is set; expected a "
                        f"real address at this lane"
                    ),
                )
            if (not bit_set) and (not slot_is_sentinel):
                return None, ValidationResult(
                    ok=False,
                    typed_reason="lane_position_mismatch",
                    detail=(
                        f"{record_path}: address[{addr_index}] "
                        f"lane {lane_index} reconstructs to "
                        f"non-sentinel value 0x{slot:x} but bit "
                        f"{lane_index} of "
                        f"(active_mask & predicate_mask)="
                        f"0x{and_mask:x} is clear; expected "
                        f"LANE_INACTIVE_SENTINEL at this lane"
                    ),
                )
        if (
            address_pb2_mod is not None
            and addr.HasField("address_space")
            and addr.address_space not in known_spaces
        ):
            return None, ValidationResult(
                ok=False,
                typed_reason="unknown_enum_value",
                detail=(
                    f"{record_path}: address[{addr_index}] "
                    f"address_space carries value {addr.address_space} "
                    f"unknown to the committed protobuf schema"
                ),
            )
        vectors.append(vector)
    return vectors, None


class TraceContractBindingsMissing(RuntimeError):
    """Compatibility base for hard protobuf dependency failures."""


class TraceContractTrackedBindingsCorrupt(TraceContractBindingsMissing):
    """A tracked `_pb2.py` authority is absent, damaged, or misresolved."""


class TraceContractRuntimePrerequisiteMissing(TraceContractBindingsMissing):
    """The project-pinned Python protobuf runtime cannot load the bindings."""


_TRACKED_BINDING_NAMES: tuple[str, ...] = (
    "address_pb2.py",
    "cuda_stream_pb2.py",
    "dim3d_pb2.py",
    "gpu_device_pb2.py",
    "instruction_pb2.py",
    "kernel_pb2.py",
    "threadblock_pb2.py",
    "trace_pb2.py",
    "warp_pb2.py",
)


def _binding_restore_guidance(paths: list[Path]) -> str:
    relative = " ".join(
        str(path.relative_to(_REPO_ROOT)) for path in sorted(paths)
    )
    return (
        "tracked protobuf authority is missing or unusable; restore its "
        "byte-exact committed content with `git restore --source=HEAD "
        f"--worktree -- {relative}`; do not regenerate bindings in the "
        "canonical checkout"
    )


def _import_pb_modules() -> tuple[Any, Any, Any]:
    """Import ``threadblock_pb2``, ``instruction_pb2`` and ``address_pb2``
    from the generated bindings under
    ``workloads/simbench_bw/_pb_generated/``.

    Raises a distinct hard exception for a runtime prerequisite failure or
    tracked repository corruption. Callers must not translate either into a
    graceful or typed-unavailable trace result.
    """
    try:
        import google.protobuf  # noqa: F401  # pyright: ignore[reportMissingModuleSource, reportUnusedImport]
    except ImportError as exc:
        raise TraceContractRuntimePrerequisiteMissing(
            "protobuf runtime prerequisite is missing; install the exact "
            "version pinned in requirements.txt before consuming the "
            "tracked dynamic-trace bindings"
        ) from exc
    required = [_PB_GENERATED_DIR / name for name in _TRACKED_BINDING_NAMES]
    missing = [
        path for path in required if not path.is_file() or path.is_symlink()
    ]
    if missing:
        raise TraceContractTrackedBindingsCorrupt(
            _binding_restore_guidance(missing)
        )
    if str(_PB_GENERATED_DIR) not in sys.path:
        sys.path.insert(0, str(_PB_GENERATED_DIR))
    try:
        import threadblock_pb2  # pyright: ignore[reportMissingImports]
        import instruction_pb2  # pyright: ignore[reportMissingImports]
        import address_pb2  # pyright: ignore[reportMissingImports]
    except Exception as exc:
        if exc.__class__.__module__.startswith("google.protobuf"):
            raise TraceContractRuntimePrerequisiteMissing(
                "protobuf runtime prerequisite is incompatible with the "
                f"tracked bindings: {type(exc).__name__}: {exc}"
            ) from exc
        raise TraceContractTrackedBindingsCorrupt(
            f"{_binding_restore_guidance(required)}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    for module, filename in (
        (threadblock_pb2, "threadblock_pb2.py"),
        (instruction_pb2, "instruction_pb2.py"),
        (address_pb2, "address_pb2.py"),
    ):
        expected = (_PB_GENERATED_DIR / filename).resolve()
        module_file = getattr(module, "__file__", None)
        if module_file is None or Path(module_file).resolve() != expected:
            raise TraceContractTrackedBindingsCorrupt(
                f"tracked protobuf authority resolved outside its owner: "
                f"{module.__name__} -> {getattr(module, '__file__', None)}; "
                f"{_binding_restore_guidance([_PB_GENERATED_DIR / filename])}"
            )
    for filename in _TRACKED_BINDING_NAMES:
        module = sys.modules.get(filename.removesuffix(".py"))
        if module is None:
            continue
        expected = (_PB_GENERATED_DIR / filename).resolve()
        module_file = getattr(module, "__file__", None)
        if module_file is None or Path(module_file).resolve() != expected:
            raise TraceContractTrackedBindingsCorrupt(
                f"tracked protobuf authority {module.__name__} resolved "
                f"outside its owner: {getattr(module, '__file__', None)}; "
                f"{_binding_restore_guidance([_PB_GENERATED_DIR / filename])}"
            )
    return threadblock_pb2, instruction_pb2, address_pb2


@dataclass
class ValidationResult:
    """Outcome of validating a ``threadblock`` against the AC4
    contract.

    On success ``ok`` is True and ``typed_reason`` is empty.

    On failure, ``typed_reason`` is one of the strings documented at
    the top of this module and ``detail`` carries the offending
    record path plus a short human-readable description.
    """

    ok: bool
    typed_reason: str = ""
    detail: str = ""


def load_allocations_sidecar(
    traces_dir: Path,
) -> tuple[dict[int, AllocationEntry] | None, ValidationResult | None]:
    """Load and validate ``<traces_dir>/extra_info/allocations.json``.

    Returns:
      - ``(table, None)`` on success, where ``table`` maps
        ``allocation_id`` to :class:`AllocationEntry`.
      - ``(None, None)`` when the sidecar file is missing. The
        caller decides whether that's acceptable (it is, as long as
        no memory record carries ``allocation_id != 0``).
      - ``(None, ValidationResult(...))`` on a hard failure
        (malformed entry / duplicate id).

    Per-entry validation:
      - All six fields must be present.
      - ``allocation_id`` must be a positive ``int``.
      - ``context_id`` and ``device_id`` must be ``int``.
        ``device_id`` may legitimately be 0 (CUDA device 0) or
        negative (uninstrumented device id).
      - ``base_address`` must be a string parseable as ``0x``-prefixed
        hex.
      - ``size_bytes`` must be a positive ``int``.
      - ``alive`` must be a ``bool``.
      - Two entries must not share an ``allocation_id``.
    """
    sidecar_path = traces_dir / "extra_info" / "allocations.json"
    if not sidecar_path.is_file():
        return None, None
    try:
        with sidecar_path.open(encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return None, ValidationResult(
            ok=False,
            typed_reason="allocations_sidecar_malformed",
            detail=(
                f"could not read allocations sidecar "
                f"{sidecar_path}: {exc}"
            ),
        )
    if not isinstance(raw, list):
        return None, ValidationResult(
            ok=False,
            typed_reason="allocations_sidecar_malformed",
            detail=(
                f"{sidecar_path}: top-level value is not a JSON list"
            ),
        )
    table: dict[int, AllocationEntry] = {}
    required = (
        "allocation_id",
        "context_id",
        "device_id",
        "base_address",
        "size_bytes",
        "alive",
    )
    required_set = frozenset(required)
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            return None, ValidationResult(
                ok=False,
                typed_reason="allocations_sidecar_malformed",
                detail=(
                    f"{sidecar_path}: entry[{index}] is not a JSON "
                    f"object"
                ),
            )
        for field in required:
            if field not in entry:
                return None, ValidationResult(
                    ok=False,
                    typed_reason="allocations_sidecar_malformed",
                    detail=(
                        f"{sidecar_path}: entry[{index}] is missing "
                        f"required field {field!r}"
                    ),
                )
        # Exact-shape gate: the producer (`AllocationRecord` ->
        # `serialize_history_json`) writes exactly six fields. An
        # extra field signals either a hostile sidecar, a producer
        # drift the consumer was not updated for, or a manual edit
        # that introduced a typo (e.g., `size_byte` not caught by
        # the required-field loop because the canonical field is
        # also present). Reject so the wire shape stays exact.
        extra_keys = set(entry.keys()) - required_set
        if extra_keys:
            return None, ValidationResult(
                ok=False,
                typed_reason="allocations_sidecar_malformed",
                detail=(
                    f"{sidecar_path}: entry[{index}] carries "
                    f"unexpected field(s) "
                    f"{sorted(extra_keys)!r}; the allocation "
                    f"sidecar shape must match the producer's "
                    f"`AllocationRecord` exactly "
                    f"({sorted(required)!r})"
                ),
            )
        # Reject bool subtype where int is required (bool is an int
        # subclass in Python; an `alive: 1` slip would otherwise
        # admit silently).
        alloc_id = entry["allocation_id"]
        ctx_id = entry["context_id"]
        dev_id = entry["device_id"]
        size = entry["size_bytes"]
        alive = entry["alive"]
        base_str = entry["base_address"]
        if (
            isinstance(alloc_id, bool)
            or not isinstance(alloc_id, int)
            or alloc_id <= 0
        ):
            return None, ValidationResult(
                ok=False,
                typed_reason="allocations_sidecar_malformed",
                detail=(
                    f"{sidecar_path}: entry[{index}] allocation_id="
                    f"{alloc_id!r} is not a positive integer"
                ),
            )
        if isinstance(ctx_id, bool) or not isinstance(ctx_id, int):
            return None, ValidationResult(
                ok=False,
                typed_reason="allocations_sidecar_malformed",
                detail=(
                    f"{sidecar_path}: entry[{index}] context_id="
                    f"{ctx_id!r} is not an integer"
                ),
            )
        if isinstance(dev_id, bool) or not isinstance(dev_id, int):
            return None, ValidationResult(
                ok=False,
                typed_reason="allocations_sidecar_malformed",
                detail=(
                    f"{sidecar_path}: entry[{index}] device_id="
                    f"{dev_id!r} is not an integer"
                ),
            )
        if not isinstance(base_str, str):
            return None, ValidationResult(
                ok=False,
                typed_reason="allocations_sidecar_malformed",
                detail=(
                    f"{sidecar_path}: entry[{index}] base_address="
                    f"{base_str!r} is not a string"
                ),
            )
        if not (
            base_str.startswith("0x") or base_str.startswith("0X")
        ):
            return None, ValidationResult(
                ok=False,
                typed_reason="allocations_sidecar_malformed",
                detail=(
                    f"{sidecar_path}: entry[{index}] base_address="
                    f"{base_str!r} is not 0x-prefixed hex"
                ),
            )
        try:
            base_address = int(base_str, 16)
        except ValueError:
            return None, ValidationResult(
                ok=False,
                typed_reason="allocations_sidecar_malformed",
                detail=(
                    f"{sidecar_path}: entry[{index}] base_address="
                    f"{base_str!r} does not parse as hex"
                ),
            )
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            return None, ValidationResult(
                ok=False,
                typed_reason="allocations_sidecar_malformed",
                detail=(
                    f"{sidecar_path}: entry[{index}] size_bytes="
                    f"{size!r} is not a positive integer"
                ),
            )
        if not isinstance(alive, bool):
            return None, ValidationResult(
                ok=False,
                typed_reason="allocations_sidecar_malformed",
                detail=(
                    f"{sidecar_path}: entry[{index}] alive="
                    f"{alive!r} is not a boolean"
                ),
            )
        if alloc_id in table:
            return None, ValidationResult(
                ok=False,
                typed_reason="allocation_id_duplicate",
                detail=(
                    f"{sidecar_path}: entry[{index}] allocation_id="
                    f"{alloc_id} duplicates a prior entry"
                ),
            )
        table[alloc_id] = AllocationEntry(
            allocation_id=alloc_id,
            context_id=ctx_id,
            device_id=dev_id,
            base_address=base_address,
            size_bytes=size,
            alive=alive,
        )
    return table, None


def _expected_address_spaces(
    memory_op_class: int,
    instruction_pb2_mod: Any,
    address_pb2_mod: Any,
) -> tuple[frozenset[int], str] | None:
    """Return the explicit effective spaces admitted for one op class.

    ``None`` means this validator has no address-carrying replay contract for
    the class; unsupported classes remain the downstream consumer's typed
    responsibility. Missing ``address_space`` never reaches this helper and
    retains the legacy instruction-class fallback.
    """
    global_spaces = frozenset(
        (
            address_pb2_mod.MEMORY_SPACE_GLOBAL,
            address_pb2_mod.MEMORY_SPACE_GENERIC,
        )
    )
    concrete_spaces = frozenset(
        (
            address_pb2_mod.MEMORY_SPACE_GLOBAL,
            address_pb2_mod.MEMORY_SPACE_GENERIC,
            address_pb2_mod.MEMORY_SPACE_SHARED,
            address_pb2_mod.MEMORY_SPACE_LOCAL,
        )
    )
    if memory_op_class == instruction_pb2_mod.MEMORY_OP_CLASS_GLOBAL:
        return global_spaces, "GLOBAL or GENERIC"
    if memory_op_class == instruction_pb2_mod.MEMORY_OP_CLASS_SHARED:
        return frozenset((address_pb2_mod.MEMORY_SPACE_SHARED,)), "SHARED"
    if memory_op_class == instruction_pb2_mod.MEMORY_OP_CLASS_LOCAL:
        return frozenset((address_pb2_mod.MEMORY_SPACE_LOCAL,)), "LOCAL"
    if memory_op_class == instruction_pb2_mod.MEMORY_OP_CLASS_ATOMIC:
        return concrete_spaces, "GLOBAL, GENERIC, SHARED, or LOCAL"
    if memory_op_class == instruction_pb2_mod.MEMORY_OP_CLASS_ASYNC_COPY:
        return global_spaces, "GLOBAL or GENERIC source"
    return None


def _check_instruction(
    inst: Any,
    instruction_pb2_mod: Any,
    record_path: str,
    allocations: dict[int, AllocationEntry] | None = None,
    address_pb2_mod: Any = None,
) -> ValidationResult | None:
    """Apply the AC4 contract clauses to a single ``instruction``.

    Returns ``None`` on success, a typed ``ValidationResult`` on the
    first violation. The walk is deliberately fail-fast so the
    operator gets a clean record-path pointer to the bad instruction
    instead of a wall of follow-on errors.
    """
    unspecified = instruction_pb2_mod.MEMORY_OP_CLASS_UNSPECIFIED

    # "Looks like memory": either signal is sufficient.
    #   - non-empty addresses list: the tracer recorded at least one
    #     active-lane access for this instruction.
    #   - memory_op_class field present: the producer classified the
    #     opcode and emitted the class even when no addresses were
    #     emitted on this particular dynamic instance (e.g. a
    #     predicated-off all-lanes-inactive memory op).
    looks_like_memory = (
        len(inst.addresses) > 0 or inst.HasField("memory_op_class")
    )

    # data_width: required only on memory instructions. Non-memory
    # records (ALU, branch, control-flow) do not carry a per-lane
    # access width; the device-side stub reports 0 for them and the
    # producer leaves data_width absent so the wire stays honest.
    if looks_like_memory:
        if not inst.HasField("data_width"):
            return ValidationResult(
                ok=False,
                typed_reason="data_width_missing",
                detail=(
                    f"{record_path}: data_width absent on a memory "
                    f"instruction (addresses non-empty or "
                    f"memory_op_class present)"
                ),
            )
        if inst.data_width <= 0:
            return ValidationResult(
                ok=False,
                typed_reason="data_width_invalid",
                detail=(
                    f"{record_path}: data_width={inst.data_width} "
                    f"is not strictly positive"
                ),
            )

    # warpid_tb: required, present.
    if not inst.HasField("warpid_tb"):
        return ValidationResult(
            ok=False,
            typed_reason="warpid_tb_missing",
            detail=(
                f"{record_path}: warpid_tb absent on the wire "
                f"(warp 0 is a valid id but must be distinguishable "
                f"from 'tracer did not emit')"
            ),
        )

    # function_unique_id: required, must be > 0. The tracer's
    # `instrument_function_if_needed` increments
    # `next_candidate_unique_function_id` (initial 0) before
    # assigning, so the first instrumented function gets id 1.
    # A wire value of 0 means the producer did not classify the
    # function (or the record was synthesized by a non-conforming
    # path); reject so a regression in the function-id wiring
    # surfaces as a typed contract failure.
    if inst.function_unique_id <= 0:
        return ValidationResult(
            ok=False,
            typed_reason="function_unique_id_invalid",
            detail=(
                f"{record_path}: function_unique_id="
                f"{inst.function_unique_id} is not strictly positive; "
                f"the tracer assigns ids starting at 1, so 0 means "
                f"the producer did not emit this field"
            ),
        )

    # active_mask / predicate_mask: presence checks fire BEFORE
    # the lane-mask consistency value check below. Marked
    # `optional` in the proto so HasField distinguishes "tracer
    # did not emit" (returns false) from "tracer emitted value
    # 0" (returns true, valid for predicated-off all-lanes-
    # inactive case). A regression that dropped the producer's
    # `inst->set_active_mask(...)` /
    # `inst->set_predicate_mask(...)` calls would otherwise
    # silently admit at the value-only check below.
    if not inst.HasField("active_mask"):
        return ValidationResult(
            ok=False,
            typed_reason="active_mask_missing",
            detail=(
                f"{record_path}: active_mask absent on the wire "
                f"(value 0 is a valid lane mask but must be "
                f"distinguishable from 'tracer did not emit')"
            ),
        )
    if not inst.HasField("predicate_mask"):
        return ValidationResult(
            ok=False,
            typed_reason="predicate_mask_missing",
            detail=(
                f"{record_path}: predicate_mask absent on the wire "
                f"(value 0 is a valid lane mask but must be "
                f"distinguishable from 'tracer did not emit')"
            ),
        )

    # vpc: presence check fires BEFORE the bitmask consistency
    # value check below. Marked `optional` in the proto so the
    # validator can distinguish "tracer did not emit vpc"
    # (HasField false) from "vpc=0" (legitimate value when
    # pc=0 for the first instruction in a kernel).
    if not inst.HasField("vpc"):
        return ValidationResult(
            ok=False,
            typed_reason="vpc_missing",
            detail=(
                f"{record_path}: vpc absent on the wire (vpc must "
                f"be populated to match pc; the producer sets both "
                f"from the same per-instruction PC offset)"
            ),
        )

    # vpc: full per-instruction virtual PC. The producer sets
    # both `pc` (uint32, function-relative offset) and `vpc`
    # (uint64) from the same `ma->vpc` value; a wire record
    # where the low 32 bits diverge from `pc` means the tracer
    # either failed to populate `vpc` (proto3 zero default) or
    # corrupted the record. Mask the comparison to 32 bits so
    # the queued channel-side widening to a true 64-bit virtual
    # PC continues to satisfy this contract: a future emitter
    # that sets the high 32 bits is still admitted as long as
    # the low 32 bits agree with `pc`.
    if (inst.vpc & 0xFFFFFFFF) != inst.pc:
        return ValidationResult(
            ok=False,
            typed_reason="vpc_inconsistent_with_pc",
            detail=(
                f"{record_path}: vpc=0x{inst.vpc:x} low 32 bits "
                f"(0x{inst.vpc & 0xFFFFFFFF:x}) do not match pc="
                f"0x{inst.pc:x}; the producer sets both fields from "
                f"the same per-instruction PC offset, so a divergence "
                f"means vpc was not emitted (proto3 zero default) or "
                f"the record was corrupted"
            ),
        )

    # memory_op_class owns the instruction-level operation/space family and
    # must be valid before any per-address space can influence allocation
    # identity. Missing/UNSPECIFIED/schema-skew therefore precede compatibility
    # and allocation diagnostics, including on class-only zero-active records.
    if not inst.HasField("memory_op_class"):
        if len(inst.addresses) > 0:
            return ValidationResult(
                ok=False,
                typed_reason=(
                    "memory_op_class_missing_on_memory_instruction"
                ),
                detail=(
                    f"{record_path}: instruction has "
                    f"{len(inst.addresses)} address record(s) but "
                    f"memory_op_class is absent"
                ),
            )
    else:
        known_classes = frozenset(instruction_pb2_mod.MemoryOpClass.values())
        if inst.memory_op_class not in known_classes:
            return ValidationResult(
                ok=False,
                typed_reason="unknown_enum_value",
                detail=(
                    f"{record_path}: memory_op_class carries value "
                    f"{inst.memory_op_class} unknown to the committed "
                    f"protobuf schema"
                ),
            )
        if inst.memory_op_class == unspecified:
            return ValidationResult(
                ok=False,
                typed_reason="memory_op_class_unspecified",
                detail=(
                    f"{record_path}: memory_op_class is present but "
                    f"UNSPECIFIED; the tracer should have emitted an "
                    f"unsupported_memory_path exclusion sidecar instead"
                ),
            )

    and_mask = inst.active_mask & inst.predicate_mask

    # Per-record positional identity check via reconstruction.
    # For every address record (regardless of compression format),
    # reconstruct the 32-entry per-lane vector and assert that
    # slot i is sentinel iff bit i of (active_mask & predicate_mask)
    # is clear. This is the strictly-stronger check that
    # `lane_mask_inconsistent` and `lane_vector_length_invalid`
    # only approximate. Two failure modes:
    #   - reconstruction-internal failure (delta count mismatch,
    #     non-contiguous base_stride active lanes, unknown
    #     compression format) -> compressed_format_malformed.
    #   - per-slot sentinel/non-sentinel mismatch with the AND
    #     mask -> lane_position_mismatch. This catches the
    #     malformed list_all wire shape Codex flagged in the
    #     R17 review (real address in slot whose AND-mask
    #     bit is clear, or sentinel in slot whose AND-mask bit
    #     is set).
    decoded_vectors, structure_error = _reconstruct_instruction_addresses(
        inst,
        and_mask,
        record_path,
        address_pb2_mod,
    )
    if structure_error is not None:
        return structure_error
    assert decoded_vectors is not None

    # Explicit address spaces are additive. Complete lane structure and enum
    # decoding precede this semantic gate; compatibility then precedes allocation
    # identity so a contradictory space cannot reroute the record.
    if inst.HasField("memory_op_class") and address_pb2_mod is not None:
        expected_spaces = _expected_address_spaces(
            inst.memory_op_class,
            instruction_pb2_mod,
            address_pb2_mod,
        )
        if expected_spaces is not None:
            allowed_spaces, expected_label = expected_spaces
            for addr_index, addr in enumerate(inst.addresses):
                if (
                    addr.HasField("address_space")
                    and addr.address_space not in allowed_spaces
                ):
                    return ValidationResult(
                        ok=False,
                        typed_reason=(
                            "memory_op_class_address_space_conflict"
                        ),
                        detail=(
                            f"{record_path}: address[{addr_index}] "
                            f"carries address_space={addr.address_space}, "
                            f"which is incompatible with memory_op_class="
                            f"{inst.memory_op_class}; expected "
                            f"{expected_label}. Missing address_space "
                            f"retains the legacy class fallback, but an "
                            f"explicit contradictory space is rejected"
                        ),
                    )

    for addr_index, (addr, vector) in enumerate(
        zip(inst.addresses, decoded_vectors)
    ):
        # Per-record allocation/range identity check (task11d).
        # Gate: memory_op_class present AND class is GLOBAL AND the
        # address record carries an explicit allocation_id != 0. The
        # producer-side per-lane resolution gate (task11b) guarantees
        # that when the producer stamps an allocation_id, every
        # active+predicated lane resolves to the SAME id (the
        # multi-id case routes through the
        # `cross_allocation_lane_span` typed-exclusion sidecar
        # instead). The validator verifies the stamped id is known
        # to the allocations sidecar AND that each non-sentinel
        # active+predicated lane address is in
        # ``[base_address, base_address + size_bytes)``.
        # Per-address global-identity discriminator (address-carrying
        # contract). When the record carries an explicit `address_space`
        # (post-transition traces), key identity on the SPACE — GLOBAL /
        # GENERIC require an allocation_id, SHARED / LOCAL are exempt — so a
        # global atomic / LDGSTS-global-source record is enforced while a
        # shared atomic (ATOMS) / LDGSTS-shared-destination is not forced
        # through the global rule. When the field is ABSENT (legacy pre-
        # transition traces), fall back to the instruction-level class
        # (GLOBAL only), exactly as before the transition, so the frozen M1
        # anchor and every old trace validate unchanged. `address_pb2_mod`
        # may be None if a caller did not thread it (only the legacy path is
        # then reachable). See plans/sm86_ldgsts_atoms_operand_adjudication.md.
        if addr.HasField("address_space") and address_pb2_mod is not None:
            requires_global_identity = addr.address_space in (
                address_pb2_mod.MEMORY_SPACE_GLOBAL,
                address_pb2_mod.MEMORY_SPACE_GENERIC,
            )
        else:
            requires_global_identity = (
                inst.HasField("memory_op_class")
                and inst.memory_op_class
                == instruction_pb2_mod.MEMORY_OP_CLASS_GLOBAL
            )
        global_class = requires_global_identity
        # Required-identity clause: a GLOBAL record with at least
        # one active+predicated lane MUST carry a non-zero
        # `allocation_id`. The producer either stamps the id when
        # all active lanes resolve to the same allocation, or
        # skips memory-payload emission entirely (any-zero ->
        # `unmapped_memory_range`; multi-non-zero ->
        # `cross_allocation_lane_span`). A wire-side GLOBAL record
        # with active lanes AND no allocation_id is therefore a
        # producer regression — fail closed instead of silently
        # admitting it as a regular memory record. Predicated-off
        # GLOBAL records (and_mask == 0) admit because no lane
        # address requires identity.
        if (
            global_class
            and and_mask != 0
            and (
                not addr.HasField("allocation_id")
                or addr.allocation_id == 0
            )
        ):
            return ValidationResult(
                ok=False,
                typed_reason="allocation_id_missing",
                detail=(
                    f"{record_path}: address[{addr_index}] is a "
                    f"GLOBAL memory record with "
                    f"(active_mask & predicate_mask)="
                    f"0x{and_mask:x} (non-zero) but "
                    f"`allocation_id` is "
                    f"{'absent' if not addr.HasField('allocation_id') else 'zero'}; "
                    f"the producer's per-lane resolution gate "
                    f"either stamps a non-zero id or routes the "
                    f"record through `unmapped_memory_range` / "
                    f"`cross_allocation_lane_span` — a missing id "
                    f"on an active GLOBAL record indicates a "
                    f"producer-side regression"
                ),
            )
        if (
            global_class
            and addr.HasField("allocation_id")
            and addr.allocation_id != 0
        ):
            if allocations is None:
                return ValidationResult(
                    ok=False,
                    typed_reason="allocations_sidecar_missing",
                    detail=(
                        f"{record_path}: address[{addr_index}] "
                        f"carries allocation_id="
                        f"{addr.allocation_id} but no allocations "
                        f"sidecar was loaded; expected "
                        f"<traces_dir>/extra_info/allocations.json"
                    ),
                )
            entry = allocations.get(addr.allocation_id)
            if entry is None:
                return ValidationResult(
                    ok=False,
                    typed_reason="allocation_id_unknown",
                    detail=(
                        f"{record_path}: address[{addr_index}] "
                        f"allocation_id={addr.allocation_id} is "
                        f"not present in the allocations sidecar"
                    ),
                )
            base = entry.base_address
            end_excl = base + entry.size_bytes
            for lane_index in range(_LANES_PER_WARP):
                if not ((and_mask >> lane_index) & 1):
                    continue
                slot = vector[lane_index]
                if slot == LANE_INACTIVE_SENTINEL:
                    continue
                # The wire stores per-lane addresses as int64; the
                # producer fills them from CUDA's uint64 virtual
                # addresses, so reinterpret the (potentially
                # negative-looking) int64 back to uint64 before
                # the range check.
                lane_addr = slot & 0xFFFFFFFFFFFFFFFF
                if not (base <= lane_addr < end_excl):
                    return ValidationResult(
                        ok=False,
                        typed_reason=(
                            "lane_address_outside_allocation_range"
                        ),
                        detail=(
                            f"{record_path}: address[{addr_index}] "
                            f"lane {lane_index} address "
                            f"0x{lane_addr:x} is outside allocation "
                            f"id={entry.allocation_id} range "
                            f"[0x{base:x}, 0x{end_excl:x}) "
                            f"(size_bytes={entry.size_bytes})"
                        ),
                    )

    # memory_op_access_kind: producer-side direction tag at proto
    # tag 12. Additive widening — legacy traces emitted before the
    # producer-side access-kind classifier landed leave the field
    # absent on the wire and admit; the consumer-side reader in
    # `tools/hrd_profiler/observation_builder.py::_walk_launch`
    # falls back to the UNSPECIFIED bucket for absent fields.
    # However, when the field IS present on a memory instruction
    # AND its value is UNSPECIFIED, this is a producer-side
    # fail-closed event analogous to `memory_op_class_unspecified`:
    # the producer encountered a memory instruction whose opcode
    # head the classifier did not recognize and should have routed
    # it through the typed-exclusion sidecar instead of emitting a
    # kind-tagged-with-UNSPECIFIED wire record. Reject so the
    # contract validator's fail-fast walk surfaces it.
    if len(inst.addresses) > 0 and inst.HasField("memory_op_access_kind"):
        access_kind_unspecified = (
            instruction_pb2_mod.MEMORY_OP_ACCESS_KIND_UNSPECIFIED
        )
        if inst.memory_op_access_kind == access_kind_unspecified:
            return ValidationResult(
                ok=False,
                typed_reason="memory_op_access_kind_unspecified",
                detail=(
                    f"{record_path}: memory_op_access_kind is "
                    f"present but UNSPECIFIED; the tracer should "
                    f"have emitted an unsupported_memory_path "
                    f"exclusion sidecar instead"
                ),
            )

    return None


def validate_threadblock(
    tb: Any,
    *,
    label: str = "threadblock",
    allocations: dict[int, AllocationEntry] | None = None,
) -> ValidationResult:
    """Validate ``tb`` (a parsed ``dynamic_trace.threadblock`` proto)
    against the AC4 per-instruction contract.

    ``label`` prefixes the offending record path on failure so the
    detail string locates the bad record across both the in-memory
    test fixtures (``label='threadblock'``) and the on-disk walk
    (``label=<file path>``).

    ``allocations`` is the optional table loaded from
    ``<traces_dir>/extra_info/allocations.json`` via
    :func:`load_allocations_sidecar`. When provided, the per-record
    allocation/range identity check fires on GLOBAL records carrying
    an ``allocation_id``; when ``None``, the check is skipped UNLESS
    a record carries ``allocation_id != 0`` (in which case
    ``allocations_sidecar_missing`` is returned).

    Returns a ``ValidationResult``. The first record that violates
    the contract short-circuits the walk; remaining records in this
    threadblock are not inspected.
    """
    _threadblock_pb2, instruction_pb2_mod, address_pb2_mod = _import_pb_modules()

    for warp_index in sorted(tb.warps.keys()):
        warp = tb.warps[warp_index]
        for inst_index, inst in enumerate(warp.instructions):
            record_path = (
                f"{label} warp={warp_index} inst={inst_index}"
            )
            result = _check_instruction(
                inst,
                instruction_pb2_mod,
                record_path,
                allocations=allocations,
                address_pb2_mod=address_pb2_mod,
            )
            if result is not None:
                return result

    return ValidationResult(ok=True)


def _infer_traces_dir_from_pb_path(pb_path: Path) -> Path | None:
    """Walk parents of ``pb_path`` looking for a ``threadblocks``
    ancestor (the conventional trace layout places per-block ``.pb``
    files under ``<traces_dir>/threadblocks/device_<id>/stream_<id>/
    kernel_<id>/``). Returns the directory containing that subtree
    (the conventional trace root) when found; otherwise ``None``.
    """
    for parent in pb_path.parents:
        if parent.name == "threadblocks":
            return parent.parent
    return None


def validate_threadblock_path(
    path: Path,
    *,
    allocations: dict[int, AllocationEntry] | None = None,
) -> ValidationResult:
    """Read ``path`` as a serialized ``threadblock`` proto and
    validate. Returns a ``ValidationResult`` with
    ``typed_reason='trace_unreadable'`` when the file is missing or
    its bytes do not parse as a ``threadblock``.

    ``allocations`` is forwarded to :func:`validate_threadblock` when
    the caller passes an explicit table. When the caller leaves
    ``allocations`` as ``None`` AND ``path`` sits inside a conventional
    trace layout (i.e., an ancestor named ``threadblocks`` exists),
    the validator auto-infers the trace root and attempts to load
    ``<traces_dir>/extra_info/allocations.json`` via
    :func:`load_allocations_sidecar` so single-file validation matches
    the behavior of :func:`validate_trace_dir`. A malformed sidecar
    surfaces as ``allocations_sidecar_malformed`` /
    ``allocation_id_duplicate`` rather than being silently dropped;
    a missing sidecar leaves ``allocations`` as ``None`` and the
    per-record allocation_id != 0 check remains gated through the
    existing ``allocations_sidecar_missing`` fail-closed path. The
    inference fires only when the caller did not pass
    ``allocations`` — explicit callers continue to control the table
    fully.
    """
    threadblock_pb2_mod, _instruction_pb2, _address_pb2 = _import_pb_modules()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return ValidationResult(
            ok=False,
            typed_reason="trace_unreadable",
            detail=f"could not read {path}: {exc}",
        )
    tb = threadblock_pb2_mod.threadblock()
    try:
        tb.ParseFromString(raw)
    except Exception as exc:
        return ValidationResult(
            ok=False,
            typed_reason="trace_unreadable",
            detail=f"could not parse {path} as threadblock: {exc}",
        )
    if allocations is None:
        inferred_traces_dir = _infer_traces_dir_from_pb_path(path)
        if inferred_traces_dir is not None:
            inferred_allocations, load_err = load_allocations_sidecar(
                inferred_traces_dir,
            )
            if load_err is not None:
                return load_err
            allocations = inferred_allocations
    return validate_threadblock(
        tb, label=str(path), allocations=allocations
    )


def validate_trace_dir(traces_dir: Path) -> ValidationResult:
    """Walk every ``*.pb`` under ``<traces_dir>/threadblocks/`` and
    apply the contract. Returns the first failing record's
    ``ValidationResult`` (or ``ok=True`` if every threadblock
    passes / the directory holds zero threadblock files).

    Loads ``<traces_dir>/extra_info/allocations.json`` first via
    :func:`load_allocations_sidecar` so the per-record
    allocation/range identity check (task11d) can verify that every
    GLOBAL record carrying an ``allocation_id`` resolves to a known
    table entry AND that each non-sentinel active+predicated lane
    address falls in the resolved range. A malformed sidecar or a
    duplicate allocation_id short-circuits the walk before any
    threadblocks are read; a missing sidecar is tolerated as long
    as no record carries ``allocation_id != 0``.

    The conventional on-disk layout is:
    ``<traces_dir>/threadblocks/device_<id>/stream_<id>/
    kernel_<id>/<tb_x>_<tb_y>_<tb_z>.pb``. The walk is recursive
    so any tree shape with ``threadblock`` records under
    ``threadblocks/`` is accepted.
    """
    threadblocks_root = traces_dir / "threadblocks"
    if not threadblocks_root.is_dir():
        return ValidationResult(
            ok=False,
            typed_reason="trace_unreadable",
            detail=(
                f"traces_dir {traces_dir} has no threadblocks/ "
                f"subtree"
            ),
        )
    allocations, load_err = load_allocations_sidecar(traces_dir)
    if load_err is not None:
        return load_err
    for tb_path in sorted(threadblocks_root.rglob("*.pb")):
        result = validate_threadblock_path(
            tb_path, allocations=allocations
        )
        if not result.ok:
            return result
    return ValidationResult(ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an NVBit tracer .pb against the AC4 per-"
            "instruction contract: every memory instruction carries "
            "a memory_op_class != UNSPECIFIED, every instruction "
            "carries a positive data_width and a present warpid_tb."
        )
    )
    parser.add_argument(
        "target",
        type=Path,
        help=(
            "Either a single threadblock .pb file or a traces "
            "directory containing a 'threadblocks/' subtree."
        ),
    )
    args = parser.parse_args(argv)

    target: Path = args.target
    try:
        if target.is_dir():
            result = validate_trace_dir(target)
        else:
            result = validate_threadblock_path(target)
    except TraceContractTrackedBindingsCorrupt as exc:
        sys.stderr.write(f"import_trace_contract: repository_corruption: {exc}\n")
        return 2
    except TraceContractRuntimePrerequisiteMissing as exc:
        sys.stderr.write(f"import_trace_contract: prerequisite_failure: {exc}\n")
        return 2

    if not result.ok:
        sys.stderr.write(
            f"import_trace_contract: {result.typed_reason}: "
            f"{result.detail}\n"
        )
        return 1
    sys.stdout.write(
        "ok: trace conforms to AC4 per-instruction contract\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
