"""Counting-pass statistics for a captured trace directory.

Walks a trace dir and produces the per-cell statistics the admissibility gate +
the sampler consume: per-block record counts, op-class mix, the placement
(`sm_id` per block), the `launch_index`-ordered pseudo-wave reconstruction, a
per-SM slot map, block-count skew (top-1% share + Gini), and atomic density +
top-sector concentration.

Wave reconstruction is on `launch_index` (CUDA row-major flatten of `block_id`),
NOT arrival/start order: the simulator has no start-order concept and orders
per-SM blocks by launch index, so this is the only reconstruction re-derivable
from the dumped trace. Resident-blocks-per-SM is estimated from the observed
`warpid_sm` slot occupancy (max slot + 1) / warps-per-block, floored at 1.

Targets-never-features boundary: no counter/label value is read; this consumes
only the trace wire.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
_NVBIT = _REPO / "tools" / "nvbit_tracer"
_LOCAL_LABELS = _REPO / "tools" / "local_labels"
for _p in (str(_REPO), str(_NVBIT), str(_LOCAL_LABELS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.nvbit_tracer.proto_binding_loader import (  # noqa: E402
    load_dynamic_trace_bindings,
)

# MemoryOpClass wire values (instruction.proto).
GLOBAL, SHARED, LOCAL, ATOMIC, ASYNC_COPY = 1, 2, 3, 6, 8
_SECTOR_BYTES = 32
_MEMORY_SPACE_GLOBAL, _MEMORY_SPACE_SHARED = 1, 2
_MEMORY_SPACE_LOCAL, _MEMORY_SPACE_GENERIC = 3, 4
_UINT64_MAX = (1 << 64) - 1
TRACER_VERSION = 5
TRACE_SAMPLING_PASS_VERSION = 2
# sm_86 resident-warp capacity per SM (arch constant; used only to bound r_s).
SM86_MAX_WARPS_PER_SM = 48


def _require_int(value: Any, ctx: str) -> int:
    """Return `value` only if it is a genuine JSON integer. `bool` is a subclass
    of `int`, so `type(value) is int` (NOT isinstance) rejects `True`/`False` as
    well as `float` (incl. `4.0`), `str`, and `None` — no coercion or rounding, so
    a malformed/stale sidecar value can never alias to a valid shape or block id."""
    if type(value) is not int:
        raise ValueError(
            f"{ctx}: expected an integer, got {type(value).__name__} {value!r}")
    return value


def _json_int(obj: Any, key: str, ctx: str) -> int:
    """Strict-integer read of `obj[key]` (a JSON object field)."""
    if not isinstance(obj, dict) or key not in obj:
        raise ValueError(f"{ctx}: missing integer field {key!r}")
    return _require_int(obj[key], f"{ctx}.{key}")


def _json_bool(obj: Any, key: str, ctx: str) -> bool:
    """Strict-boolean read of ``obj[key]`` (no integer/string aliases)."""
    if not isinstance(obj, dict) or key not in obj:
        raise ValueError(f"{ctx}: missing boolean field {key!r}")
    value = obj[key]
    if type(value) is not bool:
        raise ValueError(
            f"{ctx}.{key}: expected a boolean, got "
            f"{type(value).__name__} {value!r}")
    return value


@dataclass
class BlockStat:
    block_id: int  # flattened launch index (CUDA row-major)
    sm_id: int
    records: int
    sector_accesses: int = 0
    serialized_bytes: int = 0
    op_class_counts: dict[int, int] = field(default_factory=dict)
    warps_per_block: int = 0
    max_warp_slot: int = -1  # max observed warpid_sm on this block
    wave: int = -1  # filled by wave reconstruction


@dataclass
class CellStats:
    cell_id: str
    kernel_name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    n_blocks: int
    total_records: int
    op_class_totals: dict[int, int]
    blocks: list[BlockStat]
    per_sm_resident: dict[int, int]  # sm_id -> estimated resident blocks (r_s)
    per_sm_block_ids: dict[int, list[int]]  # sm_id -> sorted block_ids
    n_waves: int
    atomic_records: int
    atomic_top256_sector_share: float
    per_wave_record_totals: dict[int, int]  # pseudo-wave -> total records (homogeneity)
    per_sm_block_counts: dict[int, int]     # sm_id -> block count (homogeneity)
    cooperative: bool = False
    # 0 is deliberately "unproven" for synthetic/legacy constructors.  Both
    # real decoder paths stamp the version they actually produced/parsed.
    stats_pass_version: int = 0
    serialized_bytes_exact: bool = False

    @property
    def atomic_density(self) -> float:
        return self.atomic_records / self.total_records if self.total_records else 0.0

    @property
    def skew_top1pct_share(self) -> float:
        return _top_fraction_share([b.records for b in self.blocks], 0.01)

    @property
    def gini(self) -> float:
        return _gini([b.records for b in self.blocks])

    @property
    def total_sector_accesses(self) -> int:
        return sum(b.sector_accesses for b in self.blocks)

    @property
    def total_serialized_bytes(self) -> int:
        return sum(b.serialized_bytes for b in self.blocks)


def _gini(values: list[int]) -> float:
    """Gini coefficient of a non-negative distribution (0 = uniform)."""
    xs = sorted(v for v in values if v >= 0)
    n = len(xs)
    s = sum(xs)
    if n == 0 or s == 0:
        return 0.0
    cum = 0
    for i, x in enumerate(xs, start=1):
        cum += i * x
    # G = (2*Σ i*x_i)/(n*Σx) - (n+1)/n
    return (2.0 * cum) / (n * s) - (n + 1.0) / n


def _top_fraction_share(values: list[int], frac: float) -> float:
    """Share of total carried by the heaviest ceil(frac*n) items."""
    xs = sorted((v for v in values), reverse=True)
    n = len(xs)
    s = sum(xs)
    if n == 0 or s == 0:
        return 0.0
    k = max(1, int(-(-n * frac // 1)))  # ceil(n*frac)
    return sum(xs[:k]) / s


def _resident_from_slots(max_warp_slot: int, warps_per_block: int) -> int:
    """Estimate resident blocks per SM from the observed max warp slot.

    The SM packs resident warps into slots [0, capacity); a block occupies
    `warps_per_block` slots, so the number of co-resident blocks is
    (max_slot + 1) / warps_per_block, bounded by the arch warp capacity."""
    if warps_per_block <= 0:
        return 1
    cap_warps = min(max_warp_slot + 1, SM86_MAX_WARPS_PER_SM) if max_warp_slot >= 0 \
        else warps_per_block
    return max(1, cap_warps // warps_per_block)


def _reconstruct_waves(blocks: list[BlockStat]) -> tuple[dict[int, int], dict[int, list[int]], int]:
    """Assign each block a pseudo-wave = floor(seq_on_sm / r_s), ordering each
    SM's blocks by launch_index. Returns (per_sm_resident, per_sm_block_ids,
    n_waves)."""
    per_sm: dict[int, list[BlockStat]] = {}
    for b in blocks:
        per_sm.setdefault(b.sm_id, []).append(b)
    per_sm_resident: dict[int, int] = {}
    per_sm_block_ids: dict[int, list[int]] = {}
    n_waves = 0
    for sm, bs in per_sm.items():
        bs.sort(key=lambda b: b.block_id)
        max_slot = max((b.max_warp_slot for b in bs), default=-1)
        wpb = max((b.warps_per_block for b in bs), default=0)
        r_s = _resident_from_slots(max_slot, wpb)
        per_sm_resident[sm] = r_s
        per_sm_block_ids[sm] = [b.block_id for b in bs]
        for seq, b in enumerate(bs):
            b.wave = seq // r_s
            n_waves = max(n_waves, b.wave + 1)
    return per_sm_resident, per_sm_block_ids, n_waves


def _lowest_launch(trace):
    """Return (device_id, stream_id, kernel) of the lowest (device, stream,
    kernel) launch — the one the label measured (mirrors first_launch_dir)."""
    for d in sorted(trace.gpu_device.keys()):
        dev = trace.gpu_device[d]
        for s in sorted(dev.streams.keys()):
            for kern in dev.streams[s].kernels:
                return d, s, kern
    return None, None, None


def _lowest_kernel(trace) -> Any:
    return _lowest_launch(trace)[2]


def analyze_trace_dir(trace_dir: Path, cell_id: str, *, cooperative: bool = False) -> CellStats:
    """Decode a trace dir into CellStats. Fail-closed on missing grid_dim."""
    bindings = load_dynamic_trace_bindings()
    threadblock_pb2 = bindings.threadblock_pb2
    trace_pb2 = bindings.trace_pb2
    import import_trace_contract as itc  # type: ignore
    from replay_count_producer import first_launch_dir  # type: ignore

    trace = trace_pb2.Trace()
    trace.ParseFromString((trace_dir / "dynamic_trace.pb").read_bytes())
    kern = _lowest_kernel(trace)
    if kern is None or not kern.HasField("grid_dim"):
        raise ValueError(f"{trace_dir}: no kernel with grid_dim on the wire")
    gx, gy, gz = int(kern.grid_dim.x), int(kern.grid_dim.y), int(kern.grid_dim.z)
    bx, by, bz = (int(kern.block_dim.x), int(kern.block_dim.y), int(kern.block_dim.z)) \
        if kern.HasField("block_dim") else (0, 0, 0)
    grid_total = gx * gy * gz

    # Scope the walk to the SAME first launch the label measured (the shared
    # helper), never the whole multi-launch tree — an unscoped walk over-counts.
    first_launch_root, _first_kernel_id = first_launch_dir(trace_dir)
    seen_block_ids: set[int] = set()
    blocks: list[BlockStat] = []
    op_totals: dict[int, int] = {}
    total_records = 0
    atomic_sector_hist: dict[int, int] = {}
    atomic_records = 0
    for path in sorted(first_launch_root.rglob("*.pb")):
        tb = threadblock_pb2.threadblock()
        serialized = path.read_bytes()
        tb.ParseFromString(serialized)
        if not tb.HasField("block_id"):
            raise ValueError(f"{path}: threadblock missing block_id")
        cx, cy, cz = int(tb.block_id.x), int(tb.block_id.y), int(tb.block_id.z)
        # Per-AXIS bound: an out-of-grid CTA on one axis (e.g. (2,0,0) on a (2,2,1)
        # grid) still flattens INTO [0,grid_total) and would alias to a valid
        # index — reject each coordinate against its own extent.
        if not (0 <= cx < gx and 0 <= cy < gy and 0 <= cz < gz):
            raise ValueError(f"{path}: CTA ({cx},{cy},{cz}) outside grid ({gx},{gy},{gz})")
        idx = cx + cy * gx + cz * gx * gy
        if idx < 0 or idx >= grid_total:
            raise ValueError(f"{path}: block_id flattens to {idx} outside [0,{grid_total})")
        if idx in seen_block_ids:
            raise ValueError(f"{path}: duplicate flattened block_id {idx} in one launch")
        seen_block_ids.add(idx)
        recs = 0
        sm_id = -1
        max_slot = -1
        opc: dict[int, int] = {}
        sector_accesses = 0
        n_warps = len(tb.warps)
        for wi in tb.warps:
            warp = tb.warps[wi]
            for inst in warp.instructions:
                recs += 1
                if inst.HasField("sm_id"):
                    sm_id = int(inst.sm_id)
                if inst.HasField("warpid_sm"):
                    max_slot = max(max_slot, int(inst.warpid_sm))
                if inst.HasField("memory_op_class") and inst.addresses:
                    cls = int(inst.memory_op_class)
                    opc[cls] = opc.get(cls, 0) + 1
                    op_totals[cls] = op_totals.get(cls, 0) + 1
                    sector_accesses += _instruction_sector_accesses(
                        inst, itc, f"{path}: instruction pc={int(inst.pc)}")
                    if cls == ATOMIC:
                        atomic_records += 1
                        _accumulate_atomic_sectors(inst, itc, atomic_sector_hist)
        total_records += recs
        blocks.append(BlockStat(block_id=idx, sm_id=sm_id, records=recs,
                                sector_accesses=sector_accesses,
                                serialized_bytes=len(serialized),
                                op_class_counts=opc, warps_per_block=n_warps,
                                max_warp_slot=max_slot))

    if not blocks:
        raise ValueError(f"{trace_dir}: first launch has no threadblock payloads")

    per_sm_resident, per_sm_block_ids, n_waves = _reconstruct_waves(blocks)
    per_wave_totals: dict[int, int] = {}
    for b in blocks:
        per_wave_totals[b.wave] = per_wave_totals.get(b.wave, 0) + b.records
    per_sm_block_counts = {sm: len(ids) for sm, ids in per_sm_block_ids.items()}
    top256 = _top_n_share(atomic_sector_hist, 256)
    return CellStats(
        cell_id=cell_id, kernel_name=kern.name, grid=(gx, gy, gz), block=(bx, by, bz),
        n_blocks=len(blocks), total_records=total_records, op_class_totals=op_totals,
        blocks=blocks, per_sm_resident=per_sm_resident, per_sm_block_ids=per_sm_block_ids,
        n_waves=n_waves, atomic_records=atomic_records,
        atomic_top256_sector_share=top256, per_wave_record_totals=per_wave_totals,
        per_sm_block_counts=per_sm_block_counts, cooperative=cooperative,
        stats_pass_version=TRACE_SAMPLING_PASS_VERSION,
        serialized_bytes_exact=True)


def cell_stats_from_capture(trace_dir: Path, cell_id: str, *,
                            cooperative: bool = False) -> CellStats:
    """Build CellStats from a COUNTING-mode capture (no full trace payload): the
    grid/block from `dynamic_trace.pb` + per-block stats + the atomic sketch from
    `extra_info/trace_sampling_capture.json`. The gate consumes the same CellStats
    whether it comes from a full trace or the cheap counting pass."""
    trace_pb2 = load_dynamic_trace_bindings().trace_pb2

    trace = trace_pb2.Trace()
    trace.ParseFromString((trace_dir / "dynamic_trace.pb").read_bytes())
    dev_id, stream_id, kern = _lowest_launch(trace)
    if kern is None or not kern.HasField("grid_dim"):
        raise ValueError(f"{trace_dir}: no kernel with grid_dim on the wire")
    gx, gy, gz = int(kern.grid_dim.x), int(kern.grid_dim.y), int(kern.grid_dim.z)
    bx, by, bz = (int(kern.block_dim.x), int(kern.block_dim.y), int(kern.block_dim.z)) \
        if kern.HasField("block_dim") else (0, 0, 0)
    cap = json.loads((trace_dir / "extra_info" / "trace_sampling_capture.json").read_text())
    cap_ctx = f"{trace_dir}: counting sidecar"
    if not isinstance(cap, dict):
        raise ValueError(f"{cap_ctx}: root must be a JSON object")
    mode = _json_int(cap, "mode", cap_ctx)
    if mode not in (1, 2):
        raise ValueError(f"{cap_ctx}.mode: expected 1 or 2, got {mode}")
    tracer_version = _json_int(cap, "tracer_version", cap_ctx)
    if tracer_version != TRACER_VERSION:
        raise ValueError(
            f"{cap_ctx}.tracer_version: expected {TRACER_VERSION}, "
            f"got {tracer_version}")
    stats_pass_version = _json_int(cap, "stats_pass_version", cap_ctx)
    if stats_pass_version != TRACE_SAMPLING_PASS_VERSION:
        raise ValueError(
            f"{cap_ctx}.stats_pass_version: expected "
            f"{TRACE_SAMPLING_PASS_VERSION}, got {stats_pass_version}")
    if not _json_bool(cap, "sector_accesses_valid", cap_ctx):
        raise ValueError(f"{cap_ctx}: producer marked sector_accesses invalid")
    serialized_bytes_exact = _json_bool(
        cap, "serialized_bytes_exact", cap_ctx)
    if serialized_bytes_exact != (mode == 2):
        raise ValueError(
            f"{cap_ctx}.serialized_bytes_exact: must be true exactly in mode 2")

    # The counting sidecar must describe the SAME first launch the label + the
    # .pb walk see (the tracer scopes to the first launch; reject otherwise).
    launch = cap.get("launch")
    if launch is None:
        raise ValueError(f"{trace_dir}: counting sidecar missing launch identity")
    # Strict integer validation (no int() coercion): a malformed/stale sidecar
    # value like 4.9 / "4" must be REJECTED, never truncated/parsed to a match.
    if (_json_int(launch, "device", "launch"), _json_int(launch, "stream", "launch"),
            _json_int(launch, "kernel", "launch")) != \
            (int(dev_id), int(stream_id), int(kern.id)):
        raise ValueError(
            f"{trace_dir}: counting sidecar launch {launch} != dynamic_trace.pb "
            f"first launch ({dev_id},{stream_id},{kern.id})")
    # Shape-bind: the sidecar's grid/block must EXACTLY match the first launch's
    # (a stale/corrupt sidecar with a matching launch id but a different shape,
    # or an out-of-grid CTA, must be rejected — this path is otherwise fail-open).
    scg, scb = cap.get("grid_dim"), cap.get("block_dim")
    if scg is None or scb is None:
        raise ValueError(f"{trace_dir}: counting sidecar missing grid_dim/block_dim")
    if (_json_int(scg, "x", "grid_dim"), _json_int(scg, "y", "grid_dim"),
            _json_int(scg, "z", "grid_dim")) != (gx, gy, gz):
        raise ValueError(f"{trace_dir}: sidecar grid_dim {scg} != first launch ({gx},{gy},{gz})")
    if (_json_int(scb, "x", "block_dim"), _json_int(scb, "y", "block_dim"),
            _json_int(scb, "z", "block_dim")) != (bx, by, bz):
        raise ValueError(f"{trace_dir}: sidecar block_dim {scb} != first launch ({bx},{by},{bz})")

    grid_total = gx * gy * gz
    warps_per_block = max(1, (bx * by * bz + 31) // 32)
    blocks: list[BlockStat] = []
    total_records = 0
    atomic_records = 0
    seen: set[int] = set()
    capture_blocks = cap.get("blocks")
    if not isinstance(capture_blocks, list) or not capture_blocks:
        raise ValueError(f"{cap_ctx}: blocks must be a non-empty list")
    for block_index, b in enumerate(capture_blocks):
        block_ctx = f"{cap_ctx}.blocks[{block_index}]"
        cta = b.get("cta") if isinstance(b, dict) else None
        if not isinstance(cta, list) or len(cta) != 3:
            raise ValueError(f"{trace_dir}: sidecar block cta must be a 3-list, got {cta!r}")
        cx = _require_int(cta[0], f"{trace_dir}: cta.x")
        cy = _require_int(cta[1], f"{trace_dir}: cta.y")
        cz = _require_int(cta[2], f"{trace_dir}: cta.z")
        if not (0 <= cx < gx and 0 <= cy < gy and 0 <= cz < gz):
            raise ValueError(f"{trace_dir}: sidecar CTA ({cx},{cy},{cz}) outside grid ({gx},{gy},{gz})")
        idx = cx + cy * gx + cz * gx * gy
        if not (0 <= idx < grid_total):
            raise ValueError(f"{trace_dir}: sidecar block_id {idx} outside [0,{grid_total})")
        if idx in seen:
            raise ValueError(f"{trace_dir}: duplicate block_id {idx} in capture sidecar")
        seen.add(idx)
        sm_id = _json_int(b, "sm_id", block_ctx)
        records = _json_int(b, "records", block_ctx)
        max_warp_slot = _json_int(b, "max_warpid_sm", block_ctx)
        block_atomic_records = _json_int(b, "atomic_records", block_ctx)
        sector_accesses = _json_int(b, "sector_accesses", block_ctx)
        serialized_bytes = _json_int(b, "serialized_bytes", block_ctx)
        if sm_id < 0 or records < 1 or block_atomic_records < 0 or \
                sector_accesses < 0 or serialized_bytes < 0:
            raise ValueError(
                f"{block_ctx}: sm_id, atomic_records, sector_accesses, and "
                f"serialized_bytes must be nonnegative; records must be positive")
        if max_warp_slot < 0:
            raise ValueError(f"{block_ctx}.max_warpid_sm: must be nonnegative")
        if block_atomic_records > records:
            raise ValueError(
                f"{block_ctx}.atomic_records: cannot exceed records")
        if serialized_bytes_exact and serialized_bytes == 0:
            raise ValueError(
                f"{block_ctx}.serialized_bytes: exact mode-2 payload cannot be zero")
        blocks.append(BlockStat(
            block_id=idx, sm_id=sm_id, records=records,
            sector_accesses=sector_accesses,
            serialized_bytes=serialized_bytes,
            warps_per_block=warps_per_block, max_warp_slot=max_warp_slot))
        total_records += records
        atomic_records += block_atomic_records
    if len(seen) != grid_total:
        missing: list[int] = []
        for block_id in range(grid_total):
            if block_id not in seen:
                missing.append(block_id)
                if len(missing) == 8:
                    break
        raise ValueError(
            f"{cap_ctx}: all-block evidence incomplete; missing block_id(s) {missing}")
    per_sm_resident, per_sm_block_ids, n_waves = _reconstruct_waves(blocks)
    per_wave_totals: dict[int, int] = {}
    for b in blocks:
        per_wave_totals[b.wave] = per_wave_totals.get(b.wave, 0) + b.records
    per_sm_block_counts = {sm: len(ids) for sm, ids in per_sm_block_ids.items()}
    atomic_total = _json_int(cap, "atomic_sector_total", cap_ctx)
    atomic_top256 = _json_int(cap, "atomic_top256_touches", cap_ctx)
    if _json_bool(cap, "atomic_sectors_overflow", cap_ctx):
        raise ValueError(
            f"{cap_ctx}: atomic-sector sketch overflow makes hotline share inexact")
    if atomic_total < 0 or atomic_top256 < 0 or atomic_top256 > atomic_total:
        raise ValueError(
            f"{cap_ctx}: atomic sketch totals must satisfy "
            f"0 <= atomic_top256_touches <= atomic_sector_total")
    top256_share = (atomic_top256 / atomic_total) if atomic_total else 0.0
    return CellStats(
        cell_id=cell_id, kernel_name=kern.name, grid=(gx, gy, gz), block=(bx, by, bz),
        n_blocks=len(blocks), total_records=total_records,
        op_class_totals={ATOMIC: atomic_records} if atomic_records else {},
        blocks=blocks, per_sm_resident=per_sm_resident, per_sm_block_ids=per_sm_block_ids,
        n_waves=n_waves, atomic_records=atomic_records,
        atomic_top256_sector_share=top256_share, per_wave_record_totals=per_wave_totals,
        per_sm_block_counts=per_sm_block_counts, cooperative=cooperative,
        stats_pass_version=stats_pass_version,
        serialized_bytes_exact=serialized_bytes_exact)


def _instruction_sector_accesses(inst, itc, ctx: str) -> int:
    """Return exact 32-byte global-L2 sector transactions for one instruction.

    Sectors are deduplicated separately inside each address/operand record and
    then summed, matching the channel producer where every ``inst_trace_t`` is
    one operand record.  Only active+predicated lanes contribute.  GLOBAL,
    GLOBAL/GENERIC ATOMIC, and the GLOBAL/GENERIC source of ASYNC_COPY are
    admitted; malformed/mistyped records raise instead of aliasing to zero.
    """
    cls = int(inst.memory_op_class)
    if cls not in (GLOBAL, ATOMIC, ASYNC_COPY):
        return 0
    if not inst.HasField("active_mask") or not inst.HasField("predicate_mask"):
        raise ValueError(f"{ctx}: L2-relevant record missing active/predicate mask")
    if not inst.HasField("data_width") or int(inst.data_width) <= 0:
        raise ValueError(f"{ctx}: L2-relevant record missing/invalid data_width")
    instruction_width = int(inst.data_width)
    and_mask = int(inst.active_mask) & int(inst.predicate_mask)
    total = 0
    for operand_index, addr in enumerate(inst.addresses):
        addr_ctx = f"{ctx} address[{operand_index}]"
        has_space = addr.HasField("address_space")
        space = int(addr.address_space) if has_space else None
        if cls == GLOBAL:
            # Pre-address-space traces are still unambiguous here: the
            # instruction-level GLOBAL class is the documented fallback.  If a
            # newer producer does stamp the discriminator, it must agree.
            if has_space and space not in (
                    _MEMORY_SPACE_GLOBAL, _MEMORY_SPACE_GENERIC):
                raise ValueError(
                    f"{addr_ctx}: GLOBAL record has non-global address_space")
            relevant = True
        elif cls == ATOMIC:
            if not has_space or space not in (
                    _MEMORY_SPACE_GLOBAL, _MEMORY_SPACE_GENERIC,
                    _MEMORY_SPACE_SHARED, _MEMORY_SPACE_LOCAL):
                raise ValueError(
                    f"{addr_ctx}: ATOMIC record missing/invalid address_space")
            relevant = space in (_MEMORY_SPACE_GLOBAL, _MEMORY_SPACE_GENERIC)
        else:  # ASYNC_COPY: the recorded operand must be its global source.
            if not has_space or space not in (
                    _MEMORY_SPACE_GLOBAL, _MEMORY_SPACE_GENERIC):
                raise ValueError(
                    f"{addr_ctx}: ASYNC_COPY source missing/non-global address_space")
            relevant = True
        if not relevant:
            continue

        width = int(addr.data_width)
        if width <= 0:
            raise ValueError(f"{addr_ctx}: data_width must be positive, got {width}")
        if width != instruction_width:
            raise ValueError(
                f"{addr_ctx}: data_width {width} != instruction data_width "
                f"{instruction_width}")
        lanes, reason = itc._reconstruct_lane_vector(addr, and_mask)
        if lanes is None:
            raise ValueError(f"{addr_ctx}: malformed lane vector: {reason}")

        intervals: list[tuple[int, int]] = []
        for lane in range(32):
            if ((and_mask >> lane) & 1) == 0:
                continue
            base = int(lanes[lane])
            if base < 0 or base > _UINT64_MAX:
                raise ValueError(
                    f"{addr_ctx}: active lane {lane} has invalid address {base}")
            end = base + width - 1
            if end > _UINT64_MAX:
                raise ValueError(
                    f"{addr_ctx}: active lane {lane} address+width overflows uint64")
            intervals.append((base // _SECTOR_BYTES, end // _SECTOR_BYTES))
        if not intervals:
            continue

        intervals.sort()
        first, last = intervals[0]
        operand_total = 0
        for next_first, next_last in intervals[1:]:
            if next_first <= last + 1:
                last = max(last, next_last)
                continue
            operand_total += last - first + 1
            first, last = next_first, next_last
        total += operand_total + last - first + 1
    return total


def _accumulate_atomic_sectors(inst, itc, hist: dict[int, int]) -> None:
    active = inst.active_mask if inst.HasField("active_mask") else 0
    pred = inst.predicate_mask if inst.HasField("predicate_mask") else 0xFFFFFFFF
    and_mask = active & pred
    if and_mask == 0 or not inst.addresses:
        return
    addr = inst.addresses[0]
    lanes, _reason = itc._reconstruct_lane_vector(addr, and_mask)
    if lanes is None:
        return
    sentinel = itc.LANE_INACTIVE_SENTINEL
    for a in lanes:
        if int(a) == sentinel:
            continue
        sec = int(a) // _SECTOR_BYTES
        hist[sec] = hist.get(sec, 0) + 1


def _top_n_share(hist: dict[int, int], n: int) -> float:
    total = sum(hist.values())
    if total == 0:
        return 0.0
    top = sorted(hist.values(), reverse=True)[:n]
    return sum(top) / total
