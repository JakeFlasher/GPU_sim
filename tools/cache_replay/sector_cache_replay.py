"""Cache-only trace-replay predictor for counter-free L2 hit-ratio observability.

This module is a **cache hit-rate model**, the mechanistic sibling of the
clean-room Nugteren-2014 reuse-distance-histogram baseline in
``tools/baseline_runners/nugteren2014_rdh_baseline.py``. It consumes
already-collected NVBit sector/address traces and replays them through a
**sectored set-associative cache** (32 B sectors grouped into 128 B lines) to
emit a calibrated L2 (and diagnostic L1) cache-hit-ratio estimate with an
uncertainty interval.

The "NOT a GPU simulator" charter language that used to live here is RETIRED:
the no-simulator rule was withdrawn at the 2026-07-02 re-founding (building the
NCU-predicting simulator IS now the project). This module remains the canonical
PYTHON cache-hit-ratio engine and is the lockstep source of the ACTIVE Rust
port ``sim/mem/src/l2.rs`` (cache core only: ``replay_access_stream`` +
``_install_way``; slice hashes bit-exact, including negative line ids). Any
edit to the cache-core semantics here must update the Rust twin and the shared
parity corpus (``tools/cache_replay/test_l2_rust_parity.py`` +
``sim/mem/src/l2.rs::tests::python_parity_corpus_pins``) in the same commit.

Inputs: the per-cell NVBit dynamic trace (``threadblocks/**/*.pb``) decoded
through the validated wire-contract primitives in
``tools/nvbit_tracer/import_trace_contract.py``. The decode yields the same
byte-address-derived ordered access stream that
``tools/hrd_profiler/observation_builder.py`` walks, but the replay applies a
set-associative occupancy model (with conflict / slice / hash / replacement
realism) instead of a fully-associative reuse-distance approximation.
"""

from __future__ import annotations

import math
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_TOOLS_NVBIT = _REPO_ROOT / "tools" / "nvbit_tracer"
if str(_TOOLS_NVBIT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_NVBIT))

import import_trace_contract  # type: ignore[import-not-found]  # noqa: E402

# --- fixed granularity constants (pre-registered; sector/line are physical) ---
SECTOR_BYTES = 32
LINE_BYTES = 128
SECTORS_PER_LINE = LINE_BYTES // SECTOR_BYTES  # 4

# Memory-op classes that reach the L2 sector cache. The admitted sector stream
# carries GLOBAL and LOCAL; both route through L2. Other classes (SHARED,
# TEXTURE, SURFACE, ATOMIC, TMA, ASYNC_COPY, CLUSTER_DSMEM, TMEM) are excluded
# from the L2 occupancy model and recorded in provenance.
_MEMORY_OP_CLASS_GLOBAL = 1
_MEMORY_OP_CLASS_LOCAL = 3
DEFAULT_L2_OP_CLASSES = frozenset({_MEMORY_OP_CLASS_GLOBAL, _MEMORY_OP_CLASS_LOCAL})

# Per-address memory space (mirror of dynamic_trace.MemorySpace, address.proto).
# When a record carries an explicit space it is AUTHORITATIVE over the
# instruction-level op class: GLOBAL/GENERIC is L2 traffic (a global atomic or
# LDGSTS global source enters the stream), SHARED/LOCAL is not.
_MEMORY_SPACE_GLOBAL = 1
_MEMORY_SPACE_SHARED = 2
_MEMORY_SPACE_LOCAL = 3
_MEMORY_SPACE_GENERIC = 4
_L2_ADDRESS_SPACES = frozenset({_MEMORY_SPACE_GLOBAL, _MEMORY_SPACE_GENERIC})

REPLACEMENT_POLICIES = ("LRU", "SRRIP", "BIP")
PREFETCH_MODES = ("off", "next_line")

_SRRIP_RRPV_MAX = 3          # 2-bit re-reference prediction value
_SRRIP_INSERT_RRPV = 2       # long re-reference interval on insert
_BIP_MRU_INSERT_PERIOD = 32  # deterministic bimodal: 1/32 inserts at MRU


# --------------------------------------------------------------------------
# Slice-hash registry: candidate line -> slice mappings for the unknown
# Blackwell L2 slice geometry. Each candidate is a deterministic bit-mixing
# function of the 128 B line address. Selected from this fixed set during
# calibration; recorded verbatim in provenance.
# --------------------------------------------------------------------------
def _slice_line_mod(line_id: int, num_slices: int) -> int:
    return line_id % num_slices


def _slice_sector_mod(line_id: int, num_slices: int) -> int:
    # Mirrors the legacy ``sector_id_mod_num_l2_slices`` model at 128 B line
    # granularity (the first sector of the line).
    return (line_id * SECTORS_PER_LINE) % num_slices


def _slice_xor_folded(line_id: int, num_slices: int) -> int:
    x = line_id ^ (line_id >> 5) ^ (line_id >> 11)
    return x % num_slices


def _slice_knuth_mult(line_id: int, num_slices: int) -> int:
    # Knuth multiplicative hash (low 32 bits), then mod.
    return ((line_id * 2654435761) & 0xFFFFFFFF) % num_slices


def _slice_shift_mod(line_id: int, num_slices: int) -> int:
    return (line_id >> 1) % num_slices


def _slice_bit_reverse(line_id: int, num_slices: int) -> int:
    # Reverse the low 16 bits of the line address before mod (spreads
    # power-of-two strides across slices).
    v = line_id & 0xFFFF
    r = 0
    for _ in range(16):
        r = (r << 1) | (v & 1)
        v >>= 1
    return r % num_slices


SLICE_HASH_REGISTRY = {
    "line_mod": _slice_line_mod,
    "sector_mod": _slice_sector_mod,
    "xor_folded": _slice_xor_folded,
    "knuth_mult": _slice_knuth_mult,
    "shift_mod": _slice_shift_mod,
    "bit_reverse": _slice_bit_reverse,
}
SLICE_HASH_IDS = tuple(SLICE_HASH_REGISTRY.keys())  # 6 candidates (within 4-8)


@dataclass(frozen=True)
class CacheReplayConfig:
    """A single pre-registered Blackwell-L2 candidate configuration.

    ``capacity_bytes`` is the knowable device L2 capacity; ``capacity_scale``
    scales it to an *effective* capacity (the realizable fraction). The
    remaining fields are the unknown Blackwell geometry the calibrator selects.
    """

    capacity_bytes: int
    associativity: int
    replacement_policy: str = "LRU"
    num_slices: int = 1
    slice_hash_id: str = "line_mod"
    capacity_scale: float = 1.0
    prefetch: str = "off"
    sector_bytes: int = SECTOR_BYTES
    line_bytes: int = LINE_BYTES

    def __post_init__(self) -> None:
        if self.replacement_policy not in REPLACEMENT_POLICIES:
            raise ValueError(
                f"replacement_policy must be one of {REPLACEMENT_POLICIES}, "
                f"got {self.replacement_policy!r}"
            )
        if self.slice_hash_id not in SLICE_HASH_REGISTRY:
            raise ValueError(
                f"slice_hash_id must be one of {SLICE_HASH_IDS}, "
                f"got {self.slice_hash_id!r}"
            )
        if self.prefetch not in PREFETCH_MODES:
            raise ValueError(
                f"prefetch must be one of {PREFETCH_MODES}, got {self.prefetch!r}"
            )
        if self.associativity < 1:
            raise ValueError("associativity must be >= 1")
        if self.num_slices < 1:
            raise ValueError("num_slices must be >= 1")
        if self.capacity_bytes < self.line_bytes:
            raise ValueError("capacity_bytes must be >= line_bytes")

    @property
    def effective_capacity_bytes(self) -> int:
        return int(round(self.capacity_bytes * self.capacity_scale))

    @property
    def total_lines(self) -> int:
        return max(1, self.effective_capacity_bytes // self.line_bytes)

    @property
    def num_sets_per_slice(self) -> int:
        lines_per_slice = max(1, self.total_lines // self.num_slices)
        return max(1, lines_per_slice // self.associativity)

    def provenance(self) -> dict[str, Any]:
        return {
            "capacity_bytes": self.capacity_bytes,
            "effective_capacity_bytes": self.effective_capacity_bytes,
            "capacity_scale": self.capacity_scale,
            "sector_bytes": self.sector_bytes,
            "line_bytes": self.line_bytes,
            "sectors_per_line": self.line_bytes // self.sector_bytes,
            "num_slices": self.num_slices,
            "slice_hash_id": self.slice_hash_id,
            "associativity": self.associativity,
            "replacement_policy": self.replacement_policy,
            "prefetch": self.prefetch,
            "num_sets_per_slice": self.num_sets_per_slice,
            "total_lines": self.total_lines,
        }


# --------------------------------------------------------------------------
# Trace decode: ordered byte-address-derived (line_id, sector_offset, op_class)
# access stream per cell. Faithful to observation_builder._walk_launch pass 1
# (sector-edge accounting at 32 B granularity, block-coord CTA ordering), but
# kept self-contained so the replay engine does not depend on the
# fully-associative walker internals.
# --------------------------------------------------------------------------
@dataclass
class AccessStream:
    """Materialized ordered access stream for one cell.

    Parallel arrays kept compact for fast re-replay under many candidate
    configs (the expensive decode runs once; the replay is O(N))."""

    line_ids: list[int] = field(default_factory=list)
    sector_offsets: list[int] = field(default_factory=list)  # 0..SECTORS_PER_LINE-1
    n_accesses: int = 0
    n_global: int = 0
    n_local: int = 0
    n_excluded_op_class: int = 0
    included_op_class_counts: dict[int, int] = field(default_factory=dict)
    excluded_op_class_counts: dict[int, int] = field(default_factory=dict)
    n_launches: int = 0
    n_threadblocks: int = 0

    def __len__(self) -> int:
        return self.n_accesses


def _tb_block_coord_sort_key(p: Path):
    name = p.stem
    last = name.rfind("_")
    if last == -1:
        return (1, name)
    parts = name[last + 1:].split(",")
    if len(parts) != 3:
        return (1, name)
    try:
        return (0, int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return (1, name)


def _iter_launch_dirs(trace_dir: Path) -> list[Path]:
    """Return kernel-launch directories under ``threadblocks/`` in a stable
    order. Each ``.../kernel_<id>/`` directory is one launch."""
    tb_root = trace_dir / "threadblocks"
    if not tb_root.is_dir():
        return []
    kernel_dirs = sorted(
        {p.parent for p in tb_root.rglob("*.pb")},
        key=lambda d: str(d),
    )
    return kernel_dirs


def _decode_access_stream_sources(
    threadblock_paths: Iterable[str],
    read_payload: Callable[[str], bytes],
    *,
    op_classes: Iterable[int],
    cta_interleave: bool,
) -> AccessStream:
    threadblock_pb2, _, _ = import_trace_contract._import_pb_modules()
    # Retain the authenticated generated class for the decode batch; looking it up
    # per payload would revalidate the complete private binding state each time.
    threadblock_cls = threadblock_pb2.threadblock
    op_class_set = frozenset(op_classes)
    stream = AccessStream()
    launches: dict[str, list[str]] = {}
    for relative_path in threadblock_paths:
        parent = Path(relative_path).parent.as_posix()
        launches.setdefault(parent, []).append(relative_path)
    stream.n_launches = len(launches)

    for launch_path in sorted(launches):
        tb_paths = sorted(
            launches[launch_path],
            key=lambda path: _tb_block_coord_sort_key(Path(path)),
        )
        per_tb: list[list[tuple[int, int]]] = []
        for tb_path in tb_paths:
            stream.n_threadblocks += 1
            tb = threadblock_cls()
            tb.ParseFromString(read_payload(tb_path))
            tb_accesses: list[tuple[int, int]] = []
            for warp_index in sorted(tb.warps.keys()):
                for inst in tb.warps[warp_index].instructions:
                    if not inst.HasField("memory_op_class"):
                        continue
                    if not inst.addresses:
                        continue
                    and_mask = inst.active_mask & inst.predicate_mask
                    if and_mask == 0:
                        continue
                    op_class = int(inst.memory_op_class)
                    for addr_record in inst.addresses:
                        # Effective per-address disposition (address-carrying
                        # contract). address_space wins when present: GLOBAL/
                        # GENERIC is L2 traffic (a global atomic or LDGSTS global
                        # source enters the stream), SHARED/LOCAL is not; absent
                        # falls back to the instruction-level op_class membership.
                        # The GLOBAL/else bucket is likewise keyed on effective
                        # space so ATOMIC/ASYNC_COPY are not mis-counted as local.
                        if addr_record.HasField("address_space"):
                            space = addr_record.address_space
                            is_l2 = space in _L2_ADDRESS_SPACES
                            bucket_global = space in (
                                _MEMORY_SPACE_GLOBAL,
                                _MEMORY_SPACE_GENERIC,
                            )
                        else:
                            is_l2 = op_class in op_class_set
                            bucket_global = op_class == _MEMORY_OP_CLASS_GLOBAL
                        address_mask = import_trace_contract._address_lane_mask(
                            addr_record, and_mask
                        )
                        recon = import_trace_contract._reconstruct_lane_vector(
                            addr_record, address_mask
                        )
                        lane_vector = recon[0]
                        if lane_vector is None:
                            continue
                        data_width = (
                            addr_record.data_width
                            if addr_record.data_width
                            else inst.data_width
                        )
                        if data_width <= 0:
                            continue
                        for lane_index, lane_addr in enumerate(lane_vector):
                            if not ((address_mask >> lane_index) & 1):
                                continue
                            if lane_addr == import_trace_contract.LANE_INACTIVE_SENTINEL:
                                continue
                            access_start = int(lane_addr)
                            first_sector = access_start // SECTOR_BYTES
                            last_sector = (
                                access_start + int(data_width) - 1
                            ) // SECTOR_BYTES
                            if not is_l2:
                                stream.n_excluded_op_class += (
                                    last_sector - first_sector + 1
                                )
                                stream.excluded_op_class_counts[op_class] = (
                                    stream.excluded_op_class_counts.get(op_class, 0)
                                    + (last_sector - first_sector + 1)
                                )
                                continue
                            for sector_n in range(first_sector, last_sector + 1):
                                line_id = (sector_n * SECTOR_BYTES) // LINE_BYTES
                                sector_offset = sector_n % SECTORS_PER_LINE
                                tb_accesses.append((line_id, sector_offset))
                                stream.included_op_class_counts[op_class] = (
                                    stream.included_op_class_counts.get(op_class, 0)
                                    + 1
                                )
                                if bucket_global:
                                    stream.n_global += 1
                                else:
                                    stream.n_local += 1
            per_tb.append(tb_accesses)

        if cta_interleave and len(per_tb) > 1:
            max_len = max(len(s) for s in per_tb)
            for i in range(max_len):
                for seq in per_tb:
                    if i < len(seq):
                        stream.line_ids.append(seq[i][0])
                        stream.sector_offsets.append(seq[i][1])
        else:
            for seq in per_tb:
                for line_id, sector_offset in seq:
                    stream.line_ids.append(line_id)
                    stream.sector_offsets.append(sector_offset)

    stream.n_accesses = len(stream.line_ids)
    # Detect any retained binding mutation before the materialized stream escapes.
    import_trace_contract._import_pb_modules()
    return stream


def decode_cell_access_stream(
    trace_dir: Path,
    *,
    op_classes: Iterable[int] = DEFAULT_L2_OP_CLASSES,
    cta_interleave: bool = False,
) -> AccessStream:
    """Decode one lexical NVBit trace directory into an ordered access stream."""

    paths = [
        path.relative_to(trace_dir).as_posix()
        for launch_dir in _iter_launch_dirs(trace_dir)
        for path in sorted(
            launch_dir.glob("*.pb"), key=_tb_block_coord_sort_key
        )
    ]
    return _decode_access_stream_sources(
        paths,
        lambda relative_path: (trace_dir / relative_path).read_bytes(),
        op_classes=op_classes,
        cta_interleave=cta_interleave,
    )


def decode_admitted_access_stream(
    sources: Any,
    *,
    op_classes: Iterable[int] = DEFAULT_L2_OP_CLASSES,
    cta_interleave: bool = False,
) -> AccessStream:
    """Decode only bytes exposed by one retained admitted-trace capability."""

    paths = sources.threadblock_paths
    stream = _decode_access_stream_sources(
        paths,
        lambda relative_path: sources.read_threadblock_snapshot(
            relative_path
        ).payload,
        op_classes=op_classes,
        cta_interleave=cta_interleave,
    )
    sources.verify_unchanged()
    return stream


# --------------------------------------------------------------------------
# Sectored set-associative cache replay (the headline mechanistic model).
# --------------------------------------------------------------------------
def wilson_interval(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (hit ratio)."""
    if total <= 0:
        return (0.0, 1.0)
    p = hits / total
    z2 = z * z
    denom = 1.0 + z2 / total
    centre = (p + z2 / (2.0 * total)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class ReplayResult:
    predicted_hit_ratio: float
    hits: int
    total_accesses: int
    cold_misses: int
    conflict_misses: int
    capacity_misses: int
    ci_low: float
    ci_high: float

    def as_dict(self) -> dict[str, Any]:
        total_misses = self.cold_misses + self.conflict_misses + self.capacity_misses
        return {
            "predicted_hit_ratio": self.predicted_hit_ratio,
            "hits": self.hits,
            "total_accesses": self.total_accesses,
            "miss_breakdown": {
                "cold": self.cold_misses,
                "conflict": self.conflict_misses,
                "capacity": self.capacity_misses,
                "total": total_misses,
            },
            "uncertainty_interval": {
                "method": "wilson_95",
                "ci_low": self.ci_low,
                "ci_high": self.ci_high,
            },
        }


def replay_access_stream(
    line_ids: Sequence[int],
    sector_offsets: Sequence[int],
    config: CacheReplayConfig,
    *,
    classify_misses: bool = True,
    set_sample_modulus: int | None = None,
) -> ReplayResult:
    """Replay an ordered sector-access stream through a sectored
    set-associative cache and return the modeled L2 hit ratio.

    Set sampling: when ``set_sample_modulus`` is given, only accesses whose
    addressed set satisfies ``set_key % modulus == 0`` are processed. Because a
    set-associative cache's sets are independent, the sampled sets are replayed
    EXACTLY (no approximation within a set); the hit ratio over the sampled
    accesses is an unbiased estimator of the full-cache hit ratio, and the
    returned ``total_accesses`` is the number of sampled (processed) accesses.
    Sampling forces ``classify_misses=False`` (the conflict/capacity split uses a
    global fully-associative reference that is incompatible with set sampling).

    Hit semantics (sectored cache): a sector access hits iff its 128 B line tag
    is resident in the addressed set AND that sector's valid bit is set. A
    resident line with the requested sector not yet valid is a sector-fill
    miss. A non-resident line is a line miss that allocates a way (evicting the
    replacement victim when the set is full).

    Miss classification (the classic three-C split, at sector granularity):
      * cold      -- the (line, sector) pair has never been touched before.
      * conflict  -- a previously-seen sector missed, but a fully-associative
                     cache of the same total capacity would still hold the line.
      * capacity  -- a previously-seen sector missed and the equal-capacity
                     fully-associative reference would also miss it.
    Hits + cold + conflict + capacity == total accesses.
    """
    n = len(line_ids)
    if set_sample_modulus is not None:
        classify_misses = False
    num_slices = config.num_slices
    num_sets = config.num_sets_per_slice
    assoc = config.associativity
    policy = config.replacement_policy
    slice_hash = SLICE_HASH_REGISTRY[config.slice_hash_id]
    prefetch_next_line = config.prefetch == "next_line"

    # Each set is a list of ways. A way is a mutable list:
    #   [tag, valid_mask, recency_or_rrpv]
    # recency_or_rrpv holds an LRU timestamp (LRU/BIP) or an RRPV (SRRIP).
    sets: dict[int, list] = {}

    # cold detection: line_id -> bitmask of sectors ever touched globally.
    ever_seen: dict[int, int] = {} if classify_misses else {}

    # Equal-capacity fully-associative LINE reference for the conflict/capacity
    # split. Maintained in lockstep on every access; an OrderedDict gives O(1)
    # LRU move-to-end and O(1) eviction of the oldest line.
    ref_cap_lines = config.total_lines
    ref_lru: "OrderedDict[int, None]" = OrderedDict()

    hits = 0
    cold = 0
    conflict = 0
    capacity = 0
    clock = 0
    bip_counter = 0
    processed = 0

    for i in range(n):
        line_id = line_ids[i]
        sector_offset = sector_offsets[i]
        sbit = 1 << sector_offset
        slice_id = slice_hash(line_id, num_slices)
        set_idx = (line_id // num_slices) % num_sets
        set_key = slice_id * num_sets + set_idx
        if set_sample_modulus is not None and (set_key % set_sample_modulus) != 0:
            continue
        processed += 1
        ways = sets.get(set_key)
        clock += 1

        is_cold = False
        if classify_misses:
            seen_mask = ever_seen.get(line_id, 0)
            is_cold = not (seen_mask & sbit)
            ever_seen[line_id] = seen_mask | sbit

        resident_way = None
        if ways is not None:
            for way in ways:
                if way[0] == line_id:
                    resident_way = way
                    break

        if resident_way is not None and (resident_way[1] & sbit):
            # sector hit
            hits += 1
            if policy == "SRRIP":
                resident_way[2] = 0
            else:
                resident_way[2] = clock
        elif resident_way is not None:
            # line resident, sector not yet valid -> sector-fill miss
            resident_way[1] |= sbit
            if policy == "SRRIP":
                resident_way[2] = 0
            else:
                resident_way[2] = clock
            if classify_misses:
                if is_cold:
                    cold += 1
                else:
                    # sector re-touch on a line that survived in the set-assoc
                    # cache but whose sector bit was reset by a prior refill.
                    if line_id in ref_lru:
                        conflict += 1
                    else:
                        capacity += 1
        else:
            # line miss -> allocate a way (evict victim if full)
            if ways is None:
                ways = []
                sets[set_key] = ways
            _install_way(ways, line_id, sbit, assoc, policy, clock, bip_counter)
            if policy == "BIP":
                bip_counter += 1
            if classify_misses:
                if is_cold:
                    cold += 1
                elif line_id in ref_lru:
                    conflict += 1
                else:
                    capacity += 1

        # lockstep fully-associative LINE reference update (O(1) LRU)
        if classify_misses:
            if line_id in ref_lru:
                ref_lru.move_to_end(line_id)
            else:
                if len(ref_lru) >= ref_cap_lines:
                    ref_lru.popitem(last=False)
                ref_lru[line_id] = None

        if prefetch_next_line and resident_way is None:
            _prefetch_next_line(
                sets, line_id, num_slices, num_sets, assoc, policy, clock,
                slice_hash,
            )

    if not classify_misses:
        # cold/conflict/capacity not tracked; fold all misses into capacity
        # bucket is wrong, so leave them zero and only trust hit ratio.
        cold = conflict = capacity = 0

    total = processed if set_sample_modulus is not None else n
    hit_ratio = (hits / total) if total else 0.0
    lo, hi = wilson_interval(hits, total)
    return ReplayResult(
        predicted_hit_ratio=hit_ratio,
        hits=hits,
        total_accesses=total,
        cold_misses=cold,
        conflict_misses=conflict,
        capacity_misses=capacity,
        ci_low=lo,
        ci_high=hi,
    )


def _install_way(ways, line_id, sbit, assoc, policy, clock, bip_counter):
    """Insert a freshly-fetched line into the set, evicting per policy if full."""
    if len(ways) < assoc:
        if policy == "SRRIP":
            ways.append([line_id, sbit, _SRRIP_INSERT_RRPV])
        elif policy == "BIP":
            # Bimodal: mostly insert at LRU recency (below any existing way),
            # occasionally at MRU. Empty set => insert at MRU (clock).
            mru = (bip_counter % _BIP_MRU_INSERT_PERIOD) == 0
            recency = clock if (mru or not ways) else _min_recency(ways) - 1
            ways.append([line_id, sbit, recency])
        else:  # LRU
            ways.append([line_id, sbit, clock])
        return

    if policy == "SRRIP":
        # find a way with RRPV == max; if none, age all until one reaches max.
        while True:
            victim_idx = None
            for idx, w in enumerate(ways):
                if w[2] >= _SRRIP_RRPV_MAX:
                    victim_idx = idx
                    break
            if victim_idx is not None:
                ways[victim_idx] = [line_id, sbit, _SRRIP_INSERT_RRPV]
                return
            for w in ways:
                w[2] += 1
    else:
        # LRU / BIP eviction: evict least-recently-used (min recency).
        victim_idx = 0
        victim_rec = ways[0][2]
        for idx in range(1, len(ways)):
            if ways[idx][2] < victim_rec:
                victim_rec = ways[idx][2]
                victim_idx = idx
        if policy == "BIP":
            mru = (bip_counter % _BIP_MRU_INSERT_PERIOD) == 0
            recency = clock if mru else _min_recency(ways) - 1
        else:
            recency = clock
        ways[victim_idx] = [line_id, sbit, recency]


def _min_recency(ways) -> int:
    m = ways[0][2]
    for w in ways[1:]:
        if w[2] < m:
            m = w[2]
    return m


def _prefetch_next_line(sets, line_id, num_slices, num_sets, assoc, policy,
                        clock, slice_hash):
    nxt = line_id + 1
    slice_id = slice_hash(nxt, num_slices)
    set_idx = (nxt // num_slices) % num_sets
    set_key = slice_id * num_sets + set_idx
    ways = sets.get(set_key)
    if ways is None:
        ways = []
        sets[set_key] = ways
    for way in ways:
        if way[0] == nxt:
            return
    _install_way(ways, nxt, 1, assoc, policy, clock, 0)


def replay_cell(
    trace_dir: Path,
    config: CacheReplayConfig,
    *,
    classify_misses: bool = True,
    cta_interleave: bool = False,
) -> dict[str, Any]:
    """Decode a cell's trace and replay it; return a schema-shaped result dict.

    The returned dict carries the predicted hit ratio, the three-C miss
    breakdown, the full cache-parameter provenance, and the Wilson uncertainty
    interval. Range invariant: ``0 <= predicted_hit_ratio <= 1``.
    """
    stream = decode_cell_access_stream(trace_dir, cta_interleave=cta_interleave)
    result = replay_access_stream(
        stream.line_ids, stream.sector_offsets, config,
        classify_misses=classify_misses,
    )
    out = result.as_dict()
    out["cache_parameters"] = config.provenance()
    out["trace_stats"] = {
        "n_accesses": stream.n_accesses,
        "n_global": stream.n_global,
        "n_local": stream.n_local,
        "n_excluded_op_class": stream.n_excluded_op_class,
        "n_launches": stream.n_launches,
        "n_threadblocks": stream.n_threadblocks,
    }
    return out
