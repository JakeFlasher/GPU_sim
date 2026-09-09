// nvbit_tracer/metric_extractor/common.h
//
// Shared declarations for the on-device metric extractor.
//
// LaunchMetrics is the per-launch device-side aggregator that injected device
// functions atomicAdd into. The skeleton's empty struct has been replaced by
// the Tier-1 instruction-class histogram (15 fixed buckets, ≤ 120 B per
// launch). Subsequent rounds add sector-efficiency counters, shared-bank
// conflict tally, and the branch active-mask histogram alongside this one
// while keeping the per-launch device allocation under the 4 KB AC-11 / AC-15
// bound.

#ifndef NVBIT_TRACER_METRIC_EXTRACTOR_COMMON_H_
#define NVBIT_TRACER_METRIC_EXTRACTOR_COMMON_H_

#include <stdint.h>

// Schema-v4 context_id and stream_id are CUDA Driver identities local to the
// captured process. Stream topology may be compared across runs, but raw
// numeric IDs cannot establish cross-run correspondence.
#define METRIC_EXTRACTOR_SCHEMA_VERSION 4
#define METRIC_EXTRACTOR_TOOL_VERSION   "0.0.16-driver-stream-identity"

// Default cap on the per-process pool of LaunchMetrics slots. The slot pool
// is allocated lazily in nvbit_tool_init() (which is the only NVBit
// lifecycle slot where CUDA-API allocation is safe — see nvbit.h:278-282
// describing the deadlock when CUDA APIs run inside nvbit_at_ctx_init).
//
// 4096 slots is large enough that real evaluator runs do not have to
// recycle slot indices for short kernel bursts, while keeping the worst-
// case device allocation comfortably bounded (4096 * 120 B ≈ 480 KB total).
// The cap is configurable at runtime via the
// METRIC_EXTRACTOR_MAX_LAUNCH_SLOTS environment variable; values below 1
// fall back to this default.
#define METRIC_EXTRACTOR_DEFAULT_MAX_LAUNCH_SLOTS 4096
#define METRIC_EXTRACTOR_MAX_LAUNCH_SLOTS_ENV     "METRIC_EXTRACTOR_MAX_LAUNCH_SLOTS"

// AC-15 bounded-memory mechanics (Round 4 T38).
//
// Per-launch artifact cap (JSON-L line bytes): the writer enforces
// this before flushing. Records exceeding the cap are marked
// ``substatus.truncated = true``. Default 10 MB.
#define METRIC_EXTRACTOR_PER_LAUNCH_ARTIFACT_CAP_BYTES   (10ull * 1024ull * 1024ull)

// Per-process device-memory budget for the slot pool. nvbit_tool_init()
// refuses to allocate a pool larger than this. Default 512 MiB.
#define METRIC_EXTRACTOR_PER_PROCESS_BUDGET_BYTES       (512ull * 1024ull * 1024ull)

// Tier-2 per-PC table cap (per AC-15 documented formula). The Tier-1
// LaunchMetrics struct lives under 4 KB; when Tier-2 cache-prediction
// adds an optional per-PC table, the table is sized
// ``min(static_memory_instruction_count, MAX_INSTRUMENTED_INSTRUCTIONS)
// * sizeof(per_pc_bucket)``. Tier-2 is deferred per DEC-2, so the
// constant is defined here as the documented cap and reserved for the
// future per-PC accumulator. No code path consumes it yet.
#define METRIC_EXTRACTOR_MAX_INSTRUMENTED_INSTRUCTIONS  1024

// Canonical instruction-class bucket indices for the per-launch histogram.
// Bucket names match the JSONL_SCHEMA.md sparse-map class names and the
// AC-7 contract (TCGen05 / TMA / DSMEM held in their own buckets, an
// explicit unknown bucket for forward-compat across CUDA SASS spelling
// changes). New buckets must be appended at the end so existing slot
// layouts remain stable across rounds; the JSON-L emitter is responsible
// for translating the integer id back to the canonical bucket name.
enum InstructionClass {
    INSTR_CLASS_FFMA      = 0,
    INSTR_CLASS_MUFU      = 1,
    INSTR_CLASS_LDG       = 2,
    INSTR_CLASS_STG       = 3,
    INSTR_CLASS_LDS       = 4,
    INSTR_CLASS_STS       = 5,
    INSTR_CLASS_HMMA      = 6,
    INSTR_CLASS_WGMMA     = 7,
    INSTR_CLASS_TCGEN05   = 8,
    INSTR_CLASS_TMA       = 9,
    INSTR_CLASS_DSMEM     = 10,
    INSTR_CLASS_BRA       = 11,
    INSTR_CLASS_BARRIER   = 12,
    INSTR_CLASS_WARPSYNC  = 13,
    INSTR_CLASS_UNKNOWN   = 14,
    NUM_INSTRUCTION_CLASSES = 15,
};

// Fixed cap on the per-launch TMA descriptor handle copy. Sized to
// comfortably cover the largest TMA descriptor handle the upstream NVBit
// samples document (the largest documented `cp.async.bulk.tensor` /
// `cp.async.bulk` descriptor is well under 128 B), while keeping the
// per-launch slot pool footprint bounded for the AC-11 / AC-15 caps.
// `tma_param_handle_size` records the actual size of the captured handle;
// callers must clamp by this cap before copying. A future descriptor that
// exceeds the cap surfaces as `substatus.truncated = true` once T38 lands.
#define METRIC_EXTRACTOR_TMA_HANDLE_CAP 128

// Number of buckets in the branch active-mask histogram. The bucket is
// indexed by the popcount of the warp-collective predicate mask, which
// ranges over [0, 32] inclusive: bucket 0 counts warp-instances where
// every active lane evaluated the branch predicate to false, bucket 32
// counts fully-active branches (unconditional or always-true), and the
// canonical AC-4 16-active divergent branch lands in bucket 16.
#define METRIC_EXTRACTOR_BRANCH_ACTIVE_LANES_BUCKETS 33

// Per-launch aggregator. Allocated lazily on first use as a fixed-cap pool
// in __managed__ memory, indexed by a per-launch slot pointer that the host
// hands to instrumented device functions via nvbit_set_at_launch +
// nvbit_add_call_arg_launch_val64. Two launches in flight at the same time
// do not stomp on each other because the launch-mutex serialises slot
// reset / read; the slot is reset on launch-enter, atomicAdd-ed by injected
// functions during kernel execution, and read back after
// cudaDeviceSynchronize() on launch-exit.
//
// Per-launch byte budget so the per-process slot-pool memory cost is
// transparent and the AC-11 / AC-15 caps are honored. The field math:
//   NUM_INSTRUCTION_CLASSES * sizeof(uint64_t)            = 15 * 8 = 120 B
//   tma_count                                              = 8 B
//   tma_sample_valid / tma_sample_opcode_id /
//   tma_param_handle_size                                  = 3 * 4 B = 12 B
//   tma_sample_handle[TMA_HANDLE_CAP]                      = 128 B
//   (internal 4 B alignment pad after the uint8 handle to
//    restore 8 B alignment for the next uint64 field)      = 4 B
//   sector_{load,store}_{useful,transferred}_bytes         = 4 * 8 B = 32 B
//   bank_conflict_total                                    = 8 B
//   branch_active_lanes_hist[BRANCH_ACTIVE_LANES_BUCKETS]  = 33 * 8 = 264 B
//   saturated_counters                                     = 8 B
//   ---------------------------------------------------------------------
//   sizeof(LaunchMetrics)                                  = 584 B
// 584 B is exactly aligned to the struct's strictest member (uint64_t,
// 8 B), so no trailing pad is required. The runtime `slot pool
// allocated` log line reflects this padded size: 4096 * 584 = 2,392,064 B
// ≈ 2.28 MB total managed memory, well under both the 4 KB-per-launch
// cap (AC-11) and the 512 MB per-process budget (AC-15).
//
// The ``saturated_counters`` field tracks the number of per-launch
// counter atomicAdds that clamped at ``UINT64_MAX`` instead of wrapping.
// Every counter site in ``inject_funcs.cu`` routes through
// ``saturating_atomic_add_u64`` (see ``saturating_counter.h``); the
// helper bumps ``saturated_counters`` exactly once per overflow event.
// On Tier-1 sm_86 / sm_100a workloads this stays at zero — the
// instruction-class histogram would need > 10^19 events per bucket to
// hit ``UINT64_MAX`` — but the field is wired up real (not a literal
// ``0`` placeholder) so a future Tier-2 per-PC table that increments
// more densely can surface saturation visibly in the JSON-L.
//
// The TMA fields are populated by the sample_tma_descriptor injected device
// function for the FIRST TMA instruction observed per launch:
// `tma_sample_valid` is `atomicCAS`-flipped 0→1 so only one warp wins the
// sample slot; subsequent TMA instructions on the same launch still bump
// `tma_count` and the histogram's TMA bucket but do not overwrite the
// captured handle. On launch-exit (after `cudaDeviceSynchronize()`), the
// host decodes the captured handle via `nvbit_parse_tma_transfer_info` and
// emits the compact JSON-L descriptor summary.
//
// The sector-efficiency fields are populated by the count_global_mem_access
// injected device function on every instrumented LDG / STG. Per-warp the
// leader lane gathers the 32 lane addresses via __shfl_sync, computes the
// number of distinct 32-byte sectors hit by active+predicate-true lanes,
// then atomicAdds two values: useful_bytes = active_lanes * bytes_per_access
// (numerator of the efficiency ratio) and transferred_bytes =
// distinct_sectors * 32 (denominator). At launch-exit the host computes the
// ratio = useful_bytes / transferred_bytes when transferred > 0 and emits
// it as sector_efficiency_global_{load,store}; otherwise emits null. DSMEM
// ops are excluded by the instrument-pass `getMemorySpace() == GLOBAL`
// filter, so this aggregator only sees plain global memory traffic.
//
// The branch active-mask histogram is populated by the
// count_branch_active_mask_cb injected device function on every
// instrumented INSTR_CLASS_BRA instruction (BRA / JMP / EXIT / BRX /
// CALL / RET, etc. — every opcode the classifier routes to the `bra`
// bucket). Per warp-instance the leader lane computes
// popcount(__ballot_sync(__activemask(), predicate)) — the number of
// predicate-true active lanes — and atomicAdds 1 into
// branch_active_lanes_hist[that count]. The bucket index is bounded
// to [0, 32] by construction (popcount of a 32-bit mask), so the array
// has 33 buckets. Bucket 0 counts predicate-false-everywhere warp
// instances; bucket 32 counts fully-active warp instances
// (unconditional / always-true branches); bucket 16 is the AC-4
// canonical "predicated branch with 16 active lanes" test bucket.
struct LaunchMetrics {
    uint64_t instruction_class_histogram[NUM_INSTRUCTION_CLASSES];
    uint64_t tma_count;
    uint32_t tma_sample_valid;
    uint32_t tma_sample_opcode_id;
    uint32_t tma_param_handle_size;
    uint8_t  tma_sample_handle[METRIC_EXTRACTOR_TMA_HANDLE_CAP];
    uint64_t sector_load_useful_bytes;
    uint64_t sector_load_transferred_bytes;
    uint64_t sector_store_useful_bytes;
    uint64_t sector_store_transferred_bytes;
    uint64_t bank_conflict_total;
    uint64_t branch_active_lanes_hist[METRIC_EXTRACTOR_BRANCH_ACTIVE_LANES_BUCKETS];
    uint64_t saturated_counters;
    // Tier-2 address-dump per-launch buffer pointer. NULL when
    // METRIC_EXTRACTOR_TIER2_ADDRESS_DUMP is unset / 0. Populated at
    // launch-enter when the env gate is on by pointing into the
    // parallel TierTwoBuffer pool allocated lazily in
    // nvbit_tool_init. Tier-2 device-side capture reaches its
    // per-launch buffer through this pointer.
    uint64_t tier2_buffer_ptr;
};

// --------------------------------------------------------------------------
// Tier-2 address-dump per-launch capture buffer.
//
// The Tier-2 producer captures one CompactAccessRecord per warp-
// collective global memory access. Records are staged in a per-launch
// fixed-capacity ring (CompactAccessRecord::CAPACITY) inside
// __managed__ memory so the host can read them after
// cudaDeviceSynchronize. Over-capacity writes increment dropped_count
// and set truncated to 1; they do NOT corrupt the in-buffer prefix.
// Host-side serialization repacks the staging records into the
// variable-length CompactAccessRecordV1 wire format documented in the
// Tier-2 sidecar schema (record_len = align8(56 + 8 * sector_count)).
//
// Per-slot footprint: CompactAccessRecord is fixed-size (~958 B with
// sector_ids fixed at 64 entries + lane_addresses fixed at 32
// entries + a 128 B TMA descriptor handle that is zero on non-TMA
// records and carries the captured descriptor on op_kind=7 entries).
// At TIER2_RECORDS_PER_LAUNCH = 524288 records, the buffer is
// 524288 * 958 + 32 B header ~= 478.9 MiB per slot. Naive
// multiplication against the default 4096 slot-pool cap would
// require ~1.87 TiB of managed memory when env=1, which exceeds
// every supported host configuration. The combined Tier-1 + Tier-2
// per-slot footprint is therefore clamped against the per-process
// budget in nvbit_tool_init() when env=1, so the actual allocated
// slot count shrinks to fit the budget. The compile-time default
// budget is 512 MiB; operators can lift it via the
// METRIC_EXTRACTOR_PER_PROCESS_BUDGET_BYTES env var when running a
// pipeline that needs more concurrent instrumented launches than
// the default budget admits. Important nuance: the metric_extractor
// REFUSES to reuse a slot once g_accepted_slot_index reaches
// g_max_launch_slots (the launch-enter path explicitly routes to
// per_process_budget_exceeded rather than overwriting an
// already-flushed launch's data). So allocating enough slots to
// cover the targeted launch count is required; the smoke harness
// lifts the env-override to 4 GiB and caps
// METRIC_EXTRACTOR_MAX_LAUNCH_SLOTS to a small bounded number
// (8 by default) so the budget clamp picks a slot count large
// enough for the warmup-plus-measured launch sequence. When env=0
// the Tier-2 pool is never allocated and zero managed memory is
// spent.
// The empirical record-count evidence from the calibration smoke
// pipeline: the two warm/cold kernels emit ~49k and ~328k records
// per launch respectively (RoPE+KV-cache loops over the full cache
// tensor); a smaller cap silently truncated such traces and
// surfaced the truncation as `trace_budget_exceeded` at the
// predictor, blocking measured smoke records. The lane_addresses
// tail is populated unconditionally by the device-side capture; the
// host emits the PerLaneAddressRecordV1 wire-format record only
// when METRIC_EXTRACTOR_TIER2_PER_LANE=1. The tma_handle tail is
// populated only on TMA captures (op_kind=7); the host expands the
// handle into per-element addresses via
// nvbit_parse_tma_src_addrs_with_flags / _dst_addrs_with_flags at
// serialization time.
// --------------------------------------------------------------------------

#define METRIC_EXTRACTOR_TIER2_RECORDS_PER_LAUNCH 524288
#define METRIC_EXTRACTOR_TIER2_SECTORS_PER_RECORD 64

// Device-side staging record. The host repacks into the
// variable-length CompactAccessRecordV1 wire format when writing the
// sidecar binary; the fixed-size sector_ids tail is sized to the
// maximum sector_count the schema admits in v1.0.
struct CompactAccessRecord {
    // Identity fields.
    uint64_t sequence_id;
    uint64_t pc_or_instruction_id;
    uint32_t cta_x;
    uint32_t cta_y;
    uint32_t cta_z;
    // Per-warp identity.
    uint16_t warp_id;
    uint16_t bytes_per_access;
    uint32_t lane_mask;
    uint32_t active_mask;
    // Enum fields (small, packed into a single uint32 word).
    uint8_t  op_kind;
    uint8_t  cache_modifier;
    uint8_t  address_space;
    uint8_t  record_flags;
    // Variable-length sector list stored as count + fixed-size tail.
    // The host serializes only sector_count entries when writing.
    uint16_t sector_count;
    // L1 access policy on global loads. Disambiguates the
    // Ampere-folded `.cg` case where SASS surfaces as
    // `LDG.E.STRONG.GPU` (`cache_modifier=0` indistinguishable
    // from a default L1+L2 load). Values:
    //   0 = l1_unknown   — producer did not classify (legacy /
    //                     atomic / cp_async / TMA / spill paths
    //                     where global-load policy field is not
    //                     meaningful).
    //   1 = l1_admitted  — producer classified an ordinary L1+L2
    //                     read path (no `.cg` / no `STRONG.GPU`).
    //   2 = l1_bypassed  — producer detected an L1-bypass cache
    //                     policy on the global load (`.CG` SASS
    //                     suffix on sm_90+ OR `STRONG.GPU` Ampere
    //                     fold on sm_86).
    // Occupies what was previously the low byte of the reserved
    // u16 padding at compact-record wire offset 50; the high byte
    // (wire offset 51) and the u32 at wire offsets 52..55 stay
    // zero. Old sidecars whose wire bytes at offset 50 are zero
    // continue to parse with `l1_access_policy=0` (l1_unknown);
    // any nonzero value at offset 50 is admitted only if it
    // matches a defined enum value.
    uint8_t  l1_access_policy;
    uint8_t  _reserved_pad_hi;
    uint32_t _reserved_pad32;
    uint64_t sector_ids[METRIC_EXTRACTOR_TIER2_SECTORS_PER_RECORD];
    // Per-lane base addresses for the warp-collective access.
    // Populated unconditionally by the device-side capture (the
    // 32-lane __shfl_sync gather is required for sector-set
    // derivation anyway); inactive lanes carry the schema sentinel
    // 0xffffffffffffffff. The host emits the
    // PerLaneAddressRecordV1 wire record only when the per-lane
    // env gate (METRIC_EXTRACTOR_TIER2_PER_LANE=1) is on.
    uint64_t lane_addresses[32];
    // TMA descriptor handle bytes captured by the per-event TMA
    // capture path (op_kind=7 records). Zero on every non-TMA
    // record. The host serializer expands the handle into per-
    // element global addresses via NVBit's
    // nvbit_parse_tma_src_addrs_with_flags / _dst_addrs_with_flags
    // at serialization time; the resulting sector IDs populate
    // sector_ids / sector_count on the wire-format compact record.
    uint32_t tma_handle_size;
    uint32_t _tma_reserved_pad;
    uint8_t  tma_handle[METRIC_EXTRACTOR_TMA_HANDLE_CAP];
};

// Per-launch Tier-2 buffer. The host-side launch-enter resets the
// record_count / sequence_id_counter / dropped_count / truncated
// fields before exposing the buffer pointer through LaunchMetrics
// for the duration of the launch.
struct TierTwoBuffer {
    uint64_t            record_count;
    uint64_t            sequence_id_counter;
    uint64_t            dropped_count;
    uint64_t            truncated;
    CompactAccessRecord records[METRIC_EXTRACTOR_TIER2_RECORDS_PER_LAUNCH];
};

#endif  // NVBIT_TRACER_METRIC_EXTRACTOR_COMMON_H_
