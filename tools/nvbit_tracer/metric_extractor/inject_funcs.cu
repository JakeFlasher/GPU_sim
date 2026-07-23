// nvbit_tracer/metric_extractor/inject_funcs.cu
//
// Injected device functions invoked at every instrumented SASS instruction.
// Compiled with `-Xptxas -astoolspatch --keep-device-functions` so NVBit can
// patch them into the host application's CUfunctions at load time.
//
// Five device callbacks live here. The host-side instrumentation pass picks
// one or two per instruction based on `instr->isTMAMem()`, whether the
// instruction is a global load/store, whether the instruction is a
// plain-shared load/store, and whether the instruction is a branch-class
// instruction:
//
// count_metric (used for every non-TMA instrumented instruction):
//   1. predicate guard          — nvbit_add_call_arg_guard_pred_val
//   2. instruction-class id     — nvbit_add_call_arg_const_val32, derived
//                                  per-instr by the host classifier from
//                                  blackwell_opcode.h / ampere_opcode.h.
//   3. per-launch slot pointer  — nvbit_add_call_arg_launch_val64, loaded
//                                  from the value the host pushed via
//                                  nvbit_set_at_launch(ctx, func, slot_ptr).
//
// count_shared_bank_conflicts_cb (added IN ADDITION TO count_metric on
// every instrumented plain-shared LDS / STS, so the same instruction
// contributes to both the instruction-class histogram and the bank-
// conflict aggregator). The instrument pass installs this callback ONLY
// when `instr->isLoad() || instr->isStore()` AND `instr->getMemorySpace()
// == InstrType::MemorySpace::SHARED`; DSMEM ops have memspace
// DISTRIBUTED_SHARED and are filtered out by construction (the 32-bank
// model is wrong for DSMEM — AC-7 negative-test 2).
//   1. predicate guard          — nvbit_add_call_arg_guard_pred_val
//   2. addr                     — nvbit_add_call_arg_mref_addr64, the
//                                  per-thread shared address used by
//                                  the leader lane to gather every
//                                  lane's address via __shfl_sync and
//                                  feed count_shared_bank_conflicts().
//   3. per-launch slot pointer  — nvbit_add_call_arg_launch_val64.
//
// count_global_mem_access (added IN ADDITION TO count_metric on every
// instrumented global LDG / STG, so the same instruction contributes to
// both the instruction-class histogram and the sector-efficiency
// aggregator). The instrument pass installs this callback ONLY when
// `instr->isLoad() || instr->isStore()` AND `instr->getMemorySpace() ==
// InstrType::MemorySpace::GLOBAL`; DSMEM ops have memspace
// DISTRIBUTED_SHARED and are filtered out by construction.
//   1. predicate guard          — nvbit_add_call_arg_guard_pred_val
//   2. is_store (uint32 const)  — nvbit_add_call_arg_const_val32, 0 for
//                                  LDG and 1 for STG so the same
//                                  callback can split into the right
//                                  pair of per-launch counters.
//   3. bytes_per_access         — nvbit_add_call_arg_const_val32 from
//                                  `Instr::getSize()`; the per-lane
//                                  bytes-per-access used to compute
//                                  `useful_bytes`.
//   4. addr                     — nvbit_add_call_arg_mref_addr64, the
//                                  per-thread memory address used by
//                                  the leader lane to gather every
//                                  lane's address via __shfl_sync.
//   5. per-launch slot pointer  — nvbit_add_call_arg_launch_val64.
//
// count_branch_active_mask_cb (added IN ADDITION TO count_metric on every
// instrumented instruction the classifier routes to INSTR_CLASS_BRA —
// BRA / JMP / EXIT / BRX / CALL / RET / etc. — so each branch warp-instance
// contributes to both the instruction-class histogram and the branch
// active-lanes histogram). The bucket index is the popcount of the
// warp-collective predicate mask, in [0, 32]; bucket 0 = predicate-false
// everywhere, bucket 32 = fully-active branch, bucket 16 = AC-4 canonical
// 16-active divergent branch.
//   1. predicate guard          — nvbit_add_call_arg_guard_pred_val
//   2. per-launch slot pointer  — nvbit_add_call_arg_launch_val64.
//
// sample_tma_descriptor (used only when `instr->isTMAMem()` returns true):
//   1. predicate guard          — nvbit_add_call_arg_guard_pred_val
//   2. instruction-class id     — nvbit_add_call_arg_const_val32, always
//                                  INSTR_CLASS_TMA on this path.
//   3. opcode id                — nvbit_add_call_arg_const_val32, host
//                                  string→id mapping so the launch-exit
//                                  decoder can re-resolve the SASS mnemonic
//                                  for `nvbit_parse_tma_transfer_info`.
//   4. per-launch slot pointer  — nvbit_add_call_arg_launch_val64.
//   5. TMA param handle ptr     — pushed by
//                                  nvbit_add_call_arg_tma_param_handle_and_size,
//                                  which pushes the handle pointer (uint64)
//                                  followed by its size (uint32).
//   6. TMA param handle size    — the second arg pushed by (5).
//
// Per-warp leader-lane atomicAdd semantics: each instrumented instruction
// emits one warp-level increment into the histogram bucket selected by
// instr_class. Per-thread granularity would conflate predication / lane
// mask shape with raw instruction issue, which is not what AC-7 / AC-4
// describe. Out-of-range instr_class is clamped to INSTR_CLASS_UNKNOWN so
// future SASS opcode spellings the classifier does not recognise land in
// the explicit unknown bucket rather than corrupting a neighbouring counter
// or being silently dropped.
//
// Structurally identical to the upstream `opcode_hist/inject_funcs.cu`
// leader-lane pattern: single warp-collective ballot for the active mask,
// single ballot for the predicate mask, leader-lane comparison against
// `get_laneid()`, then a single saturating add into the per-launch counter
// slot. The saturating helper (``saturating_atomic_add_u64`` in
// ``saturating_counter.h``) clamps at ``UINT64_MAX`` on overflow and
// increments ``slot->saturated_counters`` exactly once per saturation
// event. Routing every per-launch counter atomicAdd through the helper
// instead of a raw ``atomicAdd`` is what the bounded-memory contract
// requires; a silent wrap on a Tier-2 per-PC table would make a kernel
// look benign after 2^64 events. The TMA-specific callback layers an
// ``atomicCAS``-gated sample copy on top of the same warp pattern so only
// the first warp per launch wins the 128-byte descriptor sample slot;
// subsequent TMA instructions still bump the bucket and the parallel
// ``tma_count`` (both via the saturating helper) but do not overwrite the
// sample.

#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"

#include "common.h"
#include "sector_counting.h"
#include "bank_counting.h"
#include "branch_counting.h"
#include "saturating_counter.h"

extern "C" __device__ __noinline__ void count_metric(int predicate,
                                                    int instr_class,
                                                    uint64_t p_metrics) {
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int predicate_mask = __ballot_sync(__activemask(), predicate);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    const int num_threads = __popc(predicate_mask);
    if (first_laneid == laneid && num_threads > 0 && p_metrics != 0ULL) {
        int bucket = instr_class;
        if (bucket < 0 || bucket >= (int)NUM_INSTRUCTION_CLASSES) {
            bucket = (int)INSTR_CLASS_UNKNOWN;
        }
        LaunchMetrics* slot = reinterpret_cast<LaunchMetrics*>(p_metrics);
        saturating_atomic_add_u64(
            &slot->instruction_class_histogram[bucket],
            1ULL, &slot->saturated_counters);
    }
}

extern "C" __device__ __noinline__ void sample_tma_descriptor(
    int predicate,
    int instr_class,
    int opcode_id,
    uint64_t p_metrics,
    uint64_t tma_param_handle,
    uint32_t tma_param_handle_size) {
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int predicate_mask = __ballot_sync(__activemask(), predicate);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    const int num_threads = __popc(predicate_mask);
    if (!(first_laneid == laneid && num_threads > 0 && p_metrics != 0ULL)) {
        return;
    }
    // Histogram bucket increment (mirrors count_metric for the TMA bucket
    // specifically so the JSON-L histogram view counts TMA instructions the
    // same way it counts every other class).
    int bucket = instr_class;
    if (bucket < 0 || bucket >= (int)NUM_INSTRUCTION_CLASSES) {
        bucket = (int)INSTR_CLASS_UNKNOWN;
    }
    LaunchMetrics* slot = (LaunchMetrics*)p_metrics;
    saturating_atomic_add_u64(
        &slot->instruction_class_histogram[bucket],
        1ULL, &slot->saturated_counters);
    // Parallel TMA counter so a downstream consumer can sanity-check the
    // histogram against the descriptor-sample path without cross-referencing
    // the histogram array layout.
    saturating_atomic_add_u64(&slot->tma_count, 1ULL,
                              &slot->saturated_counters);
    // First-warp wins the descriptor sample slot. The atomicCAS flips
    // tma_sample_valid from 0 to 1 exactly once per launch; every later TMA
    // instruction sees the value already 1 and skips the copy. The copy is
    // bounded by METRIC_EXTRACTOR_TMA_HANDLE_CAP so a future descriptor
    // larger than the cap surfaces as truncation (T38 follow-up); for now
    // the clamp prevents corrupting neighbouring fields.
    const unsigned prev_valid =
        atomicCAS(&slot->tma_sample_valid, 0u, 1u);
    if (prev_valid == 0u && tma_param_handle != 0ULL) {
        uint32_t copy_bytes = tma_param_handle_size;
        if (copy_bytes > (uint32_t)METRIC_EXTRACTOR_TMA_HANDLE_CAP) {
            copy_bytes = (uint32_t)METRIC_EXTRACTOR_TMA_HANDLE_CAP;
        }
        // memcpy is fine here — leader-lane execution is single-threaded
        // within the warp at this point. NVBit's TMA arg pusher hands a
        // device-readable handle pointer, so the byte-by-byte copy lands
        // in managed memory accessible from the host on launch-exit.
        const uint8_t* src = (const uint8_t*)tma_param_handle;
        for (uint32_t i = 0; i < copy_bytes; ++i) {
            slot->tma_sample_handle[i] = src[i];
        }
        slot->tma_sample_opcode_id   = (uint32_t)opcode_id;
        slot->tma_param_handle_size  = copy_bytes;
    }
    // Per-event Tier-2 capture path. Each TMA instruction (not just
    // the first-sample slot above) claims a unique sequence_id from
    // the unified counter and writes a CompactAccessRecord with
    // op_kind=7 plus the captured descriptor handle bytes. The host
    // serializer expands the handle into per-element global
    // addresses via NVBit's nvbit_parse_tma_src_addrs_with_flags /
    // _dst_addrs_with_flags at serialization time. When the env gate
    // is off, slot->tier2_buffer_ptr is null and this block is
    // unreachable; AC-13 byte-identity holds by construction.
    TierTwoBuffer* tier2 = reinterpret_cast<TierTwoBuffer*>(
        slot->tier2_buffer_ptr);
    if (tier2 == nullptr) {
        return;
    }
    unsigned long long seq_id =
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &tier2->sequence_id_counter),
                  1ULL);
    if (seq_id >= (unsigned long long)
                  METRIC_EXTRACTOR_TIER2_RECORDS_PER_LAUNCH) {
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &tier2->dropped_count),
                  1ULL);
        atomicExch(reinterpret_cast<unsigned long long*>(
                       &tier2->truncated),
                   1ULL);
        return;
    }
    CompactAccessRecord* rec = &tier2->records[seq_id];
    rec->sequence_id = (uint64_t)seq_id;
    rec->pc_or_instruction_id = (uint64_t)opcode_id;
    rec->cta_x = blockIdx.x;
    rec->cta_y = blockIdx.y;
    rec->cta_z = blockIdx.z;
    // Linearized warp id (mirrors count_tier2_global_mem_access).
    const unsigned linear_tid =
        ((threadIdx.z * blockDim.y) + threadIdx.y) * blockDim.x
        + threadIdx.x;
    rec->warp_id = (uint16_t)(linear_tid / 32u);
    rec->bytes_per_access = 0u;       // host fills from descriptor info
    rec->lane_mask = (uint32_t)active_mask;
    rec->active_mask = (uint32_t)predicate_mask;
    rec->op_kind = 7u;                // TMA
    rec->cache_modifier = 0u;
    rec->address_space = 1u;          // global; host confirms via descriptor
    rec->record_flags = 0u;
    rec->sector_count = 0u;           // host fills after descriptor expansion
    rec->l1_access_policy = 0u;       // TMA routing governed by op_kind=7
    rec->_reserved_pad_hi = 0u;
    rec->_reserved_pad32 = 0u;
    // No lane address staging for TMA -- the per-lane writer is a
    // global / shared / local memory-access concept. Fill the array
    // with the schema sentinel so the per-lane wire-format record (if
    // emitted) reports every lane as inactive for TMA records.
    for (int i = 0; i < 32; ++i) {
        rec->lane_addresses[i] = 0xffffffffffffffffULL;
    }
    // Capture the TMA descriptor handle bytes into the per-record
    // tail. The host expands these via NVBit's descriptor-expansion
    // APIs to derive real global-side per-element addresses.
    uint32_t copy_bytes_evt = tma_param_handle_size;
    if (copy_bytes_evt >
        (uint32_t)METRIC_EXTRACTOR_TMA_HANDLE_CAP) {
        copy_bytes_evt = (uint32_t)METRIC_EXTRACTOR_TMA_HANDLE_CAP;
    }
    rec->tma_handle_size = copy_bytes_evt;
    rec->_tma_reserved_pad = 0u;
    if (tma_param_handle != 0ULL) {
        const uint8_t* src_handle =
            (const uint8_t*)tma_param_handle;
        for (uint32_t i = 0; i < copy_bytes_evt; ++i) {
            rec->tma_handle[i] = src_handle[i];
        }
    }
}

extern "C" __device__ __noinline__ void count_global_mem_access(
    int predicate,
    int is_store,
    int bytes_per_access,
    uint64_t addr,
    uint64_t p_metrics) {
    // Warp-collective ballots. predicate_mask carries which lanes will
    // actually issue the global access; active_mask carries which lanes
    // are present at the call site at all. Both ballots are warp-wide
    // and must be reached by every active lane.
    const unsigned active_mask = __ballot_sync(__activemask(), 1);
    const unsigned predicate_mask =
        __ballot_sync(__activemask(), predicate);
    // Gather every lane's address via __shfl_sync BEFORE the leader-lane
    // gate. __shfl_sync is a warp-collective primitive — every lane in
    // the `mask` argument MUST execute the call with the same `srcLane`
    // value, otherwise the warp deadlocks. Mirrors the upstream
    // mem_trace/inject_funcs.cu pattern: all lanes participate in the
    // gather, then the leader lane post-processes.
    uint64_t lane_addrs[32];
    for (int i = 0; i < 32; ++i) {
        lane_addrs[i] = __shfl_sync(active_mask, addr, i);
    }
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    // Leader-lane-only post-processing: only one thread per warp
    // computes the distinct-sector count + atomicAdds. Non-leader lanes
    // return cleanly here without dropping out of the __shfl_sync above.
    if (!(first_laneid == laneid && p_metrics != 0ULL &&
          bytes_per_access > 0)) {
        return;
    }
    const int num_pred_lanes = __popc(predicate_mask);
    if (num_pred_lanes == 0) {
        // Predicated off everywhere — no useful bytes, no transferred
        // bytes. Skip the atomicAdd to avoid polluting the ratio.
        return;
    }
    // Build a contiguous array of predicate-true lane addresses so the
    // shared host+device helper in sector_counting.h can count distinct
    // 32B sectors with the interval-insertion semantics that match the
    // hardware loader. For each lane the helper iterates the byte range
    // [addr, addr + bytes_per_access) into 32B sectors, which is the
    // correct semantics for vectorised (LDG.E.128) and misaligned
    // accesses that span a sector boundary — the pre-Round-6 single-
    // sector-only logic undercounted these cases.
    uint64_t pred_addrs[32];
    int n_pred = 0;
    for (int i = 0; i < 32; ++i) {
        if (((predicate_mask >> i) & 1u) == 0u) continue;
        pred_addrs[n_pred++] = lane_addrs[i];
    }
    const int distinct_sectors =
        count_distinct_sectors(pred_addrs, n_pred, bytes_per_access);
    const uint64_t useful_bytes =
        (uint64_t)num_pred_lanes * (uint64_t)bytes_per_access;
    const uint64_t transferred_bytes =
        (uint64_t)distinct_sectors * 32ULL;
    LaunchMetrics* slot = (LaunchMetrics*)p_metrics;
    if (is_store) {
        saturating_atomic_add_u64(&slot->sector_store_useful_bytes,
                                  useful_bytes,
                                  &slot->saturated_counters);
        saturating_atomic_add_u64(&slot->sector_store_transferred_bytes,
                                  transferred_bytes,
                                  &slot->saturated_counters);
    } else {
        saturating_atomic_add_u64(&slot->sector_load_useful_bytes,
                                  useful_bytes,
                                  &slot->saturated_counters);
        saturating_atomic_add_u64(&slot->sector_load_transferred_bytes,
                                  transferred_bytes,
                                  &slot->saturated_counters);
    }
}

// count_tier2_global_mem_access — Tier-2 address-dump producer
// callback that captures one CompactAccessRecord per warp-collective
// global LDG / STG into the per-launch TierTwoBuffer reachable
// through ``LaunchMetrics::tier2_buffer_ptr``. Installed at the
// global LDG / STG hook site in metric_extractor.cu only when
// the address-dump env gate is on at process startup. With the gate
// off the host-side ``nvbit_insert_call`` is never emitted, so this
// device function never runs and Tier-1 byte identity is preserved
// by construction.
//
// Warp discipline mirrors ``count_global_mem_access``: every active
// lane participates in the warp-collective ballots and
// __shfl_sync-based address gather; only the first active lane
// (leader) performs the post-processing of sector-set derivation,
// atomic slot claim in the buffer, and field write.
//
// Over-capacity handling: when the per-launch buffer is full, the
// leader bumps ``dropped_count`` and sets ``truncated`` to 1 without
// writing into the records array. The in-buffer prefix is therefore
// preserved verbatim regardless of how many records the kernel
// would have produced beyond the capacity. The host serializes the
// truncation marker as ``TruncationRecordV1`` in the sidecar binary
// after writing the prefix records.
//
// Argument shape carries the AC-1 required enums + identity fields
// as 32-bit const values: op_kind (1=load / 2=store), address_space
// (1=global), cache_modifier (.cg=2 / .ca=1 / .cs=3 / .cv=4 / 0=default
// / 255=unknown), instruction_id (NVBit per-function instruction
// ordinal, stable across runs of the same SASS). The host
// instrumentation pass packs them via
// ``nvbit_add_call_arg_const_val32`` in the matching order.
extern "C" __device__ __noinline__ void count_tier2_global_mem_access(
    int predicate,
    int op_kind,
    int address_space,
    int cache_modifier,
    int bytes_per_access,
    int instruction_id,
    int l1_access_policy,
    uint64_t addr,
    uint64_t p_metrics) {
    // Warp-collective ballots — every active lane in the warp MUST
    // reach __ballot_sync with the same mask, otherwise the warp
    // deadlocks. Mirrors count_global_mem_access's discipline.
    const unsigned active_mask = __ballot_sync(__activemask(), 1);
    const unsigned predicate_mask =
        __ballot_sync(__activemask(), predicate);
    // Gather every lane's address via __shfl_sync BEFORE the leader-
    // lane gate so all lanes participate.
    uint64_t lane_addrs[32];
    for (int i = 0; i < 32; ++i) {
        lane_addrs[i] = __shfl_sync(active_mask, addr, i);
    }
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    if (!(first_laneid == laneid && p_metrics != 0ULL &&
          bytes_per_access > 0)) {
        return;
    }
    // Locate the per-launch Tier-2 buffer through the LaunchMetrics
    // pointer the host installed at launch-enter. NULL pointer means
    // the host elected to skip Tier-2 capture for this launch even
    // though env=1; the leader returns without writing.
    LaunchMetrics* slot = (LaunchMetrics*)p_metrics;
    TierTwoBuffer* tier2 =
        reinterpret_cast<TierTwoBuffer*>(slot->tier2_buffer_ptr);
    if (tier2 == nullptr) {
        return;
    }
    // Single unified atomic on the sequence-id counter. Use the
    // returned value as BOTH the monotonic sequence id AND the
    // in-buffer slot index. This avoids the two-atomic race where
    // warp A could claim slot 0 and warp B claim slot 1 but get
    // sequence ids in the opposite order — the host writer would
    // then serialize records 0,1 with sequence ids 1,0 and a strict
    // parser would correctly reject the frame as
    // tier2_sequence_order_violation. With one atomic both fields
    // grow monotonically in lockstep across all warp-collective
    // callsites, so any record at slot i carries sequence_id == i
    // and the host can repack-by-position safely. Over-capacity
    // claims bump dropped_count + set truncated without writing
    // into the records array. The host derives the effective
    // record_count at serialization time as min(sequence_id_counter,
    // capacity), making the field redundant on the device side; it
    // is left in the staging struct for cross-launch instrumentation
    // diagnostics only.
    unsigned long long seq_id =
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &tier2->sequence_id_counter),
                  1ULL);
    if (seq_id >= (unsigned long long)METRIC_EXTRACTOR_TIER2_RECORDS_PER_LAUNCH) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&tier2->dropped_count),
                  1ULL);
        atomicExch(reinterpret_cast<unsigned long long*>(&tier2->truncated),
                   1ULL);
        return;
    }
    CompactAccessRecord* rec = &tier2->records[seq_id];
    rec->sequence_id = (uint64_t)seq_id;
    rec->pc_or_instruction_id = (uint64_t)instruction_id;
    rec->cta_x = blockIdx.x;
    rec->cta_y = blockIdx.y;
    rec->cta_z = blockIdx.z;
    // Linearized warp id covers 2D / 3D CTA shapes. A bare
    // ``threadIdx.x / 32`` collapses distinct warps that differ
    // only in threadIdx.y or threadIdx.z; the linearization here
    // mirrors the canonical CUDA convention ``flat_tid =
    // (z * blockDim.y + y) * blockDim.x + x`` so each warp inside
    // the CTA gets a unique id.
    const unsigned linear_tid =
        ((threadIdx.z * blockDim.y) + threadIdx.y) * blockDim.x
        + threadIdx.x;
    rec->warp_id = (uint16_t)(linear_tid / 32u);
    rec->bytes_per_access = (uint16_t)bytes_per_access;
    rec->lane_mask = active_mask;
    rec->active_mask = predicate_mask;
    rec->op_kind = (uint8_t)op_kind;
    rec->cache_modifier = (uint8_t)cache_modifier;
    rec->address_space = (uint8_t)address_space;
    rec->record_flags = 0;
    rec->l1_access_policy = (uint8_t)l1_access_policy;
    rec->_reserved_pad_hi = 0;
    rec->_reserved_pad32 = 0;
    // Per-lane base address staging. The 32-lane __shfl_sync gather
    // above already produced lane_addrs[i] for every lane; for the
    // active subset, propagate the gathered address; for inactive
    // lanes, write the schema sentinel 0xffffffffffffffff. The host
    // emits the PerLaneAddressRecordV1 wire record only when the
    // METRIC_EXTRACTOR_TIER2_PER_LANE env gate is on; the staging
    // happens unconditionally because the addresses are a free
    // by-product of the sector-derivation pass.
    const uint64_t inactive_sentinel = 0xffffffffffffffffULL;
    for (int i = 0; i < 32; ++i) {
        if (((predicate_mask >> i) & 1u) != 0u) {
            rec->lane_addresses[i] = lane_addrs[i];
        } else {
            rec->lane_addresses[i] = inactive_sentinel;
        }
    }
    // Compute the ordered 32 B sector set from predicate-true lanes.
    // Mirrors count_distinct_sectors but emits the full sorted-by-
    // first-observation order into the staging tail.
    int n_sectors = 0;
    for (int i = 0; i < 32; ++i) {
        if (((predicate_mask >> i) & 1u) == 0u) continue;
        const uint64_t a = lane_addrs[i];
        const uint64_t first_sec = a >> 5;
        const uint64_t last_sec =
            (a + (uint64_t)bytes_per_access - 1ULL) >> 5;
        for (uint64_t s = first_sec; s <= last_sec; ++s) {
            bool found = false;
            for (int k = 0; k < n_sectors; ++k) {
                if (rec->sector_ids[k] == s) { found = true; break; }
            }
            if (!found && n_sectors <
                METRIC_EXTRACTOR_TIER2_SECTORS_PER_RECORD) {
                rec->sector_ids[n_sectors++] = s;
            }
        }
    }
    rec->sector_count = (uint16_t)n_sectors;
}

extern "C" __device__ __noinline__ void count_branch_active_mask_cb(
    int predicate,
    uint64_t p_metrics) {
    // Warp-collective ballots. The active mask gates the leader-lane
    // election; the predicate mask is the per-lane predicate-true subset
    // among active lanes. Both ballots are warp-wide and MUST be reached
    // by every active lane — putting them inside the leader-lane gate
    // would deadlock the warp (the Round 5 lesson, preserved here).
    const unsigned active_mask = __ballot_sync(__activemask(), 1);
    const unsigned predicate_mask =
        __ballot_sync(__activemask(), predicate);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    if (!(first_laneid == laneid && p_metrics != 0ULL)) {
        return;
    }
    // Bucket index = popcount of the predicate-true mask. The popcount
    // of a 32-bit value is bounded to [0, 32] by construction, so the
    // bucket index always lands in valid range; the defensive clamp
    // mirrors count_metric's INSTR_CLASS_UNKNOWN clamp pattern in case a
    // future signature change widens the bucket array.
    int bucket = count_active_lanes(predicate_mask);
    if (bucket < 0) bucket = 0;
    if (bucket >=
        (int)METRIC_EXTRACTOR_BRANCH_ACTIVE_LANES_BUCKETS) {
        bucket = (int)METRIC_EXTRACTOR_BRANCH_ACTIVE_LANES_BUCKETS - 1;
    }
    LaunchMetrics* slot = (LaunchMetrics*)p_metrics;
    saturating_atomic_add_u64(
        &slot->branch_active_lanes_hist[bucket],
        1ULL, &slot->saturated_counters);
}

extern "C" __device__ __noinline__ void count_shared_bank_conflicts_cb(
    int predicate,
    uint64_t addr,
    uint64_t p_metrics) {
    // Same warp-collective discipline as count_global_mem_access:
    // __ballot_sync + __shfl_sync are warp-collective primitives and
    // MUST run at all-lanes level, BEFORE the leader-lane gate.
    const unsigned active_mask = __ballot_sync(__activemask(), 1);
    const unsigned predicate_mask =
        __ballot_sync(__activemask(), predicate);
    uint64_t lane_addrs[32];
    for (int i = 0; i < 32; ++i) {
        lane_addrs[i] = __shfl_sync(active_mask, addr, i);
    }
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    if (!(first_laneid == laneid && p_metrics != 0ULL)) {
        return;
    }
    // Build a contiguous predicate-true subset and feed the host+device
    // shared helper. The helper computes bank = (addr >> 2) & 31 per
    // lane, dedups (bank, word) tuples, and returns the sum of per-bank
    // (distinct_words - 1) — the canonical 32-bank shared-memory
    // conflict count. Broadcast (all lanes hit same word) yields 0;
    // 32-way same-bank-different-words yields 31.
    uint64_t pred_addrs[32];
    int n_pred = 0;
    for (int i = 0; i < 32; ++i) {
        if (((predicate_mask >> i) & 1u) == 0u) continue;
        pred_addrs[n_pred++] = lane_addrs[i];
    }
    if (n_pred <= 0) {
        return;
    }
    const int conflicts =
        count_shared_bank_conflicts(pred_addrs, n_pred);
    if (conflicts > 0) {
        LaunchMetrics* slot = (LaunchMetrics*)p_metrics;
        saturating_atomic_add_u64(&slot->bank_conflict_total,
                                  (uint64_t)conflicts,
                                  &slot->saturated_counters);
    }
}
