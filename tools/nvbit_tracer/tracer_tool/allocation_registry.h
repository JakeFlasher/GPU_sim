// Per-context allocator registry for NVBit-traced applications.
//
// Multi-context CUDA applications can host independent address spaces
// per context. Two contexts can legitimately have allocations whose
// device-pointer base addresses overlap, and a free in one context
// must not perturb another context's intervals. The registry keys
// live intervals by (context handle, base address) so cross-context
// aliasing is impossible by construction.
//
// Allocation IDs are globally monotonic (one counter across all
// contexts) so a wire-side `allocation_id` uniquely identifies a
// historical allocation regardless of which context produced it.
// Context IDs are also globally monotonic, assigned at first
// `register_allocation` against a previously-unseen context handle
// (lookup does not allocate a context id), and surfaced in the
// sidecar so consumers can disambiguate cross-context allocations
// downstream.
//
// The context handle type is intentionally `void *` (rather than
// `CUcontext`) so this header has no CUDA dependency and the host-only
// unit test under `tracer_tool/tests/` can build with `g++` alone.
// The producer at `tracer_tool.cu` passes its `CUcontext ctx` (which
// is itself a pointer typedef in `<cuda.h>`) directly through the
// implicit pointer-to-void* conversion.
#ifndef NVBIT_TRACER_ALLOCATION_REGISTRY_H_
#define NVBIT_TRACER_ALLOCATION_REGISTRY_H_

#include <cstdint>
#include <ostream>
#include <vector>

namespace nvbit_tracer {

struct AllocationRecord {
  void *ctx;            // opaque context handle (CUcontext-shaped)
  uint32_t context_id;  // monotonic; assigned at first registration on a ctx
  int device_id;        // host-side device ordinal at allocation time
  uint64_t base;
  uint64_t size;
  uint32_t allocation_id;
  bool alive;
};

// Register a new allocation in the per-context registry. `ctx` must
// not be null. Returns the assigned allocation_id (monotonic across
// all contexts; starts at 1 so proto3 default 0 stays distinct as the
// "unresolved" sentinel for the wire field).
//
// If `base` is 0 or `size` is 0, the call is a no-op and returns 0.
// Such calls indicate a producer bug or driver oddity; the caller
// should already have filtered them.
uint32_t register_allocation(
    void *ctx, uint64_t base, uint64_t size, int device_id);

// Mark the allocation at (ctx, base) freed. Removes from the live
// per-context map but preserves the history entry with alive=false.
// Calls on an unknown (ctx, base) pair are a no-op.
void mark_allocation_freed(void *ctx, uint64_t base);

// Mark the half-open range [ptr, ptr+size) freed in `ctx`. For each
// live allocation that intersects the range, marks the original
// allocation_id dead and creates new sub-allocations (with fresh
// allocation_ids) for the parts that remain mapped:
//   * full unmap (range covers entire allocation): no survivor.
//   * base-partial unmap (ptr == base, size < orig_size): one
//     survivor for [ptr+size, base+orig_size).
//   * tail-partial unmap (ptr+size == base+orig_size, ptr > base):
//     one survivor for [base, ptr).
//   * interior unmap (base < ptr AND ptr+size < base+orig_size): two
//     survivors for [base, ptr) and [ptr+size, base+orig_size).
//   * no overlap: no-op.
// Sentinel inputs (nullptr ctx, ptr==0, size==0) are no-ops. This is
// the correct primitive for cuMemUnmap, which can legally unmap any
// sub-range of a previously cuMemMap-bound region.
void mark_allocation_freed_range(void *ctx, uint64_t ptr, uint64_t size);

// Look up the allocation containing `addr` in `ctx`'s live map.
// Returns 0 if no live allocation matches (the proto3 unresolved
// sentinel for the wire field).
uint32_t lookup_allocation_id(void *ctx, uint64_t addr);

// Record a stream-ordered async free as pending. Stream-ordered
// frees (cuMemFreeAsync, cuMemFreeAsync_ptsz) only complete after
// every stream-queued operation enqueued before the free has
// executed. A kernel that was launched on `stream` BEFORE the
// async free can therefore still legitimately access (ctx, base)
// AFTER the API call returns. Marking the allocation dead at
// API exit would cause its later-running kernel to look up the
// already-dead allocation and produce a spurious
// `unmapped_memory_range` typed exclusion. Instead, this entry
// captures the currently-live `allocation_id` for `(ctx, base)`,
// queues `(ctx, stream, base, allocation_id)` into a per-stream
// pending list, and keeps the allocation alive=true in the
// registry; the matching `retire_pending_frees_for_*` call
// (driven by a sync-point hook) flips THAT specific
// allocation_id's `alive=false` and removes the live-map entry
// only if it still points to that same allocation_id. This makes
// same-base reuse before sync correct: an `allocAsync` that
// reuses the freed base before the sync gets its own allocation_id,
// and the eventual sync retires the OLD id without touching the
// new one.
//
// `stream == nullptr` is a valid stream key — CUDA's
// stream-ordered allocator accepts the default stream (NULL)
// when a current context exists. Sentinel inputs (nullptr ctx,
// base==0) are no-ops. A pending free for (ctx, base) where the
// allocation is unknown (already freed, or never registered) is
// also a no-op so a noisy or out-of-order trace cannot corrupt
// registry state.
void register_pending_async_free(void *ctx, void *stream, uint64_t base);

// Retire all pending async frees on `(ctx, stream)`. For each
// retired entry, marks the SPECIFIC captured allocation_id
// freed (alive=false in history) and removes the live-map entry
// at (ctx, base) only if it still points to that same
// allocation_id (handles same-base reuse before retirement).
// Called from the cuStreamSynchronize / cuStreamSynchronize_ptsz
// / cuStreamQuery (success-only) hooks on success exit, where
// stream completion is proven by the sync/query. `stream` may
// be NULL (default stream). No-op when ctx is null or no
// pending entries exist for the (ctx, stream) key.
void retire_pending_frees_for_stream(void *ctx, void *stream);

// Retire all pending async frees across every stream in `ctx`.
// Called from the cuCtxSynchronize / cuCtxSynchronize_v2 hooks
// on success exit, where the sync proves every stream in the
// context has drained. Same allocation_id-aware semantics as
// the per-stream retirement, applied across all streams keyed
// under `ctx`. No-op when ctx is null or no pending entries
// exist for the context.
void retire_pending_frees_for_context(void *ctx);

// Diagnostic snapshot of the pending-async-free state, for
// host-only unit tests. Returns a vector of records in
// deterministic order (sorted by ctx, then stream, then base).
// Each record carries the `allocation_id` captured at queue time
// so tests can assert the same-base-reuse-before-sync invariant.
// Production code should not need this.
struct PendingFreeRecord {
  void *ctx;
  void *stream;
  uint64_t base;
  uint32_t allocation_id;
};
std::vector<PendingFreeRecord> snapshot_pending_frees_for_testing();

// === VMM (Virtual Memory Management) reservation tracking ===
//
// VMM allocations work in two phases:
//   1. cuMemAddressReserve carves out a virtual address range. The
//      range is NOT kernel-accessible — kernels would page-fault
//      reading from it. The registry's `register_allocation` is
//      NOT called here.
//   2. cuMemMap binds physical backing into a previously-reserved
//      range, making it kernel-accessible. THIS is when
//      `register_allocation` runs.
//   3. cuMemUnmap removes the binding (calls `mark_allocation_freed`).
//   4. cuMemAddressFree releases the reservation. If a live
//      mapping still exists in the freed range, that is a
//      producer/driver oddity; the registry defensively marks any
//      live allocation in [addr, addr+size) freed before clearing
//      the reservation entry.
//
// Reservation tracking lives separately from allocation tracking
// so a future `task11b` cross-allocation typed exclusion can
// distinguish "address points to a reserved-but-not-mapped range"
// (a producer/instrumentation gap) from "address is outside any
// reservation" (a true unmapped_memory_range case).

// Register a virtual address reservation in the per-context
// reservation tracker. NOT a live allocation — kernels cannot
// access this range until cuMemMap binds physical backing.
// Sentinel inputs (nullptr ctx, addr==0, size==0) are no-ops.
void register_vmm_reservation(void *ctx, uint64_t addr, uint64_t size);

// Release a virtual address reservation. If any live allocation
// (from cuMemMap) intersects [addr, addr+size), defensively mark
// each such allocation freed via `mark_allocation_freed` before
// clearing the reservation entry; this preserves the registry
// invariant "live map points only at currently-mapped ranges"
// even when the producer/driver permits a partial-cleanup
// sequence. Sentinel inputs are no-ops. A release on an
// unknown (ctx, addr) is also a no-op.
void release_vmm_reservation(void *ctx, uint64_t addr);

struct VmmReservationRecord {
  void *ctx;
  uint64_t addr;
  uint64_t size;
};

// Diagnostic snapshot of currently-tracked VMM reservations, for
// host-only unit tests. Sorted by ctx then addr.
std::vector<VmmReservationRecord> snapshot_vmm_reservations_for_testing();

// Snapshot the full allocation history (across all contexts and
// alive/freed states) for sidecar serialization. The vector is a
// copy; mutation does not affect the registry.
std::vector<AllocationRecord> snapshot_history();

// Reset the registry. Used by unit tests to start each test from a
// clean slate. Production code should never need this.
void reset_registry_for_testing();

// Serialize the allocation history to a JSON list on `out`. Each
// entry: {"allocation_id": <int>, "context_id": <int>,
// "device_id": <int>, "base_address": "0x<hex>",
// "size_bytes": <int>, "alive": <bool>}. Sorted by allocation_id
// ascending (the natural append order of allocation_history). The
// trailing newline is included so the file is POSIX-conforming.
void serialize_history_json(std::ostream &out);

}  // namespace nvbit_tracer

#endif  // NVBIT_TRACER_ALLOCATION_REGISTRY_H_
