// Host-only unit test for the per-context allocation registry.
//
// Builds with `g++` alone (no NVBit, no CUDA, no nvcc) so the
// cross-context isolation invariants are reachable on hosts without a
// GPU toolchain. This mirrors the existing `phase_classifier_test.cc`
// and `memory_op_class_test.cc` host-only test conventions.
//
// Test plan:
//   1. Two contexts with the same base address get different
//      allocation_ids; a lookup against each context resolves to its
//      own id.
//   2. Free in one context does not affect the other context's live
//      map.
//   3. context_id is monotonic and stable across multiple
//      registrations on the same context.
//   4. Registration on a freed (base) is allowed (re-allocation at
//      the same base after a free) and yields a new allocation_id.
//   5. Lookup outside any registered range returns 0.
//   6. Sentinel inputs (nullptr ctx, base==0, size==0) are no-ops.
//   7. snapshot_history preserves the full lifetime including freed
//      entries.
//   8. serialize_history_json emits a well-formed JSON list with
//      the expected keys.
#include "../allocation_registry.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>

using nvbit_tracer::register_allocation;
using nvbit_tracer::mark_allocation_freed;
using nvbit_tracer::mark_allocation_freed_range;
using nvbit_tracer::lookup_allocation_id;
using nvbit_tracer::snapshot_history;
using nvbit_tracer::reset_registry_for_testing;
using nvbit_tracer::serialize_history_json;
using nvbit_tracer::register_pending_async_free;
using nvbit_tracer::retire_pending_frees_for_stream;
using nvbit_tracer::retire_pending_frees_for_context;
using nvbit_tracer::snapshot_pending_frees_for_testing;
using nvbit_tracer::register_vmm_reservation;
using nvbit_tracer::release_vmm_reservation;
using nvbit_tracer::snapshot_vmm_reservations_for_testing;

namespace {

void *kCtxA = reinterpret_cast<void *>(static_cast<uintptr_t>(0x1000));
void *kCtxB = reinterpret_cast<void *>(static_cast<uintptr_t>(0x2000));
void *kStreamA = reinterpret_cast<void *>(static_cast<uintptr_t>(0xA000));
void *kStreamB = reinterpret_cast<void *>(static_cast<uintptr_t>(0xB000));

void test_same_base_different_contexts_isolated() {
  reset_registry_for_testing();
  uint32_t id_a = register_allocation(kCtxA, 0x10000, 4096, /*device=*/0);
  uint32_t id_b = register_allocation(kCtxB, 0x10000, 4096, /*device=*/1);
  assert(id_a == 1);
  assert(id_b == 2);
  assert(id_a != id_b);

  // Lookup against ctxA returns id_a; lookup against ctxB returns id_b.
  assert(lookup_allocation_id(kCtxA, 0x10000) == id_a);
  assert(lookup_allocation_id(kCtxA, 0x10800) == id_a);
  assert(lookup_allocation_id(kCtxB, 0x10000) == id_b);
  assert(lookup_allocation_id(kCtxB, 0x10800) == id_b);
}

void test_free_in_one_context_does_not_affect_other() {
  reset_registry_for_testing();
  uint32_t id_a = register_allocation(kCtxA, 0x20000, 4096, 0);
  uint32_t id_b = register_allocation(kCtxB, 0x20000, 4096, 1);
  assert(id_a != 0 && id_b != 0);

  mark_allocation_freed(kCtxA, 0x20000);

  // ctxA lookup misses; ctxB lookup still resolves.
  assert(lookup_allocation_id(kCtxA, 0x20000) == 0);
  assert(lookup_allocation_id(kCtxB, 0x20000) == id_b);

  // History reflects both registrations; ctxA's entry is alive=false.
  auto history = snapshot_history();
  assert(history.size() == 2);
  assert(history[0].allocation_id == id_a);
  assert(!history[0].alive);
  assert(history[1].allocation_id == id_b);
  assert(history[1].alive);
}

void test_context_id_monotonic_and_stable() {
  reset_registry_for_testing();
  register_allocation(kCtxA, 0x30000, 4096, 0);
  register_allocation(kCtxB, 0x40000, 4096, 1);
  register_allocation(kCtxA, 0x50000, 4096, 0);

  auto history = snapshot_history();
  assert(history.size() == 3);
  // ctxA is first registered, so context_id 1.
  assert(history[0].context_id == 1);
  // ctxB is second registered, so context_id 2.
  assert(history[1].context_id == 2);
  // ctxA again -> still context_id 1 (stable).
  assert(history[2].context_id == 1);
}

void test_realloc_at_same_base_after_free_yields_new_id() {
  reset_registry_for_testing();
  uint32_t id1 = register_allocation(kCtxA, 0x60000, 4096, 0);
  mark_allocation_freed(kCtxA, 0x60000);
  uint32_t id2 = register_allocation(kCtxA, 0x60000, 4096, 0);
  assert(id1 != id2);
  assert(id2 > id1);
  // Lookup resolves to the new id; the old id is freed in history.
  assert(lookup_allocation_id(kCtxA, 0x60000) == id2);
  auto history = snapshot_history();
  assert(history.size() == 2);
  assert(!history[0].alive);
  assert(history[1].alive);
}

void test_lookup_outside_any_range_returns_zero() {
  reset_registry_for_testing();
  register_allocation(kCtxA, 0x70000, 4096, 0);
  // Below the lowest base.
  assert(lookup_allocation_id(kCtxA, 0x60000) == 0);
  // Past the end of the range (base + size).
  assert(lookup_allocation_id(kCtxA, 0x71000) == 0);
  // At end (exclusive boundary).
  assert(lookup_allocation_id(kCtxA, 0x71000) == 0);
  // At base + size - 1 (inclusive boundary, last byte).
  assert(lookup_allocation_id(kCtxA, 0x70FFF) != 0);
}

void test_sentinel_inputs_are_noops() {
  reset_registry_for_testing();
  // null ctx / zero base / zero size all return 0 / no-op.
  assert(register_allocation(nullptr, 0x80000, 4096, 0) == 0);
  assert(register_allocation(kCtxA, 0, 4096, 0) == 0);
  assert(register_allocation(kCtxA, 0x80000, 0, 0) == 0);
  // Free with null ctx is a no-op.
  mark_allocation_freed(nullptr, 0x80000);
  // Lookup with null ctx returns 0.
  assert(lookup_allocation_id(nullptr, 0x80000) == 0);
  // History is empty.
  assert(snapshot_history().empty());
}

void test_snapshot_history_preserves_full_lifetime() {
  reset_registry_for_testing();
  register_allocation(kCtxA, 0xA0000, 4096, 0);
  register_allocation(kCtxA, 0xB0000, 8192, 0);
  mark_allocation_freed(kCtxA, 0xA0000);
  register_allocation(kCtxA, 0xC0000, 16384, 0);

  auto history = snapshot_history();
  assert(history.size() == 3);
  assert(history[0].allocation_id == 1);
  assert(!history[0].alive);  // freed
  assert(history[1].allocation_id == 2);
  assert(history[1].alive);
  assert(history[2].allocation_id == 3);
  assert(history[2].alive);
}

void test_serialize_history_json_well_formed() {
  reset_registry_for_testing();
  register_allocation(kCtxA, 0xD0000, 4096, 0);
  register_allocation(kCtxB, 0xE0000, 8192, 1);
  std::ostringstream out;
  serialize_history_json(out);
  std::string s = out.str();
  // Sanity checks on the emitted structure.
  assert(s.front() == '[');
  assert(s.find("\"allocation_id\": 1") != std::string::npos);
  assert(s.find("\"allocation_id\": 2") != std::string::npos);
  assert(s.find("\"context_id\": 1") != std::string::npos);
  assert(s.find("\"context_id\": 2") != std::string::npos);
  assert(s.find("\"device_id\": 0") != std::string::npos);
  assert(s.find("\"device_id\": 1") != std::string::npos);
  assert(s.find("\"base_address\": \"0xd0000\"") != std::string::npos);
  assert(s.find("\"base_address\": \"0xe0000\"") != std::string::npos);
  assert(s.find("\"size_bytes\": 4096") != std::string::npos);
  assert(s.find("\"size_bytes\": 8192") != std::string::npos);
  assert(s.find("\"alive\": true") != std::string::npos);
  // Last character before final newline is ']'.
  assert(s.size() >= 2);
  assert(s[s.size() - 2] == ']');
  assert(s.back() == '\n');
}

void test_pitched_size_lookup_spans_full_range() {
  // Round 22: cuMemAllocPitch_v2 hooks compute size as
  // *pPitch * Height (driver rounds WidthInBytes up to a
  // hardware-friendly pitch). Round 22 hooks register
  // allocations with this size; the registry's range invariant
  // must accept lookups across the full pitched range. Pin
  // the contract here with realistic dimensions: a 2D buffer
  // 256-byte pitch x 1024 rows = 262144 bytes.
  reset_registry_for_testing();
  uint64_t base = 0xF0000;
  uint64_t pitch = 256;
  uint64_t height = 1024;
  uint64_t size = pitch * height;  // 262144
  uint32_t id = register_allocation(kCtxA, base, size, 0);
  assert(id == 1);
  // First byte resolves.
  assert(lookup_allocation_id(kCtxA, base) == id);
  // Middle of row 0.
  assert(lookup_allocation_id(kCtxA, base + 128) == id);
  // First byte of row 1.
  assert(lookup_allocation_id(kCtxA, base + pitch) == id);
  // Middle of last row.
  assert(lookup_allocation_id(
             kCtxA, base + pitch * (height - 1) + 128) == id);
  // Last valid byte (base + size - 1).
  assert(lookup_allocation_id(kCtxA, base + size - 1) == id);
  // One past the end (exclusive boundary) does not resolve.
  assert(lookup_allocation_id(kCtxA, base + size) == 0);
  // Address far beyond the range does not resolve.
  assert(lookup_allocation_id(kCtxA, base + size + 0x1000) == 0);
}

// ===== Round 3 (task10 async/pool slice) =====
// Conservative stream-ordered async-free semantics: cuMemFreeAsync
// must NOT immediately mark the allocation dead, because kernels
// queued on the same stream BEFORE the async free can still be
// running and legitimately access the allocation. The matching
// cuStreamSynchronize / cuCtxSynchronize call is the proof of
// completion that retires the pending free.

void test_async_free_does_not_immediately_mark_dead() {
  reset_registry_for_testing();
  uint32_t id = register_allocation(kCtxA, 0x100000, 8192, 0);
  assert(id == 1);
  // Lookup before pending free still resolves.
  assert(lookup_allocation_id(kCtxA, 0x100000) == id);

  register_pending_async_free(kCtxA, kStreamA, 0x100000);

  // Allocation is still live: the async free has only queued
  // intent. A kernel using the (ctx, base) pre-free can still
  // resolve here.
  assert(lookup_allocation_id(kCtxA, 0x100000) == id);
  auto history = snapshot_history();
  assert(history.size() == 1);
  assert(history[0].alive);

  // Pending state is observable; allocation_id captured at queue
  // time is the live id at the time of the async free.
  auto pending = snapshot_pending_frees_for_testing();
  assert(pending.size() == 1);
  assert(pending[0].ctx == kCtxA);
  assert(pending[0].stream == kStreamA);
  assert(pending[0].base == 0x100000);
  assert(pending[0].allocation_id == id);
}

void test_stream_synchronize_retires_pending_async_frees() {
  reset_registry_for_testing();
  register_allocation(kCtxA, 0x110000, 4096, 0);
  register_allocation(kCtxA, 0x120000, 4096, 0);
  register_pending_async_free(kCtxA, kStreamA, 0x110000);
  register_pending_async_free(kCtxA, kStreamA, 0x120000);

  retire_pending_frees_for_stream(kCtxA, kStreamA);

  // Both pending frees retired; lookups now miss.
  assert(lookup_allocation_id(kCtxA, 0x110000) == 0);
  assert(lookup_allocation_id(kCtxA, 0x120000) == 0);
  auto history = snapshot_history();
  assert(history.size() == 2);
  assert(!history[0].alive);
  assert(!history[1].alive);
  // Pending state cleared.
  assert(snapshot_pending_frees_for_testing().empty());
}

void test_ctx_synchronize_retires_async_frees_across_streams() {
  reset_registry_for_testing();
  register_allocation(kCtxA, 0x130000, 4096, 0);
  register_allocation(kCtxA, 0x140000, 4096, 0);
  register_allocation(kCtxA, 0x150000, 4096, 0);
  register_pending_async_free(kCtxA, kStreamA, 0x130000);
  register_pending_async_free(kCtxA, kStreamB, 0x140000);
  // 0x150000 has no pending free; should remain live after sync.

  retire_pending_frees_for_context(kCtxA);

  assert(lookup_allocation_id(kCtxA, 0x130000) == 0);
  assert(lookup_allocation_id(kCtxA, 0x140000) == 0);
  assert(lookup_allocation_id(kCtxA, 0x150000) != 0);
  assert(snapshot_pending_frees_for_testing().empty());
}

void test_realloc_at_same_base_after_async_free_completion() {
  reset_registry_for_testing();
  uint32_t id1 = register_allocation(kCtxA, 0x160000, 4096, 0);
  register_pending_async_free(kCtxA, kStreamA, 0x160000);
  // Stream not yet synced; the original allocation is still live.
  assert(lookup_allocation_id(kCtxA, 0x160000) == id1);

  retire_pending_frees_for_stream(kCtxA, kStreamA);
  // Now freed; same-base reallocation is allowed and yields a
  // new monotonic id.
  uint32_t id2 = register_allocation(kCtxA, 0x160000, 4096, 0);
  assert(id2 == id1 + 1);
  assert(lookup_allocation_id(kCtxA, 0x160000) == id2);

  auto history = snapshot_history();
  assert(history.size() == 2);
  assert(history[0].allocation_id == id1);
  assert(!history[0].alive);
  assert(history[1].allocation_id == id2);
  assert(history[1].alive);
}

void test_async_pending_free_isolated_per_stream() {
  reset_registry_for_testing();
  register_allocation(kCtxA, 0x170000, 4096, 0);
  register_allocation(kCtxA, 0x180000, 4096, 0);
  register_pending_async_free(kCtxA, kStreamA, 0x170000);
  register_pending_async_free(kCtxA, kStreamB, 0x180000);

  // Synchronize streamA only; streamB's pending free remains.
  retire_pending_frees_for_stream(kCtxA, kStreamA);
  assert(lookup_allocation_id(kCtxA, 0x170000) == 0);
  assert(lookup_allocation_id(kCtxA, 0x180000) != 0);

  auto pending = snapshot_pending_frees_for_testing();
  assert(pending.size() == 1);
  assert(pending[0].stream == kStreamB);
  assert(pending[0].base == 0x180000);
}

void test_async_pending_free_isolated_per_context() {
  reset_registry_for_testing();
  uint32_t id_a = register_allocation(kCtxA, 0x190000, 4096, 0);
  uint32_t id_b = register_allocation(kCtxB, 0x190000, 4096, 1);
  // Same base in two contexts: cross-context isolation must hold
  // for pending frees too.
  register_pending_async_free(kCtxA, kStreamA, 0x190000);

  // ctxA stream sync retires only ctxA's pending; ctxB unaffected.
  retire_pending_frees_for_stream(kCtxA, kStreamA);
  assert(lookup_allocation_id(kCtxA, 0x190000) == 0);
  assert(lookup_allocation_id(kCtxB, 0x190000) == id_b);
  // id_a is in history with alive=false; id_b stays alive.
  auto history = snapshot_history();
  assert(history.size() == 2);
  assert(history[0].allocation_id == id_a);
  assert(!history[0].alive);
  assert(history[1].allocation_id == id_b);
  assert(history[1].alive);
}

void test_async_free_for_unknown_base_is_noop() {
  reset_registry_for_testing();
  // Pending free against a base that was never registered should
  // silently drop rather than queue a phantom that would never
  // retire.
  register_pending_async_free(kCtxA, kStreamA, 0x1A0000);
  assert(snapshot_pending_frees_for_testing().empty());

  // Pending free against a base that was already freed (sync
  // path) is also a no-op for the same reason.
  register_allocation(kCtxA, 0x1B0000, 4096, 0);
  mark_allocation_freed(kCtxA, 0x1B0000);
  register_pending_async_free(kCtxA, kStreamA, 0x1B0000);
  assert(snapshot_pending_frees_for_testing().empty());
}

void test_async_free_sentinel_inputs_are_noops() {
  reset_registry_for_testing();
  register_allocation(kCtxA, 0x1C0000, 4096, 0);
  // null ctx and base==0 are no-ops. NULL stream is now a VALID
  // key (default stream); separate tests cover default-stream
  // semantics.
  register_pending_async_free(nullptr, kStreamA, 0x1C0000);
  register_pending_async_free(kCtxA, kStreamA, 0);
  assert(snapshot_pending_frees_for_testing().empty());
  // Allocation remains live throughout.
  assert(lookup_allocation_id(kCtxA, 0x1C0000) != 0);
  // Retirement on null ctx is a no-op.
  retire_pending_frees_for_stream(nullptr, kStreamA);
  retire_pending_frees_for_context(nullptr);
  assert(lookup_allocation_id(kCtxA, 0x1C0000) != 0);
}

// ===== Round 4 (correctness fixes + VMM) =====

void test_same_base_reuse_before_async_sync_retires_correct_id() {
  // Codex Round 3 review's flagged bug: a pending free retiring
  // by base alone would mark the WRONG allocation dead when a
  // newer same-base alloc has overwritten the live map between
  // queue time and retirement time.
  reset_registry_for_testing();
  uint32_t id1 = register_allocation(kCtxA, 0x1D0000, 4096, 0);
  // Async free queues id1 at the current live state.
  register_pending_async_free(kCtxA, kStreamA, 0x1D0000);
  // Before stream sync, a new alloc reuses the same base. (This
  // mimics CUDA's stream-ordered allocator returning the same
  // virtual address from the pool when the previous allocation
  // is still pending free.)
  uint32_t id2 = register_allocation(kCtxA, 0x1D0000, 4096, 0);
  assert(id1 != id2);
  assert(id2 == id1 + 1);
  // The live map now points to id2.
  assert(lookup_allocation_id(kCtxA, 0x1D0000) == id2);

  // Stream sync retires the pending entry. Because the entry
  // captured id1 (not just the base), id1 is the one marked
  // dead. id2 stays live.
  retire_pending_frees_for_stream(kCtxA, kStreamA);

  auto history = snapshot_history();
  assert(history.size() == 2);
  assert(history[0].allocation_id == id1);
  assert(!history[0].alive);  // retired correctly
  assert(history[1].allocation_id == id2);
  assert(history[1].alive);   // NOT retired (the bug case)
  // Live lookup still resolves to id2.
  assert(lookup_allocation_id(kCtxA, 0x1D0000) == id2);
}

void test_async_pending_free_default_stream() {
  // CUDA's stream-ordered allocator accepts NULL (default stream)
  // when a current context exists. Round 4 stops dropping NULL
  // streams as a sentinel: null is a valid stream key.
  reset_registry_for_testing();
  uint32_t id = register_allocation(kCtxA, 0x1E0000, 4096, 0);
  register_pending_async_free(kCtxA, /*stream=*/nullptr, 0x1E0000);
  auto pending = snapshot_pending_frees_for_testing();
  assert(pending.size() == 1);
  assert(pending[0].stream == nullptr);
  assert(pending[0].base == 0x1E0000);
  assert(pending[0].allocation_id == id);
  // Default-stream sync retires.
  retire_pending_frees_for_stream(kCtxA, /*stream=*/nullptr);
  assert(lookup_allocation_id(kCtxA, 0x1E0000) == 0);
  assert(snapshot_pending_frees_for_testing().empty());
}

void test_default_stream_isolated_from_named_stream() {
  // A pending free on the default (NULL) stream and a pending free
  // on a named stream live in separate keys; synchronizing one
  // does not retire the other.
  reset_registry_for_testing();
  register_allocation(kCtxA, 0x1F0000, 4096, 0);
  register_allocation(kCtxA, 0x200000, 4096, 0);
  register_pending_async_free(kCtxA, /*stream=*/nullptr, 0x1F0000);
  register_pending_async_free(kCtxA, kStreamA, 0x200000);

  retire_pending_frees_for_stream(kCtxA, /*stream=*/nullptr);
  assert(lookup_allocation_id(kCtxA, 0x1F0000) == 0);
  assert(lookup_allocation_id(kCtxA, 0x200000) != 0);
  auto pending = snapshot_pending_frees_for_testing();
  assert(pending.size() == 1);
  assert(pending[0].stream == kStreamA);
}

void test_ctx_synchronize_retires_default_and_named_streams() {
  // Context-level sync retires every pending across all streams,
  // including the default (NULL) stream.
  reset_registry_for_testing();
  register_allocation(kCtxA, 0x210000, 4096, 0);
  register_allocation(kCtxA, 0x220000, 4096, 0);
  register_pending_async_free(kCtxA, /*stream=*/nullptr, 0x210000);
  register_pending_async_free(kCtxA, kStreamA, 0x220000);

  retire_pending_frees_for_context(kCtxA);
  assert(lookup_allocation_id(kCtxA, 0x210000) == 0);
  assert(lookup_allocation_id(kCtxA, 0x220000) == 0);
  assert(snapshot_pending_frees_for_testing().empty());
}

void test_vmm_reserve_alone_does_not_create_live_allocation() {
  // cuMemAddressReserve carves out a virtual range without
  // physical backing. The range must NOT appear in the live
  // allocation map; only cuMemMap binds backing and registers a
  // live allocation.
  reset_registry_for_testing();
  register_vmm_reservation(kCtxA, 0x300000, 0x10000);

  // Lookup against any address in the reserved range misses.
  assert(lookup_allocation_id(kCtxA, 0x300000) == 0);
  assert(lookup_allocation_id(kCtxA, 0x308000) == 0);
  assert(lookup_allocation_id(kCtxA, 0x30FFFF) == 0);
  // History is empty (no allocation was registered).
  assert(snapshot_history().empty());
  // Reservation tracker reflects the reserve.
  auto reservations = snapshot_vmm_reservations_for_testing();
  assert(reservations.size() == 1);
  assert(reservations[0].ctx == kCtxA);
  assert(reservations[0].addr == 0x300000);
  assert(reservations[0].size == 0x10000);
}

void test_vmm_map_registers_live_allocation_inside_reservation() {
  // After cuMemMap binds backing, the mapped range is a live
  // allocation. The reservation tracker still records the
  // reservation because a single reservation can host multiple
  // partial mappings at different offsets.
  reset_registry_for_testing();
  register_vmm_reservation(kCtxA, 0x310000, 0x10000);
  uint32_t id = register_allocation(kCtxA, 0x310000, 0x4000, 0);
  assert(id == 1);
  assert(lookup_allocation_id(kCtxA, 0x310000) == id);
  assert(lookup_allocation_id(kCtxA, 0x313FFF) == id);
  // Reservation entry preserved (reservation outlives a single
  // mapping).
  auto reservations = snapshot_vmm_reservations_for_testing();
  assert(reservations.size() == 1);
}

void test_vmm_unmap_marks_allocation_freed_but_keeps_reservation() {
  // cuMemUnmap is the symmetric inverse of cuMemMap: marks the
  // mapped range freed but leaves the reservation intact (so a
  // future cuMemMap can bind a different physical handle into
  // the same virtual range).
  reset_registry_for_testing();
  register_vmm_reservation(kCtxA, 0x320000, 0x10000);
  register_allocation(kCtxA, 0x320000, 0x4000, 0);
  // Symmetric cuMemUnmap is just `mark_allocation_freed`.
  mark_allocation_freed(kCtxA, 0x320000);
  assert(lookup_allocation_id(kCtxA, 0x320000) == 0);
  // Reservation preserved.
  auto reservations = snapshot_vmm_reservations_for_testing();
  assert(reservations.size() == 1);
  assert(reservations[0].addr == 0x320000);
}

void test_vmm_address_free_releases_reservation_and_cleans_live_mapping() {
  // cuMemAddressFree releases the reservation. If a live mapping
  // still exists in the freed range (a producer/driver oddity —
  // a clean cuMemUnmap should already have run), the registry
  // defensively marks any intersecting allocation freed and
  // removes the reservation entry.
  reset_registry_for_testing();
  register_vmm_reservation(kCtxA, 0x330000, 0x10000);
  uint32_t id = register_allocation(kCtxA, 0x330000, 0x4000, 0);
  // Skip cuMemUnmap and call cuMemAddressFree directly.
  release_vmm_reservation(kCtxA, 0x330000);
  // Allocation is defensively marked dead.
  auto history = snapshot_history();
  assert(history.size() == 1);
  assert(history[0].allocation_id == id);
  assert(!history[0].alive);
  // Live map miss for any address in the released range.
  assert(lookup_allocation_id(kCtxA, 0x330000) == 0);
  assert(lookup_allocation_id(kCtxA, 0x333FFF) == 0);
  // Reservation tracker cleared.
  assert(snapshot_vmm_reservations_for_testing().empty());
}

void test_vmm_cross_context_reservation_isolation() {
  // Two contexts can independently reserve overlapping virtual
  // ranges; releasing one does not perturb the other.
  reset_registry_for_testing();
  register_vmm_reservation(kCtxA, 0x340000, 0x10000);
  register_vmm_reservation(kCtxB, 0x340000, 0x10000);
  register_allocation(kCtxA, 0x340000, 0x4000, 0);
  register_allocation(kCtxB, 0x340000, 0x4000, 1);

  release_vmm_reservation(kCtxA, 0x340000);

  // ctxA's mapping was defensively cleaned up.
  assert(lookup_allocation_id(kCtxA, 0x340000) == 0);
  // ctxB's mapping is unaffected.
  assert(lookup_allocation_id(kCtxB, 0x340000) != 0);
  // Reservation tracker has only ctxB's reservation now.
  auto reservations = snapshot_vmm_reservations_for_testing();
  assert(reservations.size() == 1);
  assert(reservations[0].ctx == kCtxB);
}

void test_vmm_reservation_sentinel_inputs_are_noops() {
  reset_registry_for_testing();
  // null ctx, addr==0, size==0 are no-ops.
  register_vmm_reservation(nullptr, 0x350000, 0x1000);
  register_vmm_reservation(kCtxA, 0, 0x1000);
  register_vmm_reservation(kCtxA, 0x350000, 0);
  assert(snapshot_vmm_reservations_for_testing().empty());
  // Release on null ctx or unknown addr is a no-op.
  release_vmm_reservation(nullptr, 0x350000);
  release_vmm_reservation(kCtxA, 0x350000);  // never registered
  assert(snapshot_vmm_reservations_for_testing().empty());
}

void test_unmap_range_full_unmap_removes_entry() {
  reset_registry_for_testing();
  uint32_t id = register_allocation(kCtxA, 0x400000, 0x4000, 0);
  assert(lookup_allocation_id(kCtxA, 0x400000) == id);
  // Unmap exactly the original range.
  mark_allocation_freed_range(kCtxA, 0x400000, 0x4000);
  assert(lookup_allocation_id(kCtxA, 0x400000) == 0);
  auto hist = snapshot_history();
  assert(hist.size() == 1);
  assert(!hist[0].alive);
}

void test_unmap_range_base_partial_keeps_tail_survivor() {
  reset_registry_for_testing();
  register_allocation(kCtxA, 0x410000, 0x4000, 0);
  // Unmap the first half [0x410000, 0x412000).
  mark_allocation_freed_range(kCtxA, 0x410000, 0x2000);
  // Front half is gone.
  assert(lookup_allocation_id(kCtxA, 0x410000) == 0);
  assert(lookup_allocation_id(kCtxA, 0x411FFF) == 0);
  // Tail half [0x412000, 0x414000) survives under a new id.
  uint32_t tail_id = lookup_allocation_id(kCtxA, 0x412000);
  assert(tail_id != 0);
  assert(lookup_allocation_id(kCtxA, 0x413FFF) == tail_id);
  assert(lookup_allocation_id(kCtxA, 0x414000) == 0);
}

void test_unmap_range_tail_partial_keeps_front_survivor() {
  reset_registry_for_testing();
  register_allocation(kCtxA, 0x420000, 0x4000, 0);
  // Unmap the back half [0x422000, 0x424000).
  mark_allocation_freed_range(kCtxA, 0x422000, 0x2000);
  // Front half [0x420000, 0x422000) survives.
  uint32_t front_id = lookup_allocation_id(kCtxA, 0x420000);
  assert(front_id != 0);
  assert(lookup_allocation_id(kCtxA, 0x421FFF) == front_id);
  // Tail half is gone.
  assert(lookup_allocation_id(kCtxA, 0x422000) == 0);
  assert(lookup_allocation_id(kCtxA, 0x423FFF) == 0);
}

void test_unmap_range_interior_partial_splits_into_two() {
  reset_registry_for_testing();
  register_allocation(kCtxA, 0x430000, 0x6000, 0);
  // Unmap an interior range [0x432000, 0x434000).
  mark_allocation_freed_range(kCtxA, 0x432000, 0x2000);
  // Pre-range survivor [0x430000, 0x432000).
  uint32_t pre_id = lookup_allocation_id(kCtxA, 0x430000);
  assert(pre_id != 0);
  assert(lookup_allocation_id(kCtxA, 0x431FFF) == pre_id);
  // Interior is gone.
  assert(lookup_allocation_id(kCtxA, 0x432000) == 0);
  assert(lookup_allocation_id(kCtxA, 0x433FFF) == 0);
  // Post-range survivor [0x434000, 0x436000) under a different id.
  uint32_t post_id = lookup_allocation_id(kCtxA, 0x434000);
  assert(post_id != 0);
  assert(post_id != pre_id);
  assert(lookup_allocation_id(kCtxA, 0x435FFF) == post_id);
  assert(lookup_allocation_id(kCtxA, 0x436000) == 0);
}

void test_unmap_range_no_overlap_is_noop() {
  reset_registry_for_testing();
  uint32_t id = register_allocation(kCtxA, 0x440000, 0x2000, 0);
  // Unmap a range that does not overlap.
  mark_allocation_freed_range(kCtxA, 0x450000, 0x1000);
  // Original is untouched.
  assert(lookup_allocation_id(kCtxA, 0x440000) == id);
  assert(lookup_allocation_id(kCtxA, 0x441FFF) == id);
}

void test_unmap_range_sentinel_inputs_are_noops() {
  reset_registry_for_testing();
  uint32_t id = register_allocation(kCtxA, 0x460000, 0x2000, 0);
  // Sentinel inputs leave the registry untouched.
  mark_allocation_freed_range(nullptr, 0x460000, 0x2000);
  mark_allocation_freed_range(kCtxA, 0, 0x2000);
  mark_allocation_freed_range(kCtxA, 0x460000, 0);
  assert(lookup_allocation_id(kCtxA, 0x460000) == id);
}

}  // namespace

int main() {
  // Round 1-3 tests (registry + sync + async/pool basics).
  test_same_base_different_contexts_isolated();
  test_free_in_one_context_does_not_affect_other();
  test_context_id_monotonic_and_stable();
  test_realloc_at_same_base_after_free_yields_new_id();
  test_lookup_outside_any_range_returns_zero();
  test_sentinel_inputs_are_noops();
  test_snapshot_history_preserves_full_lifetime();
  test_serialize_history_json_well_formed();
  test_pitched_size_lookup_spans_full_range();
  test_async_free_does_not_immediately_mark_dead();
  test_stream_synchronize_retires_pending_async_frees();
  test_ctx_synchronize_retires_async_frees_across_streams();
  test_realloc_at_same_base_after_async_free_completion();
  test_async_pending_free_isolated_per_stream();
  test_async_pending_free_isolated_per_context();
  test_async_free_for_unknown_base_is_noop();
  test_async_free_sentinel_inputs_are_noops();
  // Round 4 (correctness fixes + VMM).
  test_same_base_reuse_before_async_sync_retires_correct_id();
  test_async_pending_free_default_stream();
  test_default_stream_isolated_from_named_stream();
  test_ctx_synchronize_retires_default_and_named_streams();
  test_vmm_reserve_alone_does_not_create_live_allocation();
  test_vmm_map_registers_live_allocation_inside_reservation();
  test_vmm_unmap_marks_allocation_freed_but_keeps_reservation();
  test_vmm_address_free_releases_reservation_and_cleans_live_mapping();
  test_vmm_cross_context_reservation_isolation();
  test_vmm_reservation_sentinel_inputs_are_noops();
  // Round 5 (VMM partial unmap range handling).
  test_unmap_range_full_unmap_removes_entry();
  test_unmap_range_base_partial_keeps_tail_survivor();
  test_unmap_range_tail_partial_keeps_front_survivor();
  test_unmap_range_interior_partial_splits_into_two();
  test_unmap_range_no_overlap_is_noop();
  test_unmap_range_sentinel_inputs_are_noops();
  std::printf("allocation_registry_test: 33 tests passed\n");
  return 0;
}
