#include "allocation_registry.h"

#include <cstdint>
#include <map>
#include <mutex>
#include <ostream>
#include <vector>

namespace nvbit_tracer {

namespace {

// All registry state is file-static and serialized through a single
// mutex. Lock scopes are short (one map insert / erase / upper_bound +
// bounds check) so contention is negligible relative to the cost of
// the surrounding NVBit callback or recv-thread protobuf
// serialization.
std::mutex g_mutex;

// Live intervals: per-context map keyed by base address. Inner map
// value is the allocation_id assigned at registration; the full
// record (size, device, alive) is fetched from g_history via
// `allocation_id - 1`.
std::map<void *, std::map<uint64_t, uint32_t>> g_live;

// Append-only full history. Indexed by `allocation_id - 1` since
// allocation_ids start at 1.
std::vector<AllocationRecord> g_history;

// Per-context monotonic context_id assignment. Empty until the first
// `register_allocation` on a context (lookup does not allocate a
// context id). Both producers and consumers see the same
// context_id for the same context handle for the duration of a
// trace run.
std::map<void *, uint32_t> g_context_id;

// Pending stream-ordered async frees: per-(ctx, stream) list of
// (base, allocation_id) entries awaiting completion. cuMemFreeAsync
// queues a record here capturing the currently-live allocation_id
// at queue time; the matching cuStreamSynchronize / cuStreamQuery
// (success-only) / cuCtxSynchronize / cuCtxSynchronize_v2 hook
// drains the list and marks THAT specific allocation_id dead.
// Capturing the allocation_id (rather than just the base) makes
// same-base reuse before retirement correct: if cuMemAllocAsync
// reuses the freed base before the sync, the new allocation gets
// its own allocation_id; the eventual sync retires the OLD id
// without touching the new one.
//
// The outer map is keyed by ctx so cross-context isolation is
// preserved; the inner map is keyed by stream so per-stream
// retirement does not affect other streams in the same context.
// `stream == nullptr` is a valid key (CUDA default stream).
struct PendingEntry {
  uint64_t base;
  uint32_t allocation_id;
};
std::map<void *, std::map<void *, std::vector<PendingEntry>>>
    g_pending_async_frees;

// Per-context VMM reservation tracker: records cuMemAddressReserve
// ranges that have not yet been mapped to physical backing.
// Reservations are NOT live allocations (kernel access faults on
// reserved-but-unmapped addresses); the registry's live map and
// `register_allocation` are reserved exclusively for cuMemMap-bound
// ranges. The reservation tracker exists so cuMemAddressFree on
// a still-mapped range can defensively clean up any leaked live
// allocation, and so a future task11b can distinguish
// reserved-but-unmapped from truly-unmapped addresses.
std::map<void *, std::map<uint64_t, uint64_t>> g_vmm_reservations;

uint32_t g_next_allocation_id = 1;
uint32_t g_next_context_id = 1;

// Caller must hold g_mutex.
uint32_t assign_context_id_locked(void *ctx) {
  auto it = g_context_id.find(ctx);
  if (it != g_context_id.end()) return it->second;
  uint32_t id = g_next_context_id++;
  g_context_id[ctx] = id;
  return id;
}

}  // namespace

uint32_t register_allocation(
    void *ctx, uint64_t base, uint64_t size, int device_id) {
  if (ctx == nullptr || base == 0 || size == 0) return 0;
  std::lock_guard<std::mutex> lock(g_mutex);
  uint32_t context_id = assign_context_id_locked(ctx);
  uint32_t alloc_id = g_next_allocation_id++;
  g_history.push_back(AllocationRecord{
      ctx, context_id, device_id, base, size, alloc_id, true});
  g_live[ctx][base] = alloc_id;
  return alloc_id;
}

// Caller must hold g_mutex. Performs the actual mark-freed work
// shared between the synchronous mark_allocation_freed entry
// point and the per-stream / per-ctx async retirement helpers.
// No-op when (ctx, base) is unknown or already freed.
void mark_allocation_freed_locked(void *ctx, uint64_t base) {
  auto ctx_it = g_live.find(ctx);
  if (ctx_it == g_live.end()) return;
  auto &per_ctx = ctx_it->second;
  auto base_it = per_ctx.find(base);
  if (base_it == per_ctx.end()) return;
  uint32_t alloc_id = base_it->second;
  if (alloc_id >= 1 && alloc_id - 1 < g_history.size()) {
    g_history[alloc_id - 1].alive = false;
  }
  per_ctx.erase(base_it);
}

void mark_allocation_freed(void *ctx, uint64_t base) {
  if (ctx == nullptr) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  mark_allocation_freed_locked(ctx, base);
}

void mark_allocation_freed_range(void *ctx, uint64_t ptr, uint64_t size) {
  if (ctx == nullptr || ptr == 0 || size == 0) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  auto ctx_it = g_live.find(ctx);
  if (ctx_it == g_live.end()) return;
  auto &per_ctx = ctx_it->second;
  const uint64_t unmap_end = ptr + size;
  // Collect bases of allocations that overlap [ptr, ptr+size).
  // Two-pass to avoid iterator invalidation during in-loop mutation
  // (split survivors insert into per_ctx).
  std::vector<uint64_t> overlapping_bases;
  for (const auto &kv : per_ctx) {
    uint64_t base = kv.first;
    uint32_t alloc_id = kv.second;
    if (alloc_id < 1 || alloc_id - 1 >= g_history.size()) continue;
    uint64_t alloc_end = g_history[alloc_id - 1].base +
                         g_history[alloc_id - 1].size;
    if (base < unmap_end && alloc_end > ptr) {
      overlapping_bases.push_back(base);
    }
  }
  for (uint64_t base : overlapping_bases) {
    auto it = per_ctx.find(base);
    if (it == per_ctx.end()) continue;
    uint32_t alloc_id = it->second;
    if (alloc_id < 1 || alloc_id - 1 >= g_history.size()) continue;
    const AllocationRecord rec = g_history[alloc_id - 1];  // copy
    const uint64_t alloc_end = rec.base + rec.size;
    // Retire the original record and remove its live-map entry.
    g_history[alloc_id - 1].alive = false;
    per_ctx.erase(it);
    // Pre-range survivor: [rec.base, ptr).
    if (rec.base < ptr) {
      uint64_t survivor_size = ptr - rec.base;
      uint32_t new_id = g_next_allocation_id++;
      g_history.push_back(AllocationRecord{
          ctx, rec.context_id, rec.device_id, rec.base, survivor_size,
          new_id, true});
      per_ctx[rec.base] = new_id;
    }
    // Post-range survivor: [unmap_end, alloc_end).
    if (alloc_end > unmap_end) {
      uint64_t survivor_base = unmap_end;
      uint64_t survivor_size = alloc_end - unmap_end;
      uint32_t new_id = g_next_allocation_id++;
      g_history.push_back(AllocationRecord{
          ctx, rec.context_id, rec.device_id, survivor_base,
          survivor_size, new_id, true});
      per_ctx[survivor_base] = new_id;
    }
  }
}

uint32_t lookup_allocation_id(void *ctx, uint64_t addr) {
  if (ctx == nullptr) return 0;
  std::lock_guard<std::mutex> lock(g_mutex);
  auto ctx_it = g_live.find(ctx);
  if (ctx_it == g_live.end()) return 0;
  const auto &per_ctx = ctx_it->second;
  if (per_ctx.empty()) return 0;
  auto it = per_ctx.upper_bound(addr);
  if (it == per_ctx.begin()) return 0;
  --it;
  uint32_t alloc_id = it->second;
  if (alloc_id < 1 || alloc_id - 1 >= g_history.size()) return 0;
  const AllocationRecord &rec = g_history[alloc_id - 1];
  if (addr < rec.base || addr >= rec.base + rec.size) return 0;
  return alloc_id;
}

std::vector<AllocationRecord> snapshot_history() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_history;
}

void reset_registry_for_testing() {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_live.clear();
  g_history.clear();
  g_context_id.clear();
  g_pending_async_frees.clear();
  g_vmm_reservations.clear();
  g_next_allocation_id = 1;
  g_next_context_id = 1;
}

// Caller must hold g_mutex. Marks the specific allocation_id dead
// in g_history (idempotent if already dead) and erases the live-map
// entry at (ctx, base) only if it still points to that exact id.
// This allocation_id-aware variant is the correct primitive for
// async-free retirement, where the live map at (ctx, base) may
// have been overwritten by a newer same-base allocation between
// queue time and retirement time.
void mark_allocation_id_freed_locked(
    void *ctx, uint64_t base, uint32_t allocation_id) {
  if (allocation_id < 1 || allocation_id - 1 >= g_history.size()) {
    return;
  }
  g_history[allocation_id - 1].alive = false;
  auto ctx_it = g_live.find(ctx);
  if (ctx_it == g_live.end()) return;
  auto &per_ctx = ctx_it->second;
  auto base_it = per_ctx.find(base);
  if (base_it == per_ctx.end()) return;
  // Erase live-map entry only if it still points to the same id;
  // otherwise a newer same-base allocation already took over and
  // must remain live.
  if (base_it->second == allocation_id) {
    per_ctx.erase(base_it);
  }
}

void register_pending_async_free(void *ctx, void *stream, uint64_t base) {
  // `stream == nullptr` is a valid key (CUDA default stream); only
  // null ctx and base==0 are sentinel no-ops.
  if (ctx == nullptr || base == 0) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  auto ctx_live_it = g_live.find(ctx);
  if (ctx_live_it == g_live.end()) return;
  auto base_it = ctx_live_it->second.find(base);
  if (base_it == ctx_live_it->second.end()) return;
  uint32_t allocation_id = base_it->second;
  g_pending_async_frees[ctx][stream].push_back({base, allocation_id});
}

void retire_pending_frees_for_stream(void *ctx, void *stream) {
  // `stream == nullptr` is a valid key (CUDA default stream).
  if (ctx == nullptr) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  auto ctx_it = g_pending_async_frees.find(ctx);
  if (ctx_it == g_pending_async_frees.end()) return;
  auto stream_it = ctx_it->second.find(stream);
  if (stream_it == ctx_it->second.end()) return;
  for (const PendingEntry &entry : stream_it->second) {
    mark_allocation_id_freed_locked(ctx, entry.base, entry.allocation_id);
  }
  ctx_it->second.erase(stream_it);
  if (ctx_it->second.empty()) {
    g_pending_async_frees.erase(ctx_it);
  }
}

void retire_pending_frees_for_context(void *ctx) {
  if (ctx == nullptr) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  auto ctx_it = g_pending_async_frees.find(ctx);
  if (ctx_it == g_pending_async_frees.end()) return;
  for (auto &kv : ctx_it->second) {
    for (const PendingEntry &entry : kv.second) {
      mark_allocation_id_freed_locked(ctx, entry.base, entry.allocation_id);
    }
  }
  g_pending_async_frees.erase(ctx_it);
}

std::vector<PendingFreeRecord> snapshot_pending_frees_for_testing() {
  std::lock_guard<std::mutex> lock(g_mutex);
  std::vector<PendingFreeRecord> out;
  for (const auto &ctx_kv : g_pending_async_frees) {
    for (const auto &stream_kv : ctx_kv.second) {
      for (const PendingEntry &entry : stream_kv.second) {
        out.push_back(
            {ctx_kv.first, stream_kv.first, entry.base, entry.allocation_id});
      }
    }
  }
  return out;
}

void register_vmm_reservation(void *ctx, uint64_t addr, uint64_t size) {
  if (ctx == nullptr || addr == 0 || size == 0) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  g_vmm_reservations[ctx][addr] = size;
}

void release_vmm_reservation(void *ctx, uint64_t addr) {
  if (ctx == nullptr) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  auto ctx_it = g_vmm_reservations.find(ctx);
  if (ctx_it == g_vmm_reservations.end()) return;
  auto addr_it = ctx_it->second.find(addr);
  if (addr_it == ctx_it->second.end()) return;
  uint64_t reservation_size = addr_it->second;
  // Defensive cleanup: if any live allocation intersects the
  // released reservation range, mark it freed. A clean
  // cuMemUnmap before cuMemAddressFree would have already done
  // this; this loop covers the producer-error path where a
  // mapping survives past its containing reservation's release.
  auto live_it = g_live.find(ctx);
  if (live_it != g_live.end()) {
    auto &per_ctx = live_it->second;
    uint64_t reservation_end = addr + reservation_size;
    for (auto live_entry = per_ctx.begin(); live_entry != per_ctx.end(); ) {
      if (live_entry->first >= addr && live_entry->first < reservation_end) {
        uint32_t alloc_id = live_entry->second;
        if (alloc_id >= 1 && alloc_id - 1 < g_history.size()) {
          g_history[alloc_id - 1].alive = false;
        }
        live_entry = per_ctx.erase(live_entry);
      } else {
        ++live_entry;
      }
    }
  }
  ctx_it->second.erase(addr_it);
  if (ctx_it->second.empty()) {
    g_vmm_reservations.erase(ctx_it);
  }
}

std::vector<VmmReservationRecord> snapshot_vmm_reservations_for_testing() {
  std::lock_guard<std::mutex> lock(g_mutex);
  std::vector<VmmReservationRecord> out;
  for (const auto &ctx_kv : g_vmm_reservations) {
    for (const auto &addr_kv : ctx_kv.second) {
      out.push_back({ctx_kv.first, addr_kv.first, addr_kv.second});
    }
  }
  return out;
}

void serialize_history_json(std::ostream &out) {
  std::lock_guard<std::mutex> lock(g_mutex);
  out << "[";
  bool first = true;
  for (const AllocationRecord &rec : g_history) {
    if (!first) out << ",";
    first = false;
    out << "\n  {\"allocation_id\": " << rec.allocation_id
        << ", \"context_id\": " << rec.context_id
        << ", \"device_id\": " << rec.device_id
        << ", \"base_address\": \"0x" << std::hex << rec.base
        << std::dec << "\", \"size_bytes\": " << rec.size
        << ", \"alive\": " << (rec.alive ? "true" : "false")
        << "}";
  }
  if (!first) out << "\n";
  out << "]\n";
}

}  // namespace nvbit_tracer
