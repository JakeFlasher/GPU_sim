#include "allocation_registry.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <ostream>
#include <iterator>
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

// Stream-ordered memory-pool allocations are not owned by the CUDA context
// that submitted their allocation operation.  Keep a separate process-wide
// live map so a free or a kernel submitted with another context can still
// resolve the same device pointer.  Context-bound allocations remain in
// `g_live` and may legitimately overlap this map only when the producer has
// violated CUDA's address-space contract; current-context lookup wins in that
// case and the sidecar remains conservative.
std::map<uint64_t, uint32_t> g_context_independent_live;

// Append-only full history. Indexed by `allocation_id - 1` since
// allocation_ids start at 1.
std::vector<AllocationRecord> g_history;

// Per-context monotonic context_id assignment. Empty until the first
// `register_allocation` on a context (lookup does not allocate a
// context id). Both producers and consumers see the same
// context_id for the same context handle for the duration of a
// trace run.
std::map<void *, uint32_t> g_context_id;

// Pending stream-ordered async frees: per-(ctx, stream_id) list of
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
// preserved; the inner map is keyed by stream_id so per-stream
// retirement does not affect other streams in the same context.
// Driver stream IDs are resolved by the tracer before they reach this map.
struct PendingEntry {
  uint64_t base;
  uint32_t allocation_id;
};
std::map<void *, std::map<int64_t, std::vector<PendingEntry>>>
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

bool checked_range_end(uint64_t base, uint64_t size, uint64_t &end) {
  if (size == 0 || base > std::numeric_limits<uint64_t>::max() - size) {
    return false;
  }
  end = base + size;
  return true;
}

bool has_allocation_ids_locked(std::size_t count) {
  if (count == 0) return true;
  if (g_next_allocation_id == 0) return false;
  return count <= uint64_t{std::numeric_limits<uint32_t>::max()} -
                      g_next_allocation_id + 1;
}

bool has_pending_allocation_locked(void *ctx, uint32_t allocation_id) {
  auto ctx_it = g_pending_async_frees.find(ctx);
  if (ctx_it == g_pending_async_frees.end()) return false;
  for (const auto &stream_entries : ctx_it->second) {
    for (const PendingEntry &entry : stream_entries.second) {
      if (entry.allocation_id == allocation_id) return true;
    }
  }
  return false;
}

bool has_pending_allocation_locked(uint32_t allocation_id) {
  for (const auto &ctx_entries : g_pending_async_frees) {
    for (const auto &stream_entries : ctx_entries.second) {
      for (const PendingEntry &entry : stream_entries.second) {
        if (entry.allocation_id == allocation_id) return true;
      }
    }
  }
  return false;
}

bool has_disallowed_overlap_in_map_locked(
    const std::map<uint64_t, uint32_t> &live,
    uint64_t base, uint64_t end, void *pending_ctx) {
  auto it = live.lower_bound(base);
  if (it != live.begin()) --it;
  for (; it != live.end() && it->first < end; ++it) {
    const uint32_t allocation_id = it->second;
    if (allocation_id < 1 || allocation_id - 1 >= g_history.size()) continue;
    const AllocationRecord &existing = g_history[allocation_id - 1];
    uint64_t existing_end = 0;
    if (!checked_range_end(existing.base, existing.size, existing_end) ||
        existing.base >= end || existing_end <= base) {
      continue;
    }
    // A same-base replacement is valid only when the previous allocation is
    // queued in the pending async-free table.  Context-independent entries
    // can be freed from any context, so their pending owner is not fixed.
    const bool pending = pending_ctx == nullptr
                             ? has_pending_allocation_locked(allocation_id)
                             : has_pending_allocation_locked(
                                   pending_ctx, allocation_id);
    if (existing.base != base || !pending) return true;
  }
  return false;
}

bool pending_allocation_overlaps_locked(
    void *ctx, uint64_t addr, uint32_t current_allocation_id) {
  if (current_allocation_id < 1 ||
      current_allocation_id - 1 >= g_history.size()) {
    return false;
  }
  const bool current_is_context_independent =
      g_history[current_allocation_id - 1].context_independent;
  for (const auto &ctx_entries : g_pending_async_frees) {
    // Context-bound mappings can only conflict with a pending free in their
    // own context.  Context-independent pool mappings can be freed from any
    // context, so their pending entries must be considered process-wide.
    if (!current_is_context_independent && ctx_entries.first != ctx) continue;
    for (const auto &stream_entries : ctx_entries.second) {
      for (const PendingEntry &entry : stream_entries.second) {
        if (entry.allocation_id == current_allocation_id ||
            entry.allocation_id < 1 ||
            entry.allocation_id - 1 >= g_history.size()) {
          continue;
        }
        const AllocationRecord &pending = g_history[entry.allocation_id - 1];
        if (pending.context_independent != current_is_context_independent)
          continue;
        uint64_t pending_end = 0;
        if (pending.alive && checked_range_end(
                                 pending.base, pending.size, pending_end) &&
            addr >= pending.base && addr < pending_end) {
          return true;
        }
      }
    }
  }
  return false;
}

bool has_disallowed_overlap_locked(void *ctx, uint64_t base, uint64_t end) {
  auto ctx_it = g_live.find(ctx);
  if (ctx_it == g_live.end()) return false;
  return has_disallowed_overlap_in_map_locked(ctx_it->second, base, end, ctx);
}

uint32_t allocate_allocation_id_locked() {
  if (g_next_allocation_id == 0) return 0;
  const uint32_t id = g_next_allocation_id;
  if (g_next_allocation_id == std::numeric_limits<uint32_t>::max()) {
    g_next_allocation_id = 0;
  } else {
    ++g_next_allocation_id;
  }
  return id;
}

// Caller must hold g_mutex.
uint32_t assign_context_id_locked(void *ctx) {
  auto it = g_context_id.find(ctx);
  if (it != g_context_id.end()) return it->second;
  if (g_next_context_id == 0) return 0;
  const uint32_t id = g_next_context_id;
  if (g_next_context_id == std::numeric_limits<uint32_t>::max()) {
    g_next_context_id = 0;
  } else {
    ++g_next_context_id;
  }
  g_context_id[ctx] = id;
  return id;
}

}  // namespace

uint32_t register_allocation(
    void *ctx, uint64_t base, uint64_t size, int device_id) {
  if (ctx == nullptr || base == 0 || size == 0) return 0;
  uint64_t end = 0;
  if (!checked_range_end(base, size, end)) return 0;
  (void)end;
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!has_allocation_ids_locked(1)) return 0;
  if (has_disallowed_overlap_locked(ctx, base, end)) return 0;
  const uint32_t context_id = assign_context_id_locked(ctx);
  if (context_id == 0) return 0;
  const uint32_t alloc_id = allocate_allocation_id_locked();
  if (alloc_id == 0) return 0;
  g_history.push_back(AllocationRecord{
      ctx, context_id, device_id, base, size, alloc_id, true, false});
  g_live[ctx][base] = alloc_id;
  return alloc_id;
}

uint32_t register_context_independent_allocation(
    void *ctx, uint64_t base, uint64_t size, int device_id) {
  if (ctx == nullptr || base == 0 || size == 0) return 0;
  uint64_t end = 0;
  if (!checked_range_end(base, size, end)) return 0;
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!has_allocation_ids_locked(1)) return 0;
  if (has_disallowed_overlap_in_map_locked(
          g_context_independent_live, base, end, nullptr)) {
    return 0;
  }
  const uint32_t context_id = assign_context_id_locked(ctx);
  if (context_id == 0) return 0;
  const uint32_t alloc_id = allocate_allocation_id_locked();
  if (alloc_id == 0) return 0;
  g_history.push_back(AllocationRecord{
      ctx, context_id, device_id, base, size, alloc_id, true, true});
  g_context_independent_live[base] = alloc_id;
  return alloc_id;
}

// Caller must hold g_mutex. Performs the actual mark-freed work
// shared between the synchronous mark_allocation_freed entry
// point and the per-stream / per-ctx async retirement helpers.
// No-op when (ctx, base) is unknown or already freed.
void mark_allocation_freed_locked(void *ctx, uint64_t base) {
  auto ctx_it = g_live.find(ctx);
  if (ctx_it != g_live.end()) {
    auto &per_ctx = ctx_it->second;
    auto base_it = per_ctx.find(base);
    if (base_it != per_ctx.end()) {
      uint32_t alloc_id = base_it->second;
      if (alloc_id >= 1 && alloc_id - 1 < g_history.size()) {
        g_history[alloc_id - 1].alive = false;
      }
      per_ctx.erase(base_it);
      return;
    }
  }
  auto global_it = g_context_independent_live.find(base);
  if (global_it == g_context_independent_live.end()) return;
  const uint32_t alloc_id = global_it->second;
  if (alloc_id >= 1 && alloc_id - 1 < g_history.size()) {
    g_history[alloc_id - 1].alive = false;
  }
  g_context_independent_live.erase(global_it);
}

void mark_allocation_freed(void *ctx, uint64_t base) {
  if (ctx == nullptr) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  mark_allocation_freed_locked(ctx, base);
}

void retire_context_allocations(void *ctx) {
  if (ctx == nullptr) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  auto live_it = g_live.find(ctx);
  if (live_it != g_live.end()) {
    for (const auto &entry : live_it->second) {
      const uint32_t id = entry.second;
      if (id >= 1 && id - 1 < g_history.size()) {
        g_history[id - 1].alive = false;
      }
    }
    g_live.erase(live_it);
  }
  // VMM reservations are context-bound even when their physical mappings
  // were never observed by the callback stream; discard them with the context
  // so a recycled context handle cannot inherit stale reservation identity.
  g_vmm_reservations.erase(ctx);
}

void mark_allocation_freed_range(void *ctx, uint64_t ptr, uint64_t size) {
  if (ctx == nullptr || ptr == 0 || size == 0) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  auto ctx_it = g_live.find(ctx);
  if (ctx_it == g_live.end()) return;
  auto &per_ctx = ctx_it->second;
  uint64_t unmap_end = 0;
  if (!checked_range_end(ptr, size, unmap_end)) return;
  // Collect bases of allocations that overlap [ptr, ptr+size).
  // Two-pass to avoid iterator invalidation during in-loop mutation
  // (split survivors insert into per_ctx).
  std::vector<uint64_t> overlapping_bases;
  std::size_t survivor_count = 0;
  for (const auto &kv : per_ctx) {
    uint64_t base = kv.first;
    uint32_t alloc_id = kv.second;
    if (alloc_id < 1 || alloc_id - 1 >= g_history.size()) continue;
    uint64_t alloc_end = 0;
    if (!checked_range_end(g_history[alloc_id - 1].base,
                           g_history[alloc_id - 1].size, alloc_end)) {
      continue;
    }
    if (base < unmap_end && alloc_end > ptr) {
      const std::size_t added_survivors =
          static_cast<std::size_t>(base < ptr) +
          static_cast<std::size_t>(alloc_end > unmap_end);
      if (survivor_count > std::numeric_limits<std::size_t>::max() -
                               added_survivors) return;
      survivor_count += added_survivors;
      overlapping_bases.push_back(base);
    }
  }
  // Reserve the complete operation's identity demand before changing any live
  // mapping. Exhaustion must not publish only one side of a split.
  if (!has_allocation_ids_locked(survivor_count)) return;
  for (uint64_t base : overlapping_bases) {
    auto it = per_ctx.find(base);
    if (it == per_ctx.end()) continue;
    uint32_t alloc_id = it->second;
    if (alloc_id < 1 || alloc_id - 1 >= g_history.size()) continue;
    const AllocationRecord rec = g_history[alloc_id - 1];  // copy
    uint64_t alloc_end = 0;
    if (!checked_range_end(rec.base, rec.size, alloc_end)) continue;
    // Retire the original record and remove its live-map entry.
    g_history[alloc_id - 1].alive = false;
    per_ctx.erase(it);
    // Pre-range survivor: [rec.base, ptr).
    if (rec.base < ptr) {
      uint64_t survivor_size = ptr - rec.base;
      const uint32_t new_id = allocate_allocation_id_locked();
      g_history.push_back(AllocationRecord{
          ctx, rec.context_id, rec.device_id, rec.base, survivor_size,
          new_id, true, rec.context_independent});
      per_ctx[rec.base] = new_id;
    }
    // Post-range survivor: [unmap_end, alloc_end).
    if (alloc_end > unmap_end) {
      uint64_t survivor_base = unmap_end;
      uint64_t survivor_size = alloc_end - unmap_end;
      const uint32_t new_id = allocate_allocation_id_locked();
      g_history.push_back(AllocationRecord{
          ctx, rec.context_id, rec.device_id, survivor_base,
          survivor_size, new_id, true, rec.context_independent});
      per_ctx[survivor_base] = new_id;
    }
  }
}

uint32_t lookup_allocation_id(void *ctx, uint64_t addr) {
  if (ctx == nullptr) return 0;
  std::lock_guard<std::mutex> lock(g_mutex);
  auto ctx_it = g_live.find(ctx);
  if (ctx_it != g_live.end() && !ctx_it->second.empty()) {
    const auto &per_ctx = ctx_it->second;
    auto it = per_ctx.upper_bound(addr);
    if (it != per_ctx.begin()) {
      --it;
      const uint32_t alloc_id = it->second;
      if (alloc_id >= 1 && alloc_id - 1 < g_history.size()) {
        const AllocationRecord &rec = g_history[alloc_id - 1];
        uint64_t end = 0;
        if (checked_range_end(rec.base, rec.size, end) &&
            addr >= rec.base && addr < end) {
          // An exact-base async-free reuse can leave an older allocation live
          // in the pending table while a newer allocation owns the live-map
          // slot. Without a launch/time identity, assigning either ID would
          // be a false attribution. Fail closed until retirement.
          if (pending_allocation_overlaps_locked(ctx, addr, alloc_id)) return 0;
          return alloc_id;
        }
      }
    }
  }

  // Stream-ordered pool allocations are context-independent. Under CUDA's
  // 64-bit unified-addressing contract their pointer values are process-wide,
  // so a lookup from a different context resolves through this map.
  auto global_it = g_context_independent_live.upper_bound(addr);
  if (global_it != g_context_independent_live.begin()) {
    --global_it;
    const uint32_t alloc_id = global_it->second;
    if (alloc_id >= 1 && alloc_id - 1 < g_history.size()) {
      const AllocationRecord &rec = g_history[alloc_id - 1];
      uint64_t end = 0;
      if (checked_range_end(rec.base, rec.size, end) &&
          addr >= rec.base && addr < end) {
        // A global allocation may be freed from any context, so inspect all
        // pending-free queues for a conflicting older generation.
        if (pending_allocation_overlaps_locked(ctx, addr, alloc_id)) return 0;
        return alloc_id;
      }
    }
  }
  return 0;
}

AllocationSnapshot snapshot_live_allocations(void *ctx) {
  std::lock_guard<std::mutex> lock(g_mutex);
  AllocationSnapshot out;
  auto append_live = [](const auto &live,
                        std::vector<AllocationRecord> &records) {
    for (const auto &entry : live) {
      const uint32_t id = entry.second;
      if (id < 1 || id - 1 >= g_history.size()) continue;
      const AllocationRecord &record = g_history[id - 1];
      if (record.alive) records.push_back(record);
    }
  };
  auto ctx_it = g_live.find(ctx);
  if (ctx_it != g_live.end()) append_live(ctx_it->second, out.context_bound);
  append_live(g_context_independent_live, out.context_independent);
  return out;
}

namespace {

uint32_t lookup_allocation_id_in_sorted_snapshot_section(
    const std::vector<AllocationRecord> &records, uint64_t addr) {
  const auto it = std::upper_bound(
      records.begin(), records.end(), addr,
      [](uint64_t address, const AllocationRecord &record) {
        return address < record.base;
      });
  if (it == records.begin()) return 0;
  const AllocationRecord &record = *std::prev(it);
  uint64_t end = 0;
  if (!record.alive || !checked_range_end(record.base, record.size, end) ||
      addr < record.base || addr >= end) {
    return 0;
  }
  return record.allocation_id;
}

}  // namespace

uint32_t lookup_allocation_id_in_snapshot(
    const AllocationSnapshot &snapshot, uint64_t addr) {
  const uint32_t context_id = lookup_allocation_id_in_sorted_snapshot_section(
      snapshot.context_bound, addr);
  const uint32_t context_independent_id =
      lookup_allocation_id_in_sorted_snapshot_section(
          snapshot.context_independent, addr);
  if (context_id != 0 && context_independent_id != 0 &&
      context_id != context_independent_id) {
    return 0;
  }
  return context_id != 0 ? context_id : context_independent_id;
}

std::vector<AllocationRecord> snapshot_history() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_history;
}

void reset_registry_for_testing() {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_live.clear();
  g_context_independent_live.clear();
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
  if (ctx_it != g_live.end()) {
    auto &per_ctx = ctx_it->second;
    auto base_it = per_ctx.find(base);
    if (base_it != per_ctx.end() && base_it->second == allocation_id) {
      per_ctx.erase(base_it);
      return;
    }
  }
  auto global_it = g_context_independent_live.find(base);
  if (global_it != g_context_independent_live.end() &&
      global_it->second == allocation_id) {
    g_context_independent_live.erase(global_it);
  }
}

void register_pending_async_free(void *ctx, int64_t stream_id, uint64_t base) {
  // Only null ctx and base==0 are sentinel no-ops. The stream ID is resolved
  // by the tracer through the driver API before entering this registry.
  if (ctx == nullptr || base == 0) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  auto ctx_live_it = g_live.find(ctx);
  uint32_t allocation_id = 0;
  if (ctx_live_it != g_live.end()) {
    auto base_it = ctx_live_it->second.find(base);
    if (base_it != ctx_live_it->second.end()) allocation_id = base_it->second;
  }
  if (allocation_id == 0) {
    auto global_it = g_context_independent_live.find(base);
    if (global_it != g_context_independent_live.end())
      allocation_id = global_it->second;
  }
  if (allocation_id == 0) return;
  g_pending_async_frees[ctx][stream_id].push_back({base, allocation_id});
}

void retire_pending_frees_for_stream(void *ctx, int64_t stream_id) {
  if (ctx == nullptr) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  auto ctx_it = g_pending_async_frees.find(ctx);
  if (ctx_it == g_pending_async_frees.end()) return;
  auto stream_it = ctx_it->second.find(stream_id);
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
  uint64_t end = 0;
  if (!checked_range_end(addr, size, end)) return;
  (void)end;
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
    uint64_t reservation_end = 0;
    if (!checked_range_end(addr, reservation_size, reservation_end)) return;
    for (auto live_entry = per_ctx.begin(); live_entry != per_ctx.end(); ) {
      const uint32_t allocation_id = live_entry->second;
      uint64_t allocation_end = 0;
      const bool overlaps =
          allocation_id >= 1 && allocation_id - 1 < g_history.size() &&
          checked_range_end(g_history[allocation_id - 1].base,
                            g_history[allocation_id - 1].size,
                            allocation_end) &&
          live_entry->first < reservation_end && allocation_end > addr;
      if (overlaps) {
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
