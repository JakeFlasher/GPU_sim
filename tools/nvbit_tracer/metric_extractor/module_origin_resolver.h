// Header-only field-based parser for /proc/self/maps lookups that map a
// host-side pointer to the pathname of the .so / cubin / mapped file that
// contains it. Used by metric_extractor.cu's cuModuleLoadData/DataEx/FatBinary
// at-exit arms to capture the on-disk origin of an in-memory image at
// load time (before torch's JIT tempdir .so files can be unlinked).
//
// The lookup is intentionally pure (no atomics, no I/O outside the
// caller-supplied stream) so it builds into a standalone host unit test
// without any NVBit / CUDA / nvcc surface. The thin wrapper in
// metric_extractor.cu opens "/proc/self/maps" as an std::ifstream and updates
// the appropriate atomic counters based on the returned MapsLookupKind.
//
// The pathname tail is read verbatim, preserving a trailing " (deleted)"
// suffix exactly as the kernel reports it, so the caller's filter regex
// can match either the live path or the post-unlink path explicitly.

#ifndef NVBIT_TRACER_MODULE_ORIGIN_RESOLVER_H_
#define NVBIT_TRACER_MODULE_ORIGIN_RESOLVER_H_

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <istream>
#include <sstream>
#include <string>

namespace nvbit_tracer {

enum class MapsLookupKind {
  Mapped,    // host_ptr fell inside a range whose pathname tail was a real path
  Inline,    // host_ptr fell inside an anonymous mapping (no pathname tail)
  NotFound,  // host_ptr fell outside every parsed range, or stream was empty
  BadInput,  // host_ptr was nullptr; no stream consumed
};

struct MapsLookupResult {
  std::string path;
  MapsLookupKind kind = MapsLookupKind::NotFound;
};

// Field-based parse of a /proc/self/maps-shaped stream. Each line is
// expected to start with `<lo>-<hi> <perms> <offset> <dev> <inode>` followed
// by optional leading whitespace and an optional pathname tail. Malformed
// lines are skipped silently — the parser never throws.
//
// Two distinct mapping kinds resolve to the Inline marker
// (`<inline:0x%lx>` where %lx is the target pointer):
//   1. Truly anonymous mappings — no pathname tail at all (heap arenas,
//      MAP_ANONYMOUS regions, JIT-emitted code regions before they back
//      a file).
//   2. Pseudo-path mappings whose pathname tail starts with `[` — the
//      kernel's representation for `[heap]`, `[stack]`, `[stack:NNN]`,
//      `[vdso]`, `[vvar]`, `[vsyscall]`, and any future `[anon:...]`
//      variants. These are NOT real on-disk files; an in-memory image
//      that lives inside such a region is heap-resident by definition
//      and must SKIP under the filter-active policy.
//
// Both kinds give the caller a stable Inline marker that no on-disk
// regex will accept by accident.
inline MapsLookupResult lookup_addr_in_maps_stream(const void* host_ptr,
                                                   std::istream& maps) {
  MapsLookupResult result;
  if (host_ptr == nullptr) {
    result.kind = MapsLookupKind::BadInput;
    return result;
  }
  const uintptr_t target = reinterpret_cast<uintptr_t>(host_ptr);
  std::string line;
  while (std::getline(maps, line)) {
    std::istringstream iss(line);
    std::string range, perms, offset, dev, inode;
    if (!(iss >> range >> perms >> offset >> dev >> inode)) continue;
    const auto dash = range.find('-');
    if (dash == std::string::npos) continue;
    char* endp = nullptr;
    const uintptr_t lo = std::strtoull(range.c_str(), &endp, 16);
    if (endp == range.c_str()) continue;
    const uintptr_t hi = std::strtoull(range.c_str() + dash + 1, &endp, 16);
    if (endp == range.c_str() + dash + 1) continue;
    if (target < lo || target >= hi) continue;
    std::string pathname;
    std::getline(iss, pathname);
    const size_t first_non_ws = pathname.find_first_not_of(" \t");
    const bool empty_tail = (first_non_ws == std::string::npos);
    const bool pseudo_path_tail =
        !empty_tail && pathname[first_non_ws] == '[';
    if (empty_tail || pseudo_path_tail) {
      char buf[64];
      std::snprintf(buf, sizeof(buf), "<inline:0x%lx>",
                    static_cast<unsigned long>(target));
      result.path = buf;
      result.kind = MapsLookupKind::Inline;
      return result;
    }
    result.path = pathname.substr(first_non_ws);
    result.kind = MapsLookupKind::Mapped;
    return result;
  }
  result.kind = MapsLookupKind::NotFound;
  return result;
}

}  // namespace nvbit_tracer

#endif  // NVBIT_TRACER_MODULE_ORIGIN_RESOLVER_H_
