#pragma once

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace simbench_nvbit {

inline bool format_phase_record_leaf(
    char *buffer,
    std::size_t buffer_size,
    std::uint32_t repeat_index,
    std::uint32_t launch_index) {
  if (buffer == nullptr || buffer_size == 0) {
    return false;
  }
  const int written = std::snprintf(
      buffer, buffer_size, "phase_records_%" PRIu32 "_%" PRIu32 ".bin",
      repeat_index, launch_index);
  return written >= 0 && static_cast<std::size_t>(written) < buffer_size;
}

// A flush timeout can leave useful forensic bytes, but those bytes must never
// match the canonical phase_records_<repeat>_<launch>.bin discovery contract.
// The leading dot and distinct hyphenated stem make this an explicitly
// noncanonical partial-evidence namespace.
inline bool format_partial_phase_record_leaf(
    char *buffer,
    std::size_t buffer_size,
    std::uint32_t repeat_index,
    std::uint32_t launch_index) {
  if (buffer == nullptr || buffer_size == 0) {
    return false;
  }
  const int written = std::snprintf(
      buffer, buffer_size,
      ".partial-phase-records_%" PRIu32 "_%" PRIu32 ".bin", repeat_index,
      launch_index);
  return written >= 0 && static_cast<std::size_t>(written) < buffer_size;
}

inline bool format_phase_record_path(
    char *buffer,
    std::size_t buffer_size,
    const char *traces_path,
    std::uint32_t repeat_index,
    std::uint32_t launch_index) {
  if (buffer == nullptr || buffer_size == 0 || traces_path == nullptr) {
    return false;
  }
  char leaf[96];
  if (!format_phase_record_leaf(leaf, sizeof(leaf), repeat_index,
                                launch_index)) {
    return false;
  }
  const int written =
      std::snprintf(buffer, buffer_size, "%s/%s", traces_path, leaf);
  return written >= 0 && static_cast<std::size_t>(written) < buffer_size;
}

}  // namespace simbench_nvbit
