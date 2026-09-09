#include "exclusion_aggregator.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace nvbit_tracer {

namespace {

std::mutex g_mutex;
std::map<ExclusionKey, uint64_t> g_aggregator;

// Schema constant. A future schema bump must update this AND the JSON
// schema files in lockstep.
constexpr int kSchemaVersion = 1;

// Escape a string for direct interpolation between two `"` JSON
// quotes. Handles `\\`, `\"`, and the C0 control characters
// (0x00-0x1F) per RFC 8259 §7. Production live slugs are simple
// (e.g., `kernel_cp_async_no_sync`) but kernel names CAN contain
// template-instantiation characters (`<`, `>`, `,`, `:`) and
// future workloads may surface stranger inputs; failing closed
// here keeps the sidecar JSON valid even in pathological cases.
// The escape table targets the subset RFC 8259 specifies; any
// remaining 0x20+ ASCII / valid UTF-8 byte passes through
// unchanged.
std::string escape_json_string(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '\\': out.append("\\\\"); break;
      case '"':  out.append("\\\""); break;
      case '\b': out.append("\\b");  break;
      case '\f': out.append("\\f");  break;
      case '\n': out.append("\\n");  break;
      case '\r': out.append("\\r");  break;
      case '\t': out.append("\\t");  break;
      default:
        if (c < 0x20) {
          // Other C0 controls: emit \u00XX.
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out.append(buf);
        } else {
          out.push_back(static_cast<char>(c));
        }
        break;
    }
  }
  return out;
}

bool write_one_typed_sidecar(
    const std::string &filename,
    const std::string &typed_reason,
    const std::string &device_tag,
    const std::map<uint32_t, std::string> &slug_cache,
    const std::map<ExclusionKey, uint64_t> &snapshot) {
  std::ofstream ofs(filename);
  if (!ofs) {
    std::cerr << "Failed to open " << filename
              << " for writing." << std::endl;
    return false;
  }
  ofs << "[";
  bool first = true;
  for (const auto &[key, count] : snapshot) {
    if (key.typed_reason != typed_reason) continue;
    if (!first) ofs << ",";
    first = false;
    auto slug_it = slug_cache.find(key.function_unique_id);
    const std::string &slug =
        (slug_it != slug_cache.end()) ? slug_it->second
                                      : std::string("unknown");
    ofs << "\n  {\"schema_version\": " << kSchemaVersion
        << ", \"kernel_unique_id\": " << key.function_unique_id
        << ", \"device\": \"" << escape_json_string(device_tag)
        << "\", \"slug\": \"" << escape_json_string(slug)
        << "\", \"excluded_op_class\": \""
        << escape_json_string(key.excluded_op_class)
        << "\", \"instruction_pc\": " << key.instruction_pc
        << ", \"count\": " << count
        << ", \"typed_reason\": \""
        << escape_json_string(typed_reason)
        << "\"";
    // AC-T5.2: emit path_resolved=true when the classifier
    // recognized the opcode. Omit the field for false-value
    // entries so legacy downstream readers (who treat
    // absence as "unknown / false") stay byte-compatible.
    if (key.path_resolved) {
      ofs << ", \"path_resolved\": true";
    }
    ofs << "}";
  }
  if (!first) ofs << "\n";
  ofs << "]\n";
  ofs.flush();
  if (!ofs) {
    std::cerr << "Failed while writing " << filename << std::endl;
    return false;
  }
  ofs.close();
  if (!ofs) {
    std::cerr << "Failed to close " << filename << std::endl;
    return false;
  }
  return true;
}

}  // namespace

void aggregate(
    uint32_t function_unique_id,
    const std::string &excluded_op_class,
    uint64_t instruction_pc,
    const std::string &typed_reason,
    bool path_resolved) {
  if (function_unique_id == 0 ||
      excluded_op_class.empty() ||
      typed_reason.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  ExclusionKey key{
      function_unique_id, excluded_op_class, instruction_pc,
      typed_reason, path_resolved};
  g_aggregator[std::move(key)]++;
}

bool serialize_exclusion_sidecars(
    const std::string &dir_path,
    const std::string &device_tag,
    std::function<std::string(uint32_t)> slug_of_function_id) {
  std::map<ExclusionKey, uint64_t> snapshot;
  std::set<uint32_t> unique_function_ids;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    snapshot = g_aggregator;
    for (const auto &[key, count] : snapshot) {
      (void)count;
      unique_function_ids.insert(key.function_unique_id);
    }
  }

  // Build the slug cache once at serialization time so the
  // per-instruction recv-thread path stays free of the lookup cost.
  std::map<uint32_t, std::string> slug_cache;
  for (uint32_t fid : unique_function_ids) {
    slug_cache[fid] =
        slug_of_function_id ? slug_of_function_id(fid)
                            : std::string("unknown");
  }

  // Write all typed-reason files. If a typed_reason has zero
  // entries, the file is still emitted as `[]` so consumers never
  // see "file missing" as a degenerate case.
  const bool unsupported_ok = write_one_typed_sidecar(
      dir_path + "/unsupported_memory_path.json",
      "unsupported_memory_path", device_tag, slug_cache, snapshot);
  const bool unmapped_ok = write_one_typed_sidecar(
      dir_path + "/unmapped_memory_range.json",
      "unmapped_memory_range", device_tag, slug_cache, snapshot);
  const bool cross_allocation_ok = write_one_typed_sidecar(
      dir_path + "/cross_allocation_lane_span.json",
      "cross_allocation_lane_span", device_tag, slug_cache,
      snapshot);

  // Telemetry: count entries per typed_reason for the log line.
  std::size_t unsupported_count = 0;
  std::size_t unmapped_count = 0;
  std::size_t cross_allocation_count = 0;
  for (const auto &[key, count] : snapshot) {
    if (key.typed_reason == "unsupported_memory_path")
      unsupported_count += count;
    else if (key.typed_reason == "unmapped_memory_range")
      unmapped_count += count;
    else if (key.typed_reason == "cross_allocation_lane_span")
      cross_allocation_count += count;
  }
  std::cout << "Serialized exclusion sidecars to " << dir_path
            << " (entries=" << snapshot.size()
            << " distinct keys with total count="
            << unsupported_count + unmapped_count
                + cross_allocation_count
            << "; unsupported=" << unsupported_count
            << ", unmapped=" << unmapped_count
            << ", cross_allocation=" << cross_allocation_count
            << ")"
            << std::endl;
  return unsupported_ok && unmapped_ok && cross_allocation_ok;
}

void reset_aggregator_for_testing() {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_aggregator.clear();
}

std::vector<std::pair<ExclusionKey, uint64_t>>
snapshot_for_testing() {
  std::lock_guard<std::mutex> lock(g_mutex);
  std::vector<std::pair<ExclusionKey, uint64_t>> result;
  result.reserve(g_aggregator.size());
  for (const auto &[key, count] : g_aggregator) {
    result.emplace_back(key, count);
  }
  return result;
}

}  // namespace nvbit_tracer
