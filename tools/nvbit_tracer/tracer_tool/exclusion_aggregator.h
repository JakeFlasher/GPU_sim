// Producer-side typed-exclusion aggregator for the NVBit tracer.
//
// AC4 requires unsupported memory paths, unmapped memory ranges,
// and cross-allocation lane spans to surface as typed exclusions
// in `traces/extra_info/`, NOT as silently-dropped records or
// normal records with absent / first-lane-biased identity. This
// translation unit accumulates per-(kernel, instruction_pc,
// op_class, typed_reason) counts during `recv_thread_fun` and
// serializes the totals to schema-valid JSON sidecars at trace
// finalization.
//
// Schemas the emitted JSON conforms to:
//   - the captured unsupported-memory sidecar
//   - the captured unmapped-range sidecar
//   - the captured cross-allocation sidecar
//
// All three schemas share the same per-entry shape (8 fields:
// schema_version, kernel_unique_id, device, slug, excluded_op_class,
// instruction_pc, count, typed_reason); they differ only in the
// `typed_reason` enum constant. The aggregator serializes by
// partitioning entries on `typed_reason` and writes one file per
// reason.
//
// Host-only translation unit (no NVBit / CUDA / nvcc) so the
// aggregator's invariants are unit-testable on hosts without a GPU
// toolchain — same pattern as `allocation_registry.{h,cc}` from
// R21.

#ifndef NVBIT_TRACER_EXCLUSION_AGGREGATOR_H_
#define NVBIT_TRACER_EXCLUSION_AGGREGATOR_H_

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace nvbit_tracer {

struct ExclusionKey {
  uint32_t function_unique_id;
  std::string excluded_op_class;  // "TMA", "ATOMIC", "TEXTURE", ...
  uint64_t instruction_pc;
  std::string typed_reason;       // "unsupported_memory_path",
                                  // "unmapped_memory_range", or
                                  // "cross_allocation_lane_span"
  // AC-T5.2 path_resolved arm. True when the producer-side
  // classifier recognized the opcode and assigned it a
  // memory class (one of the 7 routed-to-exclusion classes
  // per is_excluded_op_class), but the access still routes
  // through the typed-exclusion sidecar because the
  // downstream allocator can't fully model it (e.g., TMEM
  // tensor-memory has no allocator-tracked backing). False
  // when the classifier returned kUnspecified for an
  // unknown opcode head. The serializer emits the field
  // when true; absence / false is treated as "unknown
  // classifier" by downstream consumers.
  bool path_resolved;

  bool operator<(const ExclusionKey &o) const {
    if (function_unique_id != o.function_unique_id)
      return function_unique_id < o.function_unique_id;
    if (typed_reason != o.typed_reason)
      return typed_reason < o.typed_reason;
    if (excluded_op_class != o.excluded_op_class)
      return excluded_op_class < o.excluded_op_class;
    if (instruction_pc != o.instruction_pc)
      return instruction_pc < o.instruction_pc;
    return path_resolved < o.path_resolved;
  }
};

// Increment the count for the given key. Thread-safe. Sentinel input
// (function_unique_id == 0, empty op_class, empty typed_reason) is a
// no-op to defend against partially-populated inst_trace_t records
// reaching the aggregator. `path_resolved` is the AC-T5.2 arm: pass
// true when the classifier recognized the opcode (one of the 7
// excluded-class members per is_excluded_op_class), false when the
// opcode was unknown to the classifier (kUnspecified). Legacy
// call-sites that don't yet thread the arm pass false; the
// serializer omits the field for false-value entries so legacy
// downstream readers stay byte-compatible.
void aggregate(
    uint32_t function_unique_id,
    const std::string &excluded_op_class,
    uint64_t instruction_pc,
    const std::string &typed_reason,
    bool path_resolved = false);

// Serialize the aggregator state to JSON sidecars under `dir_path`,
// partitioned by `typed_reason`. Three output files are written, one
// per known typed_reason:
//   <dir_path>/unsupported_memory_path.json
//   <dir_path>/unmapped_memory_range.json
//   <dir_path>/cross_allocation_lane_span.json
//
// `device_tag` is the producer's compute capability tag (e.g.,
// "sm_86"). `slug_of_function_id` is a callback that looks up the
// kernel slug for a given function_unique_id; it is invoked once per
// unique function_unique_id at serialization time so the per-record
// O(1) hot path stays free of the lookup cost.
//
// If a typed_reason has zero entries, the corresponding output file
// is still written as an empty JSON array `[]` so consumers never see
// "file missing" as a degenerate case.
bool serialize_exclusion_sidecars(
    const std::string &dir_path,
    const std::string &device_tag,
    std::function<std::string(uint32_t)> slug_of_function_id);

// Test-only: clear the aggregator. Production code never calls this.
void reset_aggregator_for_testing();

// Test-only: snapshot the aggregator state as a sorted vector of
// (key, count) pairs. Used by unit tests to introspect the state
// without re-parsing the serialized JSON.
std::vector<std::pair<ExclusionKey, uint64_t>> snapshot_for_testing();

}  // namespace nvbit_tracer

#endif  // NVBIT_TRACER_EXCLUSION_AGGREGATOR_H_
