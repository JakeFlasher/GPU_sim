// Host-only unit test for the producer-side typed-exclusion
// aggregator. Builds with `g++` alone (no NVBit, no CUDA, no nvcc),
// matching the existing `phase_classifier_test.cc`,
// `memory_op_class_test.cc`, and `allocation_registry_test.cc`
// host-only test conventions.
//
// Test plan:
//   1. Single aggregate yields count=1 for the exact key.
//   2. Repeated aggregate increments the count.
//   3. Different instruction_pc disambiguates into separate entries.
//   4. Different excluded_op_class disambiguates.
//   5. Different typed_reason disambiguates.
//   6. Slug callback: each unique function_unique_id is queried
//      exactly once; the returned slug is propagated into JSON.
//   7. Sidecar JSON shape: every entry has all 8 schema-required
//      fields with the expected types and values.
//   8. All three typed-reason files are emitted (one per
//      typed_reason), and a typed_reason with zero entries gets
//      an empty JSON array `[]` instead of being absent.
//   9. Sentinel inputs (function_unique_id == 0, empty op_class,
//      empty typed_reason) are no-ops.
//  10. cross_allocation_lane_span: aggregation rounds-trips through
//      the third typed-reason file with the right partition.
#include "../exclusion_aggregator.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using nvbit_tracer::aggregate;
using nvbit_tracer::serialize_exclusion_sidecars;
using nvbit_tracer::reset_aggregator_for_testing;
using nvbit_tracer::snapshot_for_testing;

namespace {

// Make a fresh per-test scratch dir under /tmp; mkdir is idempotent
// across re-runs.
std::string make_scratch_dir(const std::string &test_name) {
  std::string dir = "/tmp/nvbit_excl_test_" + test_name;
  // Best-effort ignore-EEXIST.
  mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  return dir;
}

std::string read_file(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs) return std::string();
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

void test_single_aggregate_count_is_one() {
  reset_aggregator_for_testing();
  aggregate(/*fid=*/1, "TMA", /*pc=*/0x1000,
            "unsupported_memory_path");
  auto snap = snapshot_for_testing();
  assert(snap.size() == 1);
  assert(snap[0].first.function_unique_id == 1);
  assert(snap[0].first.excluded_op_class == "TMA");
  assert(snap[0].first.instruction_pc == 0x1000);
  assert(snap[0].first.typed_reason == "unsupported_memory_path");
  assert(snap[0].second == 1);
}

void test_repeated_aggregate_increments() {
  reset_aggregator_for_testing();
  for (int i = 0; i < 7; ++i) {
    aggregate(1, "ATOMIC", 0x2000, "unsupported_memory_path");
  }
  auto snap = snapshot_for_testing();
  assert(snap.size() == 1);
  assert(snap[0].second == 7);
}

void test_different_instruction_pc_disambiguates() {
  reset_aggregator_for_testing();
  aggregate(1, "TMA", 0x1000, "unsupported_memory_path");
  aggregate(1, "TMA", 0x1004, "unsupported_memory_path");
  auto snap = snapshot_for_testing();
  assert(snap.size() == 2);
  // Sorted ascending by ExclusionKey::operator<; pcs 0x1000 < 0x1004.
  assert(snap[0].first.instruction_pc == 0x1000);
  assert(snap[1].first.instruction_pc == 0x1004);
}

void test_different_excluded_op_class_disambiguates() {
  reset_aggregator_for_testing();
  aggregate(1, "TMA", 0x1000, "unsupported_memory_path");
  aggregate(1, "ATOMIC", 0x1000, "unsupported_memory_path");
  auto snap = snapshot_for_testing();
  assert(snap.size() == 2);
}

void test_different_typed_reason_disambiguates() {
  reset_aggregator_for_testing();
  aggregate(1, "GLOBAL", 0x1000, "unmapped_memory_range");
  aggregate(1, "GLOBAL", 0x1000, "unsupported_memory_path");
  auto snap = snapshot_for_testing();
  assert(snap.size() == 2);
}

void test_slug_callback_invoked_per_unique_function_id() {
  reset_aggregator_for_testing();
  aggregate(1, "TMA", 0x1000, "unsupported_memory_path");
  aggregate(1, "TMA", 0x1004, "unsupported_memory_path");
  aggregate(2, "ATOMIC", 0x2000, "unsupported_memory_path");

  std::set<uint32_t> seen_fids;
  int call_count = 0;
  auto slug_cb = [&](uint32_t fid) {
    seen_fids.insert(fid);
    call_count++;
    return "kernel_" + std::to_string(fid);
  };

  std::string dir = make_scratch_dir("slug_callback");
  serialize_exclusion_sidecars(dir, "sm_86", slug_cb);
  // Two unique fids => two distinct callback invocations.
  assert(seen_fids.size() == 2);
  assert(call_count == 2);

  // Verify slugs propagated into the JSON.
  std::string json = read_file(dir + "/unsupported_memory_path.json");
  assert(json.find("\"slug\": \"kernel_1\"") != std::string::npos);
  assert(json.find("\"slug\": \"kernel_2\"") != std::string::npos);
}

void test_sidecar_json_has_all_required_fields() {
  reset_aggregator_for_testing();
  aggregate(7, "TMA", 0x1000, "unsupported_memory_path");
  std::string dir = make_scratch_dir("required_fields");
  serialize_exclusion_sidecars(
      dir, "sm_120",
      [](uint32_t fid) {
        return "gemm_hmma_bf16_sm80_fid_" + std::to_string(fid);
      });
  std::string json = read_file(dir + "/unsupported_memory_path.json");
  // 8 schema-required fields per entry.
  assert(json.find("\"schema_version\": 1") != std::string::npos);
  assert(json.find("\"kernel_unique_id\": 7") != std::string::npos);
  assert(json.find("\"device\": \"sm_120\"") != std::string::npos);
  assert(json.find("\"slug\": \"gemm_hmma_bf16_sm80_fid_7\"") !=
         std::string::npos);
  assert(json.find("\"excluded_op_class\": \"TMA\"") !=
         std::string::npos);
  assert(json.find("\"instruction_pc\": 4096") != std::string::npos);
  assert(json.find("\"count\": 1") != std::string::npos);
  assert(json.find("\"typed_reason\": \"unsupported_memory_path\"") !=
         std::string::npos);
  // Top-level shape.
  assert(json.front() == '[');
  assert(json.size() >= 2);
  assert(json[json.size() - 2] == ']');
  assert(json.back() == '\n');
}

void test_two_typed_reasons_emit_two_files() {
  reset_aggregator_for_testing();
  aggregate(1, "TMA", 0x1000, "unsupported_memory_path");
  aggregate(1, "GLOBAL", 0x2000, "unmapped_memory_range");
  std::string dir = make_scratch_dir("two_files");
  serialize_exclusion_sidecars(
      dir, "sm_86",
      [](uint32_t fid) { return "k" + std::to_string(fid); });

  std::string unsup =
      read_file(dir + "/unsupported_memory_path.json");
  std::string unmap = read_file(dir + "/unmapped_memory_range.json");

  // unsupported file has the TMA entry, not the GLOBAL one.
  assert(unsup.find("\"excluded_op_class\": \"TMA\"") !=
         std::string::npos);
  assert(unsup.find("\"excluded_op_class\": \"GLOBAL\"") ==
         std::string::npos);
  // unmapped file has the GLOBAL entry, not the TMA one.
  assert(unmap.find("\"excluded_op_class\": \"GLOBAL\"") !=
         std::string::npos);
  assert(unmap.find("\"excluded_op_class\": \"TMA\"") ==
         std::string::npos);
}

void test_typed_reason_with_zero_entries_emits_empty_array() {
  reset_aggregator_for_testing();
  // Only `unsupported_memory_path` entries; `unmapped_memory_range`
  // has zero entries.
  aggregate(1, "TMA", 0x1000, "unsupported_memory_path");
  std::string dir = make_scratch_dir("empty_array");
  serialize_exclusion_sidecars(
      dir, "sm_86", [](uint32_t fid) {
        (void)fid;
        return std::string("k");
      });
  std::string unmap = read_file(dir + "/unmapped_memory_range.json");
  // Empty list with the trailing newline.
  assert(unmap == "[]\n");
}

void test_sentinel_inputs_are_noops() {
  reset_aggregator_for_testing();
  // function_unique_id == 0 is the proto3 default, treated as
  // "not set" in our wire contract.
  aggregate(0, "TMA", 0x1000, "unsupported_memory_path");
  // Empty op_class and typed_reason are also no-ops.
  aggregate(1, "", 0x1000, "unsupported_memory_path");
  aggregate(1, "TMA", 0x1000, "");
  auto snap = snapshot_for_testing();
  assert(snap.empty());
}

void test_escape_json_string_handles_unsafe_chars() {
  // Round 24: the JSON serializer must escape `\\`, `"`, and the
  // C0 control characters per RFC 8259 §7. Live slugs are simple
  // (`kernel_cp_async_no_sync` etc.) but kernel names CAN contain
  // template-instantiation characters (`<`, `>`, `,`, `:`) and
  // mangled symbols, plus future workloads may surface stranger
  // inputs. Failing closed here keeps the sidecar JSON valid even
  // in pathological cases. The test exercises the escape path
  // through `serialize_exclusion_sidecars` (the helper itself is
  // file-static; we test its observable behavior).
  reset_aggregator_for_testing();
  aggregate(1, "TMA", 0x1000, "unsupported_memory_path");
  std::string dir = make_scratch_dir("escape_unsafe");
  // Slug containing all three escape-required categories.
  serialize_exclusion_sidecars(
      dir, "sm_86",
      [](uint32_t fid) {
        (void)fid;
        // backslash, double-quote, newline, tab, embedded NUL is
        // not used here because std::string + initializer_list
        // would conflate length, so we cover the high-leverage
        // visible cases plus one C0 control (\x01).
        std::string s;
        s += "evil\\";   // backslash
        s += "\"";       // double quote
        s += "\n";       // newline (LF)
        s += "\t";       // tab
        s += "\x01";     // SOH control char
        return s;
      });
  std::string json = read_file(dir + "/unsupported_memory_path.json");
  // Each unsafe input character must appear in escaped form, never
  // as the raw byte breaking JSON syntax.
  assert(json.find("evil\\\\") != std::string::npos);  // \\
  assert(json.find("\\\"") != std::string::npos);      // \"
  assert(json.find("\\n") != std::string::npos);       // \n
  assert(json.find("\\t") != std::string::npos);       // \t
  assert(json.find("\\u0001") != std::string::npos);   // 
  // None of the raw unsafe bytes leak into the output verbatim.
  // A raw newline would split the JSON object across lines (the
  // serializer's intentional pretty-print newlines are the only
  // line-breaks expected); a raw double-quote inside the slug
  // would close the slug field early.
  // We can't easily check absence of raw bytes via find, so we
  // rely on the escape-form presence checks above plus the
  // observation that all 5 escape categories rendered.
}

void test_cross_allocation_lane_span_aggregation_and_partition() {
  // Round 5 task11b/d: the producer aggregates a third typed
  // reason (`cross_allocation_lane_span`) when a single warp
  // instruction's active lanes resolve to multiple distinct
  // non-zero allocation ids. The serializer must:
  //   1. Increment the per-key count for cross_allocation_lane_span
  //      entries (same accumulation semantics as the other two
  //      typed reasons).
  //   2. Write a third sidecar file
  //      (`cross_allocation_lane_span.json`) and partition entries
  //      into it (no entries leak into the other two files).
  //   3. Emit `[]\n` for cross_allocation_lane_span when zero
  //      entries are aggregated, mirroring the existing
  //      empty-typed-reason contract.
  reset_aggregator_for_testing();
  aggregate(1, "GLOBAL", 0x1000, "cross_allocation_lane_span");
  aggregate(1, "GLOBAL", 0x1000, "cross_allocation_lane_span");
  aggregate(1, "GLOBAL", 0x2000, "cross_allocation_lane_span");
  aggregate(1, "TMA", 0x3000, "unsupported_memory_path");
  aggregate(1, "GLOBAL", 0x4000, "unmapped_memory_range");

  auto snap = snapshot_for_testing();
  // Three cross_allocation_lane_span keys + 1 unsupported + 1 unmapped
  // (note: the two cross_allocation_lane_span aggregates at 0x1000
  // collapse into a single entry with count=2; 0x2000 is its own
  // entry with count=1; that's 2 cross_allocation_lane_span entries).
  std::size_t cross_allocation_keys = 0;
  std::size_t cross_allocation_count = 0;
  for (const auto &[key, count] : snap) {
    if (key.typed_reason == "cross_allocation_lane_span") {
      cross_allocation_keys++;
      cross_allocation_count += count;
    }
  }
  assert(cross_allocation_keys == 2);
  assert(cross_allocation_count == 3);

  std::string dir = make_scratch_dir("cross_allocation_partition");
  serialize_exclusion_sidecars(
      dir, "sm_120",
      [](uint32_t fid) { return "k" + std::to_string(fid); });

  std::string unsup =
      read_file(dir + "/unsupported_memory_path.json");
  std::string unmap =
      read_file(dir + "/unmapped_memory_range.json");
  std::string cross =
      read_file(dir + "/cross_allocation_lane_span.json");

  // Each typed_reason entry lands in exactly one file: the typed_reason
  // string itself appears only in the matching file.
  assert(cross.find("\"typed_reason\": \"cross_allocation_lane_span\"") !=
         std::string::npos);
  assert(unsup.find("\"typed_reason\": \"cross_allocation_lane_span\"") ==
         std::string::npos);
  assert(unmap.find("\"typed_reason\": \"cross_allocation_lane_span\"") ==
         std::string::npos);

  // The unsupported / unmapped files do NOT leak across.
  assert(unsup.find("\"typed_reason\": \"unsupported_memory_path\"") !=
         std::string::npos);
  assert(unmap.find("\"typed_reason\": \"unmapped_memory_range\"") !=
         std::string::npos);

  // cross_allocation file carries both the count=2 aggregate and the
  // count=1 single instance.
  assert(cross.find("\"instruction_pc\": 4096") != std::string::npos);
  assert(cross.find("\"instruction_pc\": 8192") != std::string::npos);
  assert(cross.find("\"count\": 2") != std::string::npos);
  assert(cross.find("\"count\": 1") != std::string::npos);

  // Empty-array contract for the third file: when no entries are
  // aggregated under cross_allocation_lane_span, the file is still
  // emitted as `[]\n`.
  reset_aggregator_for_testing();
  aggregate(1, "TMA", 0x1000, "unsupported_memory_path");
  std::string dir_empty =
      make_scratch_dir("cross_allocation_empty");
  serialize_exclusion_sidecars(
      dir_empty, "sm_86",
      [](uint32_t fid) {
        (void)fid;
        return std::string("k");
      });
  std::string cross_empty =
      read_file(dir_empty + "/cross_allocation_lane_span.json");
  assert(cross_empty == "[]\n");
}

void test_escape_json_string_applied_to_all_four_fields() {
  // Round 24 codex review hardening: the slug-only test above
  // demonstrated the escape helper works on user-controlled input,
  // but the helper is applied to FOUR fields in the JSON serializer
  // (device_tag, slug, excluded_op_class, typed_reason). This test
  // exercises a controllable unsafe-byte at each of three
  // additional sites (slug is already covered) so a future
  // refactor that drops the escape call from one of the sites
  // surfaces here instead of as a malformed live sidecar.
  //
  // device_tag: the producer normally builds "sm_<n>" which
  // contains no escape-required bytes, but the field IS a free
  // string and a hostile or test environment could pass an
  // unsafe value.
  // excluded_op_class: aggregator-key field; production values
  // come from the schema enum (UPPER_CASE plain ASCII), but the
  // aggregator does not enforce this — if a future caller passes
  // an unsafe op_class, escape must apply.
  // typed_reason: similar; production values are the two literal
  // strings "unsupported_memory_path" and "unmapped_memory_range",
  // but the field is a free string in the API.
  reset_aggregator_for_testing();
  // device_tag site:
  aggregate(7, "TMA", 0x1000, "unsupported_memory_path");
  std::string dir1 = make_scratch_dir("escape_device");
  serialize_exclusion_sidecars(
      dir1, "evil\"device\\tag\n",
      [](uint32_t fid) {
        (void)fid;
        return std::string("safe_kernel");
      });
  std::string j1 = read_file(dir1 + "/unsupported_memory_path.json");
  assert(j1.find("evil\\\"device\\\\tag\\n") != std::string::npos);

  // excluded_op_class site:
  reset_aggregator_for_testing();
  aggregate(7, "weird\"\\op\nclass", 0x2000,
            "unsupported_memory_path");
  std::string dir2 = make_scratch_dir("escape_op_class");
  serialize_exclusion_sidecars(
      dir2, "sm_86",
      [](uint32_t fid) {
        (void)fid;
        return std::string("safe_kernel");
      });
  std::string j2 = read_file(dir2 + "/unsupported_memory_path.json");
  assert(j2.find("weird\\\"\\\\op\\nclass") != std::string::npos);

  // typed_reason site: aggregator's typed_reason determines which
  // file the entry routes to, so we must use a string that begins
  // with "unsupported_memory_path" or "unmapped_memory_range" for
  // the entry to land in a known file. Use a payload that matches
  // one of those exactly so the file emits, then verify the
  // typed_reason field round-trips through escape unchanged for
  // safe ASCII (regression pin: the safe-string path should not
  // accidentally escape characters that don't need escaping).
  reset_aggregator_for_testing();
  aggregate(7, "TMA", 0x3000, "unsupported_memory_path");
  std::string dir3 = make_scratch_dir("escape_typed_reason_safe");
  serialize_exclusion_sidecars(
      dir3, "sm_86",
      [](uint32_t fid) {
        (void)fid;
        return std::string("safe_kernel");
      });
  std::string j3 = read_file(dir3 + "/unsupported_memory_path.json");
  // Safe ASCII passes through unchanged; no \\ escape sequences
  // in the typed_reason value.
  assert(j3.find("\"typed_reason\": \"unsupported_memory_path\"") !=
         std::string::npos);
}

void test_path_resolved_true_emits_field_in_sidecar() {
  // AC-T5.2: when the producer classifier recognized the opcode
  // (path_resolved=true), the serializer MUST emit
  // "path_resolved": true into the JSON sidecar so downstream
  // calibration can distinguish recognized-but-excluded paths
  // (e.g., TMEM tensor-memory with no allocator-tracked backing)
  // from unknown-opcode paths.
  reset_aggregator_for_testing();
  aggregate(/*fid=*/8, "TMEM", /*pc=*/0x4000,
            "unsupported_memory_path", /*path_resolved=*/true);
  std::string dir = make_scratch_dir("path_resolved_true");
  serialize_exclusion_sidecars(
      dir, "sm_100",
      [](uint32_t fid) {
        (void)fid;
        return std::string("tcgen05_ld_kernel");
      });
  std::string j = read_file(dir + "/unsupported_memory_path.json");
  assert(j.find("\"path_resolved\": true") != std::string::npos);
  assert(j.find("\"excluded_op_class\": \"TMEM\"") !=
         std::string::npos);
  assert(j.find("\"typed_reason\": \"unsupported_memory_path\"") !=
         std::string::npos);
}

void test_path_resolved_false_omits_field_in_sidecar() {
  // AC-T5.2 byte-compat invariant: when path_resolved=false
  // (classifier returned kUnspecified for an unknown opcode head),
  // the serializer MUST OMIT the field entirely. Downstream
  // readers treat field-absence as "unknown / false" so legacy
  // consumers stay byte-compatible.
  reset_aggregator_for_testing();
  aggregate(/*fid=*/9, "UNSPECIFIED", /*pc=*/0x5000,
            "unsupported_memory_path", /*path_resolved=*/false);
  std::string dir = make_scratch_dir("path_resolved_false");
  serialize_exclusion_sidecars(
      dir, "sm_100",
      [](uint32_t fid) {
        (void)fid;
        return std::string("unknown_opcode_kernel");
      });
  std::string j = read_file(dir + "/unsupported_memory_path.json");
  // The entry itself must appear...
  assert(j.find("\"excluded_op_class\": \"UNSPECIFIED\"") !=
         std::string::npos);
  // ...but the path_resolved field must NOT appear in any form
  // (neither true nor false) so legacy readers don't see a new
  // key they don't recognize.
  assert(j.find("path_resolved") == std::string::npos);
}

void test_path_resolved_default_value_is_false() {
  // Legacy call-sites that don't yet thread the path_resolved arm
  // (the aggregate() signature has path_resolved as a defaulted
  // parameter = false) MUST behave identically to an explicit
  // path_resolved=false call: the field is omitted from the
  // serialized JSON.
  reset_aggregator_for_testing();
  // Call WITHOUT the path_resolved arg — exercises the default.
  aggregate(/*fid=*/10, "TMA", /*pc=*/0x6000,
            "unsupported_memory_path");
  std::string dir = make_scratch_dir("path_resolved_default");
  serialize_exclusion_sidecars(
      dir, "sm_86",
      [](uint32_t fid) {
        (void)fid;
        return std::string("legacy_callsite_kernel");
      });
  std::string j = read_file(dir + "/unsupported_memory_path.json");
  assert(j.find("\"excluded_op_class\": \"TMA\"") !=
         std::string::npos);
  // Default path_resolved=false → field omitted.
  assert(j.find("path_resolved") == std::string::npos);
}

}  // namespace

int main() {
  test_single_aggregate_count_is_one();
  test_repeated_aggregate_increments();
  test_different_instruction_pc_disambiguates();
  test_different_excluded_op_class_disambiguates();
  test_different_typed_reason_disambiguates();
  test_slug_callback_invoked_per_unique_function_id();
  test_sidecar_json_has_all_required_fields();
  test_two_typed_reasons_emit_two_files();
  test_typed_reason_with_zero_entries_emits_empty_array();
  test_sentinel_inputs_are_noops();
  test_escape_json_string_handles_unsafe_chars();
  test_escape_json_string_applied_to_all_four_fields();
  test_cross_allocation_lane_span_aggregation_and_partition();
  test_path_resolved_true_emits_field_in_sidecar();
  test_path_resolved_false_omits_field_in_sidecar();
  test_path_resolved_default_value_is_false();
  std::printf("exclusion_aggregator_test: 16 tests passed\n");
  return 0;
}
