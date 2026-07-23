// Host-side regression test for the Blackwell SASS opcode map.
//
// Purpose: lock in the correct routing of two load-bearing SM120 mnemonics
// that were previously misrouted:
//   * CGAERRBAR (CGA error barrier) must route to OP_CGAERRBAR, not OP_ERRBAR.
//   * LDCU (uniform-register constant load) must route to OP_ULDC / UNIFORM_OP,
//     not OP_LDC / LOAD_OP.
//
// Also guards the existing UCGABAR.ARV / UCGABAR.WAIT entries (and their
// underscored aliases) against silent deletion.
//
// Build and run standalone from anywhere in the repo:
//
//   g++ -std=c++17 -Wall -Wextra tools/nvbit_tracer/isa_def/test_opcode_map.cc -o /tmp/test_opcode_map
//   /tmp/test_opcode_map
//
// Or via the Makefile:
//
//   make -C tools/nvbit_tracer/isa_def test_opcode_map
//
// Exits 0 on success, 1 on any assertion failure. No CUDA / NVBit headers
// are touched; only the opcode map itself and its scalar support types.

#include <cstdio>
#include <string>

// Blackwell_OpcodeMap pulls in trace_opcode.h (TraceInstrOpcode enum +
// OpcodeChar struct) and operation_type.h (uarch_op_t enum) transitively.
#include "blackwell_opcode.h"

namespace {

int failures = 0;

// Report a failure and keep going so we see every broken entry in one run,
// rather than stopping on the first miss.
void report_fail(const char *expr, const char *file, int line) {
  std::fprintf(stderr, "FAIL  %s:%d  %s\n", file, line, expr);
  ++failures;
}

#define CHECK(expr)                                  \
  do {                                               \
    if (!(expr)) report_fail(#expr, __FILE__, __LINE__); \
  } while (0)

bool map_contains(const std::string &key) {
  return Blackwell_OpcodeMap.find(key) != Blackwell_OpcodeMap.end();
}

unsigned map_opcode(const std::string &key) {
  return Blackwell_OpcodeMap.at(key).opcode;
}

unsigned map_category(const std::string &key) {
  return Blackwell_OpcodeMap.at(key).opcode_category;
}

}  // namespace

int main() {
  // ---------------------------------------------------------------------
  // AC-1 Positive Test 1: CGAERRBAR routes to OP_CGAERRBAR, not OP_ERRBAR.
  // ---------------------------------------------------------------------
  CHECK(map_contains("CGAERRBAR"));
  CHECK(map_opcode("CGAERRBAR") == OP_CGAERRBAR);
  CHECK(map_category("CGAERRBAR") == MISCELLANEOUS_QUEUE_OP);
  // Negative: must not be aliased onto OP_ERRBAR any more.
  CHECK(map_opcode("CGAERRBAR") != OP_ERRBAR);
  // Sanity: ERRBAR itself stays on OP_ERRBAR.
  CHECK(map_contains("ERRBAR"));
  CHECK(map_opcode("ERRBAR") == OP_ERRBAR);

  // ---------------------------------------------------------------------
  // AC-1 Positive Test 2: LDCU routes to the uniform load form.
  // ---------------------------------------------------------------------
  CHECK(map_contains("LDCU"));
  CHECK(map_opcode("LDCU") == OP_ULDC);
  CHECK(map_category("LDCU") == UNIFORM_OP);
  // Negative: must not be aliased onto OP_LDC / LOAD_OP any more.
  CHECK(map_opcode("LDCU") != OP_LDC);
  CHECK(map_category("LDCU") != LOAD_OP);
  // Sanity: LDC (vector-register form) stays on OP_LDC / LOAD_OP.
  CHECK(map_contains("LDC"));
  CHECK(map_opcode("LDC") == OP_LDC);
  CHECK(map_category("LDC") == LOAD_OP);
  // Sanity: ULDC (the canonical uniform load, already correct) still lives
  // at OP_ULDC. LDCU aliases onto the same target.
  CHECK(map_contains("ULDC"));
  CHECK(map_opcode("ULDC") == OP_ULDC);
  CHECK(map_category("ULDC") == UNIFORM_OP);

  // ---------------------------------------------------------------------
  // AC-1 Negative Test: UCGABAR.* entries stay untouched (both dotted and
  // underscored spellings, ARV and WAIT variants).
  // ---------------------------------------------------------------------
  CHECK(map_contains("UCGABAR.ARV"));
  CHECK(map_opcode("UCGABAR.ARV") == OP_UCGABAR_ARV);
  CHECK(map_category("UCGABAR.ARV") == UNIFORM_OP);

  CHECK(map_contains("UCGABAR.WAIT"));
  CHECK(map_opcode("UCGABAR.WAIT") == OP_UCGABAR_WAIT);
  CHECK(map_category("UCGABAR.WAIT") == UNIFORM_OP);

  CHECK(map_contains("UCGABAR_ARV"));
  CHECK(map_opcode("UCGABAR_ARV") == OP_UCGABAR_ARV);
  CHECK(map_category("UCGABAR_ARV") == UNIFORM_OP);

  CHECK(map_contains("UCGABAR_WAIT"));
  CHECK(map_opcode("UCGABAR_WAIT") == OP_UCGABAR_WAIT);
  CHECK(map_category("UCGABAR_WAIT") == UNIFORM_OP);

  if (failures == 0) {
    std::printf(
        "OK  %s  (14 checks: CGAERRBAR/LDCU routed correctly, "
        "UCGABAR.* / UCGABAR_* intact)\n",
        "blackwell_opcode.h");
    return 0;
  }
  std::fprintf(stderr, "%d assertion(s) failed\n", failures);
  return 1;
}
