// Unit tests for the three free helpers in `phase_classifier.h`.
//
// Compiles with `g++ -std=c++17` only — no NVBit, no CUDA, no nvcc.
// Driven by `make -C tools/nvbit_tracer/tracer_tool unit-tests`.

#include "../phase_classifier.h"

#include <cassert>
#include <cstdio>
#include <string_view>

int main() {
  // is_warp_barrier_opcode_short: BAR-only allow-list.
  assert(is_warp_barrier_opcode_short("BAR"));

  // Memory-fence / dependency-barrier family must NOT match.
  assert(!is_warp_barrier_opcode_short("MEMBAR"));
  assert(!is_warp_barrier_opcode_short("DEPBAR"));
  assert(!is_warp_barrier_opcode_short("LDGDEPBAR"));
  assert(!is_warp_barrier_opcode_short("ERRBAR"));
  assert(!is_warp_barrier_opcode_short("CGAERRBAR"));

  // Newer-architecture barrier-family mnemonics are out of scope for this
  // MVP: they must NOT be classified as warp barriers.
  assert(!is_warp_barrier_opcode_short("SYNCS"));
  assert(!is_warp_barrier_opcode_short("WARPGROUP"));
  assert(!is_warp_barrier_opcode_short("WARPGROUPSET"));
  assert(!is_warp_barrier_opcode_short("UCGABAR"));

  // Full-form input must NOT match. Callers consume the result of
  // `getOpcodeShort()`, never `getOpcode()`. Baking this into the test
  // surface guards against a future regression where a caller passes
  // the un-normalized full mnemonic.
  assert(!is_warp_barrier_opcode_short("BAR.SYNC"));
  assert(!is_warp_barrier_opcode_short("BAR.SYNC.ARV"));

  // is_function_terminator_opcode_short: covers RET and EXIT.
  assert(is_function_terminator_opcode_short("RET"));
  assert(is_function_terminator_opcode_short("EXIT"));

  // Pascal PRET (push-return-address) MUST NOT match RET. The legacy
  // substring `find("RET")` fired on PRET; the short-opcode helper
  // closes that hole.
  assert(!is_function_terminator_opcode_short("PRET"));

  // Function terminator must NOT be aliased to BAR or branches.
  assert(!is_function_terminator_opcode_short("BAR"));
  assert(!is_function_terminator_opcode_short("BRA"));
  assert(!is_function_terminator_opcode_short("CALL"));

  // is_branch_opcode_short: required allow-list of branch terminators.
  assert(is_branch_opcode_short("BRA"));
  assert(is_branch_opcode_short("BRX"));
  assert(is_branch_opcode_short("JMP"));
  assert(is_branch_opcode_short("BSSY"));
  assert(is_branch_opcode_short("BSYNC"));
  assert(is_branch_opcode_short("EXIT"));
  assert(is_branch_opcode_short("RET"));

  // CALL is routed via the existing CALL_OPS path; classifying it as a
  // basic-block terminator would double-count. High-value regression case.
  assert(!is_branch_opcode_short("CALL"));

  // BAR is a warp barrier, NOT a branch terminator. Distinguishes the
  // two predicates from each other.
  assert(!is_branch_opcode_short("BAR"));

  // A sample of unrelated short mnemonics must NOT match any predicate.
  assert(!is_branch_opcode_short("FADD"));
  assert(!is_branch_opcode_short("LDG"));
  assert(!is_branch_opcode_short("MEMBAR"));

  // Empty input is well-defined and must NOT match any predicate.
  assert(!is_warp_barrier_opcode_short(""));
  assert(!is_function_terminator_opcode_short(""));
  assert(!is_branch_opcode_short(""));

  std::printf("phase_classifier_test: PASS\n");
  return 0;
}
