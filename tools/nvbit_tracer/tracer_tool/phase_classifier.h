// Copyright (c) 2023-2026, Rodrigo Huerta, Mojtaba Abaie Shoushtary,
// Josep-Llorenç Cruz, Antonio González
// Universitat Politecnica de Catalunya
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The Universitat Politecnica de Catalunya nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <string_view>

// phase-injection opcode classifiers for the NVBit tracer.
//
// The three predicates below consume the canonical short mnemonic returned
// by NVBit's `Instr::getOpcodeShort()` (see
// `tools/nvbit_tracer/nvbit_release/core/nvbit.h:74-75`). Callers MUST pass
// `getOpcodeShort()`, not `getOpcode()`: the predicates exact-match against
// the short form, so a full-form input like "BAR.SYNC" or "BAR.SYNC.ARV"
// will NOT match `is_warp_barrier_opcode_short`. Normalization to the
// short form is the NVBit API's responsibility, not the predicates'.
//
// Header-only by design so the unit test
// `tools/nvbit_tracer/tracer_tool/tests/phase_classifier_test.cc` can
// `#include "phase_classifier.h"` and link with `g++` only — no NVBit
// linkage, no `nvcc`, no CUDA toolkit dependency.
//
// Allow-list rationale:
//   - Warp barrier (`phase_kind=3`) is `BAR` only. `MEMBAR`, `DEPBAR`,
//     `LDGDEPBAR`, `ERRBAR`, `CGAERRBAR` are memory-fence / dependency-
//     barrier opcodes and are NOT instrumented as warp barriers. Newer
//     architectures' `SYNCS`, `WARPGROUP`, `WARPGROUPSET`, `UCGABAR.*`
//     are also NOT instrumented in this MVP.
//   - Function terminator covers `RET` and `EXIT`. A device function
//     can return via either; both must close the function-entry pair
//     and emit the function-exit open/close pair.
//   - Branch covers the linear-MVP terminator candidates
//     `{BRA, BRX, JMP, BSSY, BSYNC, EXIT, RET}`. `CALL` is NOT a
//     basic-block terminator for phase-injection purposes; it is
//     routed through the existing CALL_OPS tracing path. The
//     structural CFG-based replacement using `nvbit_get_CFG()` is a
//     documented follow-up; this heuristic remains conservative
//     until that lands.

inline bool is_warp_barrier_opcode_short(std::string_view opcode) {
  return opcode == "BAR";
}

inline bool is_function_terminator_opcode_short(std::string_view opcode) {
  return opcode == "RET" || opcode == "EXIT";
}

inline bool is_branch_opcode_short(std::string_view opcode) {
  return opcode == "BRA" || opcode == "BRX" || opcode == "JMP" ||
         opcode == "BSSY" || opcode == "BSYNC" ||
         opcode == "EXIT" || opcode == "RET";
}

// Opt-in asynchronous-copy order selection. LDGSTS and LDGDEPBAR are exact
// short-opcode matches. DEPBAR additionally requires the canonical SASS
// operand form `DEPBAR.LE SB0, ...`; ordinary SB1--SB5 dependency waits must
// remain unnumbered. The SASS scanner accepts an optional leading predicate
// token and treats comma/semicolon as token delimiters.
inline bool is_async_order_selected_instruction(std::string_view opcode,
                                                std::string_view sass) {
  if (opcode == "LDGSTS" || opcode == "LDGDEPBAR") return true;
  if (opcode != "DEPBAR") return false;

  auto next_token = [&sass]() {
    while (!sass.empty() &&
           (sass.front() == ' ' || sass.front() == '\t' ||
            sass.front() == ',' || sass.front() == ';')) {
      sass.remove_prefix(1);
    }
    const std::size_t end = sass.find_first_of(" \t,;");
    const std::string_view token = sass.substr(0, end);
    if (end == std::string_view::npos) {
      sass = {};
    } else {
      sass.remove_prefix(end);
    }
    return token;
  };

  std::string_view mnemonic = next_token();
  if (!mnemonic.empty() && mnemonic.front() == '@') mnemonic = next_token();
  return mnemonic == "DEPBAR.LE" && next_token() == "SB0";
}
