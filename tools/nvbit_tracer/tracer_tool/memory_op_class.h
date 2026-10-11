// Copyright (c) 2026, the NVIDIA_Sims_Predict_NCU contributors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Header-only SASS-opcode -> memory-operation-class classifier.
//
// Used by `recv_thread_fun` in `tracer_tool.cu` to populate the
// `dynamic_trace::instruction.memory_op_class` field added in
// AC4 / R9. The classifier is the single source of truth for
// the producer-side mapping; the consumer-side validator
// (`tools/nvbit_tracer/import_trace_contract.py`) and downstream
// coalescer / HRD profiler trust whatever the tracer emits here.
//
// Header-only by design so the unit test
// `tools/nvbit_tracer/tracer_tool/tests/memory_op_class_test.cc`
// can `#include "memory_op_class.h"` and link with `g++` only — no
// NVBit, no protobuf, no CUDA toolkit dependency.
//
// Numeric mirror of `dynamic_trace::MemoryOpClass`:
// the enum values below MUST stay 1:1 with the proto enum in
// `tools/nvbit_tracer/traces_enhanced/dynamic_trace/instruction.proto`.
// The caller (`recv_thread_fun`) static_asserts the mapping and
// converts via `static_cast<dynamic_trace::MemoryOpClass>(...)`. The
// indirection is what lets this header stay dependency-free under
// the test build.
//
// Input contract: callers pass either the canonical short opcode
// (`"LDG"`, `"LDGSTS"`, `"UTMASTG"`) or the full mnemonic with
// qualifiers (`"LDG.E.SYS"`, `"LDG.E.128.SYS"`, `"ATOMG.ADD"`). The
// classifier extracts the prefix preceding the first `.` and maps
// against an exact-match table (or a small set of `starts_with`
// rules for opcode families like `ATOM*` and `UTMA*`). Both the
// short and full forms collapse to the same prefix and therefore
// to the same class — the caller does NOT have to normalize first.
//
// Returns `kUnspecified` for any opcode the classifier does not
// recognize. The caller MUST treat that as a contract violation:
// the tracer should have either left the record entirely (non-
// memory instruction) or emitted a typed
// `unsupported_memory_path` exclusion sidecar instead of a
// memory-classed `instruction` message with an unknown opcode.

#pragma once

#include <string_view>

namespace memory_op_class_classifier {

// Mirror of `dynamic_trace::MemoryOpClass`. Values must stay 1:1
// with the proto enum; see static_assert in `tracer_tool.cu`.
// kTmem (10) was added for Blackwell (sm_100) tensor-memory
// accesses via the `tcgen05.{ld,st}` family (mnemonics `LDT` /
// `STT`). Tensor memory is a Blackwell-new storage class
// dedicated to tensor-core operand staging; classifier routes
// LDT/STT to kTmem here.
enum Value : int {
  kUnspecified  = 0,
  kGlobal       = 1,
  kShared       = 2,
  kLocal        = 3,
  kTexture      = 4,
  kSurface      = 5,
  kAtomic       = 6,
  kTma          = 7,
  kAsyncCopy    = 8,
  kClusterDsmem = 9,
  kTmem         = 10,
};

namespace detail {

inline bool starts_with(std::string_view s, std::string_view prefix) {
  return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}

// Returns true if `qual` appears as a dot-delimited token anywhere in
// the qualifier tail of `opcode` (i.e. after the first `.`). Used to
// detect tail-decorator qualifiers like `MMIO` on `LDS`/`STS` family
// opcodes that signal cluster-distributed shared memory access on
// sm_90+ Hopper. Matching is exact and case-sensitive against each
// dot-delimited token.
inline bool has_qualifier(std::string_view opcode,
                          std::string_view qual) {
  while (true) {
    auto pos = opcode.find('.');
    if (pos == std::string_view::npos) return false;
    opcode.remove_prefix(pos + 1);
    auto next = opcode.find('.');
    auto tok = (next == std::string_view::npos)
                   ? opcode
                   : opcode.substr(0, next);
    if (tok == qual) return true;
    if (next == std::string_view::npos) return false;
  }
}

}  // namespace detail

inline Value classify(std::string_view opcode) {
  // Extract the opcode prefix (everything before the first `.`).
  // `LDG.E.SYS` -> `LDG`; `LDG` -> `LDG`; both classify identically.
  auto dot = opcode.find('.');
  std::string_view head =
      (dot == std::string_view::npos) ? opcode : opcode.substr(0, dot);

  if (head.empty()) return kUnspecified;

  // ASYNC_COPY. `LDGSTS` is the cp.async family load-global-store-
  // shared opcode. Tested before the LDG/STS prefix tests because
  // it is a single token, not a prefix of either.
  if (head == "LDGSTS") return kAsyncCopy;

  // TMA. Tensor-memory-accelerator family on Hopper+: UTMASTG,
  // UTMALDG, UTMACFLDG, UTMACFSTG, etc. Prefix-matched because
  // the TMA family already had several variants at sm_90 and is
  // expected to grow.
  if (detail::starts_with(head, "UTMA")) return kTma;

  // ATOMIC. Covers the `ATOM`, `ATOMS`, `ATOMG`, `RED`, `REDS`,
  // `REDG` families. Prefix-matched: any opcode whose head
  // begins with `ATOM` or `RED` is an atomic / reduction.
  if (detail::starts_with(head, "ATOM") ||
      detail::starts_with(head, "RED")) {
    return kAtomic;
  }

  // SURFACE. `SULD`, `SUST`, `SUATOM`, `SURED`. Exact-match list
  // — the `SU` family is small and well-defined.
  if (head == "SULD" || head == "SUST" ||
      head == "SUATOM" || head == "SURED") {
    return kSurface;
  }

  // TEXTURE. `TEX`, `TLD`, `TLD4`, `TXQ`. Exact-match list.
  if (head == "TEX" || head == "TLD" ||
      head == "TLD4" || head == "TXQ") {
    return kTexture;
  }

  // Generic LD/ST can target shared, local, or global memory. The producer
  // resolves their runtime state space before assigning a class; do not infer
  // it from the descriptor syntax or from a numerical pointer threshold.
  if (head == "LD" || head == "ST") return kUnspecified;
  if (head == "LDG" || head == "STG") return kGlobal;

  // CLUSTER_DSMEM. Distributed shared memory across a thread-block
  // cluster (sm_90+) is accessed via shared-memory ops decorated with
  // a cluster-scope tail qualifier. The most stable SASS signal that
  // surfaces in the dotted opcode tail is `MMIO` on the LDS/STS
  // family — the cluster-DSMEM access goes through the same
  // shared-memory mnemonic as plain shared but carries the `MMIO`
  // decorator that distinguishes the cluster-scope target. This
  // check MUST run before the plain `LDS`/`STS` -> kShared route
  // below; otherwise a cluster DSMEM access would be misclassified
  // as ordinary shared. The heuristic is conservative: bulk cluster
  // DSMEM copies use UTMA* (already routed to kTma above), and any
  // future qualifier we discover can be added here without churning
  // the wire encoding. Targets sm_86 traces (which never produce
  // cluster DSMEM) remain unaffected.
  if ((head == "LDS" || head == "STS" ||
       head == "LDSM" || head == "STSM") &&
      detail::has_qualifier(opcode, "MMIO")) {
    return kClusterDsmem;
  }

  // SHARED. Plain shared load/store plus the matrix-load/store
  // variants `LDSM`/`STSM` (Hopper warp-tile transfers — they
  // target the same shared-memory backing store as `LDS`/`STS`).
  if (head == "LDS" || head == "STS" ||
      head == "LDSM" || head == "STSM") {
    return kShared;
  }

  // LOCAL. Per-thread local memory. `LDL`/`STL`.
  if (head == "LDL" || head == "STL") return kLocal;

  // TMEM. Blackwell (sm_100) tensor-memory accesses via the
  // `tcgen05.{ld,st}` family — mnemonics `LDT` / `STT` plus
  // the matrix-variant `LDTM` / `STTM` (mirrors LDSM/STSM's
  // matrix-tile transfer for the tensor-memory storage class).
  // Exact-match on the head; SASS qualifier tails
  // (`.sync.aligned.b32.x1`, etc.) do not change the
  // classification. Routed BEFORE the `return kUnspecified`
  // fallback so an LDT/STT/LDTM/STTM instruction classifies as
  // kTmem rather than triggering the producer's typed
  // unsupported_memory_path exclusion sidecar.
  if (head == "LDT" || head == "STT" ||
      head == "LDTM" || head == "STTM") {
    return kTmem;
  }

  // CLUSTER_DSMEM (Blackwell uniform cluster-tensor copy/shift
  // path). `UTCCP` (uniform tensor-cluster copy) and
  // `UTCSHIFT` (uniform tensor-cluster shift) are
  // cluster-distributed shared-memory data movement
  // instructions on the uniform datapath. They access the
  // same cluster-DSMEM storage class as the Hopper
  // LDS/STS.MMIO path classified above, but the SASS head
  // is different. Exact-match so a future UTCC* family
  // member doesn't accidentally collapse here.
  if (head == "UTCCP" || head == "UTCSHIFT") return kClusterDsmem;

  return kUnspecified;
}

}  // namespace memory_op_class_classifier
