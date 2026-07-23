// Copyright (c) 2026, the NVIDIA_Sims_Predict_NCU contributors.
// SPDX-License-Identifier: BSD-3-Clause
//
// Header-only SASS-opcode -> memory-operation access-kind classifier.
//
// Orthogonal to `memory_op_class.h`. The class header tags the memory
// space (global / shared / local / texture / surface / atomic / tma /
// async-copy / cluster-dsmem); this header tags the access direction
// (load / store / atomic-read-modify-write / async-copy). Together
// the two enums let consumers distinguish write-allocate paths
// (global / local stores plus loads plus atomics on those spaces)
// from no-write-allocate paths (shared stores, async-copy
// destination side) without re-deriving the kind from the opcode at
// every consumer site.
//
// Used by `recv_thread_fun` in `tracer_tool.cu` to populate the
// `dynamic_trace::instruction.memory_op_access_kind` field at proto
// tag 12. The classifier is the single source of truth for the
// producer-side direction mapping; the consumer-side validator
// (`tools/nvbit_tracer/import_trace_contract.py`) and the HRD
// profiler trust whatever the tracer emits here.
//
// Header-only by design so the unit test
// `tools/nvbit_tracer/tracer_tool/tests/memory_op_access_kind_test.cc`
// can `#include "memory_op_access_kind.h"` and link with `g++` only
// — no NVBit, no protobuf, no CUDA toolkit dependency.
//
// Numeric mirror of `dynamic_trace::MemoryOpAccessKind`: the enum
// values below MUST stay 1:1 with the proto enum in
// `tools/nvbit_tracer/traces_enhanced/dynamic_trace/instruction.proto`.
// The caller (`recv_thread_fun`) static_asserts the mapping and
// converts via `static_cast<dynamic_trace::MemoryOpAccessKind>(...)`.
// The indirection is what lets this header stay dependency-free
// under the test build.
//
// Input contract: callers pass either the canonical short opcode
// (`"LDG"`, `"LDGSTS"`, `"UTMASTG"`) or the full mnemonic with
// qualifiers (`"LDG.E.SYS"`, `"STG.E.128.SYS"`, `"ATOMG.ADD"`). The
// classifier extracts the prefix preceding the first `.` and maps
// against an exact-match table (or a small set of `starts_with`
// rules for opcode families like `ATOM*` and `RED*`). Both the short
// and full forms collapse to the same prefix and therefore to the
// same kind — the caller does NOT have to normalize first.
//
// Returns `kUnspecified` for any opcode the classifier does not
// recognize. The caller MUST treat that as a contract violation:
// the tracer should have either left the record entirely (non-
// memory instruction) or emitted a typed
// `unsupported_memory_path` exclusion sidecar instead of a
// kind-tagged `instruction` message with an unknown opcode.

#pragma once

#include <string_view>

namespace memory_op_access_kind_classifier {

// Mirror of `dynamic_trace::MemoryOpAccessKind`. Values must stay
// 1:1 with the proto enum; see static_assert in `tracer_tool.cu`.
enum Value : int {
  kUnspecified = 0,
  kLoad        = 1,
  kStore       = 2,
  kAtomic      = 3,
  kAsyncCopy   = 4,
};

namespace detail {

inline bool starts_with(std::string_view s, std::string_view prefix) {
  return s.size() >= prefix.size() &&
         s.substr(0, prefix.size()) == prefix;
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
  // shared opcode — a single instruction that loads from global into
  // shared without going through registers. Both sides happen
  // atomically; the destination side bypasses L2 write-allocate.
  // Tested before the LD/ST prefix tests because LDGSTS is a single
  // token, not a prefix of either `LDG` or `STS`.
  if (head == "LDGSTS") return kAsyncCopy;

  // ATOMIC. Covers the `ATOM`, `ATOMS`, `ATOMG`, `RED`, `REDS`,
  // `REDG` families. Prefix-matched: any opcode whose head begins
  // with `ATOM` or `RED` is a read-modify-write. `SUATOM` and
  // `SURED` (surface atomics / reductions) handled below in the
  // surface block, but they share the atomic kind semantically —
  // both surface variants ALSO return kAtomic.
  if (detail::starts_with(head, "ATOM") ||
      detail::starts_with(head, "RED")) {
    return kAtomic;
  }

  // SURFACE family. `SULD` loads, `SUST` stores, `SUATOM`/`SURED`
  // are atomics. Each surface variant carries an explicit direction.
  if (head == "SULD") return kLoad;
  if (head == "SUST") return kStore;
  if (head == "SUATOM") return kAtomic;
  if (head == "SURED") return kAtomic;

  // TEXTURE family. `TEX` / `TLD` / `TLD4` / `TXQ` are all reads
  // (texture is a read-only path on the device; CUDA exposes no
  // texture-store opcode).
  if (head == "TEX") return kLoad;
  if (head == "TLD") return kLoad;
  if (head == "TLD4") return kLoad;
  if (head == "TXQ") return kLoad;

  // TMA family on Hopper+: `UTMALDG` / `UTMACFLDG` are loads (bulk
  // async copy from global into shared); `UTMASTG` / `UTMACFSTG`
  // are stores (bulk async copy from shared into global). The
  // direction is in the suffix `LDG` / `STG` of the UTMA name —
  // pattern-match on the trailing `LDG` / `STG` token after the
  // `UTMA*` prefix.
  if (detail::starts_with(head, "UTMA")) {
    // After the `UTMA` prefix, the trailing portion encodes the
    // direction. The two head forms in current NVIDIA SASS are
    // `UTMA<modifier>LDG` (load-global) and `UTMA<modifier>STG`
    // (store-global). Modifiers like `CF` (cluster) are optional.
    // Use suffix matching so future modifiers don't break the
    // direction tag.
    auto suffix_check = [&head](std::string_view suffix) {
      return head.size() >= suffix.size() &&
             head.substr(head.size() - suffix.size()) == suffix;
    };
    if (suffix_check("LDG")) return kLoad;
    if (suffix_check("STG")) return kStore;
    // Unknown UTMA shape — fail closed.
    return kUnspecified;
  }

  // GLOBAL. Plain global load / store. `LDGDEPBAR` is a memory-
  // ordering barrier, not a load — exact-match `LDG` / `STG`
  // excludes it because its head token is the full `LDGDEPBAR`,
  // not `LDG`.
  if (head == "LDG") return kLoad;
  if (head == "STG") return kStore;

  // SHARED. Plain shared load / store plus the matrix-load / store
  // variants `LDSM` / `STSM` (Hopper warp-tile transfers — target
  // the same shared-memory backing store as `LDS` / `STS`). All
  // four head tokens are exact-match. Tail qualifiers like `.MMIO`
  // (cluster-DSMEM scope) do not change the load / store direction,
  // so the kind classifier doesn't have to check them here. The
  // direction is in the `LD` vs `ST` prefix of the head.
  if (head == "LDS") return kLoad;
  if (head == "STS") return kStore;
  if (head == "LDSM") return kLoad;
  if (head == "STSM") return kStore;

  // LOCAL. Per-thread local memory.
  if (head == "LDL") return kLoad;
  if (head == "STL") return kStore;

  // TMEM. Blackwell (sm_100) tensor-memory accesses. `LDT`
  // (tensor-memory load via `tcgen05.ld`) and `LDTM`
  // (matrix variant) map to kLoad; `STT` (store) and `STTM`
  // (matrix store variant) map to kStore. Direction is
  // encoded by the head prefix `LD` vs `ST` just like the
  // global/shared/local families. Routed BEFORE the
  // `return kUnspecified` fallback so the direction tag is
  // populated alongside the kTmem class tag in
  // `memory_op_class.h`.
  if (head == "LDT" || head == "LDTM") return kLoad;
  if (head == "STT" || head == "STTM") return kStore;

  // CLUSTER_DSMEM (Blackwell uniform cluster-tensor copy
  // path). `UTCCP` (uniform tensor-cluster copy) is an
  // async data movement instruction — like LDGSTS, the
  // source-to-destination transfer happens atomically as
  // a single instruction. Map to kAsyncCopy to capture the
  // copy direction. `UTCSHIFT` writes the shifted result
  // back to a destination operand within the cluster-DSMEM
  // address space, so it maps to kStore.
  if (head == "UTCCP") return kAsyncCopy;
  if (head == "UTCSHIFT") return kStore;

  return kUnspecified;
}

}  // namespace memory_op_access_kind_classifier
