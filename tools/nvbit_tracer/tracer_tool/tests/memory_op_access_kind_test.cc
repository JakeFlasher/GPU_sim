// Unit tests for `classify` in `memory_op_access_kind.h`.
//
// Compiles with `g++ -std=c++17` only — no NVBit, no CUDA, no protobuf,
// no nvcc. Driven by `make -C tools/nvbit_tracer/tracer_tool unit-tests`.
//
// Mirrors the host-only build pattern of `memory_op_class_test.cc` so
// the producer-side write-policy classifier surface stays reachable on
// hosts without a GPU toolchain. The two classifiers are orthogonal:
// `memory_op_class.h` tags the memory space; this header tags the
// direction.

#include "../memory_op_access_kind.h"

#include <cassert>
#include <cstdio>
#include <string_view>

using memory_op_access_kind_classifier::classify;
using memory_op_access_kind_classifier::Value;

int main() {
  // GLOBAL loads -> kLoad. Both the canonical short form and the
  // dotted-qualifier forms must collapse to LOAD.
  assert(classify("LDG") == Value::kLoad);
  assert(classify("LDG.E") == Value::kLoad);
  assert(classify("LDG.E.SYS") == Value::kLoad);
  assert(classify("LDG.E.128.SYS") == Value::kLoad);

  // GLOBAL stores -> kStore. Same dotted-qualifier coverage.
  assert(classify("STG") == Value::kStore);
  assert(classify("STG.E") == Value::kStore);
  assert(classify("STG.E.SYS") == Value::kStore);

  // SHARED loads -> kLoad. `LDS` + dotted qualifiers + the Hopper
  // matrix-load variant `LDSM`.
  assert(classify("LDS") == Value::kLoad);
  assert(classify("LDS.U") == Value::kLoad);
  assert(classify("LDS.128") == Value::kLoad);
  assert(classify("LDSM") == Value::kLoad);
  assert(classify("LDSM.16") == Value::kLoad);

  // SHARED stores -> kStore. `STS` + dotted qualifiers + `STSM`.
  // These are the no-write-allocate paths from the predictor side:
  // shared memory has no L2 backing, so a shared store does not
  // fetch an L2 line on miss.
  assert(classify("STS") == Value::kStore);
  assert(classify("STS.128") == Value::kStore);
  assert(classify("STSM") == Value::kStore);
  assert(classify("STSM.16") == Value::kStore);

  // Tail qualifiers like `.MMIO` (cluster-DSMEM scope) do not change
  // the direction tag — they only affect the memory_op_class
  // dimension. LDS.MMIO is still a load; STS.MMIO is still a store.
  assert(classify("LDS.MMIO") == Value::kLoad);
  assert(classify("STS.MMIO") == Value::kStore);
  assert(classify("LDSM.MMIO") == Value::kLoad);
  assert(classify("STSM.MMIO") == Value::kStore);

  // LOCAL loads / stores.
  assert(classify("LDL") == Value::kLoad);
  assert(classify("LDL.E") == Value::kLoad);
  assert(classify("STL") == Value::kStore);
  assert(classify("STL.E") == Value::kStore);

  // TEXTURE family — all loads (no texture-store opcode in CUDA).
  assert(classify("TEX") == Value::kLoad);
  assert(classify("TEX.LZ") == Value::kLoad);
  assert(classify("TLD") == Value::kLoad);
  assert(classify("TLD4") == Value::kLoad);
  assert(classify("TXQ") == Value::kLoad);

  // SURFACE family. SULD loads, SUST stores, SUATOM/SURED atomics.
  assert(classify("SULD") == Value::kLoad);
  assert(classify("SULD.B") == Value::kLoad);
  assert(classify("SUST") == Value::kStore);
  assert(classify("SUST.B") == Value::kStore);
  assert(classify("SUATOM") == Value::kAtomic);
  assert(classify("SUATOM.ADD") == Value::kAtomic);
  assert(classify("SURED") == Value::kAtomic);
  assert(classify("SURED.MIN") == Value::kAtomic);

  // ATOMIC and REDUCTION families. Prefix match: any head beginning
  // with `ATOM` or `RED` is an atomic / reduction (read-modify-
  // write semantic). Includes `ATOMG` (global atomic), `ATOMS`
  // (shared atomic), and generic `ATOM`; same for `RED`.
  assert(classify("ATOM") == Value::kAtomic);
  assert(classify("ATOMG") == Value::kAtomic);
  assert(classify("ATOMS") == Value::kAtomic);
  assert(classify("ATOMG.ADD") == Value::kAtomic);
  assert(classify("ATOMS.CAS") == Value::kAtomic);
  assert(classify("RED") == Value::kAtomic);
  assert(classify("RED.E") == Value::kAtomic);
  assert(classify("REDG") == Value::kAtomic);
  assert(classify("REDS") == Value::kAtomic);

  // TMA family on Hopper+: direction is encoded in the suffix.
  // `UTMA<mod>LDG` -> kLoad; `UTMA<mod>STG` -> kStore. Modifiers
  // like `CF` (cluster) are optional and don't change the
  // direction.
  assert(classify("UTMALDG") == Value::kLoad);
  assert(classify("UTMACFLDG") == Value::kLoad);
  assert(classify("UTMASTG") == Value::kStore);
  assert(classify("UTMACFSTG") == Value::kStore);
  assert(classify("UTMASTG.E") == Value::kStore);
  assert(classify("UTMALDG.E") == Value::kLoad);
  // Unknown UTMA shape: fail closed.
  assert(classify("UTMAFLOOP") == Value::kUnspecified);

  // ASYNC_COPY. `LDGSTS` (cp.async family). Exact match — must NOT
  // be confused with the LDG or STS prefix paths.
  assert(classify("LDGSTS") == Value::kAsyncCopy);
  assert(classify("LDGSTS.E") == Value::kAsyncCopy);
  assert(classify("LDGSTS.E.BYPASS") == Value::kAsyncCopy);

  // UNSPECIFIED. Unknown / unrelated opcodes. Mirrors the
  // memory_op_class fail-closed sentinel: the caller is expected
  // to have routed non-memory ops away before calling this, so
  // any kUnspecified return means the producer encountered a
  // memory instruction it could not classify and should route it
  // through the typed-exclusion path instead of writing a bogus
  // direction.
  assert(classify("FADD") == Value::kUnspecified);
  assert(classify("IMAD") == Value::kUnspecified);
  assert(classify("BRA") == Value::kUnspecified);
  assert(classify("BAR") == Value::kUnspecified);
  assert(classify("EXIT") == Value::kUnspecified);
  assert(classify("HMMA.16816.F32") == Value::kUnspecified);

  // LDGDEPBAR is a memory-ordering barrier, NOT a load. Exact-match
  // `LDG` excludes it because its head is the full `LDGDEPBAR`
  // token. Regression case: a substring `find("LDG")` would have
  // mis-classified it as kLoad. The orthogonal-classifier mistake
  // (silent direction assignment for a fence) is exactly the
  // mis-classification negative control Codex's R17 review
  // flagged.
  assert(classify("LDGDEPBAR") == Value::kUnspecified);

  // Empty input is well-defined: kUnspecified.
  assert(classify("") == Value::kUnspecified);

  // Defensive: a leading dot collapses to an empty head ->
  // kUnspecified.
  assert(classify(".LDG") == Value::kUnspecified);

  // Mid-token tails must NOT match. `MYOP.LDG` must NOT classify
  // as kLoad — the head is `MYOP`, not `LDG`. Pins the
  // dot-prefix-based semantic.
  assert(classify("MYOP.LDG") == Value::kUnspecified);
  assert(classify("MYOP.STG") == Value::kUnspecified);
  assert(classify("NOTUTMALDG") == Value::kUnspecified);

  // Negative-control on the load-as-store regression Codex R17
  // review called out. A canonical LOAD opcode (LDG) MUST classify
  // as kLoad — if a future refactor silently swapped the LD-family
  // return value to kStore, this assertion would fail.
  assert(classify("LDG") == Value::kLoad);
  assert(classify("LDS") == Value::kLoad);
  assert(classify("LDL") == Value::kLoad);
  // Symmetric load-as-store negative-control on the LDG family.
  assert(classify("LDG") != Value::kStore);
  assert(classify("LDS") != Value::kStore);
  assert(classify("LDL") != Value::kStore);
  // And the symmetric store-as-load negative-control on the STG
  // family.
  assert(classify("STG") != Value::kLoad);
  assert(classify("STS") != Value::kLoad);
  assert(classify("STL") != Value::kLoad);

  // Blackwell (sm_100) tensor-memory family: `LDT` (tcgen05.ld)
  // is a load; `STT` (tcgen05.st) is a store. Qualifier tails
  // must not change the direction. Matrix variants `LDTM`/
  // `STTM` follow the same direction rule via head prefix.
  assert(classify("LDT") == Value::kLoad);
  assert(classify("STT") == Value::kStore);
  assert(classify("LDT.SYNC.ALIGNED.B32.X1") == Value::kLoad);
  assert(classify("STT.SYNC.ALIGNED.B32.X4") == Value::kStore);
  assert(classify("LDTM") == Value::kLoad);
  assert(classify("STTM") == Value::kStore);
  assert(classify("LDTM.SYNC") == Value::kLoad);
  // Direction negative-controls on the TMEM family.
  assert(classify("LDT") != Value::kStore);
  assert(classify("STT") != Value::kLoad);
  assert(classify("LDTM") != Value::kStore);
  assert(classify("STTM") != Value::kLoad);
  // Mid-token look-alikes must NOT match.
  assert(classify("LDTU") == Value::kUnspecified);
  assert(classify("STTV") == Value::kUnspecified);

  // Blackwell cluster-tensor copy/shift: `UTCCP` is an async
  // copy (atomic source-to-destination move on the cluster
  // datapath); `UTCSHIFT` writes the shifted result back to
  // the destination operand within cluster-DSMEM.
  assert(classify("UTCCP") == Value::kAsyncCopy);
  assert(classify("UTCSHIFT") == Value::kStore);
  assert(classify("UTCCP.E") == Value::kAsyncCopy);

  // Compute-only Blackwell opcodes return kUnspecified (no
  // direction tag because they don't touch memory).
  assert(classify("UTCBAR") == Value::kUnspecified);
  assert(classify("UTCHMMA") == Value::kUnspecified);
  assert(classify("FADD2") == Value::kUnspecified);
  assert(classify("FHADD") == Value::kUnspecified);

  std::printf("memory_op_access_kind_test: PASS\n");
  return 0;
}
