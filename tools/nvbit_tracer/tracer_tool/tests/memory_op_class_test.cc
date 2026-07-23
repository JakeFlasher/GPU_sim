// Unit tests for `classify` in `memory_op_class.h`.
//
// Compiles with `g++ -std=c++17` only — no NVBit, no CUDA, no protobuf,
// no nvcc. Driven by `make -C tools/nvbit_tracer/tracer_tool unit-tests`.

#include "../memory_op_class.h"

#include <cassert>
#include <cstdio>
#include <string_view>

using memory_op_class_classifier::classify;
using memory_op_class_classifier::Value;

int main() {
  // GLOBAL. Plain LDG/STG plus qualifier variants must all collapse to
  // kGlobal. Covers the contract example "LDG, LDG.E.SYS, LDG.E.128.SYS
  // all map to GLOBAL".
  assert(classify("LDG") == Value::kGlobal);
  assert(classify("STG") == Value::kGlobal);
  assert(classify("LDG.E") == Value::kGlobal);
  assert(classify("LDG.E.SYS") == Value::kGlobal);
  assert(classify("LDG.E.128.SYS") == Value::kGlobal);
  assert(classify("STG.E.SYS") == Value::kGlobal);

  // SHARED. Plain LDS/STS plus the Hopper LDSM/STSM matrix transfers.
  assert(classify("LDS") == Value::kShared);
  assert(classify("STS") == Value::kShared);
  assert(classify("LDS.U") == Value::kShared);
  assert(classify("STS.128") == Value::kShared);
  assert(classify("LDSM") == Value::kShared);
  assert(classify("STSM") == Value::kShared);
  assert(classify("LDSM.16") == Value::kShared);

  // LOCAL. Per-thread local memory.
  assert(classify("LDL") == Value::kLocal);
  assert(classify("STL") == Value::kLocal);
  assert(classify("LDL.E") == Value::kLocal);

  // TEXTURE. TEX / TLD / TLD4 / TXQ. Exact-match family.
  assert(classify("TEX") == Value::kTexture);
  assert(classify("TLD") == Value::kTexture);
  assert(classify("TLD4") == Value::kTexture);
  assert(classify("TXQ") == Value::kTexture);
  assert(classify("TEX.LZ") == Value::kTexture);

  // SURFACE. SU{LD,ST,ATOM,RED}.
  assert(classify("SULD") == Value::kSurface);
  assert(classify("SUST") == Value::kSurface);
  assert(classify("SUATOM") == Value::kSurface);
  assert(classify("SURED") == Value::kSurface);
  assert(classify("SUST.B") == Value::kSurface);

  // ATOMIC. ATOM* and RED* families collapse via prefix match.
  // ATOMG (global atomic), ATOMS (shared atomic), ATOM (generic),
  // RED (reduction), REDS (shared reduction), REDG (global reduction)
  // — all atomic.
  assert(classify("ATOM") == Value::kAtomic);
  assert(classify("ATOMG") == Value::kAtomic);
  assert(classify("ATOMS") == Value::kAtomic);
  assert(classify("ATOMG.ADD") == Value::kAtomic);
  assert(classify("ATOMS.CAS") == Value::kAtomic);
  assert(classify("RED") == Value::kAtomic);
  assert(classify("RED.E") == Value::kAtomic);
  assert(classify("REDG") == Value::kAtomic);

  // TMA. UTMA* prefix family (Hopper tensor-memory-accelerator).
  assert(classify("UTMASTG") == Value::kTma);
  assert(classify("UTMALDG") == Value::kTma);
  assert(classify("UTMACFLDG") == Value::kTma);
  assert(classify("UTMACFSTG") == Value::kTma);
  assert(classify("UTMASTG.E") == Value::kTma);

  // ASYNC_COPY. LDGSTS (cp.async family). Exact match — must NOT
  // be confused with the LDG/STS prefix paths.
  assert(classify("LDGSTS") == Value::kAsyncCopy);
  assert(classify("LDGSTS.E") == Value::kAsyncCopy);
  assert(classify("LDGSTS.E.BYPASS") == Value::kAsyncCopy);

  // CLUSTER_DSMEM. Hopper sm_90+ cluster-distributed shared memory
  // surfaces as the LDS/STS family with an `MMIO` tail qualifier. The
  // ordering is load-bearing: plain LDS without MMIO must still
  // classify as kShared, and the cluster-decorated form must classify
  // as kClusterDsmem. Both sides covered here.
  assert(classify("LDS.MMIO") == Value::kClusterDsmem);
  assert(classify("STS.MMIO") == Value::kClusterDsmem);
  assert(classify("LDSM.MMIO") == Value::kClusterDsmem);
  assert(classify("STSM.MMIO") == Value::kClusterDsmem);
  assert(classify("LDS.U.MMIO") == Value::kClusterDsmem);
  assert(classify("STS.MMIO.128") == Value::kClusterDsmem);

  // Plain LDS/STS without the MMIO qualifier must remain kShared,
  // not collapse to kClusterDsmem just because the head matches.
  assert(classify("LDS.U") == Value::kShared);
  assert(classify("STS.128") == Value::kShared);
  assert(classify("LDSM.16") == Value::kShared);

  // The qualifier check must look at dot-delimited tokens, not a
  // substring. An opcode whose head contains `MMIO` as a sub-string
  // of a different token must NOT match. (No real SASS opcode has
  // this shape, but we lock in the semantic.)
  assert(classify("LDS.NOTMMIOO") == Value::kShared);

  // TMEM. Blackwell (sm_100) tensor-memory family via
  // `tcgen05.{ld,st}` — SASS mnemonics `LDT`/`STT` plus
  // the matrix variants `LDTM`/`STTM`. Exact-match on the
  // head; qualifier tails like `.SYNC.ALIGNED.B32.X1`
  // must not change the classification.
  assert(classify("LDT") == Value::kTmem);
  assert(classify("STT") == Value::kTmem);
  assert(classify("LDT.SYNC") == Value::kTmem);
  assert(classify("STT.SYNC") == Value::kTmem);
  assert(classify("LDT.SYNC.ALIGNED.B32.X1") == Value::kTmem);
  assert(classify("STT.SYNC.ALIGNED.B32.X4") == Value::kTmem);
  assert(classify("LDTM") == Value::kTmem);
  assert(classify("STTM") == Value::kTmem);
  assert(classify("LDTM.SYNC") == Value::kTmem);
  assert(classify("STTM.16") == Value::kTmem);

  // CLUSTER_DSMEM (Blackwell uniform cluster-tensor copy/shift).
  // `UTCCP` and `UTCSHIFT` access the same cluster-DSMEM
  // storage class as Hopper's LDS/STS.MMIO path. Exact-match.
  assert(classify("UTCCP") == Value::kClusterDsmem);
  assert(classify("UTCSHIFT") == Value::kClusterDsmem);
  assert(classify("UTCCP.E") == Value::kClusterDsmem);
  assert(classify("UTCSHIFT.E") == Value::kClusterDsmem);

  // UNSPECIFIED. Unknown / unrelated opcodes. Distinguishes a memory
  // op the classifier failed on from a non-memory op (the caller
  // is expected to have routed non-memory ops away before calling
  // this function).
  assert(classify("FADD") == Value::kUnspecified);
  assert(classify("IMAD") == Value::kUnspecified);
  assert(classify("BRA") == Value::kUnspecified);
  assert(classify("BAR") == Value::kUnspecified);
  assert(classify("EXIT") == Value::kUnspecified);
  assert(classify("HMMA.16816.F32") == Value::kUnspecified);

  // Mid-token LDT/STT must NOT match. `LDTU` and `STTV` are
  // hypothetical look-alike heads — the prefix-match must not
  // collapse them to kTmem.
  assert(classify("LDTU") == Value::kUnspecified);
  assert(classify("STTV") == Value::kUnspecified);
  assert(classify("MYLDT") == Value::kUnspecified);

  // Compute-only Blackwell opcodes must NOT classify as
  // memory. UTCBAR is a cluster barrier (control), the four
  // UTC*MMA variants are tensor-core compute, the FP-multi
  // family is FP compute, and the uniform misc opcodes are
  // scheduling/counter ops — none are memory.
  assert(classify("UTCBAR") == Value::kUnspecified);
  assert(classify("UTCHMMA") == Value::kUnspecified);
  assert(classify("UTCIMMA") == Value::kUnspecified);
  assert(classify("UTCOMMA") == Value::kUnspecified);
  assert(classify("UTCQMMA") == Value::kUnspecified);
  assert(classify("UGETNEXTWORKID") == Value::kUnspecified);
  assert(classify("UVIRTCOUNT") == Value::kUnspecified);
  assert(classify("FADD2") == Value::kUnspecified);
  assert(classify("FFMA2") == Value::kUnspecified);
  assert(classify("FMUL2") == Value::kUnspecified);
  assert(classify("FHADD") == Value::kUnspecified);
  assert(classify("FHFMA") == Value::kUnspecified);
  assert(classify("FMNMX3") == Value::kUnspecified);

  // LDGDEPBAR is a memory-ordering barrier, NOT a global load.
  // Exact-match `LDG`/`STG` excludes it because its head is the
  // full `LDGDEPBAR` token (no leading dot to truncate). Regression
  // case: a substring `find("LDG")` would have miscategorized this
  // as kGlobal.
  assert(classify("LDGDEPBAR") == Value::kUnspecified);

  // Empty input is well-defined: kUnspecified.
  assert(classify("") == Value::kUnspecified);

  // Defensive: a leading dot collapses to an empty head -> kUnspecified.
  assert(classify(".LDG") == Value::kUnspecified);

  // Mid-token classes must NOT match: "MYOP.LDG" must NOT classify
  // as kGlobal (the head is "MYOP", not "LDG"). Establishes that the
  // classifier really is dot-prefix-based and not substring-based.
  assert(classify("MYOP.LDG") == Value::kUnspecified);

  std::printf("memory_op_class_test: PASS\n");
  return 0;
}
