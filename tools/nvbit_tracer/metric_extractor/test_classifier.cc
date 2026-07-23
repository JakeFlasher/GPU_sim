// Host-side regression test for the AC-7 instruction-class router.
//
// Locks in the bucket routing for every canonical class name documented in
// JSONL_SCHEMA.md ("ffma", "mufu", "ldg", "stg", "lds", "sts", "hmma",
// "wgmma", "tcgen05", "tma", "dsmem", "bra", "barrier", "warpsync",
// "unknown"). Two specific defects this test guards against:
//
//   * (OP_LDS, DSMEM_OP) and (OP_STS, DSMEM_OP) must route to `dsmem`,
//     NOT to `lds` / `sts`. The shipped blackwell_opcode.h maps the
//     `LDS.MMIO` and `STS.MMIO` mnemonics to those pairs; before the
//     Round 1 reorder, classify_op_pair() returned `lds` / `sts` from
//     the per-opcode switch before the category switch ever reached
//     DSMEM_OP.
//   * Each Hopper WGMMA opcode (BGMMA / HGMMA / IGMMA / QGMMA /
//     WARPGROUP / WARPGROUPSET) must route to `wgmma`, not `unknown`.
//     The mnemonic lookup chain in classify_instr() is tested by the
//     pytest wrapper that drives this binary; here we exercise the pure
//     (opcode, category) → bucket router that the lookup feeds.
//
// Build via the metric_extractor Makefile:
//   make -C tools/nvbit_tracer/metric_extractor test_classifier
//   make -C tools/nvbit_tracer/metric_extractor run-test-classifier
//
// Exits 0 on success, non-zero on any assertion failure. No CUDA / NVBit
// headers are touched; classifier.h is host-includable by design.

#include <cstdio>
#include <string>

#include "classifier.h"
#include "sector_counting.h"
#include "bank_counting.h"
#include "branch_counting.h"

namespace {

// Sentinel "this opcode is not in the per-opcode switch but DOES have a
// meaningful category" used to exercise the category-side fallback path
// without colliding with OP_UNKNOWN. OP_UNKNOWN is itself routed by the
// explicit `case OP_UNKNOWN: return INSTR_CLASS_UNKNOWN;` arm so it
// cannot be used to test the fallback through any meaningful category.
// A value beyond SASS_NUM_OPCODES is guaranteed not to collide with any
// existing TraceInstrOpcode entry today and not to be silently absorbed
// by a future enum addition (the test stays explicit about its intent).
constexpr unsigned OP_NOT_IN_SWITCH = 0xCAFE;

int failures = 0;

void report_fail(const char* expr, const char* file, int line) {
    std::fprintf(stderr, "FAIL  %s:%d  %s\n", file, line, expr);
    ++failures;
}

#define CHECK(expr)                                              \
    do {                                                         \
        if (!(expr)) report_fail(#expr, __FILE__, __LINE__);     \
    } while (0)

#define ROUTES_TO(opcode_, category_, expected_)                 \
    CHECK(classify_op_pair((opcode_), (category_)) == (expected_))

// Map an InstructionClass bucket id to its canonical sparse-map key —
// the same labels the JSON-L emitter writes. Mirrors
// ``metric_extractor.cu::instruction_class_name`` byte-for-byte; kept
// here as a duplicate because including metric_extractor.cu's
// translation unit would pull in NVBit headers, which the standalone
// test deliberately avoids.
const char* instruction_class_name_for_test(InstructionClass cls) {
    switch (cls) {
        case INSTR_CLASS_FFMA:     return "ffma";
        case INSTR_CLASS_MUFU:     return "mufu";
        case INSTR_CLASS_LDG:      return "ldg";
        case INSTR_CLASS_STG:      return "stg";
        case INSTR_CLASS_LDS:      return "lds";
        case INSTR_CLASS_STS:      return "sts";
        case INSTR_CLASS_HMMA:     return "hmma";
        case INSTR_CLASS_WGMMA:    return "wgmma";
        case INSTR_CLASS_TCGEN05:  return "tcgen05";
        case INSTR_CLASS_TMA:      return "tma";
        case INSTR_CLASS_DSMEM:    return "dsmem";
        case INSTR_CLASS_BRA:      return "bra";
        case INSTR_CLASS_BARRIER:  return "barrier";
        case INSTR_CLASS_WARPSYNC: return "warpsync";
        case INSTR_CLASS_UNKNOWN:  return "unknown";
        default:                   return "unknown";
    }
}

}  // namespace

int main(int argc, char** argv) {
    // --classify <mnemonic>: T39 / AC-14 entry point. Routes the
    // mnemonic through ``classify_mnemonic`` (the same chain
    // ``classify_instr`` uses) and prints the canonical bucket name
    // on stdout. Exit 0 on any non-empty input; the pytest wrapper
    // asserts the printed bucket matches the per-fixture expectation.
    // No regression-suite execution in this mode.
    if (argc == 3 && std::string(argv[1]) == "--classify") {
        const std::string mnemonic(argv[2]);
        const InstructionClass cls = classify_mnemonic(mnemonic);
        std::printf("%s\n", instruction_class_name_for_test(cls));
        return 0;
    }

    // ------------------------------------------------------------------
    // ffma — single-precision FFMA / FFMA32I land in INSTR_CLASS_FFMA.
    // ------------------------------------------------------------------
    ROUTES_TO(OP_FFMA, SP_OP, INSTR_CLASS_FFMA);
    ROUTES_TO(OP_FFMA32I, SP_OP, INSTR_CLASS_FFMA);

    // ------------------------------------------------------------------
    // mufu — SFU operations.
    // ------------------------------------------------------------------
    ROUTES_TO(OP_MUFU, SFU_OP, INSTR_CLASS_MUFU);

    // ------------------------------------------------------------------
    // ldg / stg — global memory loads/stores. Includes the LDGSTS /
    // LDGMC / LDGDEPBAR variants that share the LDG bucket and the
    // STAS variant on the STG side.
    // ------------------------------------------------------------------
    ROUTES_TO(OP_LDG, LOAD_OP, INSTR_CLASS_LDG);
    ROUTES_TO(OP_LDGSTS, LOAD_OP, INSTR_CLASS_LDG);
    ROUTES_TO(OP_LDGMC, LOAD_OP, INSTR_CLASS_LDG);
    ROUTES_TO(OP_LDGDEPBAR, LDGDEPBAR_OP, INSTR_CLASS_LDG);
    ROUTES_TO(OP_STG, STORE_OP, INSTR_CLASS_STG);
    ROUTES_TO(OP_STAS, STORE_OP, INSTR_CLASS_STG);

    // ------------------------------------------------------------------
    // lds / sts — shared memory loads/stores via the regular shared
    // memory space. LDSM / STSM (matrix variants) and LDSLK / STSCUL
    // (locked / culling variants) share these buckets.
    // ------------------------------------------------------------------
    ROUTES_TO(OP_LDS, LOAD_OP, INSTR_CLASS_LDS);
    ROUTES_TO(OP_LDSM, LOAD_OP, INSTR_CLASS_LDS);
    ROUTES_TO(OP_LDSLK, LOAD_OP, INSTR_CLASS_LDS);
    ROUTES_TO(OP_STS, STORE_OP, INSTR_CLASS_STS);
    ROUTES_TO(OP_STSM, STORE_OP, INSTR_CLASS_STS);
    ROUTES_TO(OP_STSCUL, STORE_OP, INSTR_CLASS_STS);

    // ------------------------------------------------------------------
    // hmma — legacy tensor-core matrix multiply.
    // ------------------------------------------------------------------
    ROUTES_TO(OP_HMMA, TENSOR_CORE_OP, INSTR_CLASS_HMMA);

    // ------------------------------------------------------------------
    // wgmma — Hopper warp-group matrix multiply (BGMMA / HGMMA / IGMMA
    // / QGMMA) and the warpgroup-state instructions. All four GMMA
    // variants share the TENSOR_CORE_OP category in Hopper_OpcodeMap;
    // the per-opcode disambiguation is what keeps them out of the
    // legacy HMMA bucket.
    // ------------------------------------------------------------------
    ROUTES_TO(OP_BGMMA, TENSOR_CORE_OP, INSTR_CLASS_WGMMA);
    ROUTES_TO(OP_HGMMA, TENSOR_CORE_OP, INSTR_CLASS_WGMMA);
    ROUTES_TO(OP_IGMMA, TENSOR_CORE_OP, INSTR_CLASS_WGMMA);
    ROUTES_TO(OP_QGMMA, TENSOR_CORE_OP, INSTR_CLASS_WGMMA);
    ROUTES_TO(OP_WARPGROUP, BARRIER_OP, INSTR_CLASS_WGMMA);
    ROUTES_TO(OP_WARPGROUPSET, BARRIER_OP, INSTR_CLASS_WGMMA);

    // ------------------------------------------------------------------
    // tcgen05 — all four Blackwell-specific sub-variants land in the
    // dedicated bucket regardless of which side (specific opcode or
    // dedicated category) the lookup table reaches first. The four
    // dedicated TCGEN05_*_OP categories also route to this bucket so a
    // future ISA map entry that uses only the category side still lands.
    // ------------------------------------------------------------------
    ROUTES_TO(OP_TCGEN05_MXF4, TCGEN05_MXF4_OP, INSTR_CLASS_TCGEN05);
    ROUTES_TO(OP_TCGEN05_MXF4NVF4, TCGEN05_MXF4NVF4_OP, INSTR_CLASS_TCGEN05);
    ROUTES_TO(OP_TCGEN05_F8F6F4, TCGEN05_F8F6F4_OP, INSTR_CLASS_TCGEN05);
    ROUTES_TO(OP_TCGEN05_SCALE_VECTOR, TCGEN05_SCALE_VECTOR_OP,
              INSTR_CLASS_TCGEN05);
    // Category-side routing only (future entries that may not have a
    // specific opcode in trace_opcode.h yet — exercised via
    // OP_NOT_IN_SWITCH so the explicit `case OP_UNKNOWN:` arm doesn't
    // short-circuit the category check):
    ROUTES_TO(OP_NOT_IN_SWITCH, TCGEN05_MXF4_OP, INSTR_CLASS_TCGEN05);
    ROUTES_TO(OP_NOT_IN_SWITCH, TCGEN05_SCALE_VECTOR_OP, INSTR_CLASS_TCGEN05);

    // ------------------------------------------------------------------
    // tma — Hopper bulk-tensor / TMA descriptor mnemonics. Both the
    // UTMA* / UBLK* opcode-side disambiguation and the TMA_OP
    // category-side fallback must land here. The category side covers
    // any future TMA opcode whose specific TraceInstrOpcode hasn't been
    // added yet.
    // ------------------------------------------------------------------
    ROUTES_TO(OP_UTMACCTL, MEMORY_MISCELLANEOUS_OP, INSTR_CLASS_TMA);
    ROUTES_TO(OP_UTMACMDFLUSH, MEMORY_MISCELLANEOUS_OP, INSTR_CLASS_TMA);
    ROUTES_TO(OP_UTMALDG, LOAD_OP, INSTR_CLASS_TMA);
    ROUTES_TO(OP_UTMAPF, MEMORY_MISCELLANEOUS_OP, INSTR_CLASS_TMA);
    ROUTES_TO(OP_UTMAREDG, MEMORY_MISCELLANEOUS_OP, INSTR_CLASS_TMA);
    ROUTES_TO(OP_UTMASTG, STORE_OP, INSTR_CLASS_TMA);
    ROUTES_TO(OP_UBLKCP, MEMORY_MISCELLANEOUS_OP, INSTR_CLASS_TMA);
    ROUTES_TO(OP_UBLKPF, MEMORY_MISCELLANEOUS_OP, INSTR_CLASS_TMA);
    ROUTES_TO(OP_UBLKRED, MEMORY_MISCELLANEOUS_OP, INSTR_CLASS_TMA);
    // Category-side fallback:
    ROUTES_TO(OP_NOT_IN_SWITCH, TMA_OP, INSTR_CLASS_TMA);

    // ------------------------------------------------------------------
    // dsmem — distributed shared memory. The DSMEM_OP category MUST
    // shadow the OP_LDS / OP_STS opcode buckets so the `LDS.MMIO` and
    // `STS.MMIO` mnemonics in blackwell_opcode.h (which map to
    // (OP_LDS, DSMEM_OP) and (OP_STS, DSMEM_OP)) land in dsmem, not
    // in lds / sts. This is the AC-7 negative test that the Round 0
    // shipping order failed.
    // ------------------------------------------------------------------
    ROUTES_TO(OP_LDS, DSMEM_OP, INSTR_CLASS_DSMEM);
    ROUTES_TO(OP_STS, DSMEM_OP, INSTR_CLASS_DSMEM);
    // Category-side fallback for entries without a specific opcode side:
    ROUTES_TO(OP_NOT_IN_SWITCH, DSMEM_OP, INSTR_CLASS_DSMEM);
    // Negative: (OP_LDS, LOAD_OP) (plain shared load) must NOT be
    // mistaken for DSMEM just because OP_LDS is shared. Only the
    // (OP_LDS, DSMEM_OP) pair routes to dsmem.
    CHECK(classify_op_pair(OP_LDS, LOAD_OP) != INSTR_CLASS_DSMEM);
    CHECK(classify_op_pair(OP_STS, STORE_OP) != INSTR_CLASS_DSMEM);

    // ------------------------------------------------------------------
    // bra — branch-family opcodes. BRX / JMP / JMX / CALL / RET /
    // EXIT all share this bucket. The BRANCH_OP category-side fallback
    // covers entries the per-opcode switch doesn't name explicitly.
    // ------------------------------------------------------------------
    ROUTES_TO(OP_BRA, BRANCH_OP, INSTR_CLASS_BRA);
    ROUTES_TO(OP_BREAK, BRANCH_OP, INSTR_CLASS_BRA);
    ROUTES_TO(OP_BRX, BRANCH_OP, INSTR_CLASS_BRA);
    ROUTES_TO(OP_JMP, BRANCH_OP, INSTR_CLASS_BRA);
    ROUTES_TO(OP_JMX, BRANCH_OP, INSTR_CLASS_BRA);
    ROUTES_TO(OP_CALL, CALL_OPS, INSTR_CLASS_BRA);
    ROUTES_TO(OP_RET, RET_OPS, INSTR_CLASS_BRA);
    ROUTES_TO(OP_EXIT, EXIT_OPS, INSTR_CLASS_BRA);
    ROUTES_TO(OP_BRXU, BRANCH_OP, INSTR_CLASS_BRA);
    ROUTES_TO(OP_JMXU, BRANCH_OP, INSTR_CLASS_BRA);
    // Category-side fallback:
    ROUTES_TO(OP_NOT_IN_SWITCH, BRANCH_OP, INSTR_CLASS_BRA);

    // ------------------------------------------------------------------
    // barrier — BAR / DEPBAR / MEMBAR / ERRBAR / CGAERRBAR / FENCE /
    // SYNCS / BSYNC / BSSY / UCGABAR.ARV / UCGABAR.WAIT share this
    // bucket. The BARRIER_OP / MEMORY_BARRIER_OP / GRID_BARRIER_OP
    // categories also route here.
    // ------------------------------------------------------------------
    ROUTES_TO(OP_BAR, BARRIER_OP, INSTR_CLASS_BARRIER);
    ROUTES_TO(OP_DEPBAR, DEPBAR_OP, INSTR_CLASS_BARRIER);
    ROUTES_TO(OP_MEMBAR, MEMORY_BARRIER_OP, INSTR_CLASS_BARRIER);
    ROUTES_TO(OP_ERRBAR, MISCELLANEOUS_QUEUE_OP, INSTR_CLASS_BARRIER);
    ROUTES_TO(OP_CGAERRBAR, MISCELLANEOUS_QUEUE_OP, INSTR_CLASS_BARRIER);
    ROUTES_TO(OP_FENCE, MEMORY_BARRIER_OP, INSTR_CLASS_BARRIER);
    ROUTES_TO(OP_SYNCS, BARRIER_OP, INSTR_CLASS_BARRIER);
    ROUTES_TO(OP_BSYNC, BARRIER_OP, INSTR_CLASS_BARRIER);
    ROUTES_TO(OP_BSSY, BARRIER_OP, INSTR_CLASS_BARRIER);
    ROUTES_TO(OP_UCGABAR_ARV, UNIFORM_OP, INSTR_CLASS_BARRIER);
    ROUTES_TO(OP_UCGABAR_WAIT, UNIFORM_OP, INSTR_CLASS_BARRIER);
    // Category-side fallback (e.g. an unmapped opcode with BARRIER_OP):
    ROUTES_TO(OP_NOT_IN_SWITCH, BARRIER_OP, INSTR_CLASS_BARRIER);
    ROUTES_TO(OP_NOT_IN_SWITCH, MEMORY_BARRIER_OP, INSTR_CLASS_BARRIER);
    ROUTES_TO(OP_NOT_IN_SWITCH, GRID_BARRIER_OP, INSTR_CLASS_BARRIER);

    // ------------------------------------------------------------------
    // warpsync — distinct from barrier so the cross-warp sync surface
    // can be analysed independently from the cross-block / cross-grid
    // barrier surface.
    // ------------------------------------------------------------------
    ROUTES_TO(OP_WARPSYNC, BARRIER_OP, INSTR_CLASS_WARPSYNC);
    // Negative: WARPSYNC must NOT collapse into the generic barrier
    // bucket even though its category is BARRIER_OP.
    CHECK(classify_op_pair(OP_WARPSYNC, BARRIER_OP) !=
          INSTR_CLASS_BARRIER);

    // ------------------------------------------------------------------
    // unknown — explicit OP_UNKNOWN with UNKNOWN_OP, and any
    // unrecognised (opcode, category) pair the classifier hasn't seen.
    // The explicit unknown bucket is mandatory per AC-7 so a future
    // SASS spelling change does not silently miscount.
    // ------------------------------------------------------------------
    ROUTES_TO(OP_UNKNOWN, UNKNOWN_OP, INSTR_CLASS_UNKNOWN);
    // An entirely fabricated (opcode, category) — the classifier
    // refuses to invent a bucket for it.
    ROUTES_TO(0xDEADBEEF, 0xCAFEBABE, INSTR_CLASS_UNKNOWN);
    // ALU_OP / INTP_OP / HALF_OP / DP_OP categories land in unknown
    // today (intentional — the canonical bucket set doesn't name
    // generic arithmetic). Pin this so a future bucket widening is a
    // deliberate test-update, not a silent route change.
    ROUTES_TO(OP_IADD, INTP_OP, INSTR_CLASS_UNKNOWN);
    ROUTES_TO(OP_HADD2, HALF_OP, INSTR_CLASS_UNKNOWN);
    ROUTES_TO(OP_DADD, DP_OP, INSTR_CLASS_UNKNOWN);

    // ------------------------------------------------------------------
    // Mnemonic-driven lookup chain (Blackwell → Hopper → Ampere with
    // full-mnemonic-then-token-0 fallback). This is the function
    // classify_instr() delegates to, so anything tested here is what an
    // actual NVBit instrument pass would see for a SASS opcode string.
    // The key AC-7 wire-up bug from Round 0 — WGMMA mnemonics absent
    // from the lookup chain — is locked in here: BGMMA / HGMMA / IGMMA
    // / QGMMA must classify_mnemonic to wgmma, not unknown.
    // ------------------------------------------------------------------
#define CLASSIFY(s) classify_mnemonic(std::string(s))

    // FFMA / MUFU / loads / stores from the Blackwell map.
    CHECK(CLASSIFY("FFMA") == INSTR_CLASS_FFMA);
    CHECK(CLASSIFY("FFMA32I") == INSTR_CLASS_FFMA);
    CHECK(CLASSIFY("MUFU") == INSTR_CLASS_MUFU);
    CHECK(CLASSIFY("LDG") == INSTR_CLASS_LDG);
    CHECK(CLASSIFY("STG") == INSTR_CLASS_STG);
    CHECK(CLASSIFY("LDS") == INSTR_CLASS_LDS);
    CHECK(CLASSIFY("STS") == INSTR_CLASS_STS);

    // Token-0 fallback for vectorised / dotted suffixes (LDG.E.128,
    // STG.E.64, etc.) — these are not in the map as exact keys; the
    // classifier must strip the suffix and find LDG / STG.
    CHECK(CLASSIFY("LDG.E.128") == INSTR_CLASS_LDG);
    CHECK(CLASSIFY("STG.E.64") == INSTR_CLASS_STG);

    // HMMA via Blackwell map.
    CHECK(CLASSIFY("HMMA") == INSTR_CLASS_HMMA);

    // WGMMA — the key Round 0 lookup-chain bug. All four mnemonics live
    // in Hopper_OpcodeMap, NOT Blackwell_OpcodeMap. Without the
    // Hopper-map step in the lookup chain these would return UNKNOWN.
    CHECK(CLASSIFY("BGMMA") == INSTR_CLASS_WGMMA);
    CHECK(CLASSIFY("HGMMA") == INSTR_CLASS_WGMMA);
    CHECK(CLASSIFY("IGMMA") == INSTR_CLASS_WGMMA);
    CHECK(CLASSIFY("QGMMA") == INSTR_CLASS_WGMMA);
    CHECK(CLASSIFY("WARPGROUP") == INSTR_CLASS_WGMMA);
    CHECK(CLASSIFY("WARPGROUPSET") == INSTR_CLASS_WGMMA);

    // TCGen05 multi-token mnemonics — the full-mnemonic lookup must hit
    // the Blackwell map entry as-is (token-0 fallback would land on
    // `TCGEN05` which is not a key).
    CHECK(CLASSIFY("TCGEN05.MMA.KIND::MXF4") == INSTR_CLASS_TCGEN05);
    CHECK(CLASSIFY("TCGEN05.MMA.KIND::MXF4NVF4") == INSTR_CLASS_TCGEN05);
    CHECK(CLASSIFY("TCGEN05.MMA.KIND::F8F6F4") == INSTR_CLASS_TCGEN05);
    CHECK(CLASSIFY("TCGEN05.MMA.BLOCK_SCALE.SCALE_VEC") ==
          INSTR_CLASS_TCGEN05);
    CHECK(CLASSIFY("TCGEN05.MMA.BLOCK16") == INSTR_CLASS_TCGEN05);
    CHECK(CLASSIFY("TCGEN05.MMA.BLOCK32") == INSTR_CLASS_TCGEN05);

    // DSMEM via the LDS.MMIO / STS.MMIO mappings in the Blackwell map.
    // The Round 0 routing bug had these landing in INSTR_CLASS_LDS /
    // INSTR_CLASS_STS because the opcode-side switch ran before the
    // category-side switch.
    CHECK(CLASSIFY("LDS.MMIO") == INSTR_CLASS_DSMEM);
    CHECK(CLASSIFY("STS.MMIO") == INSTR_CLASS_DSMEM);
    CHECK(CLASSIFY("LDS_MMIO") == INSTR_CLASS_DSMEM);
    CHECK(CLASSIFY("STS_MMIO") == INSTR_CLASS_DSMEM);

    // TMA mnemonics — UTMA* live in Hopper_OpcodeMap with TMA_OP
    // category, UBLKCP / UBLKPF / UBLKRED likewise (the Hopper bulk-copy
    // family also routes to tma because the on-device instrumentation
    // treats every TMA-flavoured opcode the same way). The lookup chain
    // must walk past Blackwell (which has UTMA* entries too via the
    // earlier ISA-classifier audit) and into Hopper if needed.
    CHECK(CLASSIFY("UTMACCTL") == INSTR_CLASS_TMA);
    CHECK(CLASSIFY("UTMACMDFLUSH") == INSTR_CLASS_TMA);
    CHECK(CLASSIFY("UTMALDG") == INSTR_CLASS_TMA);
    CHECK(CLASSIFY("UTMAPF") == INSTR_CLASS_TMA);
    CHECK(CLASSIFY("UTMAREDG") == INSTR_CLASS_TMA);
    CHECK(CLASSIFY("UTMASTG") == INSTR_CLASS_TMA);
    CHECK(CLASSIFY("UBLKCP") == INSTR_CLASS_TMA);
    CHECK(CLASSIFY("UBLKPF") == INSTR_CLASS_TMA);
    CHECK(CLASSIFY("UBLKRED") == INSTR_CLASS_TMA);

    // bra — token-0 fallback for dotted suffixes.
    CHECK(CLASSIFY("BRA") == INSTR_CLASS_BRA);
    CHECK(CLASSIFY("BRX") == INSTR_CLASS_BRA);
    CHECK(CLASSIFY("JMP") == INSTR_CLASS_BRA);
    CHECK(CLASSIFY("EXIT") == INSTR_CLASS_BRA);

    // barrier / warpsync.
    CHECK(CLASSIFY("BAR") == INSTR_CLASS_BARRIER);
    CHECK(CLASSIFY("DEPBAR") == INSTR_CLASS_BARRIER);
    CHECK(CLASSIFY("MEMBAR") == INSTR_CLASS_BARRIER);
    CHECK(CLASSIFY("WARPSYNC") == INSTR_CLASS_WARPSYNC);

    // unknown — empty string and a fabricated mnemonic both land here.
    CHECK(CLASSIFY("") == INSTR_CLASS_UNKNOWN);
    CHECK(CLASSIFY("THIS_OPCODE_DOES_NOT_EXIST") == INSTR_CLASS_UNKNOWN);

#undef CLASSIFY

    // ------------------------------------------------------------------
    // Memory-space-side classifier — the DSMEM bucket entry point that
    // T9 wires into `classify_instr(Instr*)`. DISTRIBUTED_SHARED is the
    // only memory space that owns its own bucket; every other space
    // returns the sentinel INSTR_CLASS_UNKNOWN so the caller falls
    // through to mnemonic / opcode routing. Locking in every enum value
    // here guards against accidental over-mapping when NVBit adds a
    // new memory space (e.g. a future `MULTI_GPU` / `CONFIDENTIAL`
    // space): the test fails until someone deliberately decides what
    // bucket the new space should map to.
    // ------------------------------------------------------------------
    CHECK(classify_memory_space(InstrType::MemorySpace::DISTRIBUTED_SHARED)
          == INSTR_CLASS_DSMEM);

    // Every other documented MemorySpace value MUST return the sentinel
    // INSTR_CLASS_UNKNOWN so classify_instr falls through to mnemonic
    // routing. Plain SHARED is the most important negative test — the
    // mnemonic side already routes plain LDS/STS to lds/sts, and the
    // memory-space side must not stomp that.
    CHECK(classify_memory_space(InstrType::MemorySpace::NONE)
          == INSTR_CLASS_UNKNOWN);
    CHECK(classify_memory_space(InstrType::MemorySpace::LOCAL)
          == INSTR_CLASS_UNKNOWN);
    CHECK(classify_memory_space(InstrType::MemorySpace::GENERIC)
          == INSTR_CLASS_UNKNOWN);
    CHECK(classify_memory_space(InstrType::MemorySpace::GLOBAL)
          == INSTR_CLASS_UNKNOWN);
    CHECK(classify_memory_space(InstrType::MemorySpace::SHARED)
          == INSTR_CLASS_UNKNOWN);
    CHECK(classify_memory_space(InstrType::MemorySpace::CONSTANT)
          == INSTR_CLASS_UNKNOWN);
    CHECK(classify_memory_space(InstrType::MemorySpace::GLOBAL_TO_SHARED)
          == INSTR_CLASS_UNKNOWN);
    CHECK(classify_memory_space(InstrType::MemorySpace::SURFACE)
          == INSTR_CLASS_UNKNOWN);
    CHECK(classify_memory_space(InstrType::MemorySpace::TEXTURE)
          == INSTR_CLASS_UNKNOWN);
    CHECK(classify_memory_space(InstrType::MemorySpace::TENSOR_MEM)
          == INSTR_CLASS_UNKNOWN);
    CHECK(classify_memory_space(InstrType::MemorySpace::TENSOR_CORE_MEM)
          == INSTR_CLASS_UNKNOWN);

    // Negative test for the cross-layer invariant: the mnemonic-driven
    // router still returns lds/sts for plain shared loads/stores. The
    // memory-space side does NOT preempt the mnemonic when the space is
    // plain SHARED; only DISTRIBUTED_SHARED triggers the DSMEM short-
    // circuit. This guards against a regression that accidentally
    // demotes plain shared traffic to the dsmem bucket.
    CHECK(classify_op_pair(OP_LDS, LOAD_OP) == INSTR_CLASS_LDS);
    CHECK(classify_op_pair(OP_STS, STORE_OP) == INSTR_CLASS_STS);

    // ------------------------------------------------------------------
    // count_distinct_sectors — the host+device sector-efficiency helper
    // from sector_counting.h. The pre-Round-6 single-sector-only logic
    // (`addr >> 5` once per lane) undercounted any access where the
    // byte interval `[addr, addr+bytes)` crossed a 32-byte sector
    // boundary. The fixed algorithm iterates first..last and inserts
    // every sector in the range. The test cases below cover every
    // scenario AC-4 (positive tests 1, 2, 6) calls out plus the
    // canonical pre-Round-6-fails-post-Round-6-passes case Codex named
    // in the Round-5 review.
    // ------------------------------------------------------------------

    {
        // Codex's canonical regression: a single lane at addr=24
        // accessing 16 bytes touches sectors 0 AND 1 (24..39 spans the
        // 32-byte boundary at 32). Pre-Round-6 returned 1 (only the
        // base sector); post-Round-6 returns 2.
        const uint64_t addrs[] = {24};
        CHECK(count_distinct_sectors(addrs, 1, 16) == 2);
    }

    {
        // Vectoradd-style coalesced 32x4B: lane_i at i*4 bytes, each
        // access is 4 bytes wide. Each warp's 32 lanes fit into 4
        // distinct 32B sectors (sector i contains lanes [8i .. 8i+7]).
        // Pre-Round-6 also returned 4 because no access crossed a
        // sector boundary — this case locks in that the fix does not
        // regress the working aligned-4B path.
        uint64_t addrs[32];
        for (int i = 0; i < 32; ++i) addrs[i] = (uint64_t)i * 4;
        CHECK(count_distinct_sectors(addrs, 32, 4) == 4);
    }

    {
        // Stride-2 32x4B: lane_i at i*8 bytes. Each access fits in a
        // distinct sector (lane 0 sector 0, lane 1 sector 0 also since
        // 8>>5=0, ...) — wait: lane_i*8 = 0, 8, 16, 24, 32, 40, ...
        // Sectors: 0,0,0,0,1,1,1,1,2,... 4 lanes per sector. 32 lanes /
        // 4 = 8 distinct sectors. Efficiency = (32*4) / (8*32) = 0.5.
        uint64_t addrs[32];
        for (int i = 0; i < 32; ++i) addrs[i] = (uint64_t)i * 8;
        CHECK(count_distinct_sectors(addrs, 32, 4) == 8);
    }

    {
        // Vectorised 32x16B perfectly aligned: lane_i at i*16 bytes,
        // each access 16 bytes. Lane sectors: lane 0 -> sector 0, lane
        // 1 -> sector 0 (16>>5=0), lane 2 -> sector 1, ... 2 lanes per
        // sector, 32 lanes / 2 = 16 distinct sectors. Pre-Round-6 also
        // returned 16 because no access crossed a boundary (each 16B
        // access aligned at 16B sits inside one 32B sector); post-
        // Round-6 returns 16 too — but the path now goes through the
        // interval-insertion loop, which is the algorithm AC-4 names.
        uint64_t addrs[32];
        for (int i = 0; i < 32; ++i) addrs[i] = (uint64_t)i * 16;
        CHECK(count_distinct_sectors(addrs, 32, 16) == 16);
    }

    {
        // Vectorised 32x16B misaligned by +24: lane_i at i*16 + 24
        // bytes. The +24 offset puts every lane at an in-sector
        // position where the 16B access crosses the next 32B boundary.
        //   lane 0:  addr=24, range [24,40) -> sectors {0,1}
        //   lane 1:  addr=40, range [40,56) -> sectors {1} (40>>5=1, 55>>5=1)
        //   lane 2:  addr=56, range [56,72) -> sectors {1,2}
        //   lane 3:  addr=72, range [72,88) -> sectors {2}
        //   lane 4:  addr=88, range [88,104) -> sectors {2,3}
        //   ...
        //   Even-numbered lanes start at an offset that crosses a
        //   boundary; odd-numbered lanes start at an offset that
        //   stays in one sector. Sectors visited: 0,1,1,1,2,2,2,3,3,...
        //   The full set across 32 lanes is {0, 1, 2, ..., 16} = 17
        //   distinct sectors. This is the canonical sector-crossing
        //   regression Codex named — pre-Round-6 returned 16 because
        //   even-numbered lanes only contributed their base sector.
        uint64_t addrs[32];
        for (int i = 0; i < 32; ++i) {
            addrs[i] = (uint64_t)(i * 16 + 24);
        }
        CHECK(count_distinct_sectors(addrs, 32, 16) == 17);
    }

    {
        // All-predicate-off path: n=0 returns 0 cleanly so the per-warp
        // ratio math sees `transferred_bytes == 0` and the host-side
        // emitter lands `null` for the field (no LDG / STG observed).
        // The early-return in the helper protects against pathological
        // input.
        const uint64_t empty[1] = {0};
        CHECK(count_distinct_sectors(empty, 0, 4) == 0);
    }

    {
        // bytes_per_access=0 also returns 0 (defensive — Instr::getSize()
        // should never return 0 for an actual LDG/STG, but if a future
        // ISA / hardware change yields a 0, the helper does not segfault
        // or wrap negative).
        const uint64_t a[1] = {64};
        CHECK(count_distinct_sectors(a, 1, 0) == 0);
    }

    {
        // 32x16B with stride-2 — each lane is 32 bytes from the next,
        // every lane lands in its own distinct sector. 32 distinct.
        // useful = 32*16 = 512, transferred = 32*32 = 1024, eff = 0.5.
        uint64_t addrs[32];
        for (int i = 0; i < 32; ++i) addrs[i] = (uint64_t)i * 32;
        CHECK(count_distinct_sectors(addrs, 32, 16) == 32);
    }

    // ------------------------------------------------------------------
    // count_shared_bank_conflicts — the host+device 32-bank shared-
    // memory conflict counter from bank_counting.h. Per AC-4 positive
    // tests 3 + 4: broadcast (all lanes same word) -> 0 conflicts;
    // 32-way same-bank different-words -> 31 conflicts/warp. The
    // additional cases below lock in perfect distribution (no
    // conflicts), 2-way (1 conflict), 16-active broadcast, and the
    // defensive zero-lane path.
    // ------------------------------------------------------------------

    {
        // Broadcast: all 32 lanes at the same address. distinct
        // (bank, word) tuples = 1. distinct_per_bank[0] = 1 → 0
        // conflicts. The H/W broadcasts to all lanes in a single
        // cycle.
        uint64_t addrs[32];
        for (int i = 0; i < 32; ++i) addrs[i] = (uint64_t)100;
        CHECK(count_shared_bank_conflicts(addrs, 32) == 0);
    }

    {
        // 32-way same-bank: lane i at byte address i*128. word_idx =
        // i*32; bank = (i*32) & 31 = 0 for every lane. Each lane lands
        // in bank 0 at a different word, so distinct_per_bank[0] = 32
        // and conflicts = 31. The H/W canonical worst-case shared-mem
        // conflict pattern.
        uint64_t addrs[32];
        for (int i = 0; i < 32; ++i) addrs[i] = (uint64_t)i * 128;
        CHECK(count_shared_bank_conflicts(addrs, 32) == 31);
    }

    {
        // Perfect distribution: lane i at byte address i*4. word_idx =
        // i; bank = i & 31 = i for every i in [0, 31]. Each lane in a
        // unique bank → distinct_per_bank[b] = 1 for every b → 0
        // conflicts. The ideal kernel pattern.
        uint64_t addrs[32];
        for (int i = 0; i < 32; ++i) addrs[i] = (uint64_t)i * 4;
        CHECK(count_shared_bank_conflicts(addrs, 32) == 0);
    }

    {
        // 2-way conflict: lane 0 at word 0 (bank 0); lane 1 at word 32
        // (bank 0, different word); lanes 2..31 at words 1..30 (banks
        // 1..30). Bank 0 has 2 distinct words → 1 conflict; banks
        // 1..30 each have 1 word → 0 conflicts. Total = 1.
        uint64_t addrs[32];
        addrs[0] = 0;           // bank 0, word 0
        addrs[1] = (uint64_t)128; // bank 0, word 32 (word_idx 32 & 31 = 0)
        for (int i = 2; i < 32; ++i) {
            addrs[i] = (uint64_t)(i - 1) * 4;  // words 1..30 → banks 1..30
        }
        CHECK(count_shared_bank_conflicts(addrs, 32) == 1);
    }

    {
        // 16-active broadcast: only 16 lanes participate, all hitting
        // the same address. 1 distinct (bank, word) → 0 conflicts.
        // Locks in that the algorithm handles partial-warp inputs the
        // device callback would pass after predicate filtering.
        uint64_t addrs[16];
        for (int i = 0; i < 16; ++i) addrs[i] = (uint64_t)200;
        CHECK(count_shared_bank_conflicts(addrs, 16) == 0);
    }

    {
        // n=0 → 0 (defensive). Matches the device callback's
        // num_pred_lanes==0 early-return semantics.
        const uint64_t empty[1] = {0};
        CHECK(count_shared_bank_conflicts(empty, 0) == 0);
    }

    // ------------------------------------------------------------------
    // count_active_lanes — the host+device branch-divergence bucket
    // selector from branch_counting.h. The helper maps a 32-bit
    // warp-collective predicate mask (built via __ballot_sync) to the
    // bucket index in branch_active_lanes_hist[0..32]. The bucket index
    // is the popcount of the mask, bounded to [0, 32] by construction.
    //
    // Per AC-4: "Predicated branch with 16 active lanes populates the
    // `16` bucket in branch_active_lanes_hist". The cases below pin
    // down all the canonical points plus the bucket-discrimination
    // invariant Codex Round 8 review's Directive Continuation Plan
    // item 2 explicitly requested.
    // ------------------------------------------------------------------

    {
        // All-inactive: mask 0x00000000 → bucket 0. Counts warp
        // instances where every active lane evaluated the predicate to
        // false (no lane took the branch this warp-instance).
        CHECK(count_active_lanes(0x00000000u) == 0);
    }

    {
        // All-active: mask 0xFFFFFFFF → bucket 32. The canonical
        // unconditional / always-true branch. Every active lane is
        // predicate-true.
        CHECK(count_active_lanes(0xFFFFFFFFu) == 32);
    }

    {
        // AC-4 canonical: 16 active lanes → bucket 16. Lower-half lane
        // pattern. This is the deterministic regression case Codex
        // Round 8 review named: a 16-active branch must land in bucket
        // 16, not in 15 or 17.
        CHECK(count_active_lanes(0x0000FFFFu) == 16);
    }

    {
        // 16 active lanes, upper-half pattern → still bucket 16. The
        // histogram is divergence-shape blind by design — two
        // warp-instances with different lane patterns but the same
        // popcount land in the same bucket. Locks in that the helper
        // does NOT distinguish 0x0000FFFF from 0xFFFF0000.
        CHECK(count_active_lanes(0xFFFF0000u) == 16);
    }

    {
        // 16 active lanes, every-other-lane pattern → still bucket 16.
        // A third 16-active mask shape; combined with the two above,
        // any future regression that breaks popcount semantics for one
        // mask shape but not the others is caught.
        CHECK(count_active_lanes(0x55555555u) == 16);
    }

    {
        // Sparse 2-active: lane 0 + lane 31 → bucket 2. Locks in that
        // the bucket index is popcount, not "highest lane index" or
        // anything else.
        CHECK(count_active_lanes(0x80000001u) == 2);
    }

    {
        // Single-lane (lane 0 only) → bucket 1. The minimum non-zero
        // bucket.
        CHECK(count_active_lanes(0x00000001u) == 1);
    }

    {
        // Bucket-discrimination invariant (Codex Round 8 directive
        // item 2): three 16-active warp instances with DIFFERENT lane
        // patterns must all land in bucket 16, leaving every other
        // bucket at zero. This simulates the leader-lane
        // atomicAdd(&hist[count], 1ULL) loop the device callback
        // executes warp-by-warp. If a future regression breaks the
        // bucket-selection logic for any one of the three patterns,
        // hist[16] != 3 or some other bucket becomes non-zero, and
        // this test fails.
        uint64_t hist[33];
        for (int i = 0; i < 33; ++i) hist[i] = 0;
        const uint32_t patterns[3] = {
            0x0000FFFFu, 0xFFFF0000u, 0x00FF00FFu,
        };
        for (int i = 0; i < 3; ++i) {
            const int bucket = count_active_lanes(patterns[i]);
            hist[bucket]++;
        }
        CHECK(hist[16] == 3);
        for (int i = 0; i < 33; ++i) {
            if (i != 16) CHECK(hist[i] == 0);
        }
    }

    if (failures == 0) {
        std::printf(
            "OK  classify_op_pair + classify_mnemonic + "
            "classify_memory_space + count_distinct_sectors + "
            "count_shared_bank_conflicts + count_active_lanes "
            "routing  (all 15 canonical instruction classification "
            "buckets exercised at both layers, DSMEM / WGMMA repair "
            "locked in, category-side fallbacks present, negative "
            "tests guard shared-vs-DSMEM and warpsync-vs-barrier "
            "collapses, mnemonic chain traverses Blackwell -> Hopper "
            "-> Ampere correctly, every NVBit InstrType::MemorySpace "
            "enum value routes through classify_memory_space with the "
            "DSMEM-only convention, sector-counter interval insertion "
            "locks in vectorised + misaligned cases that earlier "
            "base-sector-only logic would have undercounted, "
            "bank-conflict counter matches the canonical 32-bank "
            "model with broadcast=0 and 32-way same-bank=31, branch "
            "active-lanes bucket selector pins down all-inactive=0, "
            "all-active=32, three 16-active mask shapes -> bucket 16, "
            "sparse-2 -> bucket 2, and the bucket-discrimination "
            "invariant: three 16-active warp instances yield "
            "hist[16]==3 with every other bucket zero)\n");
        return 0;
    }
    std::fprintf(stderr, "%d assertion(s) failed\n", failures);
    return 1;
}
