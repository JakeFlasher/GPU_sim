// nvbit_tracer/metric_extractor/classifier.h
//
// Pure host-side instruction-class router: maps a (TraceInstrOpcode,
// uarch_op_t) pair from the shared ISA enums to one of the canonical
// AC-7 buckets (ffma / mufu / ldg / stg / lds / sts / hmma / wgmma /
// tcgen05 / tma / dsmem / bra / barrier / warpsync / unknown).
//
// Lives in a header (as opposed to inside metric_extractor.cu) so a host-
// only regression test can call the function directly without pulling in
// NVBit's CUDA dependencies, mirroring the discipline of
// nvbit_tracer/isa_def/test_opcode_map.cc.
//
// The function is `inline` (header-only) so multiple translation units
// that include it do not generate duplicate symbols at link time. No
// state lives here — the bucket id and class names are defined in
// common.h, and the input enums are defined in the isa_def/ headers.

#ifndef NVBIT_TRACER_METRIC_EXTRACTOR_CLASSIFIER_H_
#define NVBIT_TRACER_METRIC_EXTRACTOR_CLASSIFIER_H_

#include <string>

#include "common.h"

#include "isa_def/operation_type.h"
#include "isa_def/trace_opcode.h"
#include "isa_def/blackwell_opcode.h"
#include "isa_def/hopper_opcode.h"
#include "isa_def/ampere_opcode.h"

// NVBit's host-side memory-space enum. The header is part of NVBit's
// public interface and contains only constexpr definitions plus the
// MemorySpace / OperandType enums — no CUDA runtime dependency — so it
// is safe to include from the host-only regression test alongside the
// ISA maps above.
#include "../nvbit_release/core/instr_types.h"

// Memory-space-side classification. Returns INSTR_CLASS_DSMEM for the
// distributed-shared memory space (which the on-device path detects via
// `instr->getMemorySpace()` at instrument time), and INSTR_CLASS_UNKNOWN
// as a sentinel meaning "no memory-space-side mapping; fall through to
// the (opcode, category) / mnemonic classifier path". Sentinel-return is
// deliberate: the caller treats UNKNOWN as "this helper does not own the
// decision" rather than "the bucket is unknown" — that distinction lets
// classify_instr layer the memory-space check as a fast pre-pass without
// shadowing the canonical mnemonic-driven routing for non-DSMEM ops.
//
// The function is host-only and pure: no NVBit / CUDA dependencies, no
// global state. test_classifier.cc exercises every InstrType::MemorySpace
// enum value to lock in the DSMEM-only routing convention.
inline InstructionClass classify_memory_space(InstrType::MemorySpace ms) {
    if (ms == InstrType::MemorySpace::DISTRIBUTED_SHARED) {
        return INSTR_CLASS_DSMEM;
    }
    return INSTR_CLASS_UNKNOWN;
}

inline InstructionClass classify_op_pair(unsigned opcode,
                                          unsigned category) {
    // Blackwell-specific categories take precedence over the per-opcode
    // disambiguation below. Without this ordering, an entry like
    // `LDS.MMIO` mapped to `(OP_LDS, DSMEM_OP)` would land in the
    // INSTR_CLASS_LDS case from the opcode switch before the
    // DSMEM_OP arm of the category switch can route it correctly. The
    // category switch goes first for these three buckets only — the
    // remaining categories (TENSOR_CORE_OP for HMMA, BRANCH_OP for BRA,
    // etc.) still benefit from per-opcode disambiguation downstream so
    // HMMA / WGMMA do not collapse together.
    switch (category) {
        case TCGEN05_MXF4_OP:
        case TCGEN05_MXF4NVF4_OP:
        case TCGEN05_F8F6F4_OP:
        case TCGEN05_SCALE_VECTOR_OP:
            return INSTR_CLASS_TCGEN05;
        case TMA_OP:
            return INSTR_CLASS_TMA;
        case DSMEM_OP:
            return INSTR_CLASS_DSMEM;
        default:
            break;
    }
    switch (opcode) {
        case OP_FFMA:
        case OP_FFMA32I:
            return INSTR_CLASS_FFMA;
        case OP_MUFU:
            return INSTR_CLASS_MUFU;
        case OP_LDG:
        case OP_LDGSTS:
        case OP_LDGMC:
        case OP_LDGDEPBAR:
            return INSTR_CLASS_LDG;
        case OP_STG:
        case OP_STAS:
            return INSTR_CLASS_STG;
        case OP_LDS:
        case OP_LDSM:
        case OP_LDSLK:
            return INSTR_CLASS_LDS;
        case OP_STS:
        case OP_STSM:
        case OP_STSCUL:
            return INSTR_CLASS_STS;
        case OP_HMMA:
            return INSTR_CLASS_HMMA;
        case OP_BGMMA:
        case OP_HGMMA:
        case OP_IGMMA:
        case OP_QGMMA:
        case OP_WARPGROUP:
        case OP_WARPGROUPSET:
            return INSTR_CLASS_WGMMA;
        case OP_TCGEN05_MXF4:
        case OP_TCGEN05_MXF4NVF4:
        case OP_TCGEN05_F8F6F4:
        case OP_TCGEN05_SCALE_VECTOR:
            return INSTR_CLASS_TCGEN05;
        case OP_UTMACCTL:
        case OP_UTMACMDFLUSH:
        case OP_UTMALDG:
        case OP_UTMAPF:
        case OP_UTMAREDG:
        case OP_UTMASTG:
        case OP_UBLKCP:
        case OP_UBLKPF:
        case OP_UBLKRED:
            return INSTR_CLASS_TMA;
        case OP_WARPSYNC:
            return INSTR_CLASS_WARPSYNC;
        case OP_BAR:
        case OP_DEPBAR:
        case OP_MEMBAR:
        case OP_ERRBAR:
        case OP_CGAERRBAR:
        case OP_FENCE:
        case OP_SYNCS:
        case OP_BSYNC:
        case OP_BSSY:
        case OP_UCGABAR_ARV:
        case OP_UCGABAR_WAIT:
            return INSTR_CLASS_BARRIER;
        case OP_BRA:
        case OP_BREAK:
        case OP_BRX:
        case OP_JMP:
        case OP_JMX:
        case OP_CALL:
        case OP_RET:
        case OP_EXIT:
        case OP_BRXU:
        case OP_JMXU:
            return INSTR_CLASS_BRA;
        case OP_UNKNOWN:
            return INSTR_CLASS_UNKNOWN;
        default:
            break;
    }
    // Fall-through category mapping for entries the per-opcode switch
    // does not name explicitly (TENSOR_CORE_OP, BRANCH_OP, BARRIER_OP
    // variants, UNKNOWN_OP). Blackwell categories (TCGEN05_*, TMA_OP,
    // DSMEM_OP) are already handled by the early-out switch above.
    switch (category) {
        case BRANCH_OP:
            return INSTR_CLASS_BRA;
        case BARRIER_OP:
        case MEMORY_BARRIER_OP:
        case GRID_BARRIER_OP:
            return INSTR_CLASS_BARRIER;
        case TENSOR_CORE_OP:
            return INSTR_CLASS_HMMA;
        case UNKNOWN_OP:
            return INSTR_CLASS_UNKNOWN;
        default:
            return INSTR_CLASS_UNKNOWN;
    }
}

// Mnemonic-driven classifier. Encapsulates the same Blackwell → Hopper →
// Ampere lookup chain that classify_instr() uses (with the full-mnemonic
// then token-0 fallback), but takes a `const std::string&` instead of an
// NVBit `Instr*` so a host-only regression test can exercise the chain
// without pulling in NVBit's CUDA dependencies.
//
// Returns INSTR_CLASS_UNKNOWN when no map contains the mnemonic; otherwise
// returns classify_op_pair() applied to the looked-up (opcode, category)
// pair. Locked in by test_classifier.cc as the AC-7 routing authority.
inline InstructionClass classify_mnemonic(const std::string& mnemonic) {
    if (mnemonic.empty()) return INSTR_CLASS_UNKNOWN;
    auto lookup = [](const std::string& key) -> const OpcodeChar* {
        auto it = Blackwell_OpcodeMap.find(key);
        if (it != Blackwell_OpcodeMap.end()) return &it->second;
        auto it2 = Hopper_OpcodeMap.find(key);
        if (it2 != Hopper_OpcodeMap.end()) return &it2->second;
        auto it3 = Ampere_OpcodeMap.find(key);
        if (it3 != Ampere_OpcodeMap.end()) return &it3->second;
        return nullptr;
    };
    const OpcodeChar* oc = lookup(mnemonic);
    if (oc == nullptr) {
        const size_t dot = mnemonic.find('.');
        if (dot != std::string::npos) {
            oc = lookup(mnemonic.substr(0, dot));
        }
    }
    if (oc == nullptr) return INSTR_CLASS_UNKNOWN;
    return classify_op_pair(oc->opcode, oc->opcode_category);
}

#endif  // NVBIT_TRACER_METRIC_EXTRACTOR_CLASSIFIER_H_
