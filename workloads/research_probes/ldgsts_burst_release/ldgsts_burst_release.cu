#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t status = (call);                                      \
        if (status != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s failed: %s\n", #call,                  \
                         cudaGetErrorString(status));                           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

constexpr int kWarpSize = 32;
constexpr int kThreads = 128;
constexpr int kMaxWarps = 4;
constexpr int kPositions = 8;
constexpr int kPayloadBytes = 16;
constexpr std::size_t kChunkBytes = 512;
constexpr std::size_t kWarpDestinationBytes =
    kPositions * kChunkBytes;
constexpr std::size_t kArmDestinationBytes =
    kMaxWarps * kWarpDestinationBytes;
constexpr std::size_t kSharedDestinationBytes =
    2 * kArmDestinationBytes;
constexpr int kDefaultBursts = 16384;
constexpr int kDefaultWarmup = 64;
constexpr int kMaxBursts = 16384;
constexpr int kMaxRepetitions = 7;
constexpr int kTimingSchema = 2;
constexpr int kArmOrderBlock = 64;
constexpr std::size_t kL2Chunks = 1024;
constexpr std::size_t kChunksPerWarp =
    static_cast<std::size_t>(kPositions) * kMaxBursts;
constexpr std::size_t kSourceBytesPerWarp = kChunksPerWarp * kChunkBytes;
constexpr std::size_t kSourceBytes = kMaxWarps * kSourceBytesPerWarp;
constexpr std::uint32_t kOnSentinel = 0xa5a5a5a5u;
constexpr std::uint32_t kOffSentinel = 0x3c3c3c3cu;

static_assert(kSharedDestinationBytes == 32u * 1024u,
              "the two target arms require exactly 32 KiB");
static_assert(kSourceBytesPerWarp == 64u * 1024u * 1024u,
              "the DRAM mapping requires 64 MiB per warp");

enum Route : int {
    kL1 = 0,
    kL2 = 1,
    kDram = 2,
};

enum Assay : int {
    kLdgsts = 0,
    kLdg = 1,
};

enum RunMode : int {
    kDirect = 0,
    kNcu = 1,
    kCorrectness = 2,
};

struct CellSpec {
    int route;
    int depth;
    int active_warps;
    int gap_cycles;
};

struct alignas(16) DirectRecord {
    std::uint32_t words[16];
};

struct WarpState {
    unsigned long long errors;
    std::uint32_t smid_before;
    std::uint32_t smid_after;
    std::uint32_t checksum;
};

struct ArmSample {
    std::uint32_t issue_cycles;
    // Endpoint duration from the arm start (not the post-issue interval).
    std::uint32_t drain_cycles;
    unsigned long long gaps_low;
    unsigned long long gaps_high;
    std::uint32_t smid_before;
    std::uint32_t smid_after;
    std::uint32_t errors;
    std::uint32_t checksum;
};

// Keep every timing/identity result live in compact correctness and NCU
// launches.  Those modes do not emit the direct per-burst record, so without
// this small fold ptxas is free to delete the clock reads (and even the
// endpoint sequence) as dead values.  The fold is outside the timed device
// body; it only makes the same body observable in both launch modes.
__device__ __forceinline__ void retain_sample(const ArmSample& sample,
                                               std::uint32_t* checksum) {
    *checksum ^= sample.issue_cycles + 0x9e3779b9u;
    *checksum ^= (sample.drain_cycles << 7) |
                 (sample.drain_cycles >> 25);
    *checksum ^= static_cast<std::uint32_t>(sample.gaps_low);
    *checksum ^= static_cast<std::uint32_t>(sample.gaps_low >> 32);
    *checksum ^= static_cast<std::uint32_t>(sample.gaps_high);
    *checksum ^= static_cast<std::uint32_t>(sample.gaps_high >> 32);
    *checksum ^= sample.smid_before * 0x45d9f3bu;
    *checksum ^= sample.smid_after * 0x119de1f3u;
}

struct HostCell {
    std::string key;
    CellSpec spec;
};

__host__ __device__ __forceinline__ std::uint32_t source_word_value(
    std::size_t word_index) {
    return static_cast<std::uint32_t>(word_index) * 1664525u +
           1013904223u ^ 0x9e3779b9u;
}

__device__ __forceinline__ unsigned read_smid() {
    unsigned smid;
    asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
    return smid;
}

__device__ __forceinline__ unsigned long long read_clock() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %clock64;" : "=l"(value) :: "memory");
    return value;
}

__device__ __forceinline__ std::uint32_t narrow_cycles(
    unsigned long long cycles) {
    return cycles > 0xffffffffull ? 0xffffffffu
                                  : static_cast<std::uint32_t>(cycles);
}

// Inline the delay loop so ptxas cannot attach a hidden asynchronous-copy
// dependence wait to a helper CALL between two LDGSTS positions.  The unique
// label keeps every expansion independent; the loop itself is outside the
// zero-gap timed body and its achieved duration is retained per edge.
__device__ __forceinline__ std::uint32_t clock_spin(int requested_cycles) {
    if (requested_cycles == 0) return 0;
    const unsigned long long target =
        static_cast<unsigned long long>(requested_cycles);
    unsigned long long achieved;
    asm volatile(
        "{\n\t"
        ".reg .u64 spin_start, spin_now, spin_elapsed;\n\t"
        ".reg .pred spin_done;\n\t"
        "mov.u64 spin_start, %clock64;\n\t"
        "spin_loop%=: mov.u64 spin_now, %clock64;\n\t"
        "sub.u64 spin_elapsed, spin_now, spin_start;\n\t"
        "setp.lt.u64 spin_done, spin_elapsed, %1;\n\t"
        "@spin_done bra.uni spin_loop%=;\n\t"
        "mov.u64 %0, spin_elapsed;\n\t"
        "}"
        : "=l"(achieved)
        : "l"(target)
        : "memory");
    return narrow_cycles(achieved);
}

__device__ __forceinline__ void retain_gap(ArmSample* sample, int edge,
                                            std::uint32_t cycles) {
    const unsigned long long value =
        static_cast<unsigned long long>(cycles > 0xffffu ? 0xffffu : cycles);
    if (edge < 4) {
        sample->gaps_low |= value << (16 * edge);
    } else {
        sample->gaps_high |= value << (16 * (edge - 4));
    }
}

__device__ __forceinline__ void copy_position(unsigned destination,
                                               const char* source) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                 :
                 : "r"(destination), "l"(source)
                 : "memory");
}

// A single zero-gap body keeps all eight static positions.  CUDA 13.2 emits
// guarded CFGs when predicates are reused, so each conditional position gets a
// distinct predicate and a deliberately different but equivalent comparison.
// The supported depth domain is {1,4,5,6,8}; position zero is unpredicated in
// the enabled body, while the paired off arm uses the empty reference.
#define LDGSTS_RUNTIME_G0_ASM()                                               \
    asm volatile("{ .reg .pred p0,p1,p2,p3,p4,p5,p6;\n\t"                  \
                 ".reg .u32 depth_local, lane_local;\n\t"                    \
                 "mov.u32 depth_local, %21;\n\t"                            \
                 "mov.u32 lane_local, %laneid;\n\t"                         \
                 "xor.b32 depth_local, depth_local, lane_local;\n\t"          \
                 "xor.b32 depth_local, depth_local, lane_local;\n\t"          \
                 "setp.gt.s32 p0, depth_local, 3;\n\t"                       \
                 "setp.ge.s32 p1, depth_local, 4;\n\t"                       \
                 "setp.lt.s32 p2, depth_local, 4;\n\t"                       \
                 "setp.ge.s32 p3, depth_local, 5;\n\t"                       \
                 "setp.gt.s32 p4, depth_local, 5;\n\t"                       \
                 "setp.gt.s32 p5, depth_local, 6;\n\t"                       \
                 "setp.ge.s32 p6, depth_local, 8;\n\t"                       \
                 "mov.u32 %3, %smid;\n\t"                                   \
                 "mov.u64 %0, %clock64;\n\t"                                \
                 "cp.async.ca.shared.global [%13], [%5], 16;\n\t"          \
                 "@p0 cp.async.ca.shared.global [%14], [%6], 16;\n\t"       \
                 "@p1 cp.async.ca.shared.global [%15], [%7], 16;\n\t"       \
                 "@!p2 cp.async.ca.shared.global [%16], [%8], 16;\n\t"      \
                 "@p3 cp.async.ca.shared.global [%17], [%9], 16;\n\t"       \
                 "@p4 cp.async.ca.shared.global [%18], [%10], 16;\n\t"      \
                 "@p5 cp.async.ca.shared.global [%19], [%11], 16;\n\t"      \
                 "@p6 cp.async.ca.shared.global [%20], [%12], 16;\n\t"      \
                 "mov.u64 %1, %clock64;\n\t"                                \
                 "cp.async.commit_group;\n\t"                                \
                 "cp.async.wait_group 0;\n\t"                              \
                 "mov.u64 %2, %clock64;\n\t"                                \
                 "mov.u32 %4, %smid; }"                                      \
                 : "=l"(start), "=l"(issue_end), "=l"(drain_end),          \
                   "=r"(smid_before), "=r"(smid_after)                       \
                 : "l"(s0), "l"(s1), "l"(s2), "l"(s3), "l"(s4),         \
                   "l"(s5), "l"(s6), "l"(s7), "r"(d0), "r"(d1),         \
                   "r"(d2), "r"(d3), "r"(d4), "r"(d5), "r"(d6),         \
                   "r"(d7), "r"(depth)                                       \
                 : "memory")

__device__ __forceinline__ ArmSample ldgsts_runtime_g0_body(
    int depth, unsigned d0, unsigned d1, unsigned d2, unsigned d3,
    unsigned d4, unsigned d5, unsigned d6, unsigned d7, const char* s0,
    const char* s1, const char* s2, const char* s3, const char* s4,
    const char* s5, const char* s6, const char* s7) {
    unsigned long long start;
    unsigned long long issue_end;
    unsigned long long drain_end;
    std::uint32_t smid_before;
    std::uint32_t smid_after;
    LDGSTS_RUNTIME_G0_ASM();
    ArmSample sample{};
    sample.smid_before = smid_before;
    sample.smid_after = smid_after;
    sample.issue_cycles = narrow_cycles(issue_end - start);
    // Keep the drain endpoint relative to the same arm start as issue_cycles;
    // the reducer can then form the post-issue completion interval explicitly.
    sample.drain_cycles = narrow_cycles(drain_end - start);
    return sample;
}

template <int Depth, int Gap>
__device__ __forceinline__ ArmSample ldgsts_active_body(
    unsigned d0, unsigned d1, unsigned d2, unsigned d3, unsigned d4,
    unsigned d5, unsigned d6, unsigned d7, const char* s0, const char* s1,
    const char* s2, const char* s3, const char* s4, const char* s5,
    const char* s6, const char* s7) {
    static_assert(Depth == 1 || Depth == 4 || Depth == 5 || Depth == 6 ||
                      Depth == 8,
                  "unsupported LDGSTS depth");
    static_assert(Gap == 16 || Gap == 64,
                  "unsupported LDGSTS gap");
    ArmSample sample{};
    sample.smid_before = read_smid();
    const unsigned long long start = read_clock();

    copy_position(d0, s0);
    if constexpr (Gap != 0 && Depth > 1)
        retain_gap(&sample, 0, clock_spin(Gap));
    if constexpr (Depth > 1) copy_position(d1, s1);
    if constexpr (Gap != 0 && Depth > 2)
        retain_gap(&sample, 1, clock_spin(Gap));
    if constexpr (Depth > 2) copy_position(d2, s2);
    if constexpr (Gap != 0 && Depth > 3)
        retain_gap(&sample, 2, clock_spin(Gap));
    if constexpr (Depth > 3) copy_position(d3, s3);
    if constexpr (Gap != 0 && Depth > 4)
        retain_gap(&sample, 3, clock_spin(Gap));
    if constexpr (Depth > 4) copy_position(d4, s4);
    if constexpr (Gap != 0 && Depth > 5)
        retain_gap(&sample, 4, clock_spin(Gap));
    if constexpr (Depth > 5) copy_position(d5, s5);
    if constexpr (Gap != 0 && Depth > 6)
        retain_gap(&sample, 5, clock_spin(Gap));
    if constexpr (Depth > 6) copy_position(d6, s6);
    if constexpr (Gap != 0 && Depth > 7)
        retain_gap(&sample, 6, clock_spin(Gap));
    if constexpr (Depth > 7) copy_position(d7, s7);

    const unsigned long long issue_end = read_clock();
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    const unsigned long long drain_end = read_clock();
    sample.smid_after = read_smid();
    sample.issue_cycles = narrow_cycles(issue_end - start);
    sample.drain_cycles = narrow_cycles(drain_end - start);
    return sample;
}

#undef LDGSTS_RUNTIME_G0_ASM

__device__ __noinline__ ArmSample ldgsts_empty_body() {
    ArmSample sample{};
    sample.smid_before = read_smid();
    const unsigned long long start = read_clock();
    const unsigned long long issue_end = read_clock();
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    const unsigned long long drain_end = read_clock();
    sample.smid_after = read_smid();
    sample.issue_cycles = narrow_cycles(issue_end - start);
    sample.drain_cycles = narrow_cycles(drain_end - start);
    return sample;
}

template <int Depth, int Gap>
__device__ __noinline__ ArmSample ldgsts_gap_reference_body() {
    static_assert((Depth == 4 || Depth == 5 || Depth == 6) &&
                      (Gap == 16 || Gap == 64),
                  "unsupported gap reference");
    ArmSample sample{};
    sample.smid_before = read_smid();
    const unsigned long long start = read_clock();
    retain_gap(&sample, 0, clock_spin(Gap));
    retain_gap(&sample, 1, clock_spin(Gap));
    retain_gap(&sample, 2, clock_spin(Gap));
    if constexpr (Depth > 4) retain_gap(&sample, 3, clock_spin(Gap));
    if constexpr (Depth > 5) retain_gap(&sample, 4, clock_spin(Gap));
    const unsigned long long issue_end = read_clock();
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    const unsigned long long drain_end = read_clock();
    sample.smid_after = read_smid();
    sample.issue_cycles = narrow_cycles(issue_end - start);
    sample.drain_cycles = narrow_cycles(drain_end - start);
    return sample;
}

#define LDGSTS_ARGS                                                           \
    d0, d1, d2, d3, d4, d5, d6, d7, s0, s1, s2, s3, s4, s5, s6, s7

#define LDG_REGISTERS                                                         \
    ".reg .u32 a0,a1,a2,a3,b0,b1,b2,b3,c0,c1,c2,c3,d0,d1,d2,d3,"           \
    "e0,e1,e2,e3,f0,f1,f2,f3,g0,g1,g2,g3,h0,h1,h2,h3;\n\t"
#define LDG_1 "ld.global.ca.v4.u32 {a0,a1,a2,a3}, [%13];\n\t"
#define LDG_2 "ld.global.ca.v4.u32 {b0,b1,b2,b3}, [%14];\n\t"
#define LDG_3 "ld.global.ca.v4.u32 {c0,c1,c2,c3}, [%15];\n\t"
#define LDG_4 "ld.global.ca.v4.u32 {d0,d1,d2,d3}, [%16];\n\t"
#define LDG_5 "ld.global.ca.v4.u32 {e0,e1,e2,e3}, [%17];\n\t"
#define LDG_6 "ld.global.ca.v4.u32 {f0,f1,f2,f3}, [%18];\n\t"
#define LDG_7 "ld.global.ca.v4.u32 {g0,g1,g2,g3}, [%19];\n\t"
#define LDG_8 "ld.global.ca.v4.u32 {h0,h1,h2,h3}, [%20];\n\t"
#define STS_1 "st.shared.v4.u32 [%5], {a0,a1,a2,a3};\n\t"
#define STS_2 "st.shared.v4.u32 [%6], {b0,b1,b2,b3};\n\t"
#define STS_3 "st.shared.v4.u32 [%7], {c0,c1,c2,c3};\n\t"
#define STS_4 "st.shared.v4.u32 [%8], {d0,d1,d2,d3};\n\t"
#define STS_5 "st.shared.v4.u32 [%9], {e0,e1,e2,e3};\n\t"
#define STS_6 "st.shared.v4.u32 [%10], {f0,f1,f2,f3};\n\t"
#define STS_7 "st.shared.v4.u32 [%11], {g0,g1,g2,g3};\n\t"
#define STS_8 "st.shared.v4.u32 [%12], {h0,h1,h2,h3};\n\t"
#define LDG_PREFIX_4 LDG_1 LDG_2 LDG_3 LDG_4
#define LDG_PREFIX_5 LDG_PREFIX_4 LDG_5
#define LDG_PREFIX_6 LDG_PREFIX_5 LDG_6
#define LDG_PREFIX_8 LDG_PREFIX_6 LDG_7 LDG_8
#define STS_PREFIX_4 STS_1 STS_2 STS_3 STS_4
#define STS_PREFIX_5 STS_PREFIX_4 STS_5
#define STS_PREFIX_6 STS_PREFIX_5 STS_6
#define STS_PREFIX_8 STS_PREFIX_6 STS_7 STS_8

#define LDG_G0_ASM(LOADS, STORES)                                               \
    asm volatile("{\n\t" LDG_REGISTERS                                      \
                 "mov.u32 %3, %smid;\n\t"                                   \
                 "mov.u64 %0, %clock64;\n\t" LOADS                         \
                 "mov.u64 %1, %clock64;\n\t" STORES                        \
                 "mov.u64 %2, %clock64;\n\t"                                \
                 "mov.u32 %4, %smid;\n\t"                                  \
                 "}"                                                           \
                 : "=l"(start), "=l"(issue_end), "=l"(drain_end),          \
                   "=r"(smid_before), "=r"(smid_after)                       \
                 : "r"(d0), "r"(d1), "r"(d2), "r"(d3), "r"(d4),         \
                   "r"(d5), "r"(d6), "r"(d7), "l"(s0), "l"(s1),         \
                   "l"(s2), "l"(s3), "l"(s4), "l"(s5), "l"(s6),         \
                   "l"(s7)                                                     \
                 : "memory")

template <int Depth>
__device__ __forceinline__ ArmSample ldg_active_body(
    unsigned d0, unsigned d1, unsigned d2, unsigned d3, unsigned d4,
    unsigned d5, unsigned d6, unsigned d7, const char* s0, const char* s1,
    const char* s2, const char* s3, const char* s4, const char* s5,
    const char* s6, const char* s7) {
    static_assert(Depth == 4 || Depth == 5 || Depth == 6 || Depth == 8,
                  "unsupported LDG depth");
    unsigned long long start;
    unsigned long long issue_end;
    unsigned long long drain_end;
    std::uint32_t smid_before;
    std::uint32_t smid_after;
    if constexpr (Depth == 4) {
        LDG_G0_ASM(LDG_PREFIX_4, STS_PREFIX_4);
    } else if constexpr (Depth == 5) {
        LDG_G0_ASM(LDG_PREFIX_5, STS_PREFIX_5);
    } else if constexpr (Depth == 6) {
        LDG_G0_ASM(LDG_PREFIX_6, STS_PREFIX_6);
    } else {
        LDG_G0_ASM(LDG_PREFIX_8, STS_PREFIX_8);
    }
    ArmSample sample{};
    sample.smid_before = smid_before;
    sample.smid_after = smid_after;
    sample.issue_cycles = narrow_cycles(issue_end - start);
    sample.drain_cycles = narrow_cycles(drain_end - start);
    return sample;
}

#undef LDG_G0_ASM
#undef STS_PREFIX_8
#undef STS_PREFIX_6
#undef STS_PREFIX_5
#undef STS_PREFIX_4
#undef LDG_PREFIX_8
#undef LDG_PREFIX_6
#undef LDG_PREFIX_5
#undef LDG_PREFIX_4
#undef STS_8
#undef STS_7
#undef STS_6
#undef STS_5
#undef STS_4
#undef STS_3
#undef STS_2
#undef STS_1
#undef LDG_8
#undef LDG_7
#undef LDG_6
#undef LDG_5
#undef LDG_4
#undef LDG_3
#undef LDG_2
#undef LDG_1
#undef LDG_REGISTERS

__device__ __noinline__ ArmSample ldg_empty_body() {
    ArmSample sample{};
    sample.smid_before = read_smid();
    const unsigned long long start = read_clock();
    const unsigned long long issue_end = read_clock();
    const unsigned long long drain_end = read_clock();
    sample.smid_after = read_smid();
    sample.issue_cycles = narrow_cycles(issue_end - start);
    sample.drain_cycles = narrow_cycles(drain_end - start);
    return sample;
}

// Compact (NCU/correctness) launches bind assay/gap at compile time.  The
// zero-gap path keeps depth runtime-controlled in one inline-asm block with
// eight contiguous LDGSTS positions and no call or branch between them.
// Gapped variants retain clock-spin samples only between consecutive
// positions. Direct mode deliberately keeps its runtime dispatcher so
// multi-cell bundles stay one CTA launch.
template <int FixedAssay, int FixedDepth, int FixedGap>
__device__ __forceinline__ ArmSample dispatch_compact_body(
    bool active, int runtime_depth, unsigned d0, unsigned d1, unsigned d2,
    unsigned d3,
    unsigned d4, unsigned d5, unsigned d6, unsigned d7, const char* s0,
    const char* s1, const char* s2, const char* s3, const char* s4,
    const char* s5, const char* s6, const char* s7) {
    if constexpr (FixedAssay == kLdgsts) {
        if constexpr (FixedGap == 0) {
            if (!active) return ldgsts_empty_body();
            return ldgsts_runtime_g0_body(
                runtime_depth, d0, d1, d2, d3, d4, d5, d6, d7, s0, s1, s2,
                s3, s4, s5, s6, s7);
        } else {
            if (!active)
                return ldgsts_gap_reference_body<FixedDepth, FixedGap>();
            return ldgsts_active_body<FixedDepth, FixedGap>(
                d0, d1, d2, d3, d4, d5, d6, d7, s0, s1, s2, s3, s4, s5,
                s6, s7);
        }
    } else {
        static_assert(FixedGap == 0, "LDG control assay has no gap variant");
        if (!active) return ldg_empty_body();
        return ldg_active_body<FixedDepth>(
            d0, d1, d2, d3, d4, d5, d6, d7, s0, s1, s2, s3, s4, s5, s6,
            s7);
    }
}

__device__ __forceinline__ ArmSample dispatch_ldgsts_body(
    bool active, int depth, int gap, unsigned d0, unsigned d1, unsigned d2,
    unsigned d3, unsigned d4, unsigned d5, unsigned d6, unsigned d7,
    const char* s0, const char* s1, const char* s2, const char* s3,
    const char* s4, const char* s5, const char* s6, const char* s7) {
    if (!active) {
        if (gap == 0) return ldgsts_empty_body();
        if (gap == 16) {
            if (depth == 4) return ldgsts_gap_reference_body<4, 16>();
            if (depth == 5) return ldgsts_gap_reference_body<5, 16>();
            return ldgsts_gap_reference_body<6, 16>();
        }
        if (depth == 4) return ldgsts_gap_reference_body<4, 64>();
        if (depth == 5) return ldgsts_gap_reference_body<5, 64>();
        return ldgsts_gap_reference_body<6, 64>();
    }
    if (gap == 0) {
        return ldgsts_runtime_g0_body(
            depth, d0, d1, d2, d3, d4, d5, d6, d7, s0, s1, s2, s3, s4, s5,
            s6, s7);
    }
    if (gap == 16) {
        if (depth == 4) return ldgsts_active_body<4, 16>(LDGSTS_ARGS);
        if (depth == 5) return ldgsts_active_body<5, 16>(LDGSTS_ARGS);
        return ldgsts_active_body<6, 16>(LDGSTS_ARGS);
    }
    if (depth == 4) return ldgsts_active_body<4, 64>(LDGSTS_ARGS);
    if (depth == 5) return ldgsts_active_body<5, 64>(LDGSTS_ARGS);
    return ldgsts_active_body<6, 64>(LDGSTS_ARGS);
}

#undef LDGSTS_ARGS

__device__ __forceinline__ ArmSample dispatch_ldg_body(
    bool active, int depth, unsigned d0, unsigned d1, unsigned d2,
    unsigned d3, unsigned d4, unsigned d5, unsigned d6, unsigned d7,
    const char* s0, const char* s1, const char* s2, const char* s3,
    const char* s4, const char* s5, const char* s6, const char* s7) {
    if (!active) return ldg_empty_body();
    if (depth == 4)
        return ldg_active_body<4>(d0, d1, d2, d3, d4, d5, d6, d7, s0, s1,
                                  s2, s3, s4, s5, s6, s7);
    if (depth == 5)
        return ldg_active_body<5>(d0, d1, d2, d3, d4, d5, d6, d7, s0, s1,
                                  s2, s3, s4, s5, s6, s7);
    if (depth == 6)
        return ldg_active_body<6>(d0, d1, d2, d3, d4, d5, d6, d7, s0, s1,
                                  s2, s3, s4, s5, s6, s7);
    return ldg_active_body<8>(d0, d1, d2, d3, d4, d5, d6, d7, s0, s1, s2,
                              s3, s4, s5, s6, s7);
}

__device__ __forceinline__ std::size_t source_chunk_index(int route,
                                                           int burst,
                                                           int position) {
    if (route == kL1) return static_cast<std::size_t>(position);
    if (route == kL2) {
        return (static_cast<std::size_t>(burst) + 128u * position) &
               (kL2Chunks - 1u);
    }
    return static_cast<std::size_t>(kPositions) * burst + position;
}

__device__ __forceinline__ uint4 expected_shared_value(
    const char* source_base, const char* source, std::uint32_t sentinel,
    bool copied) {
    if (!copied) return make_uint4(sentinel, sentinel, sentinel, sentinel);
    const std::size_t word =
        static_cast<std::size_t>(source - source_base) / sizeof(unsigned);
    return make_uint4(source_word_value(word), source_word_value(word + 1),
                      source_word_value(word + 2),
                      source_word_value(word + 3));
}

__device__ __forceinline__ std::uint32_t check_shared_position(
    const char* destination, const char* source_base, const char* source,
    std::uint32_t sentinel, bool copied, std::uint32_t* checksum) {
    const volatile uint4* pointer =
        reinterpret_cast<const volatile uint4*>(destination);
    const uint4 value =
        make_uint4(pointer->x, pointer->y, pointer->z, pointer->w);
    const uint4 expected =
        expected_shared_value(source_base, source, sentinel, copied);
    *checksum ^= value.x + 0x7f4a7c15u;
    *checksum ^= value.y ^ (value.z << 3) ^ (value.w << 11);
    return value.x != expected.x || value.y != expected.y ||
                   value.z != expected.z || value.w != expected.w
               ? 1u
               : 0u;
}

__device__ __forceinline__ std::uint32_t check_target_destinations(
    const char* source_base, const char* on_destination,
    const char* off_destination, const char* s0, const char* s1,
    const char* s2, const char* s3, const char* s4, const char* s5,
    const char* s6, const char* s7, int depth, std::uint32_t* checksum) {
    const char* sources[kPositions] = {s0, s1, s2, s3, s4, s5, s6, s7};
    std::uint32_t errors = 0;
#pragma unroll
    for (int position = 0; position < kPositions; ++position) {
        errors += check_shared_position(
            on_destination + position * kChunkBytes, source_base,
            sources[position], kOnSentinel, position < depth, checksum);
        errors += check_shared_position(
            off_destination + position * kChunkBytes, source_base,
            sources[position], kOffSentinel, false, checksum);
    }
    return errors;
}

__device__ __forceinline__ std::uint32_t pack_u16(std::uint32_t low,
                                                   std::uint32_t high) {
    return (low & 0xffffu) | ((high & 0xffffu) << 16);
}

__device__ __forceinline__ void streaming_store_record(
    DirectRecord* destination, const ArmSample& on, const ArmSample& off,
    std::uint32_t error_mask, bool enabled_first) {
    const std::uint32_t* on_low =
        reinterpret_cast<const std::uint32_t*>(&on.gaps_low);
    const std::uint32_t* on_high =
        reinterpret_cast<const std::uint32_t*>(&on.gaps_high);
    const std::uint32_t* off_low =
        reinterpret_cast<const std::uint32_t*>(&off.gaps_low);
    const std::uint32_t* off_high =
        reinterpret_cast<const std::uint32_t*>(&off.gaps_high);
    std::uint32_t* out = destination->words;
    asm volatile(
        "st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(out), "r"(on.issue_cycles), "r"(on.drain_cycles),
          "r"(off.issue_cycles), "r"(off.drain_cycles)
        : "memory");
    asm volatile(
        "st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(out + 4), "r"(on_low[0]), "r"(on_low[1]),
          "r"(on_high[0]), "r"(on_high[1])
        : "memory");
    asm volatile(
        "st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(out + 8), "r"(off_low[0]), "r"(off_low[1]),
          "r"(off_high[0]), "r"(off_high[1])
        : "memory");
    asm volatile(
        "st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(out + 12), "r"(pack_u16(on.smid_before, on.smid_after)),
          "r"(pack_u16(off.smid_before, off.smid_after)), "r"(error_mask),
          "r"(enabled_first ? 0u : 1u)
        : "memory");
}

__device__ __forceinline__ void initialize_destinations(char* shared) {
    for (std::size_t offset =
             static_cast<std::size_t>(threadIdx.x) * sizeof(uint4);
         offset < kSharedDestinationBytes;
         offset += static_cast<std::size_t>(blockDim.x) * sizeof(uint4)) {
        const std::uint32_t sentinel =
            offset < kArmDestinationBytes ? kOnSentinel : kOffSentinel;
        *reinterpret_cast<uint4*>(shared + offset) =
            make_uint4(sentinel, sentinel, sentinel, sentinel);
    }
}

__device__ __forceinline__ std::uint32_t prewarm_load(const char* address,
                                                       int route) {
    std::uint32_t value;
    if (route == kL1) {
        asm volatile("ld.global.ca.u32 %0, [%1];"
                     : "=r"(value)
                     : "l"(address)
                     : "memory");
    } else {
        asm volatile("ld.global.cg.u32 %0, [%1];"
                     : "=r"(value)
                     : "l"(address)
                     : "memory");
    }
    return value;
}

__device__ __forceinline__ void prewarm_cell(const char* source,
                                              const CellSpec& cell) {
    if (cell.route == kDram) return;
    const std::size_t region_bytes =
        cell.route == kL1 ? kPositions * kChunkBytes
                          : kL2Chunks * kChunkBytes;
    const std::size_t words_per_warp = region_bytes / sizeof(unsigned);
    const std::size_t total_words =
        static_cast<std::size_t>(cell.active_warps) * words_per_warp;
    std::uint32_t token = 0;
    for (std::size_t flat = threadIdx.x; flat < total_words;
         flat += blockDim.x) {
        const std::size_t warp = flat / words_per_warp;
        const std::size_t word = flat % words_per_warp;
        token ^= prewarm_load(source + warp * kSourceBytesPerWarp +
                                  word * sizeof(unsigned),
                              cell.route);
    }
    asm volatile("" : : "r"(token) : "memory");
}

__global__ void initialize_source(std::uint32_t* source,
                                  std::size_t word_count) {
    for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < word_count; index += gridDim.x * blockDim.x) {
        source[index] = source_word_value(index);
    }
}

__global__ void prewarm_source(const char* source, CellSpec cell,
                               std::uint32_t* sink) {
    if (cell.route == kDram) return;
    const std::size_t region_bytes =
        cell.route == kL1 ? kPositions * kChunkBytes
                          : kL2Chunks * kChunkBytes;
    const std::size_t words_per_warp = region_bytes / sizeof(unsigned);
    const std::size_t total_words =
        static_cast<std::size_t>(cell.active_warps) * words_per_warp;
    std::uint32_t token = 0;
    for (std::size_t flat = blockIdx.x * blockDim.x + threadIdx.x;
         flat < total_words; flat += gridDim.x * blockDim.x) {
        const std::size_t warp = flat / words_per_warp;
        const std::size_t word = flat % words_per_warp;
        token ^= prewarm_load(source + warp * kSourceBytesPerWarp +
                                  word * sizeof(unsigned),
                              cell.route);
    }
    if (token != 0) atomicXor(sink, token);
}

template <bool Compact, int FixedAssay = -1, int FixedDepth = -1,
          int FixedGap = -1>
__global__ void run_cells(const char* source, const CellSpec* cells,
                          CellSpec compact_cell, int cell_count, int assay,
                          int measured_bursts, int warmup_bursts,
                          int repetition, DirectRecord* records,
                          WarpState* states) {
    __shared__ __align__(16) char destinations[kSharedDestinationBytes];
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;

    for (int cell_index = 0; cell_index < cell_count; ++cell_index) {
        CellSpec cell = Compact ? compact_cell : cells[cell_index];
        const int selected_assay = assay;
        initialize_destinations(destinations);
        if constexpr (!Compact) prewarm_cell(source, cell);
        __syncthreads();

        if constexpr (Compact) {
            if (warp >= cell.active_warps) return;
        }

        unsigned long long cell_errors = 0;
        std::uint32_t cell_checksum = 0;
        const std::uint32_t cell_smid_before = read_smid();
        if (warp < cell.active_warps) {
            const char* warp_source = source + warp * kSourceBytesPerWarp;
            char* on_destination =
                destinations + warp * kWarpDestinationBytes +
                lane * kPayloadBytes;
            char* off_destination =
                destinations + kArmDestinationBytes +
                warp * kWarpDestinationBytes + lane * kPayloadBytes;
            const int total_bursts = warmup_bursts + measured_bursts;
#pragma unroll 1
            for (int iteration = 0; iteration < total_bursts; ++iteration) {
                const bool measured = iteration >= warmup_bursts;
                const int burst = measured ? iteration - warmup_bursts
                                           : iteration;
                const std::size_t i0 =
                    source_chunk_index(cell.route, burst, 0);
                const std::size_t i1 =
                    source_chunk_index(cell.route, burst, 1);
                const std::size_t i2 =
                    source_chunk_index(cell.route, burst, 2);
                const std::size_t i3 =
                    source_chunk_index(cell.route, burst, 3);
                const std::size_t i4 =
                    source_chunk_index(cell.route, burst, 4);
                const std::size_t i5 =
                    source_chunk_index(cell.route, burst, 5);
                const std::size_t i6 =
                    source_chunk_index(cell.route, burst, 6);
                const std::size_t i7 =
                    source_chunk_index(cell.route, burst, 7);
                const char* s0 = warp_source + i0 * kChunkBytes +
                                 lane * kPayloadBytes;
                const char* s1 = warp_source + i1 * kChunkBytes +
                                 lane * kPayloadBytes;
                const char* s2 = warp_source + i2 * kChunkBytes +
                                 lane * kPayloadBytes;
                const char* s3 = warp_source + i3 * kChunkBytes +
                                 lane * kPayloadBytes;
                const char* s4 = warp_source + i4 * kChunkBytes +
                                 lane * kPayloadBytes;
                const char* s5 = warp_source + i5 * kChunkBytes +
                                 lane * kPayloadBytes;
                const char* s6 = warp_source + i6 * kChunkBytes +
                                 lane * kPayloadBytes;
                const char* s7 = warp_source + i7 * kChunkBytes +
                                 lane * kPayloadBytes;
                const unsigned on_d0 = static_cast<unsigned>(
                    __cvta_generic_to_shared(on_destination));
                const unsigned on_d1 = on_d0 + kChunkBytes;
                const unsigned on_d2 = on_d1 + kChunkBytes;
                const unsigned on_d3 = on_d2 + kChunkBytes;
                const unsigned on_d4 = on_d3 + kChunkBytes;
                const unsigned on_d5 = on_d4 + kChunkBytes;
                const unsigned on_d6 = on_d5 + kChunkBytes;
                const unsigned on_d7 = on_d6 + kChunkBytes;
                const bool enabled_first =
                    (((burst / kArmOrderBlock) + repetition) & 1) == 0;
                ArmSample on{};
                ArmSample off{};
#pragma unroll 1
                for (int sequence = 0; sequence < 2; ++sequence) {
                    const bool enabled = (sequence == 0) == enabled_first;
                    ArmSample sample;
                    if constexpr (Compact || FixedAssay >= 0) {
                        sample = dispatch_compact_body<
                            FixedAssay, FixedDepth, FixedGap>(
                            enabled, cell.depth, on_d0, on_d1, on_d2, on_d3,
                            on_d4, on_d5, on_d6, on_d7, s0, s1, s2, s3, s4,
                            s5, s6, s7);
                    } else if (selected_assay == kLdgsts) {
                        sample = dispatch_ldgsts_body(
                            enabled, cell.depth, cell.gap_cycles, on_d0, on_d1,
                            on_d2, on_d3, on_d4, on_d5, on_d6, on_d7, s0, s1,
                            s2, s3, s4, s5, s6, s7);
                    } else {
                        sample = dispatch_ldg_body(
                            enabled, cell.depth, on_d0, on_d1, on_d2, on_d3,
                            on_d4, on_d5, on_d6, on_d7, s0, s1, s2, s3, s4,
                            s5, s6, s7);
                    }
                    retain_sample(sample, &cell_checksum);
                    if (enabled) {
                        on = sample;
                    } else {
                        off = sample;
                    }
                }

                std::uint32_t lane_errors = on.errors + off.errors;
                std::uint32_t lane_checksum = on.checksum ^ off.checksum;
                lane_errors += check_target_destinations(
                    source, on_destination, off_destination, s0, s1, s2, s3,
                    s4, s5, s6, s7, cell.depth, &lane_checksum);
                const std::uint32_t error_mask =
                    __ballot_sync(0xffffffffu, lane_errors != 0);
                std::uint32_t warp_checksum = lane_checksum;
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    warp_checksum ^=
                        __shfl_xor_sync(0xffffffffu, warp_checksum, offset);
                }
                if (lane == 0) {
                    cell_errors += __popc(error_mask);
                    cell_checksum ^=
                        warp_checksum + static_cast<std::uint32_t>(iteration);
                }
                if constexpr (!Compact) {
                    if (measured && lane == 0) {
                    const std::size_t record_index =
                        (static_cast<std::size_t>(cell_index) *
                             measured_bursts +
                         burst) *
                            kMaxWarps +
                        warp;
                    streaming_store_record(records + record_index, on, off,
                                           error_mask, enabled_first);
                    }
                }
            }
        }

        if (lane == 0 && warp < cell.active_warps) {
            WarpState state{};
            state.errors = cell_errors;
            state.smid_before = cell_smid_before;
            state.smid_after = read_smid();
            state.checksum = cell_checksum;
            states[cell_index * kMaxWarps + warp] = state;
        }
        if constexpr (!Compact) __syncthreads();
    }
}

const char* route_name(int route) {
    switch (route) {
        case kL1: return "l1";
        case kL2: return "l2";
        case kDram: return "dram";
        default: return nullptr;
    }
}

bool parse_unsigned(const char* text, int* value) {
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (text[0] == '\0' || end[0] != '\0' || parsed < 0 ||
        parsed > 0x7fffffffL) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

bool parse_cell(const std::string& key, int assay, HostCell* cell) {
    char assay_text[16]{};
    char route_text[16]{};
    int warps = 0;
    int depth = 0;
    int gap = 0;
    char trailing = '\0';
    if (std::sscanf(key.c_str(), "%15[^_]_w%d_%15[^_]_d%d_g%d%c",
                    assay_text, &warps, route_text, &depth, &gap,
                    &trailing) != 5) {
        return false;
    }
    if ((assay == kLdgsts && std::strcmp(assay_text, "ldgsts") != 0) ||
        (assay == kLdg && std::strcmp(assay_text, "ldg") != 0)) {
        return false;
    }
    int route = -1;
    if (std::strcmp(route_text, "l1") == 0) route = kL1;
    if (std::strcmp(route_text, "l2") == 0) route = kL2;
    if (std::strcmp(route_text, "dram") == 0) route = kDram;
    if (route < 0 || (warps != 1 && warps != 4) || depth < 1 || depth > 8 ||
        (gap != 0 && gap != 16 && gap != 64)) {
        return false;
    }
    if (assay == kLdgsts &&
        (gap == 0
             ? (depth != 1 && depth != 4 && depth != 5 && depth != 6 &&
                depth != 8)
             : (depth != 4 && depth != 5 && depth != 6))) {
        return false;
    }
    if (assay == kLdg &&
        (warps != 1 || route != kL1 || gap != 0 ||
         (depth != 4 && depth != 5 && depth != 6 && depth != 8))) {
        return false;
    }
    const char* canonical_assay = assay == kLdgsts ? "ldgsts" : "ldg";
    const char* canonical_route = route == kL1 ? "l1" :
                                  route == kL2 ? "l2" : "dram";
    char canonical[64]{};
    std::snprintf(canonical, sizeof(canonical), "%s_w%d_%s_d%d_g%d",
                  canonical_assay, warps, canonical_route, depth, gap);
    if (key != canonical) return false;
    const bool frozen =
        assay == kLdg
            ? (warps == 1 && route == kL1 && depth >= 4 && depth <= 8 &&
               (depth == 4 || depth == 5 || depth == 6 || depth == 8) &&
               gap == 0)
            : (warps == 4
                   ? (depth == 4 || depth == 5 || depth == 6) && gap == 0
                   : (gap == 0 &&
                      (depth == 1 || depth == 4 || depth == 5 || depth == 6 ||
                       depth == 8) &&
                      (route == kL1 || route == kL2 || route == kDram)) ||
                         (gap != 0 && (route == kL1 || route == kDram) &&
                          (depth == 4 || depth == 5 || depth == 6)));
    if (!frozen) return false;
    cell->key = key;
    cell->spec = {route, depth, warps, gap};
    return true;
}

std::vector<std::string> split_cells(const std::string& text) {
    std::vector<std::string> values;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t comma = text.find(',', begin);
        const std::size_t end =
            comma == std::string::npos ? text.size() : comma;
        if (end == begin) return {};
        values.push_back(text.substr(begin, end - begin));
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return values;
}

void usage(const char* program) {
    std::fprintf(
        stderr,
        "usage: %s --mode direct|ncu|correctness --assay ldgsts|ldg "
        "--cells KEY[,KEY...] [--bursts N] [--warmup N] "
        "[--repetition N] [--order forward|reverse]\n",
        program);
}

template <int FixedAssay, int FixedDepth, int FixedGap>
void launch_compact_kernel(const char* source, CellSpec compact_cell,
                           int bursts, int warmup, int repetition,
                           WarpState* states) {
    run_cells<true, FixedAssay, FixedDepth, FixedGap><<<1, kThreads>>>(
        source, nullptr, compact_cell, 1, FixedAssay, bursts, warmup,
        repetition, nullptr, states);
}

void launch_compact_kernel_for_cell(const char* source, CellSpec cell,
                                    int assay, int bursts, int warmup,
                                    int repetition, WarpState* states) {
    if (assay == kLdg) {
        switch (cell.depth) {
            case 4:
                launch_compact_kernel<kLdg, 4, 0>(source, cell, bursts, warmup,
                                                 repetition, states);
                return;
            case 5:
                launch_compact_kernel<kLdg, 5, 0>(source, cell, bursts, warmup,
                                                 repetition, states);
                return;
            case 6:
                launch_compact_kernel<kLdg, 6, 0>(source, cell, bursts, warmup,
                                                 repetition, states);
                return;
            case 8:
                launch_compact_kernel<kLdg, 8, 0>(source, cell, bursts, warmup,
                                                 repetition, states);
                return;
            default: break;
        }
    } else if (assay == kLdgsts) {
        switch (cell.gap_cycles) {
            case 0:
                switch (cell.depth) {
                    case 1:
                        launch_compact_kernel<kLdgsts, 1, 0>(
                            source, cell, bursts, warmup, repetition, states);
                        return;
                    case 4:
                        launch_compact_kernel<kLdgsts, 4, 0>(
                            source, cell, bursts, warmup, repetition, states);
                        return;
                    case 5:
                        launch_compact_kernel<kLdgsts, 5, 0>(
                            source, cell, bursts, warmup, repetition, states);
                        return;
                    case 6:
                        launch_compact_kernel<kLdgsts, 6, 0>(
                            source, cell, bursts, warmup, repetition, states);
                        return;
                    case 8:
                        launch_compact_kernel<kLdgsts, 8, 0>(
                            source, cell, bursts, warmup, repetition, states);
                        return;
                    default: break;
                }
                break;
            case 16:
                switch (cell.depth) {
                    case 4:
                        launch_compact_kernel<kLdgsts, 4, 16>(
                            source, cell, bursts, warmup, repetition, states);
                        return;
                    case 5:
                        launch_compact_kernel<kLdgsts, 5, 16>(
                            source, cell, bursts, warmup, repetition, states);
                        return;
                    case 6:
                        launch_compact_kernel<kLdgsts, 6, 16>(
                            source, cell, bursts, warmup, repetition, states);
                        return;
                    default: break;
                }
                break;
            case 64:
                switch (cell.depth) {
                    case 4:
                        launch_compact_kernel<kLdgsts, 4, 64>(
                            source, cell, bursts, warmup, repetition, states);
                        return;
                    case 5:
                        launch_compact_kernel<kLdgsts, 5, 64>(
                            source, cell, bursts, warmup, repetition, states);
                        return;
                    case 6:
                        launch_compact_kernel<kLdgsts, 6, 64>(
                            source, cell, bursts, warmup, repetition, states);
                        return;
                    default: break;
                }
                break;
            default: break;
        }
    }
    std::fprintf(stderr, "unsupported compact cell: assay=%d depth=%d gap=%d\n",
                 assay, cell.depth, cell.gap_cycles);
    std::exit(2);
}

std::uint16_t unpack_gap(const std::uint32_t* words, int edge) {
    const int word = edge / 2;
    const int shift = 16 * (edge & 1);
    return static_cast<std::uint16_t>((words[word] >> shift) & 0xffffu);
}

void print_gap_array(const std::uint32_t* words, int depth) {
    std::printf("[");
    for (int edge = 0; edge < depth - 1; ++edge) {
        std::printf("%s%u", edge == 0 ? "" : ",",
                    static_cast<unsigned>(unpack_gap(words, edge)));
    }
    std::printf("]");
}

int main(int argc, char** argv) {
    int mode = -1;
    int assay = -1;
    int bursts = kDefaultBursts;
    int warmup = kDefaultWarmup;
    int repetition = 0;
    bool reverse = false;
    std::string cells_text;
    for (int index = 1; index < argc; ++index) {
        if (index + 1 >= argc) {
            usage(argv[0]);
            return 2;
        }
        const char* option = argv[index];
        const char* value = argv[++index];
        if (std::strcmp(option, "--mode") == 0) {
            if (std::strcmp(value, "direct") == 0) mode = kDirect;
            if (std::strcmp(value, "ncu") == 0) mode = kNcu;
            if (std::strcmp(value, "correctness") == 0) mode = kCorrectness;
        } else if (std::strcmp(option, "--assay") == 0) {
            if (std::strcmp(value, "ldgsts") == 0) assay = kLdgsts;
            if (std::strcmp(value, "ldg") == 0) assay = kLdg;
        } else if (std::strcmp(option, "--cells") == 0) {
            cells_text = value;
        } else if (std::strcmp(option, "--bursts") == 0) {
            if (!parse_unsigned(value, &bursts)) return 2;
        } else if (std::strcmp(option, "--warmup") == 0) {
            if (!parse_unsigned(value, &warmup)) return 2;
        } else if (std::strcmp(option, "--repetition") == 0) {
            if (!parse_unsigned(value, &repetition)) return 2;
        } else if (std::strcmp(option, "--order") == 0) {
            if (std::strcmp(value, "forward") == 0) reverse = false;
            else if (std::strcmp(value, "reverse") == 0) reverse = true;
            else return 2;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (mode < 0 || assay < 0 || cells_text.empty() || bursts < 1 ||
        bursts > kMaxBursts || warmup < 0 || warmup > kMaxBursts ||
        repetition < 1 || repetition > kMaxRepetitions) {
        usage(argv[0]);
        return 2;
    }

    std::vector<HostCell> cells;
    for (const std::string& key : split_cells(cells_text)) {
        HostCell cell;
        if (!parse_cell(key, assay, &cell)) {
            std::fprintf(stderr, "invalid cell key: %s\n", key.c_str());
            return 2;
        }
        cells.push_back(cell);
    }
    std::set<std::string> unique_keys;
    for (const HostCell& cell : cells) {
        if (!unique_keys.insert(cell.key).second) {
            std::fprintf(stderr, "duplicate cell key: %s\n", cell.key.c_str());
            return 2;
        }
    }
    if (cells.empty() ||
        ((mode == kNcu || mode == kCorrectness) && cells.size() != 1)) {
        return 2;
    }
    if (mode == kDirect) {
        std::set<int> gaps;
        for (const HostCell& cell : cells) gaps.insert(cell.spec.gap_cycles);
        if (gaps.size() != 1) {
            std::fprintf(stderr,
                         "direct invocation must keep one gap schedule\n");
            return 2;
        }
    }
    if (reverse) std::reverse(cells.begin(), cells.end());

    std::vector<CellSpec> host_specs;
    host_specs.reserve(cells.size());
    for (const HostCell& cell : cells) host_specs.push_back(cell.spec);

    void* source_allocation = nullptr;
    CUDA_CHECK(cudaMalloc(&source_allocation, kSourceBytes + kChunkBytes));
    const std::uintptr_t aligned_source =
        (reinterpret_cast<std::uintptr_t>(source_allocation) +
         kChunkBytes - 1u) &
        ~(kChunkBytes - 1u);
    char* const source = reinterpret_cast<char*>(aligned_source);
    initialize_source<<<256, 256>>>(
        reinterpret_cast<std::uint32_t*>(source),
        kSourceBytes / sizeof(std::uint32_t));
    CUDA_CHECK(cudaGetLastError());

    CellSpec* device_cells = nullptr;
    WarpState* device_states = nullptr;
    DirectRecord* device_records = nullptr;
    std::uint32_t* prewarm_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&device_cells,
                          host_specs.size() * sizeof(CellSpec)));
    CUDA_CHECK(cudaMemcpy(device_cells, host_specs.data(),
                          host_specs.size() * sizeof(CellSpec),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&device_states,
                          host_specs.size() * kMaxWarps *
                              sizeof(WarpState)));
    CUDA_CHECK(cudaMemset(device_states, 0,
                          host_specs.size() * kMaxWarps *
                              sizeof(WarpState)));
    const std::size_t record_capacity =
        host_specs.size() * static_cast<std::size_t>(bursts) * kMaxWarps;
    if (mode == kDirect) {
        CUDA_CHECK(cudaMalloc(&device_records,
                              record_capacity * sizeof(DirectRecord)));
    }
    CUDA_CHECK(cudaMalloc(&prewarm_sink, sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemset(prewarm_sink, 0, sizeof(std::uint32_t)));
    CUDA_CHECK(cudaDeviceSynchronize());

    if (mode == kNcu) {
        if (warmup != 0) {
            launch_compact_kernel_for_cell(
                source, cells[0].spec, assay, warmup, 0, repetition,
                device_states);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemset(device_states, 0,
                                  kMaxWarps * sizeof(WarpState)));
        }
        if (cells[0].spec.route != kDram) {
            prewarm_source<<<256, 256>>>(source, cells[0].spec, prewarm_sink);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaProfilerStart());
        launch_compact_kernel_for_cell(
            source, cells[0].spec, assay, bursts, 0, repetition,
            device_states);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaProfilerStop());
    } else if (mode == kCorrectness) {
        launch_compact_kernel_for_cell(
            source, cells[0].spec, assay, bursts, warmup, repetition,
            device_states);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        const bool all_zero_gap = std::all_of(
            host_specs.begin(), host_specs.end(),
            [](const CellSpec& spec) { return spec.gap_cycles == 0; });
        if (all_zero_gap && assay == kLdgsts) {
            run_cells<false, kLdgsts, 0, 0><<<1, kThreads>>>(
                source, device_cells, CellSpec{},
                static_cast<int>(host_specs.size()), assay, bursts, warmup,
                repetition, device_records, device_states);
        } else {
            run_cells<false><<<1, kThreads>>>(
                source, device_cells, CellSpec{},
                static_cast<int>(host_specs.size()), assay, bursts, warmup,
                repetition, device_records, device_states);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<WarpState> states(host_specs.size() * kMaxWarps);
    CUDA_CHECK(cudaMemcpy(states.data(), device_states,
                          states.size() * sizeof(WarpState),
                          cudaMemcpyDeviceToHost));
    std::vector<DirectRecord> records;
    if (mode == kDirect) {
        records.resize(record_capacity);
        CUDA_CHECK(cudaMemcpy(records.data(), device_records,
                              records.size() * sizeof(DirectRecord),
                              cudaMemcpyDeviceToHost));
    }

    bool correct = true;
    std::size_t record_count = 0;
    if (mode == kDirect) {
        for (std::size_t cell_index = 0; cell_index < cells.size();
             ++cell_index) {
            const HostCell& cell = cells[cell_index];
            for (int burst = 0; burst < bursts; ++burst) {
                for (int warp = 0; warp < cell.spec.active_warps; ++warp) {
                    const std::size_t record_index =
                        (cell_index * bursts + burst) * kMaxWarps + warp;
                    const DirectRecord& record = records[record_index];
                    const std::uint32_t smid_on = record.words[12];
                    const std::uint32_t smid_off = record.words[13];
                    const bool record_correct = record.words[14] == 0;
                    correct = correct && record_correct;
                    std::printf(
                        "{\"type\":\"paired\",\"cell\":\"%s\","
                        "\"timing_schema\":%d,"
                        "\"burst\":%d,\"warp\":%d,\"arm_order\":\"%s\","
                        "\"issue_on\":%u,\"issue_off\":%u,"
                        "\"drain_on\":%u,\"drain_off\":%u,"
                        "\"gaps_on\":",
                        cell.key.c_str(), kTimingSchema, burst, warp,
                        record.words[15] == 0 ? "on_off" : "off_on",
                        record.words[0], record.words[2], record.words[1],
                        record.words[3]);
                    print_gap_array(record.words + 4, cell.spec.depth);
                    std::printf(",\"gaps_off\":");
                    print_gap_array(record.words + 8, cell.spec.depth);
                    std::printf(
                        ",\"smid_on_before\":%u,\"smid_on_after\":%u,"
                        "\"smid_off_before\":%u,\"smid_off_after\":%u,"
                        "\"error_mask\":%u,\"correct\":%s}\n",
                        smid_on & 0xffffu, smid_on >> 16,
                        smid_off & 0xffffu, smid_off >> 16, record.words[14],
                        record_correct ? "true" : "false");
                    ++record_count;
                }
            }
        }
    }
    for (std::size_t cell_index = 0; cell_index < cells.size(); ++cell_index) {
        for (int warp = 0; warp < cells[cell_index].spec.active_warps; ++warp) {
            correct = correct &&
                      states[cell_index * kMaxWarps + warp].errors == 0;
        }
    }

    const char* mode_name =
        mode == kDirect ? "direct" : mode == kNcu ? "ncu" : "correctness";
    const char* assay_name = assay == kLdgsts ? "ldgsts" : "ldg";
    std::printf(
        "{\"type\":\"summary\",\"timing_schema\":%d,\"mode\":\"%s\",\"assay\":\"%s\","
        "\"cell_keys\":[",
        kTimingSchema, mode_name, assay_name);
    for (std::size_t index = 0; index < cells.size(); ++index) {
        std::printf("%s\"%s\"", index == 0 ? "" : ",",
                    cells[index].key.c_str());
    }
    std::printf(
        "],\"bursts\":%d,\"warmup\":%d,\"repetition\":%d,"
        "\"order\":\"%s\",\"record_count\":%zu,\"correct\":%s,"
        "\"cell_states\":[",
        bursts, warmup, repetition, reverse ? "reverse" : "forward",
        record_count, correct ? "true" : "false");
    bool first_state = true;
    for (std::size_t cell_index = 0; cell_index < cells.size(); ++cell_index) {
        for (int warp = 0; warp < cells[cell_index].spec.active_warps; ++warp) {
            const WarpState& state =
                states[cell_index * kMaxWarps + warp];
            std::printf(
                "%s{\"cell\":\"%s\",\"warp\":%d,\"smid_before\":%u,"
                "\"smid_after\":%u,\"errors\":%llu,\"checksum\":%u}",
                first_state ? "" : ",", cells[cell_index].key.c_str(), warp,
                state.smid_before, state.smid_after,
                static_cast<unsigned long long>(state.errors), state.checksum);
            first_state = false;
        }
    }
    std::printf("]}\n");

    CUDA_CHECK(cudaFree(prewarm_sink));
    if (device_records != nullptr) CUDA_CHECK(cudaFree(device_records));
    CUDA_CHECK(cudaFree(device_states));
    CUDA_CHECK(cudaFree(device_cells));
    CUDA_CHECK(cudaFree(source_allocation));
    return correct ? 0 : 1;
}
