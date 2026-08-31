#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t status = (call);                                      \
        if (status != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s failed: %s\n", #call,                   \
                         cudaGetErrorString(status));                           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

constexpr int kWarpSize = 32;
constexpr int kCopies = 8;
constexpr int kSourceCohorts = 8;
constexpr int kLanesPerCohort = 4;
constexpr int kWarmupGroups = 64;
constexpr int kDefaultGroups = 8192;

constexpr std::size_t kCohortBytes = 128;
constexpr std::size_t kSourceLaneStride = 32;
constexpr std::size_t kHotPositionStride = 1024;
constexpr std::size_t kHotSourceBytes = kCopies * kHotPositionStride;
constexpr std::size_t kColdStepStride = 1024;
constexpr std::size_t kColdRingBytes = 128ull << 20;
constexpr std::size_t kColdRingSteps = kColdRingBytes / kColdStepStride;
constexpr int kMaxGroups =
    static_cast<int>(kColdRingSteps / kCopies) - kWarmupGroups;
constexpr std::size_t kEvictionBytes = 128ull << 20;
constexpr std::size_t kSourcePoolAlignment = 4096;

constexpr std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

constexpr std::size_t kHotSourceOffset = 0;
constexpr std::size_t kColdSourceAOffset =
    align_up(kHotSourceOffset + kHotSourceBytes, kSourcePoolAlignment);
constexpr std::size_t kColdSourceBOffset =
    align_up(kColdSourceAOffset + kColdRingBytes, kSourcePoolAlignment);
constexpr std::size_t kSourcePoolBytes = kColdSourceBOffset + kColdRingBytes;

constexpr std::size_t kDestinationPositionStride = 4096;
constexpr std::size_t kDestinationCohortStride = 512;
constexpr std::size_t kSevereDestinationBase = 2048;
constexpr std::size_t kDestinationBytes =
    kCopies * kDestinationPositionStride;

static_assert(kDefaultGroups <= kMaxGroups);
static_assert((kWarmupGroups + kMaxGroups) * kCopies * kColdStepStride <=
              kColdRingBytes);
static_assert(kColdSourceAOffset % kSourcePoolAlignment == 0);
static_assert(kColdSourceBOffset % kSourcePoolAlignment == 0);
static_assert(kColdSourceAOffset >= kHotSourceOffset + kHotSourceBytes);
static_assert(kColdSourceBOffset >= kColdSourceAOffset + kColdRingBytes);
static_assert(4 * kDestinationCohortStride == kSevereDestinationBase);
static_assert(kSevereDestinationBase +
                      3 * kDestinationCohortStride +
                      3 * kCohortBytes <
                  kDestinationPositionStride);

struct PairingResult {
    unsigned long long issue_cycles;
    unsigned long long wait_cycles;
    unsigned long long total_cycles;
    std::uint32_t token;
    std::uint32_t mismatches;
};

enum class Variant : int {
    kSlowExpensive = 0,
    kSlowCheap = 1,
};

enum class RunMode : int {
    kPairAb = 0,
    kPairBa = 1,
    kSlowExpensive = 2,
    kSlowCheap = 3,
};

__device__ __forceinline__ unsigned long long read_clock64() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) : : "memory");
    return value;
}

__device__ __forceinline__ void copy_cg(unsigned destination,
                                        const char* source) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(destination), "l"(source)
                 : "memory");
}

#define SPACING_SAMPLE_1                                                     \
    "mov.u32 sample, %clock;\n\t"                                            \
    "xor.b32 %0, %0, sample;\n\t"
#define SPACING_SAMPLE_2 SPACING_SAMPLE_1                                    \
    "mov.u32 sample, %clock;\n\t"                                            \
    "xor.b32 %1, %1, sample;\n\t"
#define SPACING_SAMPLE_4 SPACING_SAMPLE_2                                    \
    "mov.u32 sample, %clock;\n\t"                                            \
    "xor.b32 %2, %2, sample;\n\t"                                            \
    "mov.u32 sample, %clock;\n\t"                                            \
    "xor.b32 %3, %3, sample;\n\t"

template <int Samples>
__device__ __forceinline__ void issue_spacing(uint4* state) {
    static_assert(Samples == 0 || Samples == 4);
    if constexpr (Samples == 4) {
        asm volatile("{ .reg .u32 sample;\n\t" SPACING_SAMPLE_4 "}"
                     : "+r"(state->x), "+r"(state->y), "+r"(state->z),
                       "+r"(state->w)
                     :
                     : "memory");
    }
}

__device__ __forceinline__ const char* source_for_lane(
    const char* hot_source, const char* cold_source, unsigned group,
    unsigned position, int lane) {
    const unsigned cohort = static_cast<unsigned>(lane / kLanesPerCohort);
    const unsigned lane_in_cohort =
        static_cast<unsigned>(lane % kLanesPerCohort);
    const std::size_t lane_offset = lane_in_cohort * kSourceLaneStride;
    if (cohort < 4) {
        return hot_source + position * kHotPositionStride +
               cohort * kCohortBytes + lane_offset;
    }
    const std::size_t step =
        static_cast<std::size_t>(group) * kCopies + position;
    return cold_source + step * kColdStepStride +
           (cohort - 4) * kCohortBytes + lane_offset;
}

__device__ __forceinline__ unsigned destination_for_lane(bool slow_cheap,
                                                          int lane) {
    const unsigned source_cohort =
        static_cast<unsigned>(lane / kLanesPerCohort);
    const unsigned lane_in_cohort =
        static_cast<unsigned>(lane % kLanesPerCohort);

    // XOR only changes source-to-destination pairing.  Both variants use the
    // same four compact slots and the same four severe slots exactly once.
    const unsigned destination_cohort =
        source_cohort ^ (slow_cheap ? 4u : 0u);
    const unsigned slot = destination_cohort & 3u;
    const bool severe = destination_cohort >= 4;
    return static_cast<unsigned>(
        (severe ? kSevereDestinationBase : 0) +
        slot * kDestinationCohortStride +
        lane_in_cohort * (severe ? kCohortBytes : kSourceLaneStride));
}

template <int Depth, int SpacingSamples>
__global__ void cohort_pairing_kernel(const char* hot_source,
                                      const char* cold_source,
                                      unsigned group_base, int groups,
                                      bool slow_cheap,
                                      PairingResult* result) {
    __shared__ __align__(16) char destination[kDestinationBytes];
    if (threadIdx.x >= kWarpSize) return;

    const int lane = static_cast<int>(threadIdx.x);
    const unsigned destination_base = static_cast<unsigned>(
        __cvta_generic_to_shared(destination) +
        destination_for_lane(slow_cheap, lane));
    constexpr unsigned destination_stride =
        static_cast<unsigned>(kDestinationPositionStride);
    unsigned long long issue_cycles = 0;
    unsigned long long wait_cycles = 0;
    unsigned long long total_cycles = 0;
    std::uint32_t token = 0x754a2d19u ^ static_cast<std::uint32_t>(lane);
    std::uint32_t mismatches = 0;
    uint4 spacing_state = {
        static_cast<unsigned>(lane + 1),
        static_cast<unsigned>(lane + 3),
        static_cast<unsigned>(lane + 5),
        static_cast<unsigned>(lane + 7),
    };

#pragma unroll 1
    for (int group_index = 0; group_index < groups; ++group_index) {
        const unsigned group = group_base + static_cast<unsigned>(group_index);
        const char* s0 = source_for_lane(hot_source, cold_source, group, 0, lane);
        const char* s1 = source_for_lane(hot_source, cold_source, group, 1, lane);
        const char* s2 = source_for_lane(hot_source, cold_source, group, 2, lane);
        const char* s3 = source_for_lane(hot_source, cold_source, group, 3, lane);
        const char* s4 = source_for_lane(hot_source, cold_source, group, 4, lane);
        const char* s5 = source_for_lane(hot_source, cold_source, group, 5, lane);
        const char* s6 = source_for_lane(hot_source, cold_source, group, 6, lane);
        const char* s7 = source_for_lane(hot_source, cold_source, group, 7, lane);

        const unsigned long long before_issue = read_clock64();
        if constexpr (Depth > 0) copy_cg(destination_base, s0);
        if constexpr (Depth > 1) {
            issue_spacing<SpacingSamples>(&spacing_state);
            copy_cg(destination_base + destination_stride, s1);
        }
        if constexpr (Depth > 2) {
            issue_spacing<SpacingSamples>(&spacing_state);
            copy_cg(destination_base + 2u * destination_stride, s2);
        }
        if constexpr (Depth > 3) {
            issue_spacing<SpacingSamples>(&spacing_state);
            copy_cg(destination_base + 3u * destination_stride, s3);
        }
        if constexpr (Depth > 4) {
            issue_spacing<SpacingSamples>(&spacing_state);
            copy_cg(destination_base + 4u * destination_stride, s4);
        }
        if constexpr (Depth > 5) {
            issue_spacing<SpacingSamples>(&spacing_state);
            copy_cg(destination_base + 5u * destination_stride, s5);
        }
        if constexpr (Depth > 6) {
            issue_spacing<SpacingSamples>(&spacing_state);
            copy_cg(destination_base + 6u * destination_stride, s6);
        }
        if constexpr (Depth > 7) {
            issue_spacing<SpacingSamples>(&spacing_state);
            copy_cg(destination_base + 7u * destination_stride, s7);
        }
        const unsigned long long after_issue = read_clock64();
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        const unsigned long long after_wait = read_clock64();

        issue_cycles += after_issue - before_issue;
        wait_cycles += after_wait - after_issue;
        total_cycles += after_wait - before_issue;

        const volatile uint4* value = reinterpret_cast<const volatile uint4*>(
            destination + (Depth - 1) * kDestinationPositionStride +
            destination_for_lane(slow_cheap, lane));
        const uint4 observed = {value->x, value->y, value->z, value->w};
        const std::uint32_t expected = lane < 16 ? 0x31313131u : 0x57575757u;
        const bool mismatch = observed.x != expected || observed.y != expected ||
                              observed.z != expected || observed.w != expected;
        const unsigned mismatch_mask = __ballot_sync(0xffffffffu, mismatch);
        if (lane == 0) mismatches += static_cast<std::uint32_t>(__popc(mismatch_mask));
        token ^= observed.x + (observed.y << 3) + (observed.z << 7) +
                 (observed.w << 11) + group;
    }
    token ^= spacing_state.x + (spacing_state.y << 3) +
             (spacing_state.z << 7) + (spacing_state.w << 11);

    if (lane == 0) {
        result->issue_cycles = issue_cycles;
        result->wait_cycles = wait_cycles;
        result->total_cycles = total_cycles;
        result->token = token;
        result->mismatches = mismatches;
    }
}

template <int Depth, int SpacingSamples>
void launch_kernel(const char* hot_source, const char* cold_source,
                   unsigned group_base, int groups, Variant variant,
                   PairingResult* result) {
    cohort_pairing_kernel<Depth, SpacingSamples><<<1, kWarpSize>>>(
        hot_source, cold_source, group_base, groups,
        variant == Variant::kSlowCheap, result);
}

template <int SpacingSamples>
void launch_depth(int depth, const char* hot_source, const char* cold_source,
                  unsigned group_base, int groups, Variant variant,
                  PairingResult* result) {
    switch (depth) {
        case 1:
            launch_kernel<1, SpacingSamples>(hot_source, cold_source, group_base,
                                             groups, variant, result);
            return;
        case 2:
            launch_kernel<2, SpacingSamples>(hot_source, cold_source, group_base,
                                             groups, variant, result);
            return;
        case 4:
            launch_kernel<4, SpacingSamples>(hot_source, cold_source, group_base,
                                             groups, variant, result);
            return;
        case 8:
            launch_kernel<8, SpacingSamples>(hot_source, cold_source, group_base,
                                             groups, variant, result);
            return;
        default:
            std::fprintf(stderr, "unsupported depth: %d\n", depth);
            std::exit(2);
    }
}

void launch(int depth, int spacing_samples, const char* hot_source,
            const char* cold_source, unsigned group_base, int groups,
            Variant variant, PairingResult* result) {
    if (spacing_samples == 0) {
        launch_depth<0>(depth, hot_source, cold_source, group_base, groups,
                        variant, result);
        return;
    }
    if (spacing_samples == 4) {
        launch_depth<4>(depth, hot_source, cold_source, group_base, groups,
                        variant, result);
        return;
    }
    std::fprintf(stderr, "unsupported spacing: %d\n", spacing_samples);
    std::exit(2);
}

int parse_int(const char* text) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (!text[0] || end[0] || value < 0 || value > kMaxGroups) return -1;
    return static_cast<int>(value);
}

const char* variant_name(Variant variant) {
    return variant == Variant::kSlowExpensive ? "slow_expensive" :
                                                "slow_cheap";
}

void print_result(const PairingResult& result, int groups) {
    std::printf(
        "{\"issue_cycles\":%llu,\"wait_cycles\":%llu,"
        "\"total_cycles\":%llu,\"issue_cycles_per_group\":%.9f,"
        "\"wait_cycles_per_group\":%.9f,\"total_cycles_per_group\":%.9f,"
        "\"sum_consistent\":%s,\"token\":%u,\"mismatches\":%u,"
        "\"correct\":%s}",
        result.issue_cycles, result.wait_cycles, result.total_cycles,
        static_cast<double>(result.issue_cycles) / groups,
        static_cast<double>(result.wait_cycles) / groups,
        static_cast<double>(result.total_cycles) / groups,
        result.issue_cycles + result.wait_cycles == result.total_cycles ?
            "true" : "false",
        result.token, result.mismatches,
        result.issue_cycles + result.wait_cycles == result.total_cycles &&
                result.mismatches == 0
            ? "true" : "false");
}

int main(int argc, char** argv) {
    int groups = kDefaultGroups;
    int depth = 2;
    int spacing_samples = 0;
    bool profiler_marked = false;
    RunMode mode = RunMode::kPairAb;

    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) return 2;
        if (std::strcmp(argv[i], "--groups") == 0) {
            groups = parse_int(argv[++i]);
        } else if (std::strcmp(argv[i], "--depth") == 0) {
            depth = parse_int(argv[++i]);
        } else if (std::strcmp(argv[i], "--spacing-samples") == 0) {
            spacing_samples = parse_int(argv[++i]);
        } else if (std::strcmp(argv[i], "--markers") == 0) {
            const char* value = argv[++i];
            if (std::strcmp(value, "on") == 0) profiler_marked = true;
            else if (std::strcmp(value, "off") == 0) profiler_marked = false;
            else return 2;
        } else if (std::strcmp(argv[i], "--order") == 0) {
            const char* value = argv[++i];
            if (std::strcmp(value, "ab") == 0) mode = RunMode::kPairAb;
            else if (std::strcmp(value, "ba") == 0) mode = RunMode::kPairBa;
            else return 2;
        } else if (std::strcmp(argv[i], "--variant") == 0) {
            const char* value = argv[++i];
            if (std::strcmp(value, "slow_expensive") == 0) {
                mode = RunMode::kSlowExpensive;
            } else if (std::strcmp(value, "slow_cheap") == 0) {
                mode = RunMode::kSlowCheap;
            } else {
                return 2;
            }
        } else {
            return 2;
        }
    }
    if (groups < 1 ||
        (depth != 1 && depth != 2 && depth != 4 && depth != 8) ||
        (spacing_samples != 0 && spacing_samples != 4)) {
        return 2;
    }

    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    char* source_pool = nullptr;
    char* eviction = nullptr;
    PairingResult* device_result = nullptr;
    CUDA_CHECK(cudaMalloc(&source_pool, kSourcePoolBytes));
    CUDA_CHECK(cudaMalloc(&eviction, kEvictionBytes));
    CUDA_CHECK(cudaMalloc(&device_result, sizeof(PairingResult)));
    char* const hot_source = source_pool + kHotSourceOffset;
    char* const cold_source_a = source_pool + kColdSourceAOffset;
    char* const cold_source_b = source_pool + kColdSourceBOffset;
    CUDA_CHECK(cudaMemset(hot_source, 0x31, kHotSourceBytes));
    CUDA_CHECK(cudaMemset(cold_source_a, 0x57, kColdRingBytes));
    CUDA_CHECK(cudaMemset(cold_source_b, 0x57, kColdRingBytes));
    CUDA_CHECK(cudaMemset(device_result, 0, sizeof(PairingResult)));
    CUDA_CHECK(cudaDeviceSynchronize());

    PairingResult expensive_result{};
    PairingResult cheap_result{};
    int arm_index = 0;
    const auto run_arm = [&](Variant variant, PairingResult* host_result) {
        const char* cold_source =
            arm_index == 0 ? cold_source_a : cold_source_b;
        CUDA_CHECK(cudaMemset(eviction, 0x41 + arm_index, kEvictionBytes));
        CUDA_CHECK(cudaMemset(device_result, 0, sizeof(PairingResult)));
        CUDA_CHECK(cudaDeviceSynchronize());

        launch(depth, spacing_samples, hot_source, cold_source, 0,
               kWarmupGroups, variant, device_result);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (profiler_marked) CUDA_CHECK(cudaProfilerStart());
        launch(depth, spacing_samples, hot_source, cold_source, kWarmupGroups,
               groups, variant, device_result);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        if (profiler_marked) CUDA_CHECK(cudaProfilerStop());
        CUDA_CHECK(cudaMemcpy(host_result, device_result, sizeof(PairingResult),
                              cudaMemcpyDeviceToHost));
        ++arm_index;
    };

    if (mode == RunMode::kPairAb) {
        run_arm(Variant::kSlowExpensive, &expensive_result);
        run_arm(Variant::kSlowCheap, &cheap_result);
    } else if (mode == RunMode::kPairBa) {
        run_arm(Variant::kSlowCheap, &cheap_result);
        run_arm(Variant::kSlowExpensive, &expensive_result);
    } else if (mode == RunMode::kSlowExpensive) {
        run_arm(Variant::kSlowExpensive, &expensive_result);
    } else {
        run_arm(Variant::kSlowCheap, &cheap_result);
    }

    const bool pair = mode == RunMode::kPairAb || mode == RunMode::kPairBa;
    const char* order = mode == RunMode::kPairBa ? "ba" : "ab";
    std::printf(
        "{\"workload\":\"async_cohort_pairing\",\"route\":\"l2\","
        "\"mode\":\"%s\",\"order\":\"%s\",\"depth\":%d,"
        "\"groups\":%d,\"warmup_groups\":%d,"
        "\"spacing_clock_samples_per_edge\":%d,"
        "\"source_cohorts_per_copy\":%d,"
        "\"hot_source_cohorts_per_copy\":4,"
        "\"cold_source_cohorts_per_copy\":4,"
        "\"source_sectors_per_cohort\":4,"
        "\"payload_bytes_per_cohort\":64,"
        "\"compact_destination_cohorts_per_copy\":4,"
        "\"severe_destination_cohorts_per_copy\":4,"
        "\"cold_ring_bytes_per_arm\":%zu,"
        "\"cold_step_stride_bytes\":%zu,"
        "\"max_admitted_groups\":%d,"
        "\"cold_ring_no_wrap\":true,"
        "\"source_pool_bytes\":%zu,"
        "\"source_pool_alignment_bytes\":%zu,"
        "\"hot_source_offset_bytes\":%zu,"
        "\"cold_source_a_offset_bytes\":%zu,"
        "\"cold_source_b_offset_bytes\":%zu,"
        "\"source_pool_single_allocation\":true,"
        "\"independent_cold_regions\":true,"
        "\"source_vector_preserved\":true,"
        "\"destination_multiset_preserved\":true,"
        "\"static_ldgsts_positions_preserved\":true,"
        "\"result_store_after_intervals\":true,"
        "\"device\":\"%s\",\"compute_capability\":\"%d.%d\","
        "\"l2_cache_bytes\":%d,",
        pair ? "pair" : "endpoint", order, depth, groups, kWarmupGroups,
        spacing_samples, kSourceCohorts, kColdRingBytes, kColdStepStride,
        kMaxGroups, kSourcePoolBytes, kSourcePoolAlignment, kHotSourceOffset,
        kColdSourceAOffset, kColdSourceBOffset, properties.name,
        properties.major, properties.minor, properties.l2CacheSize);
    if (pair) {
        std::printf("\"slow_expensive\":");
        print_result(expensive_result, groups);
        std::printf(",\"slow_cheap\":");
        print_result(cheap_result, groups);
        std::printf(
            ",\"paired_wait_delta_cycles_per_group\":%.9f,"
            "\"paired_issue_delta_cycles_per_group\":%.9f,"
            "\"correct\":%s}\n",
            (static_cast<double>(expensive_result.wait_cycles) -
             static_cast<double>(cheap_result.wait_cycles)) /
                groups,
            (static_cast<double>(expensive_result.issue_cycles) -
             static_cast<double>(cheap_result.issue_cycles)) /
                groups,
            expensive_result.issue_cycles + expensive_result.wait_cycles ==
                        expensive_result.total_cycles &&
                    cheap_result.issue_cycles + cheap_result.wait_cycles ==
                        cheap_result.total_cycles &&
                    expensive_result.mismatches == 0 &&
                    cheap_result.mismatches == 0
                ? "true" : "false");
    } else {
        const Variant variant = mode == RunMode::kSlowExpensive ?
                                    Variant::kSlowExpensive :
                                    Variant::kSlowCheap;
        const PairingResult& result = variant == Variant::kSlowExpensive ?
                                          expensive_result : cheap_result;
        std::printf("\"variant\":\"%s\",\"result\":",
                    variant_name(variant));
        print_result(result, groups);
        std::printf(",\"correct\":%s}\n",
                    result.issue_cycles + result.wait_cycles ==
                                result.total_cycles &&
                            result.mismatches == 0
                        ? "true" : "false");
    }

    CUDA_CHECK(cudaFree(device_result));
    CUDA_CHECK(cudaFree(eviction));
    CUDA_CHECK(cudaFree(source_pool));
    return 0;
}
