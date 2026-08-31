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
constexpr int kSourceCohorts = 8;
constexpr int kLanesPerCohort = 4;
constexpr int kWarmupGroups = 64;
constexpr int kDefaultGroups = 8192;
constexpr int kLeadClockSamples = 16;

constexpr std::size_t kSourceStepBytes = 2048;
constexpr std::size_t kSourceSlotBytes = 256;
constexpr std::size_t kSourceLineBytes = 128;
constexpr std::size_t kSourceRingBytes = 32ull << 20;
constexpr std::size_t kSourceRingSteps =
    kSourceRingBytes / kSourceStepBytes;
constexpr std::size_t kSourceSlotRows =
    kSourceRingBytes / kSourceSlotBytes;
constexpr int kMaxGroups =
    static_cast<int>(kSourceRingSteps) - kWarmupGroups;
constexpr std::size_t kSourcePoolAlignment = 128;
constexpr std::size_t kSourceArmAOffset = 0;
constexpr std::size_t kSourceArmBOffset = kSourceRingBytes;
constexpr std::size_t kSourcePoolBytes = 2 * kSourceRingBytes;
constexpr std::size_t kEvictionBytes = 128ull << 20;

constexpr std::size_t kDestinationCohortStride = 512;
constexpr std::size_t kDestinationBytes =
    kSourceCohorts * kDestinationCohortStride;

constexpr int kScalarLdgInstructionsPerCopy = 1;
constexpr int kScalarLdgSectorsPerCopy = 24;
constexpr int kScalarPreloadLinesPerCopy = 8;
constexpr int kLdgstsInstructionsPerCopy = 1;
constexpr int kLdgstsSectorsPerCopy = 32;
constexpr int kUniqueLdgstsLinesPerCopy = 12;
constexpr int kPayloadWavefrontsPerCopy = 32;
constexpr int kProjectedBypassWavefrontsPerCopy = 32;
constexpr int kDestinationLinesPerCopy = 20;
constexpr int kMaxCohortBankWavefronts = 3;
constexpr int kMaxCohortBypassWavefronts = 4;
constexpr int kCohortTopologyPenaltyCyclesPerCopy = 120;
constexpr int kResponseMembershipsPerCopy = 32;
constexpr int kLocalDeviceLtsLookupsPerCopy = 56;

static_assert(kSourceStepBytes == 8 * kSourceSlotBytes);
static_assert(kSourceRingBytes % kSourceStepBytes == 0);
static_assert(kSourceRingBytes % kSourceSlotBytes == 0);
static_assert(kDefaultGroups <= kMaxGroups);
static_assert((kWarmupGroups + kMaxGroups) * kSourceStepBytes <=
              kSourceRingBytes);
static_assert(kSourceArmAOffset % kSourcePoolAlignment == 0);
static_assert(kSourceArmBOffset % kSourcePoolAlignment == 0);
static_assert(kSourceArmBOffset >= kSourceArmAOffset + kSourceRingBytes);
static_assert(kSourcePoolBytes >= kSourceArmBOffset + kSourceRingBytes);
static_assert(kDestinationBytes == 4096);

struct OwnershipResult {
    unsigned long long issue_cycles;
    unsigned long long wait_cycles;
    unsigned long long total_cycles;
    std::uint32_t token;
    std::uint32_t mismatches;
};

enum class Orientation : int {
    kFirstLineEarly = 0,
    kSecondLineEarly = 1,
};

enum class Variant : int {
    kEarlyFanout = 0,
    kLateFanout = 1,
};

enum class RunMode : int {
    kPairAb = 0,
    kPairBa = 1,
    kEarlyFanout = 2,
    kLateFanout = 3,
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

#define LEAD_SAMPLE_4                                                       \
    "mov.u32 sample, %clock;\n\t"                                           \
    "xor.b32 %0, %0, sample;\n\t"                                          \
    "mov.u32 sample, %clock;\n\t"                                           \
    "xor.b32 %1, %1, sample;\n\t"                                          \
    "mov.u32 sample, %clock;\n\t"                                           \
    "xor.b32 %2, %2, sample;\n\t"                                          \
    "mov.u32 sample, %clock;\n\t"                                           \
    "xor.b32 %3, %3, sample;\n\t"
#define LEAD_SAMPLE_8 LEAD_SAMPLE_4 LEAD_SAMPLE_4
#define LEAD_SAMPLE_16 LEAD_SAMPLE_8 LEAD_SAMPLE_8

__device__ __forceinline__ void retained_lead_16(uint4* state) {
    asm volatile("{ .reg .u32 sample;\n\t" LEAD_SAMPLE_16 "}"
                 : "+r"(state->x), "+r"(state->y), "+r"(state->z),
                   "+r"(state->w)
                 :
                 : "memory");
}

__device__ __forceinline__ std::size_t source_cohort_base(unsigned cohort) {
    return static_cast<std::size_t>(cohort) * kSourceSlotBytes +
           static_cast<std::size_t>(cohort >= 4) * 64;
}

__device__ __forceinline__ const char* source_for_lane(
    const char* source_arm, unsigned group, int lane) {
    const unsigned cohort = static_cast<unsigned>(lane / kLanesPerCohort);
    const unsigned lane_in_cohort =
        static_cast<unsigned>(lane % kLanesPerCohort);
    return source_arm + static_cast<std::size_t>(group) * kSourceStepBytes +
           source_cohort_base(cohort) +
           static_cast<std::size_t>(lane_in_cohort) * 32;
}

template <bool FirstLineEarly>
__device__ __forceinline__ bool preload_lane_active(int lane) {
    if (lane < 16) return true;
    const unsigned lane_in_cohort = static_cast<unsigned>(lane & 3);
    return FirstLineEarly ? lane_in_cohort < 2 : lane_in_cohort >= 2;
}

__device__ __forceinline__ std::uint32_t preload_pending_source(
    const char* source, bool active) {
    std::uint32_t value = 0;
    const unsigned predicate = active ? 1u : 0u;
    asm volatile("{ .reg .pred preload_active;\n\t"
                 "setp.ne.u32 preload_active, %1, 0;\n\t"
                 "@preload_active ld.global.cg.u32 %0, [%2];\n\t"
                 "}"
                 : "+r"(value)
                 : "r"(predicate), "l"(source)
                 : "memory");
    return value;
}

__device__ __forceinline__ std::uint32_t postwait_preload_fold(
    std::uint32_t preload) {
    std::uint32_t value;
    asm volatile("xor.b32 %0, %1, 0x9e3779b9;"
                 : "=r"(value)
                 : "r"(preload)
                 : "memory");
    return value;
}

__device__ __forceinline__ std::uint32_t expected_source_word(int lane) {
    const unsigned cohort = static_cast<unsigned>(lane / 4);
    const unsigned lane_in_cohort = static_cast<unsigned>(lane & 3);
    if (cohort < 4 || lane_in_cohort < 2) return 0x31313131u;
    return 0x57575757u;
}

template <bool FirstLineEarly>
__device__ __forceinline__ unsigned destination_for_lane(
    bool late_fanout, int lane) {
    const unsigned cohort = static_cast<unsigned>(lane / 4);
    const unsigned lane_in_cohort = static_cast<unsigned>(lane & 3);
    const unsigned slot = cohort * kDestinationCohortStride;
    if (cohort < 4) {
        const unsigned offset = lane_in_cohort < 2 ? lane_in_cohort * 32 :
                                                    128 + (lane_in_cohort - 2) * 32;
        return slot + offset;
    }

    const bool early_line =
        FirstLineEarly ? lane_in_cohort < 2 : lane_in_cohort >= 2;
    const bool fanout = late_fanout ? !early_line : early_line;
    const unsigned pair_lane = lane_in_cohort & 1u;
    return slot + (fanout ? pair_lane * 128 : 256 + pair_lane * 32);
}

template <bool FirstLineEarly>
__global__ void source_line_ownership_kernel(const char* source_arm,
                                             unsigned group_base, int groups,
                                             bool late_fanout,
                                             OwnershipResult* result) {
    __shared__ __align__(16) char destination[kDestinationBytes];
    if (threadIdx.x >= kWarpSize) return;

    const int lane = static_cast<int>(threadIdx.x);
    const unsigned destination_address = static_cast<unsigned>(
        __cvta_generic_to_shared(destination) +
        destination_for_lane<FirstLineEarly>(late_fanout, lane));
    unsigned long long issue_cycles = 0;
    unsigned long long wait_cycles = 0;
    unsigned long long total_cycles = 0;
    std::uint32_t token = 0x51a7c3d9u ^ static_cast<std::uint32_t>(lane);
    std::uint32_t mismatches = 0;
    uint4 lead_state = {
        static_cast<unsigned>(lane + 1),
        static_cast<unsigned>(lane + 3),
        static_cast<unsigned>(lane + 5),
        static_cast<unsigned>(lane + 7),
    };

#pragma unroll 1
    for (int group_index = 0; group_index < groups; ++group_index) {
        const unsigned group = group_base + static_cast<unsigned>(group_index);
        const char* source = source_for_lane(source_arm, group, lane);
        const bool preload_active = preload_lane_active<FirstLineEarly>(lane);
        const std::uint32_t preload =
            preload_pending_source(source, preload_active);

        retained_lead_16(&lead_state);
        const unsigned long long before_issue = read_clock64();
        copy_cg(destination_address, source);
        const unsigned long long after_issue = read_clock64();
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        const unsigned long long after_wait = read_clock64();

        issue_cycles += after_issue - before_issue;
        wait_cycles += after_wait - after_issue;
        total_cycles += after_wait - before_issue;

        // This is intentionally the first consumer of the scalar-load result.
        // Static evidence must place its SB wait after the post-wait clock.
        const std::uint32_t preload_fold = postwait_preload_fold(preload);
        const std::uint32_t expected_preload =
            preload_active ? expected_source_word(lane) : 0u;
        const bool preload_mismatch = preload != expected_preload;

        const volatile uint4* value = reinterpret_cast<const volatile uint4*>(
            destination +
            destination_for_lane<FirstLineEarly>(late_fanout, lane));
        const uint4 observed = {value->x, value->y, value->z, value->w};
        const std::uint32_t expected_destination = expected_source_word(lane);
        const bool destination_mismatch =
            observed.x != expected_destination ||
            observed.y != expected_destination ||
            observed.z != expected_destination ||
            observed.w != expected_destination;
        const unsigned mismatch_mask = __ballot_sync(
            0xffffffffu, preload_mismatch || destination_mismatch);
        if (lane == 0) {
            mismatches += static_cast<std::uint32_t>(__popc(mismatch_mask));
        }
        token ^= preload_fold + observed.x + (observed.y << 3) +
                 (observed.z << 7) + (observed.w << 11) + group;
    }
    token ^= lead_state.x + (lead_state.y << 3) + (lead_state.z << 7) +
             (lead_state.w << 11);

    if (lane == 0) {
        result->issue_cycles = issue_cycles;
        result->wait_cycles = wait_cycles;
        result->total_cycles = total_cycles;
        result->token = token;
        result->mismatches = mismatches;
    }
}

template <bool FirstLineEarly>
void launch_kernel(const char* source_arm, unsigned group_base, int groups,
                   Variant variant, OwnershipResult* result) {
    source_line_ownership_kernel<FirstLineEarly><<<1, kWarpSize>>>(
        source_arm, group_base, groups, variant == Variant::kLateFanout,
        result);
}

void launch(Orientation orientation, const char* source_arm,
            unsigned group_base, int groups, Variant variant,
            OwnershipResult* result) {
    if (orientation == Orientation::kFirstLineEarly) {
        launch_kernel<true>(source_arm, group_base, groups, variant, result);
    } else {
        launch_kernel<false>(source_arm, group_base, groups, variant, result);
    }
}

int parse_int(const char* text) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (!text[0] || end[0] || value < 0 || value > kMaxGroups) return -1;
    return static_cast<int>(value);
}

const char* orientation_name(Orientation orientation) {
    return orientation == Orientation::kFirstLineEarly ? "first_line_early" :
                                                         "second_line_early";
}

const char* variant_name(Variant variant) {
    return variant == Variant::kEarlyFanout ? "early_fanout" : "late_fanout";
}

void print_result(const OwnershipResult& result, int groups) {
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
    bool profiler_marked = false;
    bool run_mode_selected = false;
    Orientation orientation = Orientation::kFirstLineEarly;
    RunMode mode = RunMode::kPairAb;

    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) return 2;
        if (std::strcmp(argv[i], "--groups") == 0) {
            groups = parse_int(argv[++i]);
        } else if (std::strcmp(argv[i], "--orientation") == 0) {
            const char* value = argv[++i];
            if (std::strcmp(value, "first_line_early") == 0) {
                orientation = Orientation::kFirstLineEarly;
            } else if (std::strcmp(value, "second_line_early") == 0) {
                orientation = Orientation::kSecondLineEarly;
            } else {
                return 2;
            }
        } else if (std::strcmp(argv[i], "--markers") == 0) {
            const char* value = argv[++i];
            if (std::strcmp(value, "on") == 0) profiler_marked = true;
            else if (std::strcmp(value, "off") == 0) profiler_marked = false;
            else return 2;
        } else if (std::strcmp(argv[i], "--order") == 0) {
            if (run_mode_selected) return 2;
            run_mode_selected = true;
            const char* value = argv[++i];
            if (std::strcmp(value, "ab") == 0) mode = RunMode::kPairAb;
            else if (std::strcmp(value, "ba") == 0) mode = RunMode::kPairBa;
            else return 2;
        } else if (std::strcmp(argv[i], "--variant") == 0) {
            if (run_mode_selected) return 2;
            run_mode_selected = true;
            const char* value = argv[++i];
            if (std::strcmp(value, "early_fanout") == 0) {
                mode = RunMode::kEarlyFanout;
            } else if (std::strcmp(value, "late_fanout") == 0) {
                mode = RunMode::kLateFanout;
            } else {
                return 2;
            }
        } else {
            return 2;
        }
    }
    if (groups < 1) return 2;

    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    char* source_pool = nullptr;
    char* eviction = nullptr;
    OwnershipResult* device_result = nullptr;
    CUDA_CHECK(cudaMalloc(&source_pool, kSourcePoolBytes));
    if (reinterpret_cast<std::uintptr_t>(source_pool) % kSourcePoolAlignment !=
        0) {
        std::fprintf(stderr, "source allocation is not 128-byte aligned\n");
        return 1;
    }
    CUDA_CHECK(cudaMalloc(&eviction, kEvictionBytes));
    CUDA_CHECK(cudaMalloc(&device_result, sizeof(OwnershipResult)));
    char* const source_arm_a = source_pool + kSourceArmAOffset;
    char* const source_arm_b = source_pool + kSourceArmBOffset;
    for (char* arm : {source_arm_a, source_arm_b}) {
        CUDA_CHECK(cudaMemset2D(arm, kSourceSlotBytes, 0x31,
                                kSourceLineBytes, kSourceSlotRows));
        CUDA_CHECK(cudaMemset2D(arm + kSourceLineBytes, kSourceSlotBytes,
                                0x57, kSourceLineBytes, kSourceSlotRows));
    }
    CUDA_CHECK(cudaMemset(device_result, 0, sizeof(OwnershipResult)));
    CUDA_CHECK(cudaDeviceSynchronize());

    OwnershipResult early_result{};
    OwnershipResult late_result{};
    int arm_index = 0;
    const auto run_arm = [&](Variant variant, OwnershipResult* host_result) {
        const char* source_arm = arm_index == 0 ? source_arm_a : source_arm_b;
        CUDA_CHECK(cudaMemset(eviction, 0x41 + arm_index, kEvictionBytes));
        CUDA_CHECK(cudaMemset(device_result, 0, sizeof(OwnershipResult)));
        CUDA_CHECK(cudaDeviceSynchronize());

        launch(orientation, source_arm, 0, kWarmupGroups, variant,
               device_result);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (profiler_marked) CUDA_CHECK(cudaProfilerStart());
        launch(orientation, source_arm, kWarmupGroups, groups, variant,
               device_result);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        if (profiler_marked) CUDA_CHECK(cudaProfilerStop());
        CUDA_CHECK(cudaMemcpy(host_result, device_result,
                              sizeof(OwnershipResult), cudaMemcpyDeviceToHost));
        ++arm_index;
    };

    if (mode == RunMode::kPairAb) {
        run_arm(Variant::kEarlyFanout, &early_result);
        run_arm(Variant::kLateFanout, &late_result);
    } else if (mode == RunMode::kPairBa) {
        run_arm(Variant::kLateFanout, &late_result);
        run_arm(Variant::kEarlyFanout, &early_result);
    } else if (mode == RunMode::kEarlyFanout) {
        run_arm(Variant::kEarlyFanout, &early_result);
    } else {
        run_arm(Variant::kLateFanout, &late_result);
    }

    const bool pair = mode == RunMode::kPairAb || mode == RunMode::kPairBa;
    const char* order = mode == RunMode::kPairBa ? "ba" : "ab";
    std::printf(
        "{\"workload\":\"async_source_line_ownership\",\"route\":\"l2\","
        "\"mode\":\"%s\",\"order\":\"%s\","
        "\"orientation\":\"%s\",\"depth\":1,"
        "\"groups\":%d,\"warmup_groups\":%d,\"lead_clock_samples\":%d,"
        "\"source_cohorts_per_copy\":%d,"
        "\"aligned_source_cohorts_per_copy\":4,"
        "\"split_source_cohorts_per_copy\":4,"
        "\"source_sectors_per_cohort\":4,"
        "\"payload_bytes_per_cohort\":64,"
        "\"source_step_stride_bytes\":%zu,"
        "\"source_step_max_exclusive_byte\":1968,"
        "\"source_step_guard_bytes\":80,"
        "\"scalar_ldg_instructions_per_copy\":%d,"
        "\"scalar_ldg_sectors_per_copy\":%d,"
        "\"scalar_preload_lines_per_copy\":%d,"
        "\"ldgsts_instructions_per_copy\":%d,"
        "\"ldgsts_sectors_per_copy\":%d,"
        "\"unique_ldgsts_lines_per_copy\":%d,"
        "\"payload_wavefronts_per_copy\":%d,"
        "\"projected_bypass_wavefronts_per_copy\":%d,"
        "\"destination_lines_per_copy\":%d,"
        "\"max_cohort_bank_wavefronts\":%d,"
        "\"max_cohort_bypass_wavefronts\":%d,"
        "\"cohort_topology_penalty_cycles_per_copy\":%d,"
        "\"response_memberships_per_copy\":%d,"
        "\"local_device_lts_lookups_per_copy\":%d,"
        "\"source_ring_bytes_per_arm\":%zu,"
        "\"source_ring_steps_per_arm\":%zu,"
        "\"max_admitted_groups\":%d,\"source_ring_no_wrap\":true,"
        "\"source_pool_bytes\":%zu,\"source_pool_alignment_bytes\":%zu,"
        "\"source_arm_a_offset_bytes\":%zu,"
        "\"source_arm_b_offset_bytes\":%zu,"
        "\"source_pool_single_allocation\":true,"
        "\"independent_source_arms\":true,"
        "\"source_vector_preserved\":true,"
        "\"preload_vector_preserved\":true,"
        "\"destination_multiset_preserved\":true,"
        "\"ownership_permutation_preserved\":true,"
        "\"static_ldgsts_positions_preserved\":true,"
        "\"result_store_after_intervals\":true,"
        "\"device\":\"%s\",\"compute_capability\":\"%d.%d\","
        "\"l2_cache_bytes\":%d,",
        pair ? "pair" : "endpoint", order, orientation_name(orientation),
        groups, kWarmupGroups, kLeadClockSamples, kSourceCohorts,
        kSourceStepBytes, kScalarLdgInstructionsPerCopy,
        kScalarLdgSectorsPerCopy, kScalarPreloadLinesPerCopy,
        kLdgstsInstructionsPerCopy, kLdgstsSectorsPerCopy,
        kUniqueLdgstsLinesPerCopy, kPayloadWavefrontsPerCopy,
        kProjectedBypassWavefrontsPerCopy, kDestinationLinesPerCopy,
        kMaxCohortBankWavefronts, kMaxCohortBypassWavefronts,
        kCohortTopologyPenaltyCyclesPerCopy, kResponseMembershipsPerCopy,
        kLocalDeviceLtsLookupsPerCopy, kSourceRingBytes, kSourceRingSteps,
        kMaxGroups, kSourcePoolBytes, kSourcePoolAlignment,
        kSourceArmAOffset, kSourceArmBOffset, properties.name,
        properties.major, properties.minor, properties.l2CacheSize);
    if (pair) {
        std::printf("\"early_fanout\":");
        print_result(early_result, groups);
        std::printf(",\"late_fanout\":");
        print_result(late_result, groups);
        std::printf(
            ",\"paired_wait_delta_cycles_per_group\":%.9f,"
            "\"paired_issue_delta_cycles_per_group\":%.9f,"
            "\"correct\":%s}\n",
            (static_cast<double>(early_result.wait_cycles) -
             static_cast<double>(late_result.wait_cycles)) /
                groups,
            (static_cast<double>(early_result.issue_cycles) -
             static_cast<double>(late_result.issue_cycles)) /
                groups,
            early_result.issue_cycles + early_result.wait_cycles ==
                        early_result.total_cycles &&
                    late_result.issue_cycles + late_result.wait_cycles ==
                        late_result.total_cycles &&
                    early_result.mismatches == 0 && late_result.mismatches == 0
                ? "true" : "false");
    } else {
        const Variant variant = mode == RunMode::kEarlyFanout ?
                                    Variant::kEarlyFanout : Variant::kLateFanout;
        const OwnershipResult& result = variant == Variant::kEarlyFanout ?
                                            early_result : late_result;
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
