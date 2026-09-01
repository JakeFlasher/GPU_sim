#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t status = (call);                                      \
        if (status != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s failed: %s\n", #call,                    \
                         cudaGetErrorString(status));                           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

constexpr int kWarpSize = 32;
constexpr std::size_t kLaneStride = 32;
constexpr std::size_t kSlotStride = 4096;
constexpr int kSlots = 3;
constexpr std::size_t kBytes = kSlots * kSlotStride;
constexpr int kDefaultEpochs = 8192;
constexpr int kMaxEpochs = 65536;

enum class Mode : int { Identification = 0, Holdout = 1 };

struct PrefixResult {
    unsigned long long issue_commit_cycles;
    unsigned long long prefix_wait_cycles;
    unsigned long long oldest_consume_cycles;
    unsigned long long remainder_wait_cycles;
    unsigned long long total_cycles;
    std::uint32_t token;
};

__device__ __forceinline__ unsigned long long read_clock64() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) : : "memory");
    return value;
}

__device__ __forceinline__ void copy_ca(unsigned destination,
                                        const char* source) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(destination), "l"(source)
                 : "memory");
}

__device__ __forceinline__ void copy_cg(unsigned destination,
                                        const char* source) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(destination), "l"(source)
                 : "memory");
}

__device__ __forceinline__ void commit_group() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int Pending>
__device__ __forceinline__ void wait_group() {
    static_assert(Pending >= 0 && Pending <= 2);
    asm volatile("cp.async.wait_group %0;\n" : : "n"(Pending) : "memory");
}

__device__ __forceinline__ std::uint32_t consume_slot(const char* destination,
                                                       int slot, int lane) {
    const auto* value = reinterpret_cast<const volatile uint4*>(
        destination + static_cast<std::size_t>(slot) * kSlotStride +
        static_cast<std::size_t>(lane) * kLaneStride);
    return value->x ^ (value->y << 3) ^ (value->z << 7) ^ (value->w << 11);
}

__global__ void wait_prefix_identification_kernel(const char* source, int epochs,
                                                   bool l2_old,
                                                   PrefixResult* result) {
    __shared__ __align__(16) char destination[kBytes];
    if (threadIdx.x >= kWarpSize) return;
    const int lane = static_cast<int>(threadIdx.x);
    const char* ca_source = source + static_cast<std::size_t>(lane) * kLaneStride;
    const char* cg_source = source + kSlotStride +
                            static_cast<std::size_t>(lane) * kLaneStride;
    const unsigned ca_destination = static_cast<unsigned>(__cvta_generic_to_shared(
        destination + static_cast<std::size_t>(lane) * kLaneStride));
    const unsigned cg_destination = static_cast<unsigned>(__cvta_generic_to_shared(
        destination + kSlotStride + static_cast<std::size_t>(lane) * kLaneStride));

    PrefixResult local{};
    std::uint32_t token = 0x715f8a3du ^ static_cast<std::uint32_t>(lane);
#pragma unroll 1
    for (int epoch = 0; epoch < epochs; ++epoch) {
        const unsigned long long before_issue = read_clock64();
        if (l2_old) {
            copy_cg(cg_destination, cg_source);
            commit_group();
            copy_ca(ca_destination, ca_source);
            commit_group();
        } else {
            copy_ca(ca_destination, ca_source);
            commit_group();
            copy_cg(cg_destination, cg_source);
            commit_group();
        }
        const unsigned long long before_prefix_wait = read_clock64();
        wait_group<1>();
        const unsigned long long after_prefix_wait = read_clock64();

        token ^= consume_slot(destination, l2_old ? 1 : 0, lane) +
                 static_cast<std::uint32_t>(epoch);
        const unsigned long long before_remainder_wait = read_clock64();
        wait_group<0>();
        const unsigned long long after_remainder_wait = read_clock64();
        token ^= (consume_slot(destination, l2_old ? 0 : 1, lane) << 1);

        local.issue_commit_cycles += before_prefix_wait - before_issue;
        local.prefix_wait_cycles += after_prefix_wait - before_prefix_wait;
        local.oldest_consume_cycles += before_remainder_wait - after_prefix_wait;
        local.remainder_wait_cycles += after_remainder_wait - before_remainder_wait;
        local.total_cycles += after_remainder_wait - before_issue;
    }
    if (lane == 0) {
        local.token = token;
        *result = local;
    }
}

__global__ void wait_prefix_holdout_kernel(const char* source, int epochs,
                                           bool l2_old,
                                           PrefixResult* result) {
    __shared__ __align__(16) char destination[kBytes];
    if (threadIdx.x >= kWarpSize) return;
    const int lane = static_cast<int>(threadIdx.x);
    const char* ca0_source = source + static_cast<std::size_t>(lane) * kLaneStride;
    const char* cg_source = source + kSlotStride +
                            static_cast<std::size_t>(lane) * kLaneStride;
    const char* ca1_source = source + 2 * kSlotStride +
                             static_cast<std::size_t>(lane) * kLaneStride;
    const unsigned ca0_destination = static_cast<unsigned>(__cvta_generic_to_shared(
        destination + static_cast<std::size_t>(lane) * kLaneStride));
    const unsigned cg_destination = static_cast<unsigned>(__cvta_generic_to_shared(
        destination + kSlotStride + static_cast<std::size_t>(lane) * kLaneStride));
    const unsigned ca1_destination = static_cast<unsigned>(__cvta_generic_to_shared(
        destination + 2 * kSlotStride + static_cast<std::size_t>(lane) * kLaneStride));

    PrefixResult local{};
    std::uint32_t token = 0x9cb1642fu ^ static_cast<std::uint32_t>(lane);
#pragma unroll 1
    for (int epoch = 0; epoch < epochs; ++epoch) {
        const unsigned long long before_issue = read_clock64();
        if (l2_old) {
            copy_cg(cg_destination, cg_source);
            commit_group();
            copy_ca(ca0_destination, ca0_source);
            commit_group();
            copy_ca(ca1_destination, ca1_source);
            commit_group();
        } else {
            copy_ca(ca0_destination, ca0_source);
            commit_group();
            copy_ca(ca1_destination, ca1_source);
            commit_group();
            copy_cg(cg_destination, cg_source);
            commit_group();
        }
        const unsigned long long before_prefix_wait = read_clock64();
        wait_group<2>();
        const unsigned long long after_prefix_wait = read_clock64();

        token ^= consume_slot(destination, l2_old ? 1 : 0, lane) +
                 static_cast<std::uint32_t>(epoch);
        const unsigned long long before_remainder_wait = read_clock64();
        wait_group<0>();
        const unsigned long long after_remainder_wait = read_clock64();
        if (l2_old) {
            token ^= consume_slot(destination, 0, lane) << 1;
            token ^= consume_slot(destination, 2, lane) << 2;
        } else {
            token ^= consume_slot(destination, 2, lane) << 1;
            token ^= consume_slot(destination, 1, lane) << 2;
        }

        local.issue_commit_cycles += before_prefix_wait - before_issue;
        local.prefix_wait_cycles += after_prefix_wait - before_prefix_wait;
        local.oldest_consume_cycles += before_remainder_wait - after_prefix_wait;
        local.remainder_wait_cycles += after_remainder_wait - before_remainder_wait;
        local.total_cycles += after_remainder_wait - before_issue;
    }
    if (lane == 0) {
        local.token = token;
        *result = local;
    }
}

__global__ void prewarm_kernel(const char* source, std::uint32_t* sink) {
    if (threadIdx.x >= kWarpSize) return;
    const int lane = static_cast<int>(threadIdx.x);
    std::uint32_t value = 0;
#pragma unroll
    for (int slot = 0; slot < kSlots; ++slot) {
        const auto* item = reinterpret_cast<const uint4*>(
            source + static_cast<std::size_t>(slot) * kSlotStride +
            static_cast<std::size_t>(lane) * kLaneStride);
        const uint4 loaded = *item;
        value ^= loaded.x ^ loaded.y ^ loaded.z ^ loaded.w;
    }
    if (lane == 0) *sink = value;
}

void launch_target(Mode mode, bool l2_old, const char* source, int epochs,
                   PrefixResult* result) {
    if (mode == Mode::Identification) {
        wait_prefix_identification_kernel<<<1, kWarpSize>>>(source, epochs, l2_old, result);
    } else {
        wait_prefix_holdout_kernel<<<1, kWarpSize>>>(source, epochs, l2_old, result);
    }
}

int parse_epochs(const char* text) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (!text[0] || end[0] || value < 1 || value > kMaxEpochs) return -1;
    return static_cast<int>(value);
}

std::uint32_t fold_source(const std::vector<std::uint32_t>& source, int slot) {
    const std::size_t first = static_cast<std::size_t>(slot) * kSlotStride /
                              sizeof(std::uint32_t);
    return source[first] ^ (source[first + 1] << 3) ^
           (source[first + 2] << 7) ^ (source[first + 3] << 11);
}

std::uint32_t expected_token(const std::vector<std::uint32_t>& source, int epochs,
                             Mode mode, bool l2_old) {
    std::uint32_t token = mode == Mode::Identification ? 0x715f8a3du : 0x9cb1642fu;
    const int oldest = l2_old ? 1 : 0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        token ^= fold_source(source, oldest) + static_cast<std::uint32_t>(epoch);
        if (mode == Mode::Identification) {
            token ^= fold_source(source, l2_old ? 0 : 1) << 1;
        } else if (l2_old) {
            token ^= fold_source(source, 0) << 1;
            token ^= fold_source(source, 2) << 2;
        } else {
            token ^= fold_source(source, 2) << 1;
            token ^= fold_source(source, 1) << 2;
        }
    }
    return token;
}

int main(int argc, char** argv) {
    int epochs = kDefaultEpochs;
    bool l2_old = true;
    Mode mode = Mode::Identification;
    bool profiler_marked = false;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) return 2;
        if (std::strcmp(argv[i], "--epochs") == 0) {
            epochs = parse_epochs(argv[++i]);
        } else if (std::strcmp(argv[i], "--variant") == 0) {
            const char* value = argv[++i];
            if (std::strcmp(value, "l2_old") == 0) l2_old = true;
            else if (std::strcmp(value, "l1_old") == 0) l2_old = false;
            else return 2;
        } else if (std::strcmp(argv[i], "--mode") == 0) {
            const char* value = argv[++i];
            if (std::strcmp(value, "identification") == 0) mode = Mode::Identification;
            else if (std::strcmp(value, "holdout") == 0) mode = Mode::Holdout;
            else return 2;
        } else if (std::strcmp(argv[i], "--markers") == 0) {
            const char* value = argv[++i];
            if (std::strcmp(value, "on") == 0) profiler_marked = true;
            else if (std::strcmp(value, "off") == 0) profiler_marked = false;
            else return 2;
        } else {
            return 2;
        }
    }
    if (epochs < 1) return 2;

    std::vector<std::uint32_t> host_source(kBytes / sizeof(std::uint32_t));
    for (std::size_t i = 0; i < host_source.size(); ++i) {
        host_source[i] = (static_cast<std::uint32_t>(i) * 1664525u + 1013904223u) ^
                         0x9e3779b9u;
    }
    char* source = nullptr;
    PrefixResult* device_result = nullptr;
    std::uint32_t* device_sink = nullptr;
    PrefixResult host_result{};
    CUDA_CHECK(cudaMalloc(&source, kBytes));
    CUDA_CHECK(cudaMalloc(&device_result, sizeof(PrefixResult)));
    CUDA_CHECK(cudaMalloc(&device_sink, sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemcpy(source, host_source.data(), kBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_result, 0, sizeof(PrefixResult)));
    prewarm_kernel<<<1, kWarpSize>>>(source, device_sink);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (profiler_marked) CUDA_CHECK(cudaProfilerStart());
    launch_target(mode, l2_old, source, epochs, device_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (profiler_marked) CUDA_CHECK(cudaProfilerStop());
    CUDA_CHECK(cudaMemcpy(&host_result, device_result, sizeof(PrefixResult),
                          cudaMemcpyDeviceToHost));

    const unsigned long long reconstructed =
        host_result.issue_commit_cycles + host_result.prefix_wait_cycles +
        host_result.oldest_consume_cycles + host_result.remainder_wait_cycles;
    const std::uint32_t expected = expected_token(host_source, epochs, mode, l2_old);
    const bool correct = reconstructed == host_result.total_cycles &&
                         host_result.token == expected;
    std::printf(
        "{\"workload\":\"async_wait_prefix_order\",\"mode\":\"%s\","
        "\"variant\":\"%s\",\"epochs\":%d,\"groups_per_epoch\":%d,"
        "\"wait_group_immediate\":%d,\"issue_commit_cycles\":%llu,"
        "\"prefix_wait_cycles\":%llu,\"oldest_consume_cycles\":%llu,"
        "\"remainder_wait_cycles\":%llu,\"total_cycles\":%llu,"
        "\"sum_consistent\":%s,\"token\":%u,\"expected_token\":%u,"
        "\"result_store_after_intervals\":true,\"correct\":%s}\n",
        mode == Mode::Identification ? "identification" : "holdout",
        l2_old ? "l2_old" : "l1_old", epochs,
        mode == Mode::Identification ? 2 : 3,
        mode == Mode::Identification ? 1 : 2,
        host_result.issue_commit_cycles, host_result.prefix_wait_cycles,
        host_result.oldest_consume_cycles, host_result.remainder_wait_cycles,
        host_result.total_cycles,
        reconstructed == host_result.total_cycles ? "true" : "false",
        host_result.token, expected, correct ? "true" : "false");

    CUDA_CHECK(cudaFree(device_sink));
    CUDA_CHECK(cudaFree(device_result));
    CUDA_CHECK(cudaFree(source));
    return correct ? 0 : 1;
}
