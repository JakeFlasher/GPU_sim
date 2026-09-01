#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
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
constexpr int kPrefixSteps = 128;
constexpr int kWarmupPairs = 3;
constexpr unsigned long long kQueueGateNs = 250000ull;
constexpr std::size_t kL2EvictionBytes = 64u * 1024u * 1024u;

enum ModeKind : int {
    kControl = 0,
    kStoreNone = 1,
    kStoreDevice = 2,
    kModeCount = 3,
    kPaired = 3,
};

enum LayoutKind : int {
    kLineUnique = 0,
    kLineReuse = 1,
};

__device__ __forceinline__ std::uint32_t dependent_step(std::uint32_t value) {
    std::uint32_t next;
    asm volatile("mad.lo.u32 %0, %1, 1664525, 1013904223;\n"
                 : "=r"(next)
                 : "r"(value));
    return next;
}

std::uint32_t host_dependent_step(std::uint32_t value) {
    return value * 1664525u + 1013904223u;
}

__device__ __forceinline__ unsigned long long read_globaltimer() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(value) : : "memory");
    return value;
}

__device__ __forceinline__ unsigned read_smid() {
    unsigned value;
    asm volatile("mov.u32 %0, %%smid;\n" : "=r"(value));
    return value;
}

__host__ __device__ __forceinline__ std::size_t target_word(
    int layout, int depth, int block, int operation, int lane) {
    if (layout == kLineUnique) {
        return ((static_cast<std::size_t>(block) * depth + operation) *
                    kWarpSize +
                lane) *
               kWarpSize;
    }
    return (static_cast<std::size_t>(block) * kWarpSize + lane) * kWarpSize;
}

__global__ __launch_bounds__(kWarpSize) void store_ack_probe(
    std::uint32_t* words, unsigned long long* timestamps,
    unsigned long long* checksums, int mode, int depth, int active_lanes,
    int layout, int gap_steps) {
    const int lane = static_cast<int>(threadIdx.x);
    const int block = static_cast<int>(blockIdx.x);
    const std::size_t global_thread =
        static_cast<std::size_t>(block) * kWarpSize + lane;
    const bool memory_active = mode != kControl && lane < active_lanes;
    std::uint32_t value =
        0x9e3779b9u ^ static_cast<std::uint32_t>(global_thread * 131 + lane);
    unsigned long long checksum = global_thread + 1;

#pragma unroll 1
    for (int step = 0; step < kPrefixSteps; ++step) {
        value = dependent_step(value);
    }

    unsigned long long start = 0;
    if (lane == 0) {
        start = read_globaltimer();
    }
#pragma unroll 1
    for (int operation = 0; operation < depth; ++operation) {
        value = dependent_step(value);
        const std::size_t word =
            target_word(layout, depth, block, operation, lane);
        checksum ^= static_cast<unsigned long long>(word) + value;
        if (memory_active) {
            const std::uint32_t stored = value | 1u;
            asm volatile("st.global.wb.u32 [%0], %1;\n"
                         :
                         : "l"(words + word), "r"(stored)
                         : "memory");
        }
#pragma unroll 1
        for (int gap = 0; gap < gap_steps; ++gap) {
            value = dependent_step(value);
        }
    }
    unsigned long long after_issue = 0;
    if (lane == 0) {
        after_issue = read_globaltimer();
    }
    if (mode == kStoreDevice) {
        asm volatile("membar.gl;\n" : : : "memory");
    }
    unsigned long long after_fence = 0;
    if (lane == 0) {
        after_fence = read_globaltimer();
    }

    checksums[global_thread] = checksum ^ value;
    if (lane == 0) {
        const std::size_t base = static_cast<std::size_t>(block) * 4;
        timestamps[base + 0] = start;
        timestamps[base + 1] = after_issue;
        timestamps[base + 2] = after_fence;
        timestamps[base + 3] = read_smid();
    }
}

__global__ void store_ack_sentinel(const std::uint32_t* words,
                                   unsigned long long* observation,
                                   std::size_t anchor_word) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const unsigned long long start = read_globaltimer();
    std::uint32_t value;
    asm volatile("ld.global.cg.u32 %0, [%1];\n"
                 : "=r"(value)
                 : "l"(words + anchor_word)
                 : "memory");
    observation[0] = start;
    observation[1] = value;
}

__global__ void store_ack_queue_gate(unsigned long long duration_ns,
                                     unsigned long long* timestamps) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const unsigned long long start = read_globaltimer();
    unsigned long long now = start;
    while (now - start < duration_ns) {
        now = read_globaltimer();
    }
    timestamps[0] = start;
    timestamps[1] = now;
}

const char* mode_name(int mode) {
    switch (mode) {
        case kControl: return "control";
        case kStoreNone: return "store_none";
        case kStoreDevice: return "store_device";
        case kPaired: return "paired";
        default: return nullptr;
    }
}

const char* layout_name(int layout) {
    switch (layout) {
        case kLineUnique: return "line_unique";
        case kLineReuse: return "line_reuse";
        default: return nullptr;
    }
}

template <typename Name>
bool parse_named(const char* value, int begin, int end, Name name, int* result) {
    for (int candidate = begin; candidate <= end; ++candidate) {
        const char* candidate_name = name(candidate);
        if (candidate_name != nullptr && std::strcmp(value, candidate_name) == 0) {
            *result = candidate;
            return true;
        }
    }
    return false;
}

std::size_t allocation_words(int layout, int blocks, int depth) {
    if (layout == kLineUnique) {
        return static_cast<std::size_t>(blocks) * depth * kWarpSize * kWarpSize;
    }
    return static_cast<std::size_t>(blocks) * kWarpSize * kWarpSize;
}

template <typename T>
T median(std::vector<T> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

struct CtaResult {
    int block = 0;
    unsigned sm_id = 0;
    unsigned long long start_ns = 0;
    unsigned long long after_issue_ns = 0;
    unsigned long long after_fence_ns = 0;
};

struct TrialResult {
    int trial = 0;
    int order_index = 0;
    int mode = kControl;
    unsigned long long issue_min_ns = 0;
    unsigned long long issue_median_ns = 0;
    unsigned long long issue_max_ns = 0;
    unsigned long long fence_min_ns = 0;
    unsigned long long fence_median_ns = 0;
    unsigned long long fence_max_ns = 0;
    unsigned long long grid_span_ns = 0;
    unsigned long long handoff_ns = 0;
    double pair_event_ns = 0.0;
    unsigned long long gate_observed_ns = 0;
    unsigned long long host_enqueue_ns = 0;
    unsigned long long checksum = 0;
    std::uint32_t sentinel_value = 0;
    bool queue_preloaded = false;
    bool trace_only = false;
    bool correct = false;
    std::vector<CtaResult> ctas;
};

void initialize_trial(cudaStream_t stream, std::uint32_t* words,
                      std::size_t word_count, unsigned long long* timestamps,
                      std::size_t timestamp_count,
                      unsigned long long* checksums,
                      std::size_t checksum_count,
                      unsigned long long* sentinel,
                      unsigned long long* gate_timestamps,
                      std::uint32_t* l2_eviction, int fill) {
    CUDA_CHECK(cudaMemsetAsync(words, 0, word_count * sizeof(*words), stream));
    CUDA_CHECK(cudaMemsetAsync(timestamps, 0,
                               timestamp_count * sizeof(*timestamps), stream));
    CUDA_CHECK(cudaMemsetAsync(checksums, 0,
                               checksum_count * sizeof(*checksums), stream));
    CUDA_CHECK(cudaMemsetAsync(sentinel, 0, 2 * sizeof(*sentinel), stream));
    CUDA_CHECK(cudaMemsetAsync(gate_timestamps, 0,
                               2 * sizeof(*gate_timestamps), stream));
    CUDA_CHECK(cudaMemsetAsync(l2_eviction, fill, kL2EvictionBytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

std::uint32_t expected_anchor_value(int mode, int depth, int active_lanes,
                                    int gap_steps, int blocks) {
    if (mode == kControl || active_lanes == 0) {
        return 0;
    }
    const int block = blocks - 1;
    const int lane = 0;
    const std::size_t global_thread =
        static_cast<std::size_t>(block) * kWarpSize + lane;
    std::uint32_t value =
        0x9e3779b9u ^ static_cast<std::uint32_t>(global_thread * 131 + lane);
    for (int step = 0; step < kPrefixSteps; ++step) {
        value = host_dependent_step(value);
    }
    for (int operation = 0; operation < depth; ++operation) {
        value = host_dependent_step(value);
        if (operation == depth - 1) {
            return value | 1u;
        }
        for (int gap = 0; gap < gap_steps; ++gap) {
            value = host_dependent_step(value);
        }
    }
    return 0;
}

bool verify_words(const std::vector<std::uint32_t>& observed, int mode,
                  int layout, int blocks, int depth, int active_lanes,
                  int gap_steps) {
    if (mode == kControl) {
        return std::all_of(observed.begin(), observed.end(),
                           [](std::uint32_t value) { return value == 0; });
    }
    bool correct = true;
    for (int block = 0; block < blocks; ++block) {
        for (int lane = 0; lane < kWarpSize; ++lane) {
            const std::size_t global_thread =
                static_cast<std::size_t>(block) * kWarpSize + lane;
            std::uint32_t value =
                0x9e3779b9u ^
                static_cast<std::uint32_t>(global_thread * 131 + lane);
            for (int step = 0; step < kPrefixSteps; ++step) {
                value = host_dependent_step(value);
            }
            for (int operation = 0; operation < depth; ++operation) {
                value = host_dependent_step(value);
                const std::size_t word =
                    target_word(layout, depth, block, operation, lane);
                if (layout == kLineUnique || operation == depth - 1) {
                    if (lane < active_lanes) {
                        correct = correct && observed[word] == (value | 1u);
                    } else {
                        correct = correct && observed[word] == 0;
                    }
                }
                for (int gap = 0; gap < gap_steps; ++gap) {
                    value = host_dependent_step(value);
                }
            }
        }
    }
    return correct;
}

TrialResult run_trial(
    int trial, int order_index, int mode, int layout, int blocks, int depth,
    int active_lanes, int gap_steps, cudaStream_t stream, std::uint32_t* words,
    std::size_t word_count, unsigned long long* timestamps,
    unsigned long long* checksums, unsigned long long* sentinel,
    unsigned long long* gate_timestamps, std::uint32_t* l2_eviction,
    cudaEvent_t pair_start, cudaEvent_t pair_stop, bool verify_payload,
    bool trace_only) {
    const std::size_t timestamp_count = static_cast<std::size_t>(blocks) * 4;
    const std::size_t checksum_count = static_cast<std::size_t>(blocks) * kWarpSize;
    initialize_trial(stream, words, word_count, timestamps, timestamp_count,
                     checksums, checksum_count, sentinel, gate_timestamps,
                     l2_eviction,
                     trial * kModeCount + mode + 1);

    const std::size_t anchor_word =
        target_word(layout, depth, blocks - 1, depth - 1, 0);
    const auto enqueue_begin = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point enqueue_end;
    if (trace_only) {
        CUDA_CHECK(cudaProfilerStart());
        CUDA_CHECK(cudaEventRecord(pair_start, stream));
        store_ack_probe<<<blocks, kWarpSize, 0, stream>>>(
            words, timestamps, checksums, mode, depth, active_lanes, layout,
            gap_steps);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(pair_stop, stream));
        enqueue_end = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventSynchronize(pair_stop));
        CUDA_CHECK(cudaProfilerStop());
        store_ack_sentinel<<<1, kWarpSize, 0, stream>>>(words, sentinel,
                                                        anchor_word);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        store_ack_queue_gate<<<1, kWarpSize, 0, stream>>>(kQueueGateNs,
                                                          gate_timestamps);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(pair_start, stream));
        store_ack_probe<<<blocks, kWarpSize, 0, stream>>>(
            words, timestamps, checksums, mode, depth, active_lanes, layout,
            gap_steps);
        CUDA_CHECK(cudaGetLastError());
        store_ack_sentinel<<<1, kWarpSize, 0, stream>>>(words, sentinel,
                                                        anchor_word);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(pair_stop, stream));
        enqueue_end = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventSynchronize(pair_stop));
    }

    float pair_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&pair_ms, pair_start, pair_stop));
    std::vector<unsigned long long> observed_timestamps(timestamp_count);
    std::vector<unsigned long long> observed_checksums(checksum_count);
    std::array<unsigned long long, 2> observed_sentinel{};
    std::array<unsigned long long, 2> observed_gate{};
    CUDA_CHECK(cudaMemcpy(observed_timestamps.data(), timestamps,
                          timestamp_count * sizeof(*timestamps),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(observed_checksums.data(), checksums,
                          checksum_count * sizeof(*checksums),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(observed_sentinel.data(), sentinel,
                          observed_sentinel.size() * sizeof(sentinel[0]),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(observed_gate.data(), gate_timestamps,
                          observed_gate.size() * sizeof(gate_timestamps[0]),
                          cudaMemcpyDeviceToHost));

    bool correct = true;
    std::vector<unsigned long long> issue;
    std::vector<unsigned long long> fence;
    unsigned long long earliest_start = ~0ull;
    unsigned long long latest_after_fence = 0;
    std::vector<CtaResult> ctas;
    ctas.reserve(static_cast<std::size_t>(blocks));
    for (int block = 0; block < blocks; ++block) {
        const std::size_t base = static_cast<std::size_t>(block) * 4;
        const unsigned long long start = observed_timestamps[base + 0];
        const unsigned long long after_issue = observed_timestamps[base + 1];
        const unsigned long long after_fence = observed_timestamps[base + 2];
        correct = correct && start != 0 && after_issue >= start &&
                  after_fence >= after_issue;
        issue.push_back(after_issue - start);
        fence.push_back(after_fence - after_issue);
        earliest_start = std::min(earliest_start, start);
        latest_after_fence = std::max(latest_after_fence, after_fence);
        ctas.push_back(CtaResult {
            block,
            static_cast<unsigned>(observed_timestamps[base + 3]),
            start,
            after_issue,
            after_fence,
        });
    }
    correct = correct && observed_sentinel[0] >= latest_after_fence;
    const unsigned long long gate_observed_ns =
        trace_only ? 0 : observed_gate[1] - observed_gate[0];
    const auto host_enqueue_ns_signed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(enqueue_end -
                                                             enqueue_begin)
            .count();
    const unsigned long long host_enqueue_ns =
        host_enqueue_ns_signed < 0
            ? 0
            : static_cast<unsigned long long>(host_enqueue_ns_signed);
    const bool queue_preloaded =
        trace_only ||
        (observed_gate[0] != 0 && gate_observed_ns >= kQueueGateNs &&
         host_enqueue_ns < gate_observed_ns);
    correct = correct && queue_preloaded;
    const std::uint32_t expected_anchor =
        expected_anchor_value(mode, depth, active_lanes, gap_steps, blocks);
    correct = correct &&
              static_cast<std::uint32_t>(observed_sentinel[1]) ==
                  expected_anchor;

    if (verify_payload) {
        std::vector<std::uint32_t> observed_words(word_count);
        CUDA_CHECK(cudaMemcpy(observed_words.data(), words,
                              word_count * sizeof(*words),
                              cudaMemcpyDeviceToHost));
        correct = correct && verify_words(observed_words, mode, layout, blocks,
                                          depth, active_lanes, gap_steps);
    }

    unsigned long long checksum = 0;
    for (unsigned long long value : observed_checksums) {
        checksum = checksum * 131u + value;
    }
    const auto [issue_min, issue_max] =
        std::minmax_element(issue.begin(), issue.end());
    const auto [fence_min, fence_max] =
        std::minmax_element(fence.begin(), fence.end());
    TrialResult result;
    result.trial = trial;
    result.order_index = order_index;
    result.mode = mode;
    result.issue_min_ns = *issue_min;
    result.issue_median_ns = median(issue);
    result.issue_max_ns = *issue_max;
    result.fence_min_ns = *fence_min;
    result.fence_median_ns = median(fence);
    result.fence_max_ns = *fence_max;
    result.grid_span_ns = latest_after_fence - earliest_start;
    result.handoff_ns = observed_sentinel[0] - latest_after_fence;
    result.pair_event_ns = static_cast<double>(pair_ms) * 1000000.0;
    result.gate_observed_ns = gate_observed_ns;
    result.host_enqueue_ns = host_enqueue_ns;
    result.checksum = checksum;
    result.sentinel_value =
        static_cast<std::uint32_t>(observed_sentinel[1]);
    result.queue_preloaded = queue_preloaded;
    result.trace_only = trace_only;
    result.correct = correct;
    result.ctas = std::move(ctas);
    return result;
}

void print_trial(const TrialResult& result, int layout, int blocks, int depth,
                 int active_lanes, int gap_steps) {
    std::set<unsigned> unique_sms;
    for (const CtaResult& cta : result.ctas) {
        unique_sms.insert(cta.sm_id);
        std::printf(
            "{\"record\":\"cta\",\"mode\":\"%s\","
            "\"layout\":\"%s\",\"blocks\":%d,\"depth\":%d,"
            "\"active_lanes\":%d,\"gap_steps\":%d,\"trial\":%d,"
            "\"order_index\":%d,\"block\":%d,\"sm_id\":%u,"
            "\"before_issue_ns\":%llu,\"after_issue_ns\":%llu,"
            "\"after_fence_ns\":%llu}\n",
            mode_name(result.mode), layout_name(layout), blocks, depth,
            active_lanes, gap_steps, result.trial, result.order_index,
            cta.block, cta.sm_id, cta.start_ns, cta.after_issue_ns,
            cta.after_fence_ns);
    }
    std::printf(
        "{\"record\":\"trial\",\"mode\":\"%s\",\"layout\":\"%s\","
        "\"blocks\":%d,\"depth\":%d,\"active_lanes\":%d,"
        "\"gap_steps\":%d,\"trial\":%d,\"order_index\":%d,"
        "\"issue_ns_min\":%llu,\"issue_ns_median\":%llu,"
        "\"issue_ns_max\":%llu,\"fence_ns_min\":%llu,"
        "\"fence_ns_median\":%llu,\"fence_ns_max\":%llu,"
        "\"grid_span_ns\":%llu,\"handoff_ns\":%llu,"
        "\"pair_event_ns\":%.3f,\"gate_observed_ns\":%llu,"
        "\"host_enqueue_ns\":%llu,\"unique_sms\":%zu,"
        "\"queue_preloaded\":%s,"
        "\"trace_only\":%s,\"checksum\":%llu,"
        "\"sentinel_value\":%u,\"correct\":%s}\n",
        mode_name(result.mode), layout_name(layout), blocks, depth,
        active_lanes, gap_steps, result.trial, result.order_index,
        result.issue_min_ns, result.issue_median_ns, result.issue_max_ns,
        result.fence_min_ns, result.fence_median_ns, result.fence_max_ns,
        result.grid_span_ns, result.handoff_ns, result.pair_event_ns,
        result.gate_observed_ns, result.host_enqueue_ns,
        unique_sms.size(),
        result.queue_preloaded ? "true" : "false",
        result.trace_only ? "true" : "false", result.checksum,
        result.sentinel_value,
        result.correct ? "true" : "false");
}

void print_summary(int mode, int layout, int blocks, int depth,
                   int active_lanes, int gap_steps,
                   const std::vector<TrialResult>& results) {
    std::vector<unsigned long long> issue_max;
    std::vector<unsigned long long> fence_max;
    std::vector<unsigned long long> grid_span;
    std::vector<unsigned long long> handoff;
    std::vector<double> pair_event;
    unsigned long long maximum_host_enqueue_ns = 0;
    unsigned long long minimum_gate_observed_ns = ~0ull;
    bool correct = !results.empty();
    for (const TrialResult& result : results) {
        issue_max.push_back(result.issue_max_ns);
        fence_max.push_back(result.fence_max_ns);
        grid_span.push_back(result.grid_span_ns);
        handoff.push_back(result.handoff_ns);
        pair_event.push_back(result.pair_event_ns);
        maximum_host_enqueue_ns =
            std::max(maximum_host_enqueue_ns, result.host_enqueue_ns);
        minimum_gate_observed_ns =
            std::min(minimum_gate_observed_ns, result.gate_observed_ns);
        correct = correct && result.correct;
    }
    const auto [handoff_min, handoff_max] =
        std::minmax_element(handoff.begin(), handoff.end());
    std::printf(
        "{\"record\":\"summary\",\"mode\":\"%s\","
        "\"layout\":\"%s\",\"blocks\":%d,\"depth\":%d,"
        "\"active_lanes\":%d,\"gap_steps\":%d,\"trials\":%zu,"
        "\"prefix_steps\":%d,\"l2_eviction_bytes\":%zu,"
        "\"payload_store_instructions\":%llu,"
        "\"payload_store_sectors\":%llu,"
        "\"issue_ns_max_median\":%llu,"
        "\"fence_ns_max_median\":%llu,"
        "\"grid_span_ns_median\":%llu,"
        "\"handoff_ns_min\":%llu,\"handoff_ns_median\":%llu,"
        "\"handoff_ns_max\":%llu,\"pair_event_ns_median\":%.3f,"
        "\"minimum_gate_observed_ns\":%llu,"
        "\"maximum_host_enqueue_ns\":%llu,"
        "\"correct\":%s}\n",
        mode_name(mode), layout_name(layout), blocks, depth, active_lanes,
        gap_steps, results.size(), kPrefixSteps, kL2EvictionBytes,
        mode == kControl ? 0ull
                         : static_cast<unsigned long long>(blocks) * depth,
        mode == kControl
            ? 0ull
            : static_cast<unsigned long long>(blocks) * depth * active_lanes,
        median(issue_max), median(fence_max), median(grid_span), *handoff_min,
        median(handoff), *handoff_max, median(pair_event),
        minimum_gate_observed_ns, maximum_host_enqueue_ns,
        correct ? "true" : "false");
}

int main(int argc, char** argv) {
    if (argc != 8) {
        std::fprintf(
            stderr,
            "usage: %s MODE LAYOUT BLOCKS DEPTH ACTIVE_LANES GAP_STEPS TRIALS\n",
            argv[0]);
        return 2;
    }
    int selected_mode = -1;
    int layout = -1;
    const int blocks = std::atoi(argv[3]);
    const int depth = std::atoi(argv[4]);
    const int active_lanes = std::atoi(argv[5]);
    const int gap_steps = std::atoi(argv[6]);
    const int trials = std::atoi(argv[7]);
    if (!parse_named(argv[1], kControl, kPaired, mode_name, &selected_mode) ||
        !parse_named(argv[2], kLineUnique, kLineReuse, layout_name, &layout) ||
        blocks < 1 || blocks > 168 || depth < 1 || depth > 128 ||
        active_lanes < 0 || active_lanes > kWarpSize || gap_steps < 0 ||
        gap_steps > 256 || trials < 1 || trials > 129 ||
        (trials % 2) == 0) {
        return 2;
    }
    const bool trace_only = std::getenv("STORE_ACK_TRACE_ONLY") != nullptr ||
                            std::getenv("CUDA_INJECTION64_PATH") != nullptr;
    if (trace_only && (selected_mode == kPaired || trials != 1)) {
        std::fprintf(stderr,
                     "STORE_ACK_TRACE_ONLY requires one explicit arm and one trial\n");
        return 2;
    }

    const std::size_t word_count = allocation_words(layout, blocks, depth);
    const std::size_t timestamp_count = static_cast<std::size_t>(blocks) * 4;
    const std::size_t checksum_count = static_cast<std::size_t>(blocks) * kWarpSize;
    std::uint32_t* words = nullptr;
    unsigned long long* timestamps = nullptr;
    unsigned long long* checksums = nullptr;
    unsigned long long* sentinel = nullptr;
    unsigned long long* gate_timestamps = nullptr;
    std::uint32_t* l2_eviction = nullptr;
    CUDA_CHECK(cudaMalloc(&words, word_count * sizeof(*words)));
    CUDA_CHECK(cudaMalloc(&timestamps, timestamp_count * sizeof(*timestamps)));
    CUDA_CHECK(cudaMalloc(&checksums, checksum_count * sizeof(*checksums)));
    CUDA_CHECK(cudaMalloc(&sentinel, 2 * sizeof(*sentinel)));
    CUDA_CHECK(cudaMalloc(&gate_timestamps, 2 * sizeof(*gate_timestamps)));
    CUDA_CHECK(cudaMalloc(&l2_eviction, kL2EvictionBytes));
    cudaStream_t stream = nullptr;
    cudaEvent_t pair_start = nullptr;
    cudaEvent_t pair_stop = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&pair_start));
    CUDA_CHECK(cudaEventCreate(&pair_stop));

    const int first_mode = selected_mode == kPaired ? kControl : selected_mode;
    const int last_mode = selected_mode == kPaired ? kStoreDevice : selected_mode;
    for (int warmup = 0; warmup < (trace_only ? 0 : kWarmupPairs); ++warmup) {
        for (int mode = first_mode; mode <= last_mode; ++mode) {
            initialize_trial(stream, words, word_count, timestamps,
                             timestamp_count, checksums, checksum_count,
                             sentinel, gate_timestamps, l2_eviction,
                             warmup * kModeCount + mode + 1);
            const std::size_t anchor_word =
                target_word(layout, depth, blocks - 1, depth - 1, 0);
            store_ack_probe<<<blocks, kWarpSize, 0, stream>>>(
                words, timestamps, checksums, mode, depth, active_lanes,
                layout, gap_steps);
            CUDA_CHECK(cudaGetLastError());
            store_ack_sentinel<<<1, kWarpSize, 0, stream>>>(words, sentinel,
                                                            anchor_word);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    std::array<std::vector<TrialResult>, kModeCount> by_mode;
    bool correct = true;
    for (int trial = 0; trial < trials; ++trial) {
        const int mode_count = selected_mode == kPaired ? kModeCount : 1;
        for (int order_index = 0; order_index < mode_count; ++order_index) {
            const int mode = selected_mode == kPaired
                                 ? (trial + order_index) % kModeCount
                                 : selected_mode;
            TrialResult result = run_trial(
                trial, order_index, mode, layout, blocks, depth, active_lanes,
                gap_steps, stream, words, word_count, timestamps, checksums,
                sentinel, gate_timestamps, l2_eviction, pair_start, pair_stop,
                trial == trials - 1, trace_only);
            correct = correct && result.correct;
            print_trial(result, layout, blocks, depth, active_lanes, gap_steps);
            by_mode[mode].push_back(result);
        }
    }
    for (int mode = first_mode; mode <= last_mode; ++mode) {
        print_summary(mode, layout, blocks, depth, active_lanes, gap_steps,
                      by_mode[mode]);
    }

    CUDA_CHECK(cudaEventDestroy(pair_stop));
    CUDA_CHECK(cudaEventDestroy(pair_start));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(l2_eviction));
    CUDA_CHECK(cudaFree(gate_timestamps));
    CUDA_CHECK(cudaFree(sentinel));
    CUDA_CHECK(cudaFree(checksums));
    CUDA_CHECK(cudaFree(timestamps));
    CUDA_CHECK(cudaFree(words));
    return correct ? 0 : 1;
}
