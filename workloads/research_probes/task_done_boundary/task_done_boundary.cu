#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t status = (call);                                      \
        if (status != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s failed: %s\n", #call,                   \
                         cudaGetErrorString(status));                           \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

constexpr int kThreads = 256;
constexpr std::size_t kLineBytes = 128;
constexpr std::size_t kWordsPerLine = kLineBytes / sizeof(std::uint32_t);
constexpr int kWarmups = 2;

enum Mode : int { kControl = 0, kStore = 1, kStoreDevice = 2 };

__host__ __device__ __forceinline__ std::uint32_t lcg(std::uint32_t value) {
    return value * 1664525u + 1013904223u;
}

__device__ __forceinline__ unsigned long long globaltimer() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(value) : : "memory");
    return value;
}

__host__ __device__ __forceinline__ std::uint32_t expected_value(
    std::size_t line) {
    std::uint32_t value = 0x6a09e667u ^
                          static_cast<std::uint32_t>(line * 0x27d4eb2du);
    return lcg(lcg(value)) | 1u;
}

__host__ __device__ __forceinline__ std::uint32_t stored_value(
    std::size_t line) {
    return lcg(expected_value(line));
}

__global__ __launch_bounds__(kThreads) void producer_kernel(
    std::uint32_t* data, std::size_t line_count, int mode,
    unsigned long long* timestamps) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    if (blockIdx.x == 0 && threadIdx.x == 0) timestamps[0] = globaltimer();
    for (std::size_t line = tid; line < line_count; line += stride) {
        std::uint32_t value = stored_value(line);
        if (mode != kControl) {
            asm volatile("st.global.wb.u32 [%0], %1;\n"
                         :
                         : "l"(data + line * kWordsPerLine), "r"(value)
                         : "memory");
        }
    }
    if (mode == kStoreDevice) {
        asm volatile("membar.gl;\n" : : : "memory");
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) timestamps[1] = globaltimer();
}

__global__ void thrash_kernel(std::uint32_t* data, std::size_t word_count,
                              unsigned long long* timestamps) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    if (blockIdx.x == 0 && threadIdx.x == 0) timestamps[2] = globaltimer();
    std::uint32_t value = static_cast<std::uint32_t>(tid + 17);
    for (std::size_t word = tid; word < word_count; word += stride) {
        value = lcg(value);
        asm volatile("st.global.wb.u32 [%0], %1;\n"
                     :
                     : "l"(data + word), "r"(value)
                     : "memory");
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) timestamps[3] = globaltimer();
}

__global__ void consumer_kernel(const std::uint32_t* data,
                                std::size_t line_count,
                                unsigned long long* timestamps,
                                unsigned long long* partials) {
    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    if (blockIdx.x == 0 && threadIdx.x == 0) timestamps[4] = globaltimer();
    unsigned long long sum = 0;
    for (std::size_t line = tid; line < line_count; line += stride) {
        std::uint32_t value;
        asm volatile("ld.global.cg.u32 %0, [%1];\n"
                     : "=r"(value)
                     : "l"(data + line * kWordsPerLine)
                     : "memory");
        sum += value;
    }
    partials[tid] = sum;
    if (blockIdx.x == 0 && threadIdx.x == 0) timestamps[5] = globaltimer();
}

const char* mode_name(Mode mode) {
    switch (mode) {
        case kControl: return "control";
        case kStore: return "store";
        case kStoreDevice: return "store_device";
    }
    return "unknown";
}

bool parse_mode(const char* text, Mode* mode) {
    if (std::strcmp(text, "control") == 0) {
        *mode = kControl;
        return true;
    }
    if (std::strcmp(text, "store") == 0) {
        *mode = kStore;
        return true;
    }
    if (std::strcmp(text, "store_device") == 0) {
        *mode = kStoreDevice;
        return true;
    }
    return false;
}

struct Trial {
    int trial = 0;
    Mode mode = kControl;
    unsigned long long producer_block_global_ns = 0;
    unsigned long long thrash_block_global_ns = 0;
    unsigned long long consumer_block_global_ns = 0;
    float producer_event_ms = 0.0f;
    float total_event_ms = 0.0f;
    float copy_event_ms = 0.0f;
    unsigned long long host_total_ns = 0;
    unsigned long long producer_to_consumer_ns = 0;
    unsigned long long checksum = 0;
    unsigned long long expected_checksum = 0;
    bool correct = false;
};

Trial run_trial(int trial, Mode mode, std::size_t line_count,
                std::size_t thrash_words, int blocks, cudaStream_t stream,
                std::uint32_t* data, std::uint32_t* thrash,
                unsigned long long* timestamps,
                unsigned long long* partials, std::uint32_t* host_data,
                cudaEvent_t producer_start, cudaEvent_t producer_stop,
                cudaEvent_t total_stop, cudaEvent_t copy_stop) {
    const std::size_t byte_count = line_count * kWordsPerLine * sizeof(std::uint32_t);
    CUDA_CHECK(cudaMemsetAsync(data, 0, byte_count, stream));
    CUDA_CHECK(cudaMemsetAsync(thrash, 0,
                               std::max<std::size_t>(thrash_words, 1) * sizeof(std::uint32_t),
                               stream));
    CUDA_CHECK(cudaMemsetAsync(timestamps, 0, 6 * sizeof(*timestamps), stream));
    CUDA_CHECK(cudaMemsetAsync(partials, 0,
                               static_cast<std::size_t>(blocks) * kThreads * sizeof(*partials),
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const auto host_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(producer_start, stream));
    producer_kernel<<<blocks, kThreads, 0, stream>>>(data, line_count, mode, timestamps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(producer_stop, stream));
    thrash_kernel<<<blocks, kThreads, 0, stream>>>(thrash, thrash_words, timestamps);
    CUDA_CHECK(cudaGetLastError());
    consumer_kernel<<<blocks, kThreads, 0, stream>>>(data, line_count, timestamps, partials);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(total_stop, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_data, data, byte_count,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(copy_stop, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto host_end = std::chrono::steady_clock::now();

    float producer_ms = 0.0f;
    float total_ms = 0.0f;
    float copy_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&producer_ms, producer_start, producer_stop));
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, producer_start, total_stop));
    CUDA_CHECK(cudaEventElapsedTime(&copy_ms, total_stop, copy_stop));

    std::vector<unsigned long long> host_timestamps(6);
    CUDA_CHECK(cudaMemcpy(host_timestamps.data(), timestamps,
                          host_timestamps.size() * sizeof(*timestamps),
                          cudaMemcpyDeviceToHost));
    std::vector<unsigned long long> host_partials(
        static_cast<std::size_t>(blocks) * kThreads);
    CUDA_CHECK(cudaMemcpy(host_partials.data(), partials,
                          host_partials.size() * sizeof(*partials),
                          cudaMemcpyDeviceToHost));

    Trial result;
    result.trial = trial;
    result.mode = mode;
    result.producer_block_global_ns = host_timestamps[1] - host_timestamps[0];
    result.thrash_block_global_ns = host_timestamps[3] - host_timestamps[2];
    result.consumer_block_global_ns = host_timestamps[5] - host_timestamps[4];
    result.producer_event_ms = producer_ms;
    result.total_event_ms = total_ms;
    result.copy_event_ms = copy_ms;
    result.host_total_ns = static_cast<unsigned long long>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(host_end - host_start).count());
    result.producer_to_consumer_ns = host_timestamps[4] - host_timestamps[1];
    for (unsigned long long value : host_partials) result.checksum += value;
    for (std::size_t line = 0; line < line_count; ++line) {
        const std::uint32_t expected = mode == kControl ? 0 : stored_value(line);
        result.expected_checksum += expected;
    }
    result.correct = result.checksum == result.expected_checksum;
    const std::size_t sample_step = std::max<std::size_t>(line_count / 1024, 1);
    for (std::size_t line = 0; line < line_count; line += sample_step) {
        const std::uint32_t expected = mode == kControl ? 0 : stored_value(line);
        if (host_data[line * kWordsPerLine] != expected) {
            result.correct = false;
            break;
        }
    }
    if (line_count != 0) {
        const std::size_t line = line_count - 1;
        const std::uint32_t expected = mode == kControl ? 0 : stored_value(line);
        result.correct = result.correct && host_data[line * kWordsPerLine] == expected;
    }
    return result;
}

void print_trial(const Trial& result) {
    std::printf(
        "{\"record\":\"trial\",\"trial\":%d,\"mode\":\"%s\","
        "\"producer_block_global_ns\":%llu,\"thrash_block_global_ns\":%llu,"
        "\"consumer_block_global_ns\":%llu,\"producer_event_ms\":%.6f,"
        "\"total_event_ms\":%.6f,\"copy_event_ms\":%.6f,"
        "\"host_total_ns\":%llu,\"producer_to_consumer_ns\":%llu,"
        "\"checksum\":%llu,\"expected_checksum\":%llu,\"correct\":%s}\n",
        result.trial, mode_name(result.mode), result.producer_block_global_ns,
        result.thrash_block_global_ns, result.consumer_block_global_ns,
        result.producer_event_ms, result.total_event_ms, result.copy_event_ms,
        result.host_total_ns, result.producer_to_consumer_ns, result.checksum,
        result.expected_checksum, result.correct ? "true" : "false");
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr,
                     "usage: %s MODE FOOTPRINT_MB THRASH_MB BLOCKS TRIALS\n",
                     argv[0]);
        return 2;
    }
    Mode mode;
    const int footprint_mb = std::atoi(argv[2]);
    const int thrash_mb = std::atoi(argv[3]);
    const int blocks = std::atoi(argv[4]);
    const int trials = std::atoi(argv[5]);
    if (!parse_mode(argv[1], &mode) || footprint_mb < 1 || footprint_mb > 256 ||
        thrash_mb < 0 || thrash_mb > 512 || blocks < 1 || blocks > 168 ||
        trials < 1 || trials > 31 || (trials % 2) == 0) {
        return 2;
    }
    const std::size_t footprint_bytes = static_cast<std::size_t>(footprint_mb) * 1024u * 1024u;
    const std::size_t line_count = footprint_bytes / kLineBytes;
    const std::size_t thrash_words = static_cast<std::size_t>(thrash_mb) * 1024u * 1024u / sizeof(std::uint32_t);
    std::uint32_t* data = nullptr;
    std::uint32_t* thrash = nullptr;
    unsigned long long* timestamps = nullptr;
    unsigned long long* partials = nullptr;
    std::uint32_t* host_data = nullptr;
    CUDA_CHECK(cudaMalloc(&data, line_count * kWordsPerLine * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&thrash, std::max<std::size_t>(thrash_words, 1) * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&timestamps, 6 * sizeof(*timestamps)));
    CUDA_CHECK(cudaMalloc(&partials, static_cast<std::size_t>(blocks) * kThreads * sizeof(*partials)));
    CUDA_CHECK(cudaMallocHost(&host_data, line_count * kWordsPerLine * sizeof(std::uint32_t)));
    cudaStream_t stream = nullptr;
    cudaEvent_t producer_start = nullptr;
    cudaEvent_t producer_stop = nullptr;
    cudaEvent_t total_stop = nullptr;
    cudaEvent_t copy_stop = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&producer_start));
    CUDA_CHECK(cudaEventCreate(&producer_stop));
    CUDA_CHECK(cudaEventCreate(&total_stop));
    CUDA_CHECK(cudaEventCreate(&copy_stop));

    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        (void)run_trial(-warmup - 1, mode, line_count, thrash_words, blocks,
                        stream, data, thrash, timestamps, partials, host_data,
                        producer_start, producer_stop, total_stop, copy_stop);
    }
    bool correct = true;
    for (int trial = 0; trial < trials; ++trial) {
        Trial result = run_trial(trial, mode, line_count, thrash_words, blocks,
                                 stream, data, thrash, timestamps, partials,
                                 host_data, producer_start, producer_stop,
                                 total_stop, copy_stop);
        correct = correct && result.correct;
        print_trial(result);
    }
    std::printf(
        "{\"record\":\"summary\",\"mode\":\"%s\","
        "\"footprint_mb\":%d,\"thrash_mb\":%d,\"blocks\":%d,"
        "\"trials\":%d,\"line_count\":%zu,\"correct\":%s}\n",
        mode_name(mode), footprint_mb, thrash_mb, blocks, trials, line_count,
        correct ? "true" : "false");

    CUDA_CHECK(cudaEventDestroy(copy_stop));
    CUDA_CHECK(cudaEventDestroy(total_stop));
    CUDA_CHECK(cudaEventDestroy(producer_stop));
    CUDA_CHECK(cudaEventDestroy(producer_start));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(host_data));
    CUDA_CHECK(cudaFree(partials));
    CUDA_CHECK(cudaFree(timestamps));
    CUDA_CHECK(cudaFree(thrash));
    CUDA_CHECK(cudaFree(data));
    return correct ? 0 : 1;
}
