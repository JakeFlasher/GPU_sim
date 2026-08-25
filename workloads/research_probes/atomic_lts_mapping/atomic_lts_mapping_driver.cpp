#include <cuda.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define DRIVER_CHECK(call)                                                      \
    do {                                                                        \
        const CUresult status = (call);                                         \
        if (status != CUDA_SUCCESS) {                                           \
            const char* name = nullptr;                                         \
            const char* description = nullptr;                                  \
            cuGetErrorName(status, &name);                                      \
            cuGetErrorString(status, &description);                             \
            std::fprintf(stderr, "%s failed: %s (%s)\n", #call,              \
                         name == nullptr ? "unknown" : name,                   \
                         description == nullptr ? "no description"             \
                                                : description);                 \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

constexpr int kThreads = 32;
constexpr int kMappingBlocks = 49;
constexpr int kCompatibilityMaxBlocks = 168;
constexpr int kCompatibilityMaxThreads = kThreads * kCompatibilityMaxBlocks;
constexpr std::uint32_t kAnchorGenerations = 48;
constexpr const char* kFunction = "atomic_lts_mapping_probe";
constexpr const char* kClearFunction = "atomic_lts_mapping_clear";

bool parse_u32_csv(const std::string& text, std::vector<std::uint32_t>* values) {
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty() || token.front() == '-') {
            return false;
        }
        char* end = nullptr;
        const unsigned long long value = std::strtoull(token.c_str(), &end, 10);
        if (end == token.c_str() || *end != '\0' ||
            value > std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }
        values->push_back(static_cast<std::uint32_t>(value));
    }
    return !values->empty();
}

bool load_queries(const std::string& path, std::vector<std::uint32_t>* values) {
    std::ifstream input(path);
    if (!input) {
        return false;
    }
    std::string text{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
    }
    return parse_u32_csv(text, values);
}

int main(int argc, char** argv) {
    if (argc != 8) {
        std::fprintf(
            stderr,
            "usage: %s CUBIN QUERY_REGIONS_FILE REPRESENTATIVES_CSV "
            "FIXED_DATA_BYTES ALLOCATION_PREFIX_BYTES REGION_STRIDE_BYTES "
            "VARIANT\n",
            argv[0]);
        return 2;
    }
    std::vector<std::uint32_t> query_regions;
    std::vector<std::uint32_t> representatives;
    if (!load_queries(argv[2], &query_regions) ||
        !parse_u32_csv(argv[3], &representatives)) {
        return 2;
    }
    const std::size_t fixed_data_bytes =
        static_cast<std::size_t>(std::strtoull(argv[4], nullptr, 10));
    const std::size_t allocation_prefix_bytes =
        static_cast<std::size_t>(std::strtoull(argv[5], nullptr, 10));
    const std::uint32_t region_stride_bytes =
        static_cast<std::uint32_t>(std::strtoul(argv[6], nullptr, 10));
    if (query_regions.size() > 65535 || representatives.size() != 48 ||
        std::unordered_set<std::uint32_t>(representatives.begin(),
                                          representatives.end()).size() !=
            representatives.size() ||
        fixed_data_bytes == 0 ||
        fixed_data_bytes % sizeof(std::uint32_t) != 0 ||
        allocation_prefix_bytes > (std::size_t{512} << 20) ||
        region_stride_bytes == 0 ||
        region_stride_bytes % sizeof(std::uint32_t) != 0) {
        return 2;
    }

    const std::uint32_t maximum_region = std::max(
        *std::max_element(query_regions.begin(), query_regions.end()),
        *std::max_element(representatives.begin(), representatives.end()));
    const std::uint64_t maximum_address_bytes =
        static_cast<std::uint64_t>(maximum_region) * region_stride_bytes;
    if (maximum_address_bytes >
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) *
            sizeof(std::uint32_t)) {
        return 2;
    }
    const std::size_t required_words =
        static_cast<std::size_t>(maximum_address_bytes / sizeof(std::uint32_t)) + 1;
    const std::size_t required_data_bytes = required_words * sizeof(std::uint32_t);
    if (required_data_bytes > fixed_data_bytes) {
        return 2;
    }

    std::vector<std::uint32_t> indices(kMappingBlocks * kThreads);
    std::vector<std::uint32_t> limits(kMappingBlocks);
    limits[0] = kAnchorGenerations;
    std::iota(limits.begin() + 1, limits.end(), 0u);
    const auto populate_indices = [&](std::uint32_t anchor) {
        for (int block = 0; block < kMappingBlocks; ++block) {
            const std::uint32_t region =
                block == 0 ? anchor : representatives[block - 1];
            const std::uint32_t index = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(region) * region_stride_bytes /
                sizeof(std::uint32_t));
            for (int lane = 0; lane < kThreads; ++lane) {
                indices[block * kThreads + lane] = index;
            }
        }
    };

    std::vector<std::uint32_t> zeros(fixed_data_bytes / sizeof(std::uint32_t), 0);
    std::vector<std::uint32_t> data(zeros.size());

    DRIVER_CHECK(cuInit(0));
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;
    CUfunction clear_function;
    DRIVER_CHECK(cuDeviceGet(&device, 0));
    DRIVER_CHECK(cuCtxCreate(&context, nullptr, 0, device));
    DRIVER_CHECK(cuModuleLoad(&module, argv[1]));
    DRIVER_CHECK(cuModuleGetFunction(&function, module, kFunction));
    DRIVER_CHECK(cuModuleGetFunction(&clear_function, module, kClearFunction));

    CUdeviceptr elapsed_device;
    CUdeviceptr returned_device;
    CUdeviceptr allocation_prefix_device = 0;
    CUdeviceptr data_device;
    CUdeviceptr indices_device;
    CUdeviceptr limits_device;
    DRIVER_CHECK(cuMemAlloc(&elapsed_device,
                            kCompatibilityMaxBlocks * sizeof(unsigned long long)));
    DRIVER_CHECK(cuMemAlloc(&returned_device,
                            kCompatibilityMaxThreads * sizeof(unsigned long long)));
    if (allocation_prefix_bytes != 0) {
        DRIVER_CHECK(cuMemAlloc(&allocation_prefix_device, allocation_prefix_bytes));
        DRIVER_CHECK(cuMemsetD8(allocation_prefix_device, 0,
                               allocation_prefix_bytes));
    }
    DRIVER_CHECK(cuMemAlloc(&data_device, fixed_data_bytes));
    DRIVER_CHECK(cuMemAlloc(&indices_device,
                            indices.size() * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemAlloc(&limits_device,
                            limits.size() * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemcpyHtoD(data_device, zeros.data(), fixed_data_bytes));
    DRIVER_CHECK(cuMemcpyHtoD(limits_device, limits.data(),
                              limits.size() * sizeof(std::uint32_t)));

    void* arguments[] = {&elapsed_device, &returned_device, &data_device,
                         &indices_device, &limits_device};
    void* clear_arguments[] = {&data_device, &indices_device};
    for (const std::uint32_t query : query_regions) {
        populate_indices(query);
        DRIVER_CHECK(cuMemcpyHtoD(indices_device, indices.data(),
                                  indices.size() * sizeof(std::uint32_t)));
        DRIVER_CHECK(cuLaunchKernel(clear_function, 1, 1, 1, 64, 1, 1, 0,
                                    nullptr, clear_arguments, nullptr));
        DRIVER_CHECK(cuLaunchKernel(function, kMappingBlocks, 1, 1, kThreads, 1, 1,
                                    0, nullptr, arguments, nullptr));
        DRIVER_CHECK(cuCtxSynchronize());
    }
    DRIVER_CHECK(cuMemcpyDtoH(data.data(), data_device, fixed_data_bytes));

    const std::uint64_t operations =
        static_cast<std::uint64_t>(kThreads) *
        std::accumulate(limits.begin(), limits.end(), std::uint64_t{0});
    const std::uint64_t data_sum =
        std::accumulate(data.begin(), data.end(), std::uint64_t{0});
    std::unordered_map<std::uint32_t, std::uint64_t> expected_by_region;
    for (const std::uint32_t query : query_regions) {
        expected_by_region[query] = 0;
        for (const std::uint32_t representative : representatives) {
            expected_by_region[representative] = 0;
        }
        expected_by_region[query] +=
            static_cast<std::uint64_t>(kThreads) * kAnchorGenerations;
        for (std::size_t index = 0; index < representatives.size(); ++index) {
            expected_by_region[representatives[index]] +=
                static_cast<std::uint64_t>(kThreads) * index;
        }
    }
    const std::uint64_t expected_data_sum = std::accumulate(
        expected_by_region.begin(), expected_by_region.end(), std::uint64_t{0},
        [](std::uint64_t total, const auto& entry) {
            return total + entry.second;
        });
    const bool correct = data_sum == expected_data_sum;
    std::printf(
        "{\"variant\":\"%s\",\"query_count\":%zu,"
        "\"blocks_per_query\":%d,\"anchor_generations\":%u,"
        "\"representative_generations_min\":0,"
        "\"representative_generations_max\":47,"
        "\"fixed_data_bytes\":%zu,\"required_data_bytes\":%zu,"
        "\"allocation_prefix_bytes\":%zu,\"region_stride_bytes\":%u,"
        "\"data_device_address\":%llu,\"operations\":%llu,"
        "\"data_sum\":%llu,\"expected_data_sum\":%llu,"
        "\"correct\":%s}\n",
        argv[7], query_regions.size(), kMappingBlocks, kAnchorGenerations,
        fixed_data_bytes, required_data_bytes, allocation_prefix_bytes,
        region_stride_bytes, static_cast<unsigned long long>(data_device),
        static_cast<unsigned long long>(operations),
        static_cast<unsigned long long>(data_sum),
        static_cast<unsigned long long>(expected_data_sum),
        correct ? "true" : "false");
    return correct ? 0 : 1;
}
