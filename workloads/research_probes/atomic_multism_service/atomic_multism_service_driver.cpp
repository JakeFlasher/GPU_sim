#include <cuda.h>

#include <algorithm>
#include <cerrno>
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
constexpr int kMaxBlocks = 168;
constexpr int kMaxThreads = kThreads * kMaxBlocks;
constexpr const char* kFunction = "_Z20atomic_multism_probePyS_PjPKj";

bool parse_int(const char* text, int* value) {
    if (text == nullptr || *text == '\0' || value == nullptr) return false;
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) return false;
    *value = static_cast<int>(parsed);
    return true;
}

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

int main(int argc, char** argv) {
    if (argc < 9 || argc > 17) {
        std::fprintf(
            stderr,
            "usage: %s CUBIN BLOCKS GENERATIONS TRIALS VARIANT RETURNING "
            "SPACING_BYTES PERMUTED [ADDRESS_GROUPS [BASE_BYTES "
            "[FIXED_DATA_BYTES [REGION_IDS_CSV "
            "[GROUP_BLOCK_COUNTS_CSV [ALLOCATION_PREFIX_BYTES "
            "[BATCH_REGION_IDS_SEMICOLON_CSV "
            "[REGION_STRIDE_BYTES]]]]]]]]\n",
            argv[0]);
        return 2;
    }
    int blocks = 0;
    int generations = 0;
    int trials = 0;
    int returning_int = 0;
    if (!parse_int(argv[2], &blocks) || !parse_int(argv[3], &generations) ||
        !parse_int(argv[4], &trials) || !parse_int(argv[6], &returning_int)) {
        return 2;
    }
    const bool returning = returning_int != 0;
    const std::uint32_t spacing_bytes =
        static_cast<std::uint32_t>(std::strtoul(argv[7], nullptr, 10));
    const bool permuted = std::atoi(argv[8]) != 0;
    const int address_groups = argc >= 10 ? std::atoi(argv[9]) : blocks;
    const std::uint32_t base_bytes = argc >= 11
        ? static_cast<std::uint32_t>(std::strtoul(argv[10], nullptr, 10))
        : 0;
    const std::size_t fixed_data_bytes = argc >= 12
        ? static_cast<std::size_t>(std::strtoull(argv[11], nullptr, 10))
        : 0;
    std::vector<std::uint32_t> region_ids;
    if (argc >= 13 && !parse_u32_csv(argv[12], &region_ids)) {
        return 2;
    }
    std::vector<std::uint32_t> group_block_counts;
    if (argc >= 14 && !parse_u32_csv(argv[13], &group_block_counts)) {
        return 2;
    }
    if (std::any_of(group_block_counts.begin(), group_block_counts.end(),
                    [](std::uint32_t value) { return value == 0; })) {
        return 2;
    }
    const std::size_t allocation_prefix_bytes = argc >= 15
        ? static_cast<std::size_t>(std::strtoull(argv[14], nullptr, 10))
        : 0;
    std::vector<std::vector<std::uint32_t>> batch_region_ids;
    if (argc >= 16) {
        std::string encoded_queries = argv[15];
        if (!encoded_queries.empty() && encoded_queries.front() == '@') {
            std::ifstream input(encoded_queries.substr(1));
            if (!input) {
                return 2;
            }
            encoded_queries.assign(
                std::istreambuf_iterator<char>(input),
                std::istreambuf_iterator<char>());
            while (!encoded_queries.empty() &&
                   (encoded_queries.back() == '\n' ||
                    encoded_queries.back() == '\r')) {
                encoded_queries.pop_back();
            }
        }
        std::stringstream stream(encoded_queries);
        std::string query;
        while (std::getline(stream, query, ';')) {
            std::vector<std::uint32_t> values;
            if (!parse_u32_csv(query, &values)) {
                return 2;
            }
            batch_region_ids.push_back(std::move(values));
        }
    }
    const std::uint32_t region_stride_bytes = argc >= 17
        ? static_cast<std::uint32_t>(std::strtoul(argv[16], nullptr, 10))
        : 256u;
    const std::uint64_t configured_blocks =
        group_block_counts.empty()
        ? 0
        : std::accumulate(group_block_counts.begin(), group_block_counts.end(),
                          std::uint64_t{0});
    if (blocks < 1 || blocks > kMaxBlocks ||
        generations != 1024 || trials < 1 ||
        (returning_int != 0 && returning_int != 1) ||
        (batch_region_ids.empty() ? trials > 32 : trials > 65535) ||
        spacing_bytes % sizeof(std::uint32_t) != 0 ||
        base_bytes % sizeof(std::uint32_t) != 0 || address_groups < 1 ||
        region_stride_bytes == 0 ||
        region_stride_bytes % sizeof(std::uint32_t) != 0 ||
        address_groups > blocks ||
        (group_block_counts.empty() && blocks % address_groups != 0) ||
        fixed_data_bytes % sizeof(std::uint32_t) != 0 ||
        (!region_ids.empty() &&
         region_ids.size() != static_cast<std::size_t>(address_groups)) ||
        (!region_ids.empty() &&
         std::unordered_set<std::uint32_t>(region_ids.begin(), region_ids.end()).size() !=
             region_ids.size()) ||
        (!group_block_counts.empty() &&
         group_block_counts.size() != static_cast<std::size_t>(address_groups)) ||
        (!group_block_counts.empty() &&
         configured_blocks != static_cast<std::uint64_t>(blocks)) ||
        allocation_prefix_bytes > (std::size_t{512} << 20) ||
        (!batch_region_ids.empty() &&
         batch_region_ids.size() != static_cast<std::size_t>(trials)) ||
        (!batch_region_ids.empty() && batch_region_ids.front() != region_ids) ||
        std::any_of(
            batch_region_ids.begin(), batch_region_ids.end(),
            [address_groups](const std::vector<std::uint32_t>& values) {
                return values.size() != static_cast<std::size_t>(address_groups) ||
                       std::unordered_set<std::uint32_t>(values.begin(), values.end())
                               .size() != values.size();
            })) {
        return 2;
    }
    std::vector<std::uint32_t> group_ends;
    if (!group_block_counts.empty()) {
        std::uint32_t end = 0;
        group_ends.reserve(group_block_counts.size());
        for (const std::uint32_t count : group_block_counts) {
            end += count;
            group_ends.push_back(end);
        }
    }
    const int threads = blocks * kThreads;
    std::vector<std::uint32_t> indices(threads);
    const auto populate_indices = [&](const std::vector<std::uint32_t>& regions) {
        for (int block = 0; block < blocks; ++block) {
            const std::uint32_t logical_block = permuted
                ? (static_cast<std::uint32_t>(block) * 67u) %
                      static_cast<std::uint32_t>(blocks)
                : static_cast<std::uint32_t>(block);
            const std::uint32_t group = group_block_counts.empty()
                ? logical_block % static_cast<std::uint32_t>(address_groups)
                : static_cast<std::uint32_t>(
                      std::upper_bound(group_ends.begin(), group_ends.end(),
                                       logical_block) -
                      group_ends.begin());
            const std::uint64_t group_bytes = regions.empty()
                ? static_cast<std::uint64_t>(group) * spacing_bytes
                : static_cast<std::uint64_t>(regions[group]) *
                      region_stride_bytes;
            const std::uint64_t address_bytes =
                static_cast<std::uint64_t>(base_bytes) + group_bytes;
            const std::uint32_t index = static_cast<std::uint32_t>(
                address_bytes / sizeof(std::uint32_t));
            for (int lane = 0; lane < kThreads; ++lane) {
                indices[block * kThreads + lane] = index;
            }
        }
    };
    std::uint64_t maximum_address_bytes = base_bytes;
    const auto observe_maximum_address = [&](
                                             const std::vector<std::uint32_t>& regions) {
        if (regions.empty()) {
            maximum_address_bytes = std::max(
                maximum_address_bytes,
                static_cast<std::uint64_t>(base_bytes) +
                    static_cast<std::uint64_t>(address_groups - 1) * spacing_bytes);
        } else {
            for (const std::uint32_t region : regions) {
                maximum_address_bytes = std::max(
                    maximum_address_bytes,
                    static_cast<std::uint64_t>(base_bytes) +
                        static_cast<std::uint64_t>(region) *
                            region_stride_bytes);
            }
        }
    };
    observe_maximum_address(region_ids);
    for (const auto& regions : batch_region_ids) {
        observe_maximum_address(regions);
    }
    if (maximum_address_bytes >
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) *
            sizeof(std::uint32_t)) {
        return 2;
    }
    const std::uint32_t maximum_index = static_cast<std::uint32_t>(
        maximum_address_bytes / sizeof(std::uint32_t));
    populate_indices(region_ids);
    const std::size_t required_words =
        static_cast<std::size_t>(maximum_index) + 1;
    const std::size_t required_data_bytes =
        required_words * sizeof(std::uint32_t);
    const std::size_t data_bytes =
        std::max(required_data_bytes, fixed_data_bytes);
    if (data_bytes > (std::size_t{512} << 20)) return 3;
    const std::size_t words = data_bytes / sizeof(std::uint32_t);
    std::vector<unsigned long long> elapsed(blocks);
    std::vector<std::vector<unsigned long long>> samples(
        trials, std::vector<unsigned long long>(blocks));
    std::vector<unsigned long long> returned(threads);
    std::vector<std::uint32_t> zeros(words, 0);
    std::vector<std::uint32_t> data(words);

    DRIVER_CHECK(cuInit(0));
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;
    DRIVER_CHECK(cuDeviceGet(&device, 0));
    DRIVER_CHECK(cuCtxCreate(&context, nullptr, 0, device));
    DRIVER_CHECK(cuModuleLoad(&module, argv[1]));
    DRIVER_CHECK(cuModuleGetFunction(&function, module, kFunction));
    CUdeviceptr elapsed_device;
    CUdeviceptr returned_device;
    CUdeviceptr data_device;
    CUdeviceptr indices_device;
    DRIVER_CHECK(cuMemAlloc(&elapsed_device,
                            kMaxBlocks * sizeof(unsigned long long)));
    DRIVER_CHECK(cuMemAlloc(&returned_device,
                            kMaxThreads * sizeof(unsigned long long)));
    CUdeviceptr allocation_prefix_device = 0;
    if (allocation_prefix_bytes != 0) {
        DRIVER_CHECK(cuMemAlloc(&allocation_prefix_device,
                                allocation_prefix_bytes));
        DRIVER_CHECK(cuMemsetD8(allocation_prefix_device, 0,
                               allocation_prefix_bytes));
    }
    DRIVER_CHECK(cuMemAlloc(&data_device, data_bytes));
    DRIVER_CHECK(cuMemAlloc(&indices_device,
                            indices.size() * sizeof(std::uint32_t)));
    if (batch_region_ids.empty()) {
        DRIVER_CHECK(cuMemcpyHtoD(indices_device, indices.data(),
                                  indices.size() * sizeof(std::uint32_t)));
    } else {
        DRIVER_CHECK(cuMemcpyHtoD(data_device, zeros.data(), data_bytes));
    }
    void* arguments[] = {
        &elapsed_device, &returned_device, &data_device, &indices_device};
    bool elapsed_valid = true;
    for (int trial = 0; trial < trials; ++trial) {
        DRIVER_CHECK(cuMemsetD8(elapsed_device, 0xff,
                                kMaxBlocks * sizeof(unsigned long long)));
        DRIVER_CHECK(cuMemsetD8(returned_device, 0xa5,
                                kMaxThreads * sizeof(unsigned long long)));
        if (batch_region_ids.empty()) {
            DRIVER_CHECK(cuMemcpyHtoD(data_device, zeros.data(), data_bytes));
        } else {
            populate_indices(batch_region_ids[trial]);
            DRIVER_CHECK(cuMemcpyHtoD(indices_device, indices.data(),
                                      indices.size() * sizeof(std::uint32_t)));
            DRIVER_CHECK(cuMemsetD32(data_device, 0, required_words));
        }
        DRIVER_CHECK(cuLaunchKernel(function, blocks, 1, 1, kThreads, 1, 1, 0,
                                    nullptr, arguments, nullptr));
        DRIVER_CHECK(cuCtxSynchronize());
        DRIVER_CHECK(cuMemcpyDtoH(samples[trial].data(), elapsed_device,
                                  blocks * sizeof(unsigned long long)));
        for (int block = 0; block < blocks; ++block) {
            elapsed_valid = elapsed_valid && samples[trial][block] != UINT64_MAX &&
                            samples[trial][block] > 0;
        }
    }
    DRIVER_CHECK(cuMemcpyDtoH(returned.data(), returned_device,
                              threads * sizeof(unsigned long long)));
    DRIVER_CHECK(cuMemcpyDtoH(data.data(), data_device, data_bytes));

    const std::uint64_t operations =
        static_cast<std::uint64_t>(threads) * generations;
    std::uint64_t data_sum = 0;
    for (std::uint32_t value : data) {
        data_sum += value;
    }
    std::uint64_t returned_total = 0;
    for (unsigned long long value : returned) {
        returned_total += value;
    }
    std::unordered_map<std::uint32_t, std::uint64_t> threads_per_word;
    for (std::uint32_t index : indices) {
        ++threads_per_word[index];
    }
    std::uint64_t expected_returned = 0;
    if (returning) {
        for (const auto& [index, count] : threads_per_word) {
            (void)index;
            const std::uint64_t group_operations = count * generations;
            expected_returned +=
                group_operations * (group_operations - 1) / 2;
        }
    }
    const bool returned_present = !returning ||
        std::all_of(returned.begin(), returned.end(), [](unsigned long long value) {
            return value != 0xa5a5a5a5a5a5a5a5ull;
        });
    const bool correct = data_sum == operations && returned_total == expected_returned &&
                         elapsed_valid && returned_present;

    std::printf(
        "{\"variant\":\"%s\",\"blocks\":%d,\"warps\":%d,"
        "\"generations\":%d,\"returning\":%s,"
        "\"spacing_bytes\":%u,\"permuted\":%s,\"address_groups\":%d,"
        "\"base_bytes\":%u",
        argv[5], blocks, blocks, generations, returning ? "true" : "false",
        spacing_bytes, permuted ? "true" : "false", address_groups, base_bytes);
    if (fixed_data_bytes != 0) {
        std::printf(
            ",\"fixed_data_bytes\":%zu,\"required_data_bytes\":%zu,"
            "\"data_device_address\":%llu,\"data_device_region_256\":%llu",
            fixed_data_bytes, required_data_bytes,
            static_cast<unsigned long long>(data_device),
            static_cast<unsigned long long>(data_device >> 8));
    }
    if (allocation_prefix_bytes != 0) {
        std::printf(",\"allocation_prefix_bytes\":%zu",
                    allocation_prefix_bytes);
    }
    if (region_stride_bytes != 256u) {
        std::printf(",\"region_stride_bytes\":%u", region_stride_bytes);
    }
    if (!region_ids.empty()) {
        std::printf(",\"region_ids\":[");
        for (std::size_t index = 0; index < region_ids.size(); ++index) {
            std::printf("%s%u", index == 0 ? "" : ",", region_ids[index]);
        }
        std::printf("]");
    }
    if (!group_block_counts.empty()) {
        std::printf(",\"group_block_counts\":[");
        for (std::size_t index = 0; index < group_block_counts.size(); ++index) {
            std::printf("%s%u", index == 0 ? "" : ",",
                        group_block_counts[index]);
        }
        std::printf("]");
    }
    if (!batch_region_ids.empty()) {
        std::printf(",\"batch_region_ids\":[");
        for (std::size_t query = 0; query < batch_region_ids.size(); ++query) {
            std::printf("%s[", query == 0 ? "" : ",");
            for (std::size_t group = 0; group < batch_region_ids[query].size();
                 ++group) {
                std::printf("%s%u", group == 0 ? "" : ",",
                            batch_region_ids[query][group]);
            }
            std::printf("]");
        }
        std::printf("]");
        std::printf(",\"samples_max_cycles\":[");
        for (int trial = 0; trial < trials; ++trial) {
            const unsigned long long maximum = *std::max_element(
                samples[trial].begin(), samples[trial].end());
            std::printf("%s%llu", trial == 0 ? "" : ",", maximum);
        }
        std::printf("]");
    } else {
        std::printf(",\"samples_cycles\":[");
        for (int trial = 0; trial < trials; ++trial) {
            std::printf("%s[", trial == 0 ? "" : ",");
            for (int block = 0; block < blocks; ++block) {
                std::printf("%s%llu", block == 0 ? "" : ",",
                            samples[trial][block]);
            }
            std::printf("]");
        }
        std::printf("]");
    }
    if (!batch_region_ids.empty()) {
        std::printf(",\"kernel_launches\":%d,\"total_operations\":%llu",
                    trials,
                    static_cast<unsigned long long>(operations) * trials);
    }
    std::printf(
        ",\"operations\":%llu,\"data_sum\":%llu,"
        "\"returned_sum\":%llu,\"expected_returned_sum\":%llu,"
        "\"unique_words\":%zu,\"correct\":%s,\"elapsed_samples_valid\":%s,"
        "\"returned_output_present\":%s,\"allocation_bytes\":%zu}\n",
        static_cast<unsigned long long>(operations),
        static_cast<unsigned long long>(data_sum),
        static_cast<unsigned long long>(returned_total),
        static_cast<unsigned long long>(expected_returned),
        threads_per_word.size(), correct ? "true" : "false",
        elapsed_valid ? "true" : "false", returned_present ? "true" : "false",
        data_bytes);
    DRIVER_CHECK(cuMemFree(indices_device));
    DRIVER_CHECK(cuMemFree(data_device));
    DRIVER_CHECK(cuMemFree(returned_device));
    DRIVER_CHECK(cuMemFree(elapsed_device));
    if (allocation_prefix_device != 0) DRIVER_CHECK(cuMemFree(allocation_prefix_device));
    DRIVER_CHECK(cuModuleUnload(module));
    DRIVER_CHECK(cuCtxDestroy(context));
    return correct ? 0 : 1;
}
