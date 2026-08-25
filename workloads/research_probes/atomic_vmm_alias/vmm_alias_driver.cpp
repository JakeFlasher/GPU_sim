#include <cuda.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <string>
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
constexpr int kBlocks = 168;
constexpr int kAliases = 7;
constexpr int kGenerations = 64;
constexpr std::array<int, kAliases> kGroupBlocks = {105, 1, 2, 4, 8, 16, 32};
constexpr const char* kFunction = "vmm_alias_read_probe";

enum class Mode {
    AliasSameOffset,
    AliasDistinctOffsets,
    SingleAliasSameOffset,
    SingleAliasDistinctOffsets,
    SingleMappingAliasStride,
};

bool parse_mode(const std::string& text, Mode* mode) {
    if (text == "alias_same_offset") {
        *mode = Mode::AliasSameOffset;
    } else if (text == "alias_distinct_offsets") {
        *mode = Mode::AliasDistinctOffsets;
    } else if (text == "single_alias_same_offset") {
        *mode = Mode::SingleAliasSameOffset;
    } else if (text == "single_alias_distinct_offsets") {
        *mode = Mode::SingleAliasDistinctOffsets;
    } else if (text == "single_mapping_alias_stride") {
        *mode = Mode::SingleMappingAliasStride;
    } else {
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(
            stderr,
            "usage: %s CUBIN TRIALS VARIANT MODE PHYSICAL_OFFSET_BYTES\n",
            argv[0]);
        return 2;
    }
    const int trials = std::atoi(argv[2]);
    Mode mode;
    if (!parse_mode(argv[4], &mode)) {
        return 2;
    }
    const std::size_t physical_offset =
        static_cast<std::size_t>(std::strtoull(argv[5], nullptr, 10));
    if (trials < 1 || trials > 32 || physical_offset % 128 != 0) {
        return 2;
    }

    DRIVER_CHECK(cuInit(0));
    CUdevice device;
    CUcontext context;
    DRIVER_CHECK(cuDeviceGet(&device, 0));
    DRIVER_CHECK(cuCtxCreate(&context, nullptr, 0, device));
    int vmm_supported = 0;
    DRIVER_CHECK(cuDeviceGetAttribute(
        &vmm_supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
        device));
    if (vmm_supported == 0) {
        std::fprintf(stderr, "device does not support CUDA VMM\n");
        return 2;
    }

    CUmemAllocationProp properties{};
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = device;
    properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    size_t granularity = 0;
    DRIVER_CHECK(cuMemGetAllocationGranularity(
        &granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    if (physical_offset + (kAliases - 1) * 128 + sizeof(std::uint32_t) >
        granularity) {
        return 2;
    }

    const std::size_t reservation_bytes = granularity * kAliases;
    const std::size_t physical_allocation_bytes =
        mode == Mode::SingleMappingAliasStride ? reservation_bytes : granularity;
    CUmemGenericAllocationHandle allocation;
    DRIVER_CHECK(
        cuMemCreate(&allocation, physical_allocation_bytes, &properties, 0));
    CUdeviceptr reservation = 0;
    DRIVER_CHECK(cuMemAddressReserve(&reservation, reservation_bytes, granularity,
                                     0, 0));
    std::array<CUdeviceptr, kAliases> aliases{};
    for (int index = 0; index < kAliases; ++index) {
        aliases[index] = reservation + static_cast<std::size_t>(index) * granularity;
    }
    if (mode == Mode::SingleMappingAliasStride) {
        DRIVER_CHECK(
            cuMemMap(reservation, reservation_bytes, 0, allocation, 0));
    } else {
        for (const CUdeviceptr alias : aliases) {
            DRIVER_CHECK(cuMemMap(alias, granularity, 0, allocation, 0));
        }
    }
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    DRIVER_CHECK(cuMemSetAccess(reservation, reservation_bytes, &access, 1));
    DRIVER_CHECK(cuMemsetD8(aliases[0], 1, physical_allocation_bytes));
    DRIVER_CHECK(cuCtxSynchronize());

    std::array<CUdeviceptr, kAliases> group_addresses{};
    for (int group = 0; group < kAliases; ++group) {
        if (mode == Mode::AliasSameOffset) {
            group_addresses[group] = aliases[group] + physical_offset;
        } else if (mode == Mode::AliasDistinctOffsets) {
            group_addresses[group] =
                aliases[group] + physical_offset + group * 128;
        } else if (mode == Mode::SingleAliasSameOffset) {
            group_addresses[group] = aliases[0] + physical_offset;
        } else if (mode == Mode::SingleAliasDistinctOffsets) {
            group_addresses[group] =
                aliases[0] + physical_offset + group * 128;
        } else {
            group_addresses[group] = aliases[group] + physical_offset;
        }
    }

    std::vector<unsigned long long> addresses(kBlocks * kThreads);
    int block = 0;
    for (int group = 0; group < kAliases; ++group) {
        for (int count = 0; count < kGroupBlocks[group]; ++count, ++block) {
            for (int lane = 0; lane < kThreads; ++lane) {
                addresses[block * kThreads + lane] = group_addresses[group];
            }
        }
    }
    if (block != kBlocks) {
        return 2;
    }

    CUmodule module;
    CUfunction function;
    DRIVER_CHECK(cuModuleLoad(&module, argv[1]));
    DRIVER_CHECK(cuModuleGetFunction(&function, module, kFunction));
    CUdeviceptr elapsed_device;
    CUdeviceptr returned_device;
    CUdeviceptr addresses_device;
    DRIVER_CHECK(cuMemAlloc(&elapsed_device,
                            kBlocks * sizeof(unsigned long long)));
    DRIVER_CHECK(cuMemAlloc(&returned_device,
                            addresses.size() * sizeof(unsigned long long)));
    DRIVER_CHECK(cuMemAlloc(&addresses_device,
                            addresses.size() * sizeof(unsigned long long)));
    DRIVER_CHECK(cuMemcpyHtoD(addresses_device, addresses.data(),
                              addresses.size() * sizeof(unsigned long long)));
    void* arguments[] = {&elapsed_device, &returned_device, &addresses_device};

    std::vector<unsigned long long> elapsed(kBlocks);
    std::vector<unsigned long long> returned(addresses.size());
    std::vector<unsigned long long> sample_maxima;
    for (int trial = 0; trial < trials; ++trial) {
        DRIVER_CHECK(cuLaunchKernel(function, kBlocks, 1, 1, kThreads, 1, 1, 0,
                                    nullptr, arguments, nullptr));
        DRIVER_CHECK(cuCtxSynchronize());
        DRIVER_CHECK(cuMemcpyDtoH(elapsed.data(), elapsed_device,
                                  elapsed.size() * sizeof(unsigned long long)));
        sample_maxima.push_back(
            *std::max_element(elapsed.begin(), elapsed.end()));
    }
    DRIVER_CHECK(cuMemcpyDtoH(returned.data(), returned_device,
                              returned.size() * sizeof(unsigned long long)));
    const unsigned long long word = 0x0101'0101ull;
    const unsigned long long expected_per_thread = word * kGenerations;
    const bool correct = std::all_of(
        returned.begin(), returned.end(),
        [expected_per_thread](unsigned long long value) {
            return value == expected_per_thread;
        });

    std::printf(
        "{\"variant\":\"%s\",\"mode\":\"%s\",\"trials\":%d,"
        "\"blocks\":%d,\"generations\":%d,\"vmm_supported\":true,"
        "\"allocation_granularity_bytes\":%zu,"
        "\"physical_allocation_bytes\":%zu,\"reservation_bytes\":%zu,"
        "\"reservation_address\":%llu,\"physical_offset_bytes\":%zu,"
        "\"alias_addresses\":[",
        argv[3], argv[4], trials, kBlocks, kGenerations, granularity,
        physical_allocation_bytes,
        reservation_bytes, static_cast<unsigned long long>(reservation),
        physical_offset);
    for (int index = 0; index < kAliases; ++index) {
        std::printf("%s%llu", index == 0 ? "" : ",",
                    static_cast<unsigned long long>(aliases[index]));
    }
    std::printf("],\"group_addresses\":[");
    for (int index = 0; index < kAliases; ++index) {
        std::printf("%s%llu", index == 0 ? "" : ",",
                    static_cast<unsigned long long>(group_addresses[index]));
    }
    std::printf("],\"group_block_counts\":[");
    for (int index = 0; index < kAliases; ++index) {
        std::printf("%s%d", index == 0 ? "" : ",", kGroupBlocks[index]);
    }
    std::printf("],\"samples_max_cycles\":[");
    for (int trial = 0; trial < trials; ++trial) {
        std::printf("%s%llu", trial == 0 ? "" : ",", sample_maxima[trial]);
    }
    std::printf(
        "],\"returned_value_per_thread\":%llu,\"correct\":%s}\n",
        expected_per_thread, correct ? "true" : "false");

    DRIVER_CHECK(cuMemFree(addresses_device));
    DRIVER_CHECK(cuMemFree(returned_device));
    DRIVER_CHECK(cuMemFree(elapsed_device));
    DRIVER_CHECK(cuModuleUnload(module));
    if (mode == Mode::SingleMappingAliasStride) {
        DRIVER_CHECK(cuMemUnmap(reservation, reservation_bytes));
    } else {
        for (const CUdeviceptr alias : aliases) {
            DRIVER_CHECK(cuMemUnmap(alias, granularity));
        }
    }
    DRIVER_CHECK(cuMemAddressFree(reservation, reservation_bytes));
    DRIVER_CHECK(cuMemRelease(allocation));
    DRIVER_CHECK(cuCtxDestroy(context));
    return correct ? 0 : 1;
}
