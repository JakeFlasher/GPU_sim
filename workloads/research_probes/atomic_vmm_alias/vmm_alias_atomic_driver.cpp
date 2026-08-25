#include <cuda.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
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
constexpr int kBlocks = 49;
constexpr int kAliases = 7;
constexpr int kValidationLaunches = 48;
constexpr std::uint32_t kAnchorGenerations = 48;
constexpr std::size_t kReferenceBytes = 32 * 1024 * 1024;
constexpr std::array<std::uint32_t, 48> kRepresentatives = {
    0,   2,   8,   10,  16,  18,  24,  26,  32,  34,  40,  42,
    48,  50,  56,  58,  80,  82,  88,  90,  96,  98,  104, 106,
    128, 130, 136, 138, 144, 146, 152, 154, 160, 162, 168, 170,
    176, 178, 184, 186, 208, 210, 216, 218, 224, 226, 232, 234};
constexpr const char* kFunction = "atomic_vmm_mapping_probe";
constexpr const char* kClearFunction = "atomic_vmm_mapping_clear";

bool load_offsets(const std::string& path, std::vector<std::size_t>* offsets) {
    std::ifstream input(path);
    if (!input) {
        return false;
    }
    std::string text{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty() || token.front() == '-') {
            return false;
        }
        char* end = nullptr;
        const unsigned long long value = std::strtoull(token.c_str(), &end, 10);
        if (end == token.c_str() || (*end != '\0' && *end != '\n' && *end != '\r')) {
            return false;
        }
        offsets->push_back(static_cast<std::size_t>(value));
    }
    return !offsets->empty() && offsets->size() <= 65'000;
}

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        std::fprintf(stderr,
                     "usage: %s CUBIN PHYSICAL_OFFSET_BYTES VARIANT "
                     "[PHYSICAL_OFFSETS_FILE]\n",
                     argv[0]);
        return 2;
    }
    const std::size_t physical_offset =
        static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
    std::vector<std::size_t> mapping_offsets;
    if (argc == 5) {
        if (!load_offsets(argv[4], &mapping_offsets)) {
            return 2;
        }
    } else {
        mapping_offsets.assign(kAliases, physical_offset);
    }
    if (physical_offset % 128 != 0 ||
        std::any_of(mapping_offsets.begin(), mapping_offsets.end(),
                    [](std::size_t offset) { return offset % 128 != 0; })) {
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
    const std::size_t maximum_offset =
        *std::max_element(mapping_offsets.begin(), mapping_offsets.end());
    if (maximum_offset >
            std::numeric_limits<std::size_t>::max() - sizeof(std::uint32_t) ||
        physical_offset >
            std::numeric_limits<std::size_t>::max() - sizeof(std::uint32_t)) {
        return 2;
    }
    const std::size_t required_bytes =
        maximum_offset + sizeof(std::uint32_t);
    if (required_bytes >
        std::numeric_limits<std::size_t>::max() - (granularity - 1)) {
        return 2;
    }
    const std::size_t physical_allocation_bytes =
        ((required_bytes + granularity - 1) / granularity) * granularity;
    if (physical_offset + sizeof(std::uint32_t) > physical_allocation_bytes ||
        required_bytes > physical_allocation_bytes) {
        return 2;
    }
    if (physical_allocation_bytes >
        std::numeric_limits<std::size_t>::max() / kAliases) {
        return 2;
    }
    CUmemGenericAllocationHandle allocation;
    DRIVER_CHECK(
        cuMemCreate(&allocation, physical_allocation_bytes, &properties, 0));
    const std::size_t reservation_bytes = physical_allocation_bytes * kAliases;
    CUdeviceptr reservation = 0;
    DRIVER_CHECK(cuMemAddressReserve(&reservation, reservation_bytes, granularity,
                                     0, 0));
    std::array<CUdeviceptr, kAliases> aliases{};
    for (int index = 0; index < kAliases; ++index) {
        aliases[index] =
            reservation + static_cast<std::size_t>(index) * physical_allocation_bytes;
        DRIVER_CHECK(
            cuMemMap(aliases[index], physical_allocation_bytes, 0, allocation, 0));
    }
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    DRIVER_CHECK(cuMemSetAccess(reservation, reservation_bytes, &access, 1));
    DRIVER_CHECK(cuMemsetD8(aliases[0], 0, physical_allocation_bytes));
    DRIVER_CHECK(cuCtxSynchronize());

    CUdeviceptr reference_device;
    DRIVER_CHECK(cuMemAlloc(&reference_device, kReferenceBytes));
    DRIVER_CHECK(cuMemsetD8(reference_device, 0, kReferenceBytes));

    CUmodule module;
    CUfunction function;
    CUfunction clear_function;
    DRIVER_CHECK(cuModuleLoad(&module, argv[1]));
    DRIVER_CHECK(cuModuleGetFunction(&function, module, kFunction));
    DRIVER_CHECK(cuModuleGetFunction(&clear_function, module, kClearFunction));

    std::vector<unsigned long long> addresses(kBlocks * kThreads);
    std::array<std::uint32_t, kBlocks> limits{};
    limits[0] = kAnchorGenerations;
    std::iota(limits.begin() + 1, limits.end(), 0u);
    CUdeviceptr elapsed_device;
    CUdeviceptr returned_device;
    CUdeviceptr addresses_device;
    CUdeviceptr limits_device;
    DRIVER_CHECK(cuMemAlloc(&elapsed_device,
                            kBlocks * sizeof(unsigned long long)));
    DRIVER_CHECK(cuMemAlloc(&returned_device,
                            addresses.size() * sizeof(unsigned long long)));
    DRIVER_CHECK(cuMemAlloc(&addresses_device,
                            addresses.size() * sizeof(unsigned long long)));
    DRIVER_CHECK(cuMemAlloc(&limits_device,
                            limits.size() * sizeof(std::uint32_t)));
    DRIVER_CHECK(cuMemcpyHtoD(limits_device, limits.data(),
                              limits.size() * sizeof(std::uint32_t)));

    const auto configure_addresses = [&](CUdeviceptr target) {
        for (int block = 0; block < kBlocks; ++block) {
            const CUdeviceptr address = block == 0
                ? target
                : reference_device +
                      static_cast<std::size_t>(kRepresentatives[block - 1]) * 128;
            for (int lane = 0; lane < kThreads; ++lane) {
                addresses[block * kThreads + lane] = address;
            }
        }
        DRIVER_CHECK(cuMemcpyHtoD(addresses_device, addresses.data(),
                                  addresses.size() * sizeof(unsigned long long)));
    };
    void* clear_arguments[] = {&addresses_device};
    void* arguments[] = {&elapsed_device, &returned_device, &addresses_device,
                         &limits_device};
    const auto launch = [&](CUdeviceptr target) {
        configure_addresses(target);
        DRIVER_CHECK(cuLaunchKernel(clear_function, 1, 1, 1, 64, 1, 1, 0,
                                    nullptr, clear_arguments, nullptr));
        DRIVER_CHECK(cuLaunchKernel(function, kBlocks, 1, 1, kThreads, 1, 1, 0,
                                    nullptr, arguments, nullptr));
        DRIVER_CHECK(cuCtxSynchronize());
    };

    for (int index = 0; index < kValidationLaunches; ++index) {
        launch(reference_device +
               static_cast<std::size_t>(kRepresentatives[index]) * 128);
    }
    for (std::size_t index = 0; index < mapping_offsets.size(); ++index) {
        launch(aliases[index % aliases.size()] + mapping_offsets[index]);
    }

    std::vector<unsigned long long> returned(addresses.size());
    DRIVER_CHECK(cuMemcpyDtoH(returned.data(), returned_device,
                              returned.size() * sizeof(unsigned long long)));
    bool correct = std::all_of(returned.begin(), returned.end(),
                               [](unsigned long long value) { return value == 0; });
    std::uint32_t target_value = 0;
    const std::size_t final_mapping_index = mapping_offsets.size() - 1;
    DRIVER_CHECK(cuMemcpyDtoH(
        &target_value,
        aliases[final_mapping_index % aliases.size()] + mapping_offsets.back(),
        sizeof(target_value)));
    correct = correct && target_value == kThreads * kAnchorGenerations;
    for (std::size_t index = 0; index < kRepresentatives.size(); ++index) {
        std::uint32_t value = 0;
        DRIVER_CHECK(cuMemcpyDtoH(
            &value,
            reference_device +
                static_cast<std::size_t>(kRepresentatives[index]) * 128,
            sizeof(value)));
        correct = correct && value == kThreads * index;
    }

    std::printf(
        "{\"variant\":\"%s\",\"vmm_supported\":true,"
        "\"allocation_granularity_bytes\":%zu,"
        "\"physical_allocation_bytes\":%zu,\"reservation_bytes\":%zu,"
        "\"reservation_address\":%llu,\"physical_offset_bytes\":%zu,"
        "\"reference_address\":%llu,\"validation_launches\":%d,"
        "\"alias_launches\":%zu,\"mapping_launches\":%zu,"
        "\"batch_offsets\":%s,"
        "\"alias_addresses\":[",
        argv[3], granularity, physical_allocation_bytes, reservation_bytes,
        static_cast<unsigned long long>(reservation), physical_offset,
        static_cast<unsigned long long>(reference_device), kValidationLaunches,
        mapping_offsets.size(), kValidationLaunches + mapping_offsets.size(),
        argc == 5 ? "true" : "false");
    for (int index = 0; index < kAliases; ++index) {
        std::printf("%s%llu", index == 0 ? "" : ",",
                    static_cast<unsigned long long>(aliases[index]));
    }
    std::printf("],\"representative_regions\":[");
    for (std::size_t index = 0; index < kRepresentatives.size(); ++index) {
        std::printf("%s%u", index == 0 ? "" : ",", kRepresentatives[index]);
    }
    std::printf("],\"mapping_offsets_bytes\":[");
    for (std::size_t index = 0; index < mapping_offsets.size(); ++index) {
        std::printf("%s%zu", index == 0 ? "" : ",", mapping_offsets[index]);
    }
    std::printf(
        "],\"target_value\":%u,\"correct\":%s}\n", target_value,
        correct ? "true" : "false");

    DRIVER_CHECK(cuMemFree(limits_device));
    DRIVER_CHECK(cuMemFree(addresses_device));
    DRIVER_CHECK(cuMemFree(returned_device));
    DRIVER_CHECK(cuMemFree(elapsed_device));
    DRIVER_CHECK(cuModuleUnload(module));
    DRIVER_CHECK(cuMemFree(reference_device));
    for (const CUdeviceptr alias : aliases) {
        DRIVER_CHECK(cuMemUnmap(alias, physical_allocation_bytes));
    }
    DRIVER_CHECK(cuMemAddressFree(reservation, reservation_bytes));
    DRIVER_CHECK(cuMemRelease(allocation));
    DRIVER_CHECK(cuCtxDestroy(context));
    return correct ? 0 : 1;
}
