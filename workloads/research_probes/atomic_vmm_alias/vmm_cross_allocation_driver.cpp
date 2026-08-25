#include <cuda.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
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
constexpr int kAllocations = 8;
constexpr int kValidationLaunches = 48;
constexpr int kTargetsPerAllocation = 48;
constexpr int kTargetLaunches = kAllocations * kTargetsPerAllocation;
constexpr int kLaunches = kValidationLaunches + kTargetLaunches;
constexpr std::uint32_t kAnchorGenerations = 48;
constexpr std::size_t kReferenceBytes = 32 * 1024 * 1024;
constexpr std::array<std::uint32_t, 48> kRepresentatives = {
    0,   2,   8,   10,  16,  18,  24,  26,  32,  34,  40,  42,
    48,  50,  56,  58,  80,  82,  88,  90,  96,  98,  104, 106,
    128, 130, 136, 138, 144, 146, 152, 154, 160, 162, 168, 170,
    176, 178, 184, 186, 208, 210, 216, 218, 224, 226, 232, 234};
constexpr const char* kFunction = "atomic_vmm_mapping_probe";
constexpr const char* kClearFunction = "atomic_vmm_mapping_clear";

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s CUBIN VARIANT\n", argv[0]);
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
    const std::size_t reservation_bytes = granularity * kAllocations;
    CUdeviceptr reservation = 0;
    DRIVER_CHECK(cuMemAddressReserve(&reservation, reservation_bytes, granularity,
                                     0, 0));
    std::array<CUmemGenericAllocationHandle, kAllocations> allocations{};
    std::array<CUdeviceptr, kAllocations> allocation_addresses{};
    for (int index = 0; index < kAllocations; ++index) {
        DRIVER_CHECK(cuMemCreate(&allocations[index], granularity, &properties, 0));
        allocation_addresses[index] =
            reservation + static_cast<std::size_t>(index) * granularity;
        DRIVER_CHECK(cuMemMap(allocation_addresses[index], granularity, 0,
                              allocations[index], 0));
    }
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    DRIVER_CHECK(cuMemSetAccess(reservation, reservation_bytes, &access, 1));
    for (const CUdeviceptr address : allocation_addresses) {
        DRIVER_CHECK(cuMemsetD8(address, 0, granularity));
    }
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
    for (const std::uint32_t region : kRepresentatives) {
        launch(reference_device + static_cast<std::size_t>(region) * 128);
    }
    for (const CUdeviceptr allocation_address : allocation_addresses) {
        for (const std::uint32_t region : kRepresentatives) {
            launch(allocation_address + static_cast<std::size_t>(region) * 128);
        }
    }

    std::vector<unsigned long long> returned(addresses.size());
    DRIVER_CHECK(cuMemcpyDtoH(returned.data(), returned_device,
                              returned.size() * sizeof(unsigned long long)));
    bool correct = std::all_of(returned.begin(), returned.end(),
                               [](unsigned long long value) { return value == 0; });
    const CUdeviceptr final_target = allocation_addresses.back() +
        static_cast<std::size_t>(kRepresentatives.back()) * 128;
    std::uint32_t target_value = 0;
    DRIVER_CHECK(cuMemcpyDtoH(&target_value, final_target, sizeof(target_value)));
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
        "\"allocation_granularity_bytes\":%zu,\"allocation_count\":%d,"
        "\"reservation_address\":%llu,\"reservation_bytes\":%zu,"
        "\"reference_address\":%llu,\"validation_launches\":%d,"
        "\"target_launches\":%d,\"mapping_launches\":%d,"
        "\"allocation_addresses\":[",
        argv[2], granularity, kAllocations,
        static_cast<unsigned long long>(reservation), reservation_bytes,
        static_cast<unsigned long long>(reference_device), kValidationLaunches,
        kTargetLaunches, kLaunches);
    for (int index = 0; index < kAllocations; ++index) {
        std::printf("%s%llu", index == 0 ? "" : ",",
                    static_cast<unsigned long long>(allocation_addresses[index]));
    }
    std::printf("],\"representative_regions\":[");
    for (std::size_t index = 0; index < kRepresentatives.size(); ++index) {
        std::printf("%s%u", index == 0 ? "" : ",", kRepresentatives[index]);
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
    for (int index = 0; index < kAllocations; ++index) {
        DRIVER_CHECK(cuMemUnmap(allocation_addresses[index], granularity));
    }
    DRIVER_CHECK(cuMemAddressFree(reservation, reservation_bytes));
    for (const CUmemGenericAllocationHandle allocation : allocations) {
        DRIVER_CHECK(cuMemRelease(allocation));
    }
    DRIVER_CHECK(cuCtxDestroy(context));
    return correct ? 0 : 1;
}
