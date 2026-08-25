#include <cuda_runtime.h>

#include <cstdint>

#ifndef RESULT_LOAD_PRIORITY_GENERATIONS
constexpr int kResultLoadPriorityGenerations = 1024;
#else
constexpr int kResultLoadPriorityGenerations = RESULT_LOAD_PRIORITY_GENERATIONS;
#endif

__global__ void result_load_priority_search(unsigned long long* elapsed,
                                            std::uint32_t* sink,
                                            const std::uint32_t* inputs) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const std::uint32_t* address = inputs + threadIdx.x;
    std::uint32_t warm = *address;
    std::uint32_t increment = inputs[threadIdx.x + blockDim.x];
    std::uint32_t acc = 0x3f800000u;
    std::uint32_t load_acc = 0;
    std::uint32_t live_pad = warm;
    std::uint32_t live_pad2 = warm ^ 0xdeadbeefu;
    std::uint32_t loaded = warm;
    asm volatile("lop3.b32 %0, %0, %1, %1, 0x96;"
                 : "+r"(acc)
                 : "r"(warm)
                 : "memory");
    asm volatile("" : "+r"(increment));

    const unsigned long long start = clock64();
#pragma unroll 1
    for (int generation = 0; generation < kResultLoadPriorityGenerations;
         ++generation) {
        asm volatile("ld.global.ca.u32 %0, [%1];"
                     : "+r"(loaded)
                     : "l"(address)
                     : "memory");
        asm volatile("add.rn.f16x2 %0, %0, %1;"
                     : "+r"(acc)
                     : "r"(increment)
                     : "memory");
        asm volatile("bar.warp.sync 0xffffffff;" : : : "memory");
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;"
                     : "+r"(load_acc)
                     : "r"(loaded), "r"(live_pad)
                     : "memory");
        asm volatile("lop3.b32 %0, %0, %1, %2, 0x3c;"
                     : "+r"(acc)
                     : "r"(load_acc), "r"(live_pad2)
                     : "memory");
    }
    const unsigned long long stop = clock64();

    if (lane == 0) {
        elapsed[warp] = stop - start;
        sink[warp] = acc ^ load_acc;
    }
}
