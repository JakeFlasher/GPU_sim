#include <cuda_runtime.h>

#include <cstdint>

#define IBUF_CLOCK_GROUP(N)                                                   \
    "mov.u64 %" #N ", %%clock64;\n\t"                                      \
    "bar.warp.sync 0xffffffff;\n\t"                                        \
    "bar.warp.sync 0xffffffff;\n\t"                                        \
    "bar.warp.sync 0xffffffff;\n\t"

// The paper identifies the issued warp from CLOCK timestamps and inserts NOPs
// because CLOCK cannot issue in consecutive cycles. Three full-mask warp syncs
// lower to three SASS NOPs between reads, producing eight four-instruction
// groups (32 instructions) without a branch.
__global__ void ibuffer_timeline_probe(unsigned long long* timeline,
                                       std::uint32_t* sink) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    unsigned long long t0;
    unsigned long long t1;
    unsigned long long t2;
    unsigned long long t3;
    unsigned long long t4;
    unsigned long long t5;
    unsigned long long t6;
    unsigned long long t7;
    unsigned long long t8;
    asm volatile(
        IBUF_CLOCK_GROUP(0) IBUF_CLOCK_GROUP(1) IBUF_CLOCK_GROUP(2)
            IBUF_CLOCK_GROUP(3) IBUF_CLOCK_GROUP(4) IBUF_CLOCK_GROUP(5)
                IBUF_CLOCK_GROUP(6) IBUF_CLOCK_GROUP(7) IBUF_CLOCK_GROUP(8)
        : "=l"(t0), "=l"(t1), "=l"(t2), "=l"(t3), "=l"(t4),
          "=l"(t5), "=l"(t6), "=l"(t7), "=l"(t8)
        :
        : "memory");
    if (lane == 0) {
        unsigned long long* warp_timeline = timeline + warp * 9;
        warp_timeline[0] = t0;
        warp_timeline[1] = t1;
        warp_timeline[2] = t2;
        warp_timeline[3] = t3;
        warp_timeline[4] = t4;
        warp_timeline[5] = t5;
        warp_timeline[6] = t6;
        warp_timeline[7] = t7;
        warp_timeline[8] = t8;
        warp_timeline[8] = t8;
        sink[warp] = 0x1b0f0000u + static_cast<std::uint32_t>(warp);
    }
}

#undef IBUF_CLOCK_GROUP
