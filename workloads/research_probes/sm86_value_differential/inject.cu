#include "capture.h"
#include <cstdarg>
extern "C" __device__ __noinline__ void snapshot(
    uint64_t pointer, uint32_t pc, uint32_t guard, uint32_t predicates,
    uint32_t ur4, uint32_t ur5, uint32_t nregs, ...) {
    auto *capture = reinterpret_cast<Capture *>(pointer);
    unsigned lane = threadIdx.x;
    unsigned index = capture->count[lane]++;
    if (index >= kFrames || nregs > kRegisters) {
        atomicAdd(&capture->overflow, 1u);
        return;
    }
    Frame &frame = capture->frames[lane][index];
    frame.pc = pc; frame.guard = guard; frame.active = __activemask();
    frame.predicates = predicates; frame.ur4 = ur4; frame.ur5 = ur5;
    va_list args;
    va_start(args, nregs);
    for (unsigned r = 0; r < nregs; ++r) frame.regs[r] = va_arg(args, uint32_t);
    va_end(args);
}
extern "C" __device__ __noinline__ void capture_constants(
    uint64_t pointer, uint32_t stack, uint32_t desc_lo, uint32_t desc_hi,
    uint32_t in_lo, uint32_t in_hi, uint32_t out_lo, uint32_t out_hi,
    uint32_t delta) {
    if (threadIdx.x != 0) return;
    auto *c = reinterpret_cast<Capture *>(pointer);
    c->constants[0] = stack; c->constants[1] = desc_lo; c->constants[2] = desc_hi;
    c->constants[3] = in_lo; c->constants[4] = in_hi;
    c->constants[5] = out_lo; c->constants[6] = out_hi; c->constants[7] = delta;
}
extern "C" __device__ __noinline__ void snapshot_memory(
    uint64_t pointer, uint32_t pc, uint32_t guard, uint64_t address,
    uint32_t kind, uint32_t value) {
    auto *capture = reinterpret_cast<Capture *>(pointer);
    unsigned lane = threadIdx.x;
    unsigned index = capture->memory_count[lane]++;
    if (index >= kMemoryFrames) {
        atomicAdd(&capture->memory_overflow, 1u);
        return;
    }
    auto &frame = capture->memory[lane][index];
    frame.pc = pc;
    frame.guard = guard;
    frame.address = static_cast<uint32_t>(address);
    frame.kind = kind;
    frame.value = value;
}
