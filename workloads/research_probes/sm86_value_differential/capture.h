#pragma once
#include <stdint.h>
// Deliberately one CTA/warp; every count and capacity is checked by the consumer.
constexpr unsigned kLanes = 32, kFrames = 64, kRegisters = 32;
constexpr unsigned kMemoryFrames = 16;
struct Frame {
    uint32_t pc, guard, active, predicates, regs[kRegisters], ur4, ur5;
};
struct MemoryFrame {
    uint32_t pc, guard, value, kind;
    uint64_t address;
};
struct Capture {
    uint32_t count[kLanes], memory_count[kLanes], overflow, memory_overflow, constants[8];
    Frame frames[kLanes][kFrames];
    MemoryFrame memory[kLanes][kMemoryFrames];
};
