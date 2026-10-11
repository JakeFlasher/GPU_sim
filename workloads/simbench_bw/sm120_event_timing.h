#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

inline int sm120_event_reps() {
    const char *raw = getenv("SM120_EVENT_REPS");
    if (!raw || !*raw) return 1;
    char *end = nullptr;
    long value = strtol(raw, &end, 10);
    if (end == raw || value < 1 || value > 1000000) return 1;
    return (int)value;
}

// Optional source-level event timing for prospective SM120 evidence.  The
// caller places one scope around the measured kernel region; host setup and
// output copies remain outside the event interval.
struct Sm120CudaEventScope {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    bool active{false};

    explicit Sm120CudaEventScope(cudaStream_t stream = 0) {
        if (cudaEventCreate(&start) != cudaSuccess) return;
        if (cudaEventCreate(&stop) != cudaSuccess) {
            cudaEventDestroy(start);
            start = nullptr;
            return;
        }
        if (cudaEventRecord(start, stream) != cudaSuccess) return;
        active = true;
    }

    void finish(const char *label, cudaStream_t stream = 0) {
        if (!active) return;
        if (cudaEventRecord(stop, stream) != cudaSuccess) return;
        if (cudaEventSynchronize(stop) != cudaSuccess) return;
        float milliseconds = 0.0f;
        if (cudaEventElapsedTime(&milliseconds, start, stop) != cudaSuccess) return;
        printf("SM120_CUDA_EVENT label=%s elapsed_ms=%.9g\n", label, milliseconds);
        fflush(stdout);
        active = false;
    }

    ~Sm120CudaEventScope() {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
    }
};
