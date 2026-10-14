#pragma once

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
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

// One prewarmed launch per event pair. Host submission may still contribute to
// very short CUDA-event intervals; it is reported, never equated to NCU time.
// reset() restores any in-place input outside the measured bracket.
template<class Launch, class Reset>
inline void sm120_measure_launches(const char* label, Launch launch, Reset reset,
                                  cudaStream_t stream = 0) {
    auto require = [](cudaError_t error, const char* action) {
        if (error != cudaSuccess) {
            fprintf(stderr, "%s: %s\n", action, cudaGetErrorString(error));
            exit(2);
        }
    };
    for (int warm = 0; warm < 3; ++warm) {
        reset(); launch();
        require(cudaGetLastError(), "warm-up launch");
    }
    require(cudaStreamSynchronize(stream), "warm-up stream");
    cudaEvent_t begin{}, end{};
    require(cudaEventCreate(&begin), "create start event");
    require(cudaEventCreate(&end), "create stop event");
    const int repetitions = sm120_event_reps();
    for (int rep = 0; rep < repetitions; ++rep) {
        reset();
        require(cudaStreamSynchronize(stream), "input reset stream");
        require(cudaProfilerStart(), "start measured launch range");
        require(cudaEventRecord(begin, stream), "record start event");
        launch();
        require(cudaGetLastError(), "measured launch");
        require(cudaEventRecord(end, stream), "record stop event");
        require(cudaEventSynchronize(end), "wait stop event");
        require(cudaProfilerStop(), "stop measured launch range");
        float ms = 0;
        require(cudaEventElapsedTime(&ms, begin, end), "elapsed event time");
        printf("SM120_CUDA_EVENT label=%s repetition=%d launches=1 warmups=3 elapsed_ms=%.9g\n",
               label, rep, ms);
    }
    require(cudaEventDestroy(begin), "destroy start event");
    require(cudaEventDestroy(end), "destroy stop event");
}
