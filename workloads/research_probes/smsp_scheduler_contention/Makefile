CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v

.PHONY: all clean

all: smsp_scheduler_contention smsp_scheduler_contention_trace

smsp_scheduler_contention: smsp_scheduler_contention.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

smsp_scheduler_contention_trace: smsp_scheduler_contention.cu
	$(NVCC) $(NVCC_FLAGS) -DSMSP_SCHEDULER_GENERATIONS=64 -o $@ $<

clean:
	rm -f smsp_scheduler_contention smsp_scheduler_contention_trace

