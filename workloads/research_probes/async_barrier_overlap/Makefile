CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v

.PHONY: all clean

all: async_barrier_overlap async_barrier_overlap_trace

async_barrier_overlap: async_barrier_overlap.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

async_barrier_overlap_trace: async_barrier_overlap.cu
	$(NVCC) $(NVCC_FLAGS) -DASYNC_BARRIER_GROUPS=64 -o $@ $<

clean:
	rm -f async_barrier_overlap async_barrier_overlap_trace
