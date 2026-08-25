CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v

.PHONY: all clean

all: fixed_latency_allocate fixed_latency_allocate_trace fixed_latency_allocate.cubin fixed_latency_allocate_trace.cubin fixed_latency_allocate_driver

fixed_latency_allocate: fixed_latency_allocate.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

fixed_latency_allocate_trace: fixed_latency_allocate.cu
	$(NVCC) $(NVCC_FLAGS) -DFIXED_ALLOCATE_GENERATIONS=64 -o $@ $<

fixed_latency_allocate.cubin: fixed_latency_allocate.cu
	$(NVCC) $(NVCC_FLAGS) -cubin -o $@ $<

fixed_latency_allocate_trace.cubin: fixed_latency_allocate.cu
	$(NVCC) $(NVCC_FLAGS) -DFIXED_ALLOCATE_GENERATIONS=64 -cubin -o $@ $<

fixed_latency_allocate_driver: fixed_latency_allocate_driver.cpp
	$(NVCC) -std=c++17 -O3 -o $@ $< -lcuda

clean:
	rm -f fixed_latency_allocate fixed_latency_allocate_trace fixed_latency_allocate.cubin fixed_latency_allocate_trace.cubin fixed_latency_allocate_driver
