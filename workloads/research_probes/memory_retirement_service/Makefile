CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v

.PHONY: all clean

all: memory_retirement_service memory_retirement_service_trace

memory_retirement_service: memory_retirement_service.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

memory_retirement_service_trace: memory_retirement_service.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f memory_retirement_service memory_retirement_service_trace

