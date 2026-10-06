CUDA_BIN ?= /usr/local/cuda-13.2/bin
HOST_CXX ?= /usr/bin/g++-14
NVCC = $(CUDA_BIN)/nvcc -ccbin $(HOST_CXX)

.PHONY: all clean
all: runner value.sm_86.cubin

runner: runner.cpp
	$(NVCC) -std=c++17 -O3 -o $@ $< -lcuda

value.sm_86.cubin: value.ptx
	$(CUDA_BIN)/ptxas -arch=sm_86 -o $@ $<

clean:
	rm -f runner value.sm_86.cubin
