CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC ?= $(CUDA_BIN)/nvcc
HOST_CXX ?= /usr/bin/g++-14
NVCC_FLAGS ?= -ccbin=$(HOST_CXX) -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v

.PHONY: all clean

all: tf32_issue_probe

tf32_issue_probe: tf32_issue_probe.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f tf32_issue_probe
