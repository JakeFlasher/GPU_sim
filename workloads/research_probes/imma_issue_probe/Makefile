CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC ?= $(CUDA_BIN)/nvcc
HOST_CXX ?= /usr/bin/g++-14
NVCC_FLAGS ?= -ccbin=$(HOST_CXX) -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v

.PHONY: all clean
all: imma_issue_probe

imma_issue_probe: imma_issue_probe.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f imma_issue_probe
