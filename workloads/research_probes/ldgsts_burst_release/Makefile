CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC ?= $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v

.PHONY: all clean

all: ldgsts_burst_release

ldgsts_burst_release: ldgsts_burst_release.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f ldgsts_burst_release
