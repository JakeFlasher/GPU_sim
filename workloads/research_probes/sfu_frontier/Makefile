CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v

.PHONY: all clean

all: sfu_frontier

sfu_frontier: sfu_frontier.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f sfu_frontier
