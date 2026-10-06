CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
# Ignore the ambient GCC 16 selection; a make command-line override still works.
NVCC_CCBIN := /usr/bin/g++-14
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v -ccbin $(NVCC_CCBIN)

.PHONY: all clean

all: sfu_frontier

sfu_frontier: sfu_frontier.cu Makefile
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f sfu_frontier
