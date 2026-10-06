CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC_CCBIN ?= /usr/bin/g++-14
NVCC := $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v

.PHONY: all clean

all: constant_load_path

constant_load_path: constant_load_path.cu
	$(NVCC) -ccbin $(NVCC_CCBIN) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f constant_load_path
