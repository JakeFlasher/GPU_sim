CUDA_BIN ?= /usr/local/cuda-13.2/bin
HOST_CXX ?= /usr/bin/g++-14
NVCC = $(CUDA_BIN)/nvcc -ccbin $(HOST_CXX)

.PHONY: all clean
all: main

main: main.cu
	$(NVCC) -std=c++17 -O3 -arch=sm_86 -lineinfo -o $@ $<

clean:
	rm -f main
