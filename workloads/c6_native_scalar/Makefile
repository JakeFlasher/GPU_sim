CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC_CCBIN := /usr/bin/g++-14

.PHONY: all clean

all: scalar_kernel.cubin scalar_driver

scalar_kernel.cubin: scalar_kernel.ptx
	$(CUDA_BIN)/ptxas -arch=sm_86 -o $@ $<

scalar_driver: scalar_driver.cpp scalar_kernel.cubin Makefile
	$(CUDA_BIN)/nvcc -std=c++17 -ccbin $(NVCC_CCBIN) -O2 -arch=sm_86 -o $@ $< -lcuda

clean:
	rm -f scalar_kernel.cubin scalar_driver
