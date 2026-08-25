CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v

.PHONY: all clean
all: atomic_lts_mapping_driver atomic_lts_mapping.cubin

atomic_lts_mapping_driver: atomic_lts_mapping_driver.cpp
	$(NVCC) -std=c++17 -O3 -o $@ $< -lcuda

atomic_lts_mapping.cubin: atomic_lts_mapping.cu
	$(NVCC) $(NVCC_FLAGS) -cubin -o $@ $<

clean:
	rm -f atomic_lts_mapping_driver atomic_lts_mapping.cubin
