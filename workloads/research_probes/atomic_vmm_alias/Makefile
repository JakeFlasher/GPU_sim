CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v

.PHONY: all clean
all: vmm_alias_driver vmm_alias_read.cubin vmm_alias_atomic_driver vmm_alias_atomic.cubin vmm_cross_allocation_driver

vmm_alias_driver: vmm_alias_driver.cpp
	$(NVCC) -std=c++17 -O3 -o $@ $< -lcuda

vmm_alias_read.cubin: vmm_alias_read.cu
	$(NVCC) $(NVCC_FLAGS) -cubin -o $@ $<

vmm_alias_atomic_driver: vmm_alias_atomic_driver.cpp
	$(NVCC) -std=c++17 -O3 -o $@ $< -lcuda

vmm_alias_atomic.cubin: vmm_alias_atomic.cu
	$(NVCC) $(NVCC_FLAGS) -cubin -o $@ $<

vmm_cross_allocation_driver: vmm_cross_allocation_driver.cpp
	$(NVCC) -std=c++17 -O3 -o $@ $< -lcuda

clean:
	rm -f vmm_alias_driver vmm_alias_read.cubin vmm_alias_atomic_driver vmm_alias_atomic.cubin vmm_cross_allocation_driver
