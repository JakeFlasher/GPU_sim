CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v
PTXAS_FLAGS ?= --opt-level=1

.PHONY: all valid-imc-probe clean

all: immediate_constant_frontier valid-imc-probe

immediate_constant_frontier: immediate_constant_frontier.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Hand-authored PTX keeps the four parameter loads inside the timed loop;
# ptxas lowers them to SASS CBANK operands without introducing local spills.
valid-imc-probe: imc_kernel.cubin imc_valid_driver

imc_kernel.cubin: imc_kernel.ptx
	$(CUDA_BIN)/ptxas -arch=sm_86 $(PTXAS_FLAGS) -o $@ $<

imc_valid_driver: imc_valid_driver.cpp imc_kernel.cubin
	$(NVCC) -std=c++17 -O2 -o $@ $< -lcuda

clean:
	rm -f immediate_constant_frontier
