CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v

.PHONY: all clean

all: launch_cycle_semantics launch_cycle_semantics_trace

launch_cycle_semantics: launch_cycle_semantics.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# The executable is intentionally identical at the CUDA/SASS level.  The
# separate name keeps NVBit capture commands and retained artifacts explicit.
launch_cycle_semantics_trace: launch_cycle_semantics.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f launch_cycle_semantics launch_cycle_semantics_trace

