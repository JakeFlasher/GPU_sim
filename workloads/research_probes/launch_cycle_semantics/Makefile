CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC = $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v

.PHONY: all clean

all: launch_cycle_semantics launch_cycle_semantics_trace ncu_launch_envelope imma_issue_probe

launch_cycle_semantics: launch_cycle_semantics.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# The executable is intentionally identical at the CUDA/SASS level.  The
# separate name keeps NVBit capture commands and retained artifacts explicit.
launch_cycle_semantics_trace: launch_cycle_semantics.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# This executable deliberately omits cudaProfilerStart/Stop.  It matches the
# demo campaign's NCU protocol: one ordinary warm-up launch followed by one
# scored launch selected with --launch-skip 1 --launch-count 1.
ncu_launch_envelope: ncu_launch_envelope.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

imma_issue_probe: imma_issue_probe.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f launch_cycle_semantics launch_cycle_semantics_trace ncu_launch_envelope imma_issue_probe
