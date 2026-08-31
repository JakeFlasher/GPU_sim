CUDA_BIN ?= /usr/local/cuda-13.2/bin
NVCC ?= $(CUDA_BIN)/nvcc
NVCC_FLAGS ?= -std=c++17 -O3 -arch=sm_86 -lineinfo -Xptxas=-v

.PHONY: all clean

all: async_source_only async_visibility_phase async_issue_spacing async_cohort_pairing

async_source_only: async_source_only.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

async_visibility_phase: async_visibility_phase.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

async_issue_spacing: async_issue_spacing.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

async_cohort_pairing: async_cohort_pairing.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f async_source_only async_visibility_phase async_issue_spacing async_cohort_pairing
