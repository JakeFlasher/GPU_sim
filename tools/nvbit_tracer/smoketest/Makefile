# tools/nvbit_tracer/smoketest/Makefile
#
# Builds the self-contained backprop_enriched smoke test binary used as the
# NVBit tracer validation target. Single .cu file, no external source pulls.
#
# Default architecture: sm_120 (cloud GeForce RTX 5090 Blackwell — the
# project's cloud target; compute 12.0). The local smoke-test host on
# this branch is RTX 3090 Ti / sm_86 (Ampere); override with
# `make ARCH=sm_86` to build for local execution. Other valid overrides
# include `make ARCH=sm_90` for H100.

CUDA_INSTALL_PATH ?= /usr/local/cuda-13.2
NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc
CXX  ?= /usr/bin/g++-14
ARCH ?= sm_120

NVCC_FLAGS = -O3 -std=c++20 -arch=$(ARCH) -ccbin=$(CXX) -lineinfo

all: backprop_enriched phase_injection_microbench

backprop_enriched: backprop_enriched.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Three-kernel microbench driving the surgical phase-injection bug-fix
# patch's smoke pytest. cp.async-only / __syncthreads-only / divergent-
# branch kernels exercise the four phase-injection boundary classes.
# Driven by `workloads/simbench_bw/tests/test_phase_injection_smoke.py`
# which builds the binary directly via nvcc; this Makefile target is
# offered for manual rebuilds.
phase_injection_microbench: phase_injection_microbench.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

run: backprop_enriched
	./backprop_enriched

clean:
	rm -f backprop_enriched phase_injection_microbench
	rm -rf traces

.PHONY: all clean run
