# Makefile for the warm/cold direct-NCU-harvest binaries.
#
# These are standalone calibration executables compiled from the
# deployed `baselines/L1/<slug>/cuda_cpp/kernel.cu` kernels (per the
# warm/cold cache-protocol design dossier section 5 — the binary IS
# the deployed baseline, not a cohort-B surrogate). Each binary loops
# its single target kernel 6 times so `ncu --launch-skip 5
# --launch-count 1` deterministically lands on the 6th invocation
# (BL-20260521-ncu-launch-index-single-target-probes). The result of
# the NCU sweep is the read-only sector L1/L2 hit-rate row that
# `run_ncu_harvest_batch.py` merges into
# `scripts/baselines_harness/calibration/cache_prediction/label_ledger_v1_full_partition.toml`.
#
# Variables (matching the nvbit_tracer/metric_extractor/tests/
# calibration_cohort_b/Makefile + calibration_warm_cold/Makefile
# conventions):
#   ARCH         GPU arch (default sm_86 for the local 3090 Ti host;
#                sm_100a when cross-compiling for the cloud B200).
#   NVCC         nvcc binary (default `nvcc`).
#   NVCC_CCBIN   Host C++ compiler nvcc invokes (default g++-14).
#
# Targets:
#   make ARCH=sm_86       — local build for the 3090 Ti.
#   make ARCH=sm_100a     — cross-compile for the cloud B200 target.
#   make clean            — drop the binaries.

ARCH ?= sm_86
NVCC ?= nvcc
NVCC_CCBIN ?= g++-14

NVCC_FLAGS = \
	-ccbin $(NVCC_CCBIN) \
	-O3 \
	-std=c++17 \
	-arch=$(ARCH) \
	-Xcompiler -Wall

KERNELS = \
	problem_018_fused_rope_with_qk_norm_and_kv_cache_update \
	problem_071_kv_cache_update_with_rope

.PHONY: all clean
all: $(KERNELS)

%: %.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(KERNELS)
