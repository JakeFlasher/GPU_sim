# SimBench-BW source build helpers.
# Per-kernel Makefiles live under l1/ l2/ l3/ l4/<slug> and include common.mk.
# Acceptance is complete real-trace replay, not a fixture or schema gate.

PYTHON ?= python3

.PHONY: build
build:
	@echo "Build a selected workload with its per-kernel Makefile."
	@echo "Example: make -C l1/fp32_fma_dep_chain"
