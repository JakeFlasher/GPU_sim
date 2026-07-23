# tools/nvbit_tracer/Makefile
#
# Top-level entry point for the standalone host-build NVBit tracer tool.
# `make` here is equivalent to `make -C tracer_tool`, with CUDA_INSTALL_PATH
# auto-detected if not provided by the caller.
#
# Produces:
#   tracer_tool/tracer_tool.so     — the LD_PRELOAD-able NVBit instrumentation tool
#   tracer_tool/trace_printer      — CLI to inspect dynamic_trace.pb output
#
# Host toolchain assumed:
#   nvcc    — CUDA 13.2 (the custom tracer does not compile with host 13.3)
#   g++     — 14.x for nvcc host compilation on the certified host
#   protoc  — exactly 34.1
#   libprotobuf — exactly 34.1.0 (runtime and headers)
#   NVBit 1.8 at ./nvbit_release/
# Source pinned_build_env.sh before invoking this Makefile; it selects and
# validates these exact tools while leaving the system CUDA 13.3 installation
# untouched.

# H3 preserves the current tracer and metric-extractor artifacts.  Mutating
# make goals are permitted only from an explicit non-Git temporary copy; the
# read-only environment probe remains usable in the canonical checkout.
NVBIT_MUTATING_GOALS := $(strip $(filter-out check-pinned-env,$(if $(MAKECMDGOALS),$(MAKECMDGOALS),all)))
ifneq ($(NVBIT_MUTATING_GOALS),)
NVBIT_INSIDE_GIT_WORKTREE := $(shell git rev-parse --is-inside-work-tree 2>/dev/null || true)
ifeq ($(NVBIT_INSIDE_GIT_WORKTREE),true)
$(error H3 refuses tracer build/clean in every Git worktree; use a verified non-Git temporary copy)
endif
ifneq ($(NVBIT_TEMPCOPY_BUILD_OK),1)
$(error non-Git temporary-copy build requires NVBIT_TEMPCOPY_BUILD_OK=1)
endif
endif

# Auto-detect CUDA_INSTALL_PATH from the location of nvcc if the caller did
# not supply one. Typical host installs: /opt/cuda, /usr/local/cuda.
CUDA_INSTALL_PATH ?= $(shell dirname $(shell dirname $(shell which nvcc)))
export CUDA_INSTALL_PATH

PKG_CONFIG?=pkg-config
PKG_CONFIG_FLAGS?=--define-prefix
PB_PREFIX?=$(abspath ../../.toolchain/pb)
PROTOC?=protoc
NVCC_CCBIN?=$(CXX)

# Keep the documented bare `make -C tools/nvbit_tracer` entry point building
# the tool even though the validation target is defined first below.
.DEFAULT_GOAL := all

check-pinned-env:
	@nvcc_release="$$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -n 1)"; \
	protoc_release="$$("$(PROTOC)" --version 2>/dev/null | awk '{print $$2}')"; \
	host_cxx_major="$$("$(NVCC_CCBIN)" -dumpversion 2>/dev/null | cut -d. -f1)"; \
	protobuf_version="$$("$(PKG_CONFIG)" $(PKG_CONFIG_FLAGS) --modversion protobuf 2>/dev/null)"; \
	protobuf_prefix="$$("$(PKG_CONFIG)" $(PKG_CONFIG_FLAGS) --variable=prefix protobuf 2>/dev/null)"; \
	if [ "$$nvcc_release" != 13.2 ]; then \
		echo "ERROR: custom NVBit tracer requires nvcc 13.2, got $${nvcc_release:-unknown}; source tools/nvbit_tracer/pinned_build_env.sh" >&2; exit 2; \
	fi; \
	if [ "$$protoc_release" != 34.1 ]; then \
		echo "ERROR: custom NVBit tracer requires protoc 34.1, got $${protoc_release:-unknown}; source tools/nvbit_tracer/pinned_build_env.sh" >&2; exit 2; \
	fi; \
	if [ "$$host_cxx_major" != 14 ]; then \
		echo "ERROR: custom NVBit tracer requires g++-14, got major $${host_cxx_major:-unknown}; source tools/nvbit_tracer/pinned_build_env.sh" >&2; exit 2; \
	fi; \
	if [ "$$protobuf_version" != 34.1.0 ] || [ "$$protobuf_prefix" != "$(PB_PREFIX)" ]; then \
		echo "ERROR: expected protobuf 34.1.0 under $(PB_PREFIX), got version=$${protobuf_version:-unknown} prefix=$${protobuf_prefix:-unknown}; source tools/nvbit_tracer/pinned_build_env.sh" >&2; exit 2; \
	fi

all: check-pinned-env
	$(MAKE) -C tracer_tool

clean:
	$(MAKE) -C tracer_tool clean

.PHONY: all check-pinned-env clean
