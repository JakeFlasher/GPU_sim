# Direct in-tree build for the enhanced NVBit tracer.
# Override tool paths on the make command line when needed.

# The local SM86 campaign is pinned to CUDA 13.2. Prefer that exact toolkit
# when installed; PATH may point at a newer nvcc whose front-end is incompatible
# with the pinned protobuf/Abseil headers.
CUDA_INSTALL_PATH ?= $(if $(wildcard /usr/local/cuda-13.2/bin/nvcc),/usr/local/cuda-13.2,$(shell dirname $(shell dirname $(shell command -v nvcc))))
export CUDA_INSTALL_PATH

PKG_CONFIG ?= pkg-config
PKG_CONFIG_FLAGS ?= --define-prefix
PB_PREFIX ?= $(abspath ../../.toolchain/pb)
PKG_CONFIG_PATH ?= $(PB_PREFIX)/lib/pkgconfig
PROTOC ?= $(PB_PREFIX)/bin/protoc
# GCC 15+ triggers an NVCC/Abseil template parser failure in the generated
# protobuf surface. GCC 14 is the newest locally validated host compiler.
NVCC_CCBIN ?= $(if $(wildcard /usr/bin/g++-14),/usr/bin/g++-14,$(CXX))

export PKG_CONFIG PKG_CONFIG_FLAGS PKG_CONFIG_PATH PROTOC

.DEFAULT_GOAL := all

all:
	$(MAKE) -C tracer_tool

test:
	$(MAKE) -C tracer_tool unit-tests

clean:
	$(MAKE) -C tracer_tool clean

.PHONY: all test clean
