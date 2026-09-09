# Direct in-tree build for the enhanced NVBit tracer.
CUDA_INSTALL_PATH ?= $(if $(wildcard /usr/local/cuda-13.2/bin/nvcc),/usr/local/cuda-13.2,$(shell dirname $(shell dirname $(shell command -v nvcc))))
PKG_CONFIG ?= pkg-config
PKG_CONFIG_FLAGS ?= --define-prefix
PB_PREFIX ?= $(abspath ../../.toolchain/pb)
PKG_CONFIG_PATH ?= $(PB_PREFIX)/lib/pkgconfig
PROTOC ?= $(PB_PREFIX)/bin/protoc
NVCC_CCBIN ?= $(if $(wildcard /usr/bin/g++-14),/usr/bin/g++-14,$(CXX))
export CUDA_INSTALL_PATH PKG_CONFIG PKG_CONFIG_FLAGS PKG_CONFIG_PATH PROTOC
.DEFAULT_GOAL := all
all:
	$(MAKE) -C tracer_tool
clean:
	$(MAKE) -C tracer_tool clean
.PHONY: all clean
