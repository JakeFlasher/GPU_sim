# Direct in-tree build for the enhanced NVBit tracer.
# Override tool paths on the make command line when needed.

CUDA_INSTALL_PATH ?= $(shell dirname $(shell dirname $(shell command -v nvcc)))
export CUDA_INSTALL_PATH

PKG_CONFIG ?= pkg-config
PKG_CONFIG_FLAGS ?= --define-prefix
PB_PREFIX ?= $(abspath ../../.toolchain/pb)
PKG_CONFIG_PATH ?= $(PB_PREFIX)/lib/pkgconfig
PROTOC ?= $(PB_PREFIX)/bin/protoc
NVCC_CCBIN ?= $(CXX)

export PKG_CONFIG PKG_CONFIG_FLAGS PKG_CONFIG_PATH PROTOC

.DEFAULT_GOAL := all

all:
	$(MAKE) -C tracer_tool

test:
	$(MAKE) -C tracer_tool unit-tests

clean:
	$(MAKE) -C tracer_tool clean

.PHONY: all test clean
