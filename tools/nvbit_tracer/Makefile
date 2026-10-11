ifeq ($(origin CXX),default)
CXX := /usr/bin/g++-14
endif
CXX ?= /usr/bin/g++-14
ifeq ($(origin CC),default)
CC := /usr/bin/gcc-14
endif
CC ?= /usr/bin/gcc-14

# Direct in-tree build for the enhanced NVBit tracer.
CUDA_INSTALL_PATH ?= $(if $(wildcard /usr/local/cuda-13.2/bin/nvcc),/usr/local/cuda-13.2,$(shell dirname $(shell dirname $(shell command -v nvcc))))
PKG_CONFIG ?= pkg-config
PKG_CONFIG_FLAGS ?= --define-prefix
PB_PREFIX ?= $(abspath $(CURDIR)/../../.toolchain/pb)
PKG_CONFIG_PATH ?= $(PB_PREFIX)/lib/pkgconfig
PROTOC ?= $(PB_PREFIX)/bin/protoc
NVCC_CCBIN ?= /usr/bin/g++-14
export CC CXX NVCC_CCBIN CUDA_INSTALL_PATH PKG_CONFIG PKG_CONFIG_FLAGS PKG_CONFIG_PATH PROTOC
.DEFAULT_GOAL := all
all:
	$(MAKE) -C tracer_tool
clean:
	$(MAKE) -C tracer_tool clean
.PHONY: all clean
