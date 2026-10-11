CUDA_ROOT := /usr/local/cuda-13.2
NVCC := $(CUDA_ROOT)/bin/nvcc
CC := /usr/bin/gcc-14
CXX := /usr/bin/g++-14
CUTLASS_ROOT ?= /root/cutlass_latest
TARGET := libsm120_fp8_groupwise.so
.PHONY: all
all: $(TARGET)
$(TARGET): fp8_groupwise.cu
	$(NVCC) -std=c++17 -O3 -lineinfo -ccbin=$(CXX) -arch=sm_120a --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler=-fPIC -shared -I$(CUDA_ROOT)/include -I$(CUTLASS_ROOT)/include -I$(CUTLASS_ROOT)/tools/util/include -L$(CUDA_ROOT)/lib64 --cudart=shared $< -o $@
