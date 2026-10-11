CUDA_INSTALL_PATH ?= /usr/local/cuda-13.2
NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc
ifeq ($(origin CXX),default)
CXX := /usr/bin/g++-14
endif
CXX ?= /usr/bin/g++-14
CUTLASS_ROOT ?= /root/cutlass_latest
CUTLASS_INCLUDE = -I$(CUTLASS_ROOT)/include -I$(CUTLASS_ROOT)/tools/util/include
NVCC_FLAGS = -std=c++17 -O3 --expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_120a,code=sm_120a -ccbin=$(CXX) $(CUTLASS_INCLUDE) -lineinfo
TARGET = cutlass_nvfp4_grouped_gemm_run
.PHONY: all clean sass
all: $(TARGET)
$(TARGET): cutlass_nvfp4_grouped_gemm.cu helper.h
	$(NVCC) $(NVCC_FLAGS) $< -o $@
sass: $(TARGET)
	$(CUDA_INSTALL_PATH)/bin/cuobjdump --dump-sass $(TARGET) > $(TARGET).sass
clean:
	python3 -c 'from pathlib import Path; [p.unlink() for p in Path(".").glob("$(TARGET)*") if p.is_file()]'
