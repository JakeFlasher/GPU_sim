ARCH_GATE = sm_86
include ../../common.mk

TARGET = vendor_sgemm_strided_batched_cutlass
SRC = $(TARGET).cu
RUN = $(TARGET)_run
REPO_ROOT := $(abspath ../../../..)
CUTLASS_DIR ?= $(REPO_ROOT)/third_party/cutlass
NVCC_CCBIN ?= g++-14
CUTLASS_INCLUDES = -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include

.PHONY: all clean sass run-bin
all: run-bin
run-bin: $(RUN)

$(RUN): $(SRC) main.cu
	$(NVCC) $(NVCC_COMMON) $(NVCC_ARCH_FOR_GATE) -ccbin $(NVCC_CCBIN) $(CUTLASS_INCLUDES) $(SRC) main.cu -o $(RUN) -lcudart

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_COMMON) $(NVCC_ARCH_FOR_GATE) -ccbin $(NVCC_CCBIN) $(CUTLASS_INCLUDES) -c $(SRC) -o $(TARGET).o

sass: $(RUN)
	@set -e; mkdir -p sass_archive
	$(CUOBJDUMP) --dump-sass $(RUN) > sass_archive/$(TARGET).sass

clean:
	rm -f $(TARGET) $(TARGET).o $(RUN)
	rm -rf sass_archive
