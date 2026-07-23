ARCH_GATE = sm_90a sm_120
include ../../common.mk

TARGET = tma_tensor_copy_2d
SRC = $(TARGET).cu
RUN = $(TARGET)_run

.PHONY: all clean sass run-bin
all: run-bin

run-bin: $(RUN)

# This kernel uses `cuTensorMapEncodeTiled` (driver API), so we link
# against `-lcuda` in addition to the usual runtime library.
$(RUN): $(SRC) main.cu
	$(NVCC) $(NVCC_COMMON) $(NVCC_ARCH_FOR_GATE) $(SRC) main.cu -o $(RUN) -lcudart -lcuda

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_COMMON) $(NVCC_ARCH_FOR_GATE) -c $(SRC) -o $(TARGET).o

sass: $(RUN)
	@set -e; mkdir -p sass_archive
	$(CUOBJDUMP) --dump-sass $(RUN) > sass_archive/$(TARGET).sass

clean:
	rm -f $(TARGET) $(TARGET).o $(RUN)
	rm -rf sass_archive
