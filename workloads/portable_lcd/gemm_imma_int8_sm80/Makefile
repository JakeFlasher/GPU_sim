ARCH_GATE = lcd
include ../common.mk

TARGET = gemm_imma_int8_sm80
SRC    = $(TARGET).cu
RUN    = $(TARGET)_run

.PHONY: all clean sass run-bin
all: run-bin
run-bin: $(RUN)

$(RUN): $(SRC) main.cu
	$(NVCC) $(NVCC_COMMON) $(NVCC_ARCH_FOR_GATE) $(SRC) main.cu -o $(RUN) -lcudart

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_COMMON) $(NVCC_ARCH_FOR_GATE) -c $(SRC) -o $(TARGET).o

sass: $(RUN)
	@set -e; mkdir -p sass_archive
	$(CUOBJDUMP) --dump-sass $(RUN) > sass_archive/$(TARGET).sass

clean:
	rm -f $(TARGET).o $(RUN)
	rm -rf sass_archive results traces
