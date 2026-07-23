ARCH_GATE = lcd
include ../common.mk

TARGET = ac4_mixed_long_sm80
SRC    = $(TARGET).cu
RUN    = $(TARGET)_run

.PHONY: all clean sass run-bin
all: run-bin
run-bin: $(RUN)

$(RUN): $(SRC)
	$(NVCC) $(NVCC_COMMON) $(NVCC_ARCH_FOR_GATE) $(SRC) -o $(RUN) -lcudart -ldl

sass: $(RUN)
	@set -e; mkdir -p sass_archive
	$(CUOBJDUMP) --dump-sass $(RUN) > sass_archive/$(TARGET).sass

clean:
	rm -f $(RUN)
	rm -rf sass_archive results traces
