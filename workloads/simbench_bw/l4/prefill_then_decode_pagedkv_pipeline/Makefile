ARCH_GATE = sm_90a sm_120
include ../../common.mk

TARGET = prefill_then_decode_pagedkv_pipeline
SRC = $(TARGET).cu
RUN = $(TARGET)_run

.PHONY: all clean sass run-bin
all: run-bin

run-bin: $(RUN)

$(RUN): $(SRC) main.cu
	$(NVCC) $(NVCC_COMMON) -I/usr/include -L/usr/lib/x86_64-linux-gnu $(NVCC_ARCH_FOR_GATE) $(SRC) main.cu -o $(RUN) -lcudart -lcrypto

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_COMMON) $(NVCC_ARCH_FOR_GATE) -c $(SRC) -o $(TARGET).o

sass: $(RUN)
	@set -e; mkdir -p sass_archive
	$(CUOBJDUMP) --dump-sass $(RUN) > sass_archive/$(TARGET).sass

clean:
	rm -f $(TARGET) $(TARGET).o $(RUN)
	rm -rf sass_archive
