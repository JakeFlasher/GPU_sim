ARCH_GATE = sm_120a
include ../../common.mk

TARGET = fp6_e2m3_saturation
SRC = $(TARGET).cu
RUN = $(TARGET)_run

.PHONY: all clean sass run-bin helpers_lib
all: run-bin
run-bin: $(RUN)

$(RUN): $(SRC) main.cu
	$(NVCC) $(NVCC_COMMON) $(NVCC_ARCH_FOR_GATE) $(SRC) main.cu -o $(RUN) -lcudart

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_COMMON) $(NVCC_ARCH_FOR_GATE) -c $(SRC) -o $(TARGET).o

sass: $(RUN)
	@set -e; mkdir -p sass_archive
	$(CUOBJDUMP) --dump-sass $(RUN) > sass_archive/$(TARGET).sass

# Round 46 addition (extends the Round 43-44 compiled-helper contract
# architecture from FP8 E4M3 + FP4 E2M1 to FP6 E2M3, the third of four
# byte-contract C mirrors): build a shared library exporting `extern "C"`
# versions of the canonical Round-5 FP6 E2M3 byte-contract function for
# ctypes loading from capture tooling. See `helpers_lib.cu` for rationale +
# the Round 46 test `the retained workload contract`
# for the byte-equality enforcement.
helpers_lib: helpers_lib.so
helpers_lib.so: helpers_lib.cu
	$(NVCC) -shared -Xcompiler -fPIC -O2 -o helpers_lib.so helpers_lib.cu

clean:
	rm -f $(TARGET) $(TARGET).o $(RUN) helpers_lib.so
	rm -rf sass_archive
