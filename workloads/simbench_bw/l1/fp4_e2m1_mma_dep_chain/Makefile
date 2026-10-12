ARCH_GATE = sm_120a
include ../../common.mk

TARGET = fp4_e2m1_mma_dep_chain
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

# Round 44 addition (extends the Round 43 compiled-helper contract
# architecture from FP8 E4M3 to FP4 E2M1): build a shared library
# exporting `extern "C"` versions of the canonical Round-4 FP4 E2M1
# byte-contract functions for ctypes loading from capture tooling. See
# `helpers_lib.cu` for rationale + the Round 44 test
# `the retained workload contract` for the byte-
# equality enforcement.
helpers_lib: helpers_lib.so
helpers_lib.so: helpers_lib.cu
	$(NVCC) -shared -Xcompiler -fPIC -O2 -o helpers_lib.so helpers_lib.cu

clean:
	rm -f $(TARGET) $(TARGET).o $(RUN) helpers_lib.so
	rm -rf sass_archive
