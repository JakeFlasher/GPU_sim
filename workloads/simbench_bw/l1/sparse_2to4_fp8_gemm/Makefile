ARCH_GATE = sm_120a
include ../../common.mk

TARGET = sparse_2to4_fp8_gemm
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

# Round 43 addition (compiled-helper contract per user-authorized
# Round-42-end promotion of Codex's recurring queued #2): build a
# shared library exporting `extern "C"` versions of the canonical
# Round-31 FP8 E4M3 byte-contract functions for ctypes loading from
# capture tooling. See `helpers_lib.cu` for rationale + the Round 43
# test `the retained workload contract`
# for the byte-equality enforcement.
helpers_lib: helpers_lib.so
helpers_lib.so: helpers_lib.cu
	$(NVCC) -shared -Xcompiler -fPIC -O2 -o helpers_lib.so helpers_lib.cu

clean:
	rm -f $(TARGET) $(TARGET).o $(RUN) helpers_lib.so
	rm -rf sass_archive
