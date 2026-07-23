ARCH_GATE = sm_120a
include ../../common.mk

TARGET = fp6_e3m2_mma_dep_chain
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

# Round 46 addition (the FOURTH and final byte-contract C mirror in
# the compiled-helper enforcement layer; closes Codex's recurring
# queued #2 across all 4 surfaces — FP8 E4M3 [Round 43], FP4 E2M1
# [Round 44], FP6 E2M3 + FP6 E3M2 [Round 46]): build a shared library
# exporting `extern "C"` versions of the canonical Round-6 FP6 E3M2
# byte-contract function. See `helpers_lib.cu` for rationale + the
# Round 46 test `tests/test_fp6_e3m2_compiled_helper_enforcement.py`
# for the byte-equality enforcement.
helpers_lib: helpers_lib.so
helpers_lib.so: helpers_lib.cu
	$(NVCC) -shared -Xcompiler -fPIC -O2 -o helpers_lib.so helpers_lib.cu

clean:
	rm -f $(TARGET) $(TARGET).o $(RUN) helpers_lib.so
	rm -rf sass_archive
