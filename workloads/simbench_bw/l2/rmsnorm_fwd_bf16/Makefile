# rmsnorm_fwd_bf16 per-kernel Makefile.
#
# Portable row: builds under both sm_90a (Hopper) and sm_120 (Blackwell
# consumer / workstation). The common.mk wrapper expands ARCH_GATE into
# the matching -gencode flag set.

ARCH_GATE = sm_86 sm_90a sm_120

include ../../common.mk

TARGET = rmsnorm_fwd_bf16
SRC = $(TARGET).cu
RUN = $(TARGET)_run

# Build a single executable that links the kernel TU + host driver so the
# orchestrator can subprocess-invoke $(RUN) with shape args and parse its JSON
# output. nvcc's default --compiler-options propagate the C++ runtime.
.PHONY: all clean sass run-bin
all: run-bin

run-bin: $(RUN)

$(RUN): $(SRC) main.cu
	$(NVCC) $(NVCC_COMMON) $(NVCC_ARCH_FOR_GATE) $(SRC) main.cu -o $(RUN) -lcudart

sass: $(RUN)
	@set -e; mkdir -p sass_archive
	$(CUOBJDUMP) --dump-sass $(RUN) > sass_archive/$(TARGET).sass

clean:
	rm -f $(TARGET) $(TARGET).o $(RUN)
	rm -rf sass_archive
