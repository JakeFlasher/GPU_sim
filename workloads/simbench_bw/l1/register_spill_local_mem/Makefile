ARCH_GATE = sm_86 sm_90a sm_120
include ../../common.mk

TARGET = register_spill_local_mem
SRC = $(TARGET).cu
RUN = $(TARGET)_run

# Register-spill microbench: the kernel keeps a 64-wide live FP32 set, which
# FITS in registers at an uncapped build (no spill). Capping the per-thread
# register file forces that live set to spill to local memory -- LDL/STL in
# SASS and STACK-frame bytes > 0 in `cuobjdump -res-usage`. A register-SPILL
# microbench must therefore build under a cap, or it never spills. SPILL_MAXREG
# is overridable so the differential spill-sweep ladder
# (tools/local_labels/verify_spill_sweep.py) can build the same source at a
# range of caps and confirm spill grows as the cap drops (the highest cap must
# NOT spill, the lowest must -- proving the -maxrregcount sweep is not a
# confounded no-op). Empirically on sm_86: cap 255 -> STACK 0; 64 -> 112;
# 32 -> 424; 24 -> 472 bytes.
SPILL_MAXREG ?= 32

# The committed register_cap build-variant ladder (high -> low), mirrored in
# tools/local_labels/verify_spill_sweep.py LADDER_CAPS and the build-variant
# manifest data/local_labels/build_variants/register_spill_local_mem_sm_86.json.
SPILL_LADDER_CAPS ?= 255 64 32 24

.PHONY: all clean sass run-bin spill-variants
all: run-bin
run-bin: $(RUN)

$(RUN): $(SRC) main.cu
	$(NVCC) $(NVCC_COMMON) $(NVCC_ARCH_FOR_GATE) -maxrregcount=$(SPILL_MAXREG) $(SRC) main.cu -o $(RUN) -lcudart

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_COMMON) $(NVCC_ARCH_FOR_GATE) -maxrregcount=$(SPILL_MAXREG) -c $(SRC) -o $(TARGET).o

# Build one distinct per-cap binary per register_cap build-variant via
# SPILL_MAXREG, into spill_variants/, so the local-label collection harness can
# SELECT the matching binary before capture (or rebuild a single cap via
# `make run-bin SPILL_MAXREG=<cap>`). sm_86 single-arch keeps it fast and matches
# the build-variant manifest's identity arch.
spill-variants: $(SRC) main.cu
	@set -e; mkdir -p spill_variants
	@for cap in $(SPILL_LADDER_CAPS); do \
		echo "building register_cap=$$cap"; \
		$(NVCC) $(NVCC_COMMON) $(ARCH_SM86) -maxrregcount=$$cap $(SRC) main.cu -o spill_variants/$(RUN)__cap$$cap -lcudart; \
	done

sass: $(RUN)
	@set -e; mkdir -p sass_archive
	$(CUOBJDUMP) --dump-sass $(RUN) > sass_archive/$(TARGET).sass

clean:
	rm -f $(TARGET) $(TARGET).o $(RUN)
	rm -rf sass_archive spill_variants
