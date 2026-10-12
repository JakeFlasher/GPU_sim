ARCH_GATE = sm_86 sm_90a sm_120
include ../../common.mk

TARGET = register_spill_local_mem
SRC = $(TARGET).cu
RUN = $(TARGET)_run

# Register cap is a workload build parameter; lowering it increases local-memory pressure.
SPILL_MAXREG ?= 32

SPILL_LADDER_CAPS ?= 255 64 32 24

.PHONY: all clean sass run-bin spill-variants
all: run-bin
run-bin: $(RUN)

$(RUN): $(SRC) main.cu
	$(NVCC) $(NVCC_COMMON) $(NVCC_ARCH_FOR_GATE) -maxrregcount=$(SPILL_MAXREG) $(SRC) main.cu -o $(RUN) -lcudart

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_COMMON) $(NVCC_ARCH_FOR_GATE) -maxrregcount=$(SPILL_MAXREG) -c $(SRC) -o $(TARGET).o

# Build independently named binaries for real register-pressure measurements.
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
