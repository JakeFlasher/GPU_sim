# Makefile for nvbit_tracer/metric_extractor — counter-free per-launch metric
# extractor built on NVBit 1.8. Structurally derived from
# nvbit_release/tools/opcode_hist/Makefile, with NVBIT_PATH adjusted from
# ../../core to ../nvbit_release/core because this tool sits one level
# above the vendored nvbit_release/tools tree.
#
# Targets:
#   make ARCH=sm_86       — build for the 3090 Ti.
#   make ARCH=sm_100a     — build for B200.
#   make clean            — remove generated outputs.

.DEFAULT_GOAL := all

NVCC_CCBIN ?= $(CXX)
NVCC = nvcc -ccbin=$(NVCC_CCBIN) -D_FORCE_INLINES
PTXAS = ptxas

DEBUG ?= 0

ifeq ($(DEBUG),1)
NVCC_OPT = -O0 -g
else
NVCC_OPT = -O3
endif

# nvcc version gate. NVBit's runtime requires at least 10.1 to even compile a
# tool — instrumented apps may use older nvcc, but the tool itself cannot.
NVCC_VER_REQ = 10.1
NVCC_VER = $(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_VER_CHECK = $(shell awk -v a="${NVCC_VER}" -v b="$(NVCC_VER_REQ)" 'BEGIN{print (a+0 >= b+0) ? 1 : 0}')

ifeq ($(NVCC_VER_CHECK),0)
$(error ERROR: nvcc version >= $(NVCC_VER_REQ) required to compile an nvbit tool. Instrumented applications can still use lower versions of nvcc.)
endif

# Newer ptxas accepts -maxrregcount on injected device functions; older ptxas
# rejects it. Mirror the upstream opcode_hist/instr_count probe.
PTXAS_VER_ADD_FLAG = 12.3
PTXAS_VER = $(shell $(PTXAS) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
PTXAS_VER_CHECK = $(shell awk -v a="${PTXAS_VER}" -v b="$(PTXAS_VER_ADD_FLAG)" 'BEGIN{print (a+0 >= b+0) ? 1 : 0}')

ifeq ($(PTXAS_VER_CHECK),0)
MAXRREGCOUNT_FLAG = -maxrregcount=24
else
MAXRREGCOUNT_FLAG =
endif

# nvbit_tracer/metric_extractor → nvbit_tracer/nvbit_release/core is one ../ up.
NVBIT_PATH = ../nvbit_release/core
INCLUDES = -I$(NVBIT_PATH)

LIBS = -L$(NVBIT_PATH) -lnvbit
NVCC_LIBPATH = -L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

SOURCES = $(wildcard *.cu)
OBJECTS = $(SOURCES:.cu=.o)

ARCH ?= sm_86
ALLOWED_ARCHES = sm_86 sm_100a
ifeq ($(filter $(ARCH),$(ALLOWED_ARCHES)),)
$(error unsupported ARCH=$(ARCH); supported values are sm_86 and sm_100a)
endif

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

NVBIT_TOOL = $(current_dir).so

all: $(NVBIT_TOOL)

$(NVBIT_TOOL): $(OBJECTS) $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) $(NVCC_OPT) $(OBJECTS) $(LIBS) $(NVCC_LIBPATH) -lcuda -lcudart_static -shared -o $@

# Host-side translation units (the launch hook + per-launch JSON-L emitter
# will live in metric_extractor.cu). Compiled with -dc -c so device-function
# objects can be linked into the final shared library.
%.o: %.cu
	$(NVCC) -dc -c -std=c++11 $(INCLUDES) -Xptxas -cloning=no -Xcompiler -Wall -arch=$(ARCH) $(NVCC_OPT) -Xcompiler -fPIC $< -o $@

# Injected device functions need a different compile recipe: -astoolspatch keeps
# the patch surface intact and --keep-device-functions retains symbols NVBit
# inserts at runtime.
inject_funcs.o: inject_funcs.cu
	$(NVCC) $(INCLUDES) $(MAXRREGCOUNT_FLAG) -Xptxas -astoolspatch --keep-device-functions -arch=$(ARCH) -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

# Host-only regression test for the AC-7 instruction-class router. Mirrors
# the nvbit_tracer/isa_def/test_opcode_map.cc pattern: pure C++ unit test,
# no NVBit / CUDA dependency, single-binary executable. Run after any edit
# to classifier.h or to common.h's InstructionClass enum.
test_classifier: test_classifier.cc classifier.h common.h
	$(CXX) -std=c++17 -Wall -Wextra -I. -I.. test_classifier.cc -o $@

# Verbose form so the pytest wrapper can drive `make run-test-classifier`
# and capture the binary's stdout in one step.
.PHONY: run-test-classifier
run-test-classifier: test_classifier
	./test_classifier

# Host-only regression test for the saturating-add primitive that backs
# every per-launch counter atomicAdd in inject_funcs.cu. The compute
# helper is `__host__ __device__` so it compiles standalone with a
# pure-C++ toolchain (no nvcc / CUDA dependency); the test asserts the
# clamp / overflow / zero-delta contract directly.
test_saturating_counter: test_saturating_counter.cc saturating_counter.h
	$(CXX) -std=c++17 -Wall -Wextra -I. test_saturating_counter.cc -o $@

.PHONY: run-test-saturating-counter
run-test-saturating-counter: test_saturating_counter
	./test_saturating_counter

# Host-only known-vector test for the Tier-2 SHA-256 implementation
# that backs the per-launch sidecar digest in LaunchHeaderV1. Pure
# C++; no nvcc / CUDA / NVBit dependency. The vectors are FIPS-180-4
# Appendix B (empty string, "abc", 1M 'a' bytes) plus a streaming-vs-
# one-shot equivalence probe.
test_tier2_sha256: test_tier2_sha256.cc tier2_sha256.h
	$(CXX) -std=c++17 -Wall -Wextra -I. test_tier2_sha256.cc -o $@

.PHONY: run-test-tier2-sha256
run-test-tier2-sha256: test_tier2_sha256
	./test_tier2_sha256

# Host-only behavioral regression for Tier-2 truncation-marker boundary
# metadata. Drives tier2_frame_serializer.h with synthetic CompactAccessRecord
# and TmaExpansionResult arrays; asserts the marker emits the correct first
# omitted sequence id on every scenario (including first-record lossy TMA
# at sequence 0 which exposed the Round-14 last_sequence_id+1 bug). Pure
# C++; no nvcc / CUDA / NVBit dependency.
test_tier2_marker_metadata: test_tier2_marker_metadata.cc tier2_frame_serializer.h tier2_sha256.h common.h
	$(CXX) -std=c++17 -Wall -Wextra -I. test_tier2_marker_metadata.cc -o $@

.PHONY: run-test-tier2-marker-metadata
run-test-tier2-marker-metadata: test_tier2_marker_metadata
	./test_tier2_marker_metadata

clean:
	rm -f *.so *.o test_classifier test_saturating_counter test_tier2_sha256 test_tier2_marker_metadata

.PHONY: all clean
