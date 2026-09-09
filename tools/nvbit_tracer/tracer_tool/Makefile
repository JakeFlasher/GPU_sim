.DEFAULT_GOAL := all
CUDA_INSTALL_PATH ?= $(if $(wildcard /usr/local/cuda-13.2/bin/nvcc),/usr/local/cuda-13.2,$(shell dirname $(shell dirname $(shell command -v nvcc))))
NVCC_CCBIN?=$(if $(wildcard /usr/bin/g++-14),/usr/bin/g++-14,$(CXX))
NVCC?=$(CUDA_INSTALL_PATH)/bin/nvcc -ccbin=$(NVCC_CCBIN) -D_FORCE_INLINES
PTXAS?=$(CUDA_INSTALL_PATH)/bin/ptxas

DEBUG_TOOL?=0

# Keep direct subdirectory builds on the repository-pinned protobuf runtime.
# The parent Makefile exports these values, but invoking this Makefile directly
# previously fell through to the host protobuf (currently 7.36) and paired it
# with the checked-in 7.34 generated code.
PB_PREFIX ?= $(abspath ../../../.toolchain/pb)
PKG_CONFIG ?= pkg-config
PKG_CONFIG_FLAGS ?= --define-prefix
PKG_CONFIG_PATH ?= $(PB_PREFIX)/lib/pkgconfig
PROTOC ?= $(PB_PREFIX)/bin/protoc
export PKG_CONFIG_PATH PROTOC

OPTFLAGS=

ifneq ($(DEBUG_TOOL),1)
	OPTFLAGS +=-O3
else
	OPTFLAGS +=-g -O0
endif

NVCC_VER_REQ=10.1
NVCC_VER=$(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
# Use awk instead of bc for version comparison (awk is POSIX-required, bc is optional)
NVCC_VER_CHECK=$(shell awk -v a="${NVCC_VER}" -v b="$(NVCC_VER_REQ)" 'BEGIN{print (a+0 >= b+0) ? 1 : 0}')

ifeq ($(NVCC_VER_CHECK),0)
$(error ERROR: nvcc version >= $(NVCC_VER_REQ) required to compile an nvbit tool! Instrumented applications can still use lower versions of nvcc.)
endif


PTXAS_VER_ADD_FLAG=12.3
PTXAS_VER=$(shell $(PTXAS) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
PTXAS_VER_CHECK=$(shell awk -v a="${PTXAS_VER}" -v b="$(PTXAS_VER_ADD_FLAG)" 'BEGIN{print (a+0 >= b+0) ? 1 : 0}')

ifeq ($(PTXAS_VER_CHECK), 0)
	MAXRREGCOUNT_FLAG=-maxrregcount=24
else
	MAXRREGCOUNT_FLAG=
endif

NVBIT_PATH=../nvbit_release/core

LIBS=-L$(NVBIT_PATH) -lnvbit
NVCC_PATH=-L$(CUDA_INSTALL_PATH)/lib64

# Protobuf 34.x depends on Abseil; resolve the full transitive closure via
# pkg-config so the link line follows whatever the host protobuf runtime
# actually needs (rather than hardcoding `-lprotobuf` and getting a DSO-missing
# error at trace_printer link time or an unresolved-symbol failure when the
# tracer is LD_PRELOADed into a CUDA process).
#
# Hard-fail if pkg-config can't resolve protobuf: falling back silently to
# `-lprotobuf` was what the original docker-only build did, and that's exactly
# the link mode that breaks on modern protobuf/Abseil stacks — we want a loud
# error at make time, not a delayed DSO-missing error at trace_printer link
# or an unresolved-symbol crash at LD_PRELOAD time.
PKG_CONFIG?=pkg-config
PKG_CONFIG_FLAGS?=--define-prefix
PROTOBUF_CFLAGS:=$(shell $(PKG_CONFIG) $(PKG_CONFIG_FLAGS) --cflags protobuf 2>/dev/null)
PROTOBUF_LIBS:=$(shell $(PKG_CONFIG) $(PKG_CONFIG_FLAGS) --libs protobuf 2>/dev/null)
ifeq ($(PROTOBUF_CFLAGS),)
$(error pkg-config could not resolve protobuf compile flags; set PKG_CONFIG or PKG_CONFIG_PATH for the installed protobuf)
endif
ifeq ($(PROTOBUF_LIBS),)
$(error pkg-config could not resolve 'protobuf'. Install protobuf-dev (or equivalent) and ensure $(PKG_CONFIG) can find its .pc file. protobuf 34.x pulls in ~80 Abseil libs that bare '-lprotobuf' does not expose — silently falling back would produce a broken tracer_tool.so.)
endif

# Private compiler flags. Do not expose this as generic `INCLUDES`: command-line
# variables propagate to the traces_enhanced sub-make and can replace that
# build's generated-header paths. pkg-config --define-prefix relocates the
# copied .toolchain/pb metadata to the current checkout.
TRACER_INCLUDES=-I$(NVBIT_PATH) $(PROTOBUF_CFLAGS)

SOURCES=$(wildcard *.cu)

# Device code is architecture-specific.  Keep the architecture in the object
# name so switching ARCH cannot reuse an object compiled for another target
# (for example, an sm_86 object when linking the default ARCH=all tool).
TRACER_TOOL_OBJECT=tracer_tool.$(ARCH).o
INJECT_FUNCS_OBJECT=inject_funcs.$(ARCH).o
OBJECTS=$(filter-out tracer_tool.o inject_funcs.o,$(SOURCES:.cu=.o)) \
	$(TRACER_TOOL_OBJECT) $(INJECT_FUNCS_OBJECT)
DEPFILES=$(OBJECTS:.o=.d) $(HOST_OBJECTS:.o=.d)

# Host-only C++ translation units linked into tracer_tool.so.
#
#   allocation_registry.cc — per-context allocator registry that
#     tracer_tool.cu's cuMemAlloc/cuMemFree hooks and
#     recv_thread_fun's lookup call into.
#   exclusion_aggregator.cc — typed-exclusion sidecar accumulator
#     that recv_thread_fun increments on the 6 excluded memory-op
#     classes, serialized to traces/extra_info/*.json at trace
#     finalization (task11b producer side).
#   phase_record_writer.cc — descriptor-relative, single-device,
#     exclusive and durable publisher for canonical phase-record leaves.
HOST_OBJECTS=allocation_registry.o exclusion_aggregator.o phase_record_writer.o
ARCH?=all

ARCH_VER_REQ=11.7 # NVCC < 11.7 doesn't support -arch=all flag
ARCH_VER_CHECK=$(shell awk -v a="${NVCC_VER}" -v b="$(ARCH_VER_REQ)" 'BEGIN{print (a+0 >= b+0) ? 1 : 0}')

ifeq ($(ARCH_VER_CHECK),0)
ifeq ($(ARCH), all)
$(shell echo "${ARCH}, ${ARCH_VER_CHECK}, ${NVCC_VER}" )
$(error ERROR: ARCH environment variable must be set to "sm_XX" for nvcc < 11.7, ex: run "export ARCH=sm_70 for compute capability 70")
endif
endif

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

NVBIT_TOOL=$(current_dir).so

# MOD. Begin. ENHANCED TRACES
# Paths relative to tools/nvbit_tracer/tracer_tool/ (the standalone host-build layout).
TRACES_ENHANCED_DIR=../traces_enhanced/
TRACES_ENHANCED_PROTOCOL_BUFFER_DIR=../traces_enhanced/pb_trace
TRACES_ENHANCED_OBJ=$(wildcard $(TRACES_ENHANCED_DIR)/obj/*.o)
TRACES_ENHANCED_PROTOCOL_BUFFER_OBJ=$(wildcard $(TRACES_ENHANCED_PROTOCOL_BUFFER_DIR)/obj/*.o)
TRACE_PRINTER = trace_printer
# MOD. End. ENHANCED TRACES


all: traces_enhanced $(NVBIT_TOOL) trace_printer

$(NVBIT_TOOL): $(OBJECTS) $(HOST_OBJECTS) $(NVBIT_PATH)/libnvbit.a traces_enhanced
	$(NVCC) -arch=$(ARCH) $(OPTFLAGS) $(OBJECTS) $(HOST_OBJECTS) $(TRACES_ENHANCED_OBJ) $(TRACES_ENHANCED_PROTOCOL_BUFFER_OBJ) $(LIBS) $(NVCC_PATH) -Xlinker --no-as-needed -lcuda -Xlinker --as-needed -lcudart_static $(PROTOBUF_LIBS) -Xlinker -z -Xlinker defs -shared -o $@

# allocation_registry.cc is host-only C++ (no CUDA / NVBit / nvcc).
# Build it with g++ + -fPIC so it can link into the shared tracer .so
# AND into the host-only unit-test binary below.
allocation_registry.o: allocation_registry.cc allocation_registry.h Makefile
	$(CXX) -std=c++20 -Wall -Wextra -fPIC $(OPTFLAGS) -MMD -MP -MF $(@:.o=.d) -MT $@ -c $< -o $@

# exclusion_aggregator.cc is host-only C++ (same constraints as
# allocation_registry.cc). Builds the typed-exclusion sidecar
# accumulator that recv_thread_fun feeds; serialized to
# traces/extra_info/{unsupported_memory_path,unmapped_memory_range}.json
# at trace finalization.
exclusion_aggregator.o: exclusion_aggregator.cc exclusion_aggregator.h Makefile
	$(CXX) -std=c++20 -Wall -Wextra -fPIC $(OPTFLAGS) -MMD -MP -MF $(@:.o=.d) -MT $@ -c $< -o $@

# Descriptor-relative phase-record publication helper. This host-only object
# owns the retained TRACES_FOLDER descriptor and is linked into tracer_tool.so.
phase_record_writer.o: phase_record_writer.cc phase_record_writer.h phase_record_path.h Makefile
	$(CXX) -std=c++20 -Wall -Wextra -fPIC $(OPTFLAGS) -MMD -MP -MF $(@:.o=.d) -MT $@ -c $< -o $@

# Generated protobuf headers the tracer .cu files consume via
# `#include "../traces_enhanced/pb_trace/include/instruction.pb.h"` (and
# trace.pb.h, address.pb.h). The headers live in the traces_enhanced
# subbuild but the .cu files depend on their wire layout: when
# `instruction.proto` changes, both architecture-specific CUDA objects
# MUST be re-emitted before the .so is relinked. Without this dependency
# `make -C tools/nvbit_tracer/tracer_tool` would re-generate
# `instruction.pb.{h,cc}` (because traces_enhanced is a phony recursive
# build) but reuse a stale tracer object against new `instruction.pb.o`,
# producing a tracer .so whose protobuf layout disagrees with the .pb.o
# it was linked against — the failure mode is a glibc malloc abort the
# first time `recv_thread_fun` deserializes a message.
#
# Two pieces of dependency wiring are needed:
#
#   1. The headers are listed as ordinary prerequisites so a proto-
#      driven header change forces a rebuild via mtime comparison.
#   2. `traces_enhanced` is added as an order-only prerequisite (the
#      `|` separator) so a parallel `make -j` or a direct
#      `make $(NVBIT_TOOL)` cannot evaluate the .o rule before the
#      recursive subbuild has produced the headers. Without the
#      order-only dependency, `make` would either error out with
#      `no rule to make target 'instruction.pb.h'` on a clean tree
#      or silently use stale headers under -j parallelism.
TRACES_ENHANCED_PB_HEADERS = \
	$(TRACES_ENHANCED_PROTOCOL_BUFFER_DIR)/include/instruction.pb.h \
	$(TRACES_ENHANCED_PROTOCOL_BUFFER_DIR)/include/trace.pb.h \
	$(TRACES_ENHANCED_PROTOCOL_BUFFER_DIR)/include/address.pb.h \
	$(TRACES_ENHANCED_PROTOCOL_BUFFER_DIR)/include/cuda_stream.pb.h \
	$(TRACES_ENHANCED_PROTOCOL_BUFFER_DIR)/include/gpu_device.pb.h \
	$(TRACES_ENHANCED_PROTOCOL_BUFFER_DIR)/include/kernel.pb.h \
	$(TRACES_ENHANCED_PROTOCOL_BUFFER_DIR)/include/threadblock.pb.h \
	$(TRACES_ENHANCED_PROTOCOL_BUFFER_DIR)/include/warp.pb.h

%.o: %.cu common.h $(TRACES_ENHANCED_PB_HEADERS) | traces_enhanced
	$(NVCC) -dc -c -std=c++20 $(TRACER_INCLUDES) -Xptxas -cloning=no -Xcompiler -Wall -arch=$(ARCH) $(OPTFLAGS) -Xcompiler -fPIC $< -o $@

$(TRACER_TOOL_OBJECT): Makefile tracer_tool.cu common.h $(TRACES_ENHANCED_PB_HEADERS) \
	active_launch_gate.h module_origin_resolver.h phase_record_writer.h phase_record_path.h ../metric_extractor/tier2_sha256.h \
	../traces_enhanced/src/traced_execution.h ../traces_enhanced/src/traced_kernel.h \
	../traces_enhanced/src/traced_instruction.h ../traces_enhanced/src/traced_operand.h \
	../traces_enhanced/src/control_bits.h ../traces_enhanced/src/register_usage.h | traces_enhanced
	$(NVCC) -dc -c -MMD -MP -MF $(@:.o=.d) -MT $@ -std=c++20 $(TRACER_INCLUDES) -Xptxas -cloning=no -Xcompiler -Wall -arch=$(ARCH) $(OPTFLAGS) -Xcompiler -fPIC tracer_tool.cu -o $@

# Compatibility target for callers that explicitly request tracer_tool.o.  The
# live link uses TRACER_TOOL_OBJECT, so this copy cannot contaminate another
# architecture's build.
tracer_tool.o: $(TRACER_TOOL_OBJECT)
	cp $< $@

$(INJECT_FUNCS_OBJECT): inject_funcs.cu Makefile common.h $(TRACES_ENHANCED_PB_HEADERS) | traces_enhanced
	$(NVCC) $(TRACER_INCLUDES) $(MAXRREGCOUNT_FLAG) -Xptxas -astoolspatch --keep-device-functions -MMD -MP -MF $(@:.o=.d) -MT $@ -arch=$(ARCH) -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

# Compatibility target for callers that explicitly request inject_funcs.o.
inject_funcs.o: $(INJECT_FUNCS_OBJECT)
	cp $< $@

# MOD. Begin. ENHANCED TRACES
traces_enhanced:
	$(MAKE) -C $(TRACES_ENHANCED_DIR)
# MOD. End. ENHANCED TRACES

# Build the trace_printer executable
# First compile trace_printer's code into a shared library
trace_printer: traces_enhanced
	$(NVCC) -std=c++20 $(TRACER_INCLUDES) $(OPTFLAGS) -o $(TRACE_PRINTER) $(TRACES_ENHANCED_OBJ) $(TRACES_ENHANCED_PROTOCOL_BUFFER_OBJ) trace_printer.cc $(PROTOBUF_LIBS)

clean:
	rm -f *.so *.o *.d trace_printer $(PHASE_CLASSIFIER_TEST_BIN) $(MEMORY_OP_CLASS_TEST_BIN) $(MEMORY_OP_ACCESS_KIND_TEST_BIN) $(ALLOCATION_REGISTRY_TEST_BIN) $(EXCLUSION_AGGREGATOR_TEST_BIN) $(PATH_RESOLVER_TEST_BIN) $(ACTIVE_LAUNCH_GATE_TEST_BIN) $(MODULE_ORIGIN_RESOLVER_TEST_BIN) $(PHASE_RECORD_WRITER_TEST_BIN)
	$(MAKE) clean -C $(TRACES_ENHANCED_DIR)

.PHONY: all traces_enhanced clean

-include $(DEPFILES)
