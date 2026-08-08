NVBIT_INSIDE_GIT_WORKTREE := $(shell git rev-parse --is-inside-work-tree 2>/dev/null || true)
ifeq ($(NVBIT_INSIDE_GIT_WORKTREE),true)
$(error H3 refuses tracer_tool mutation in every Git worktree; use a verified non-Git temporary copy)
endif
ifneq ($(NVBIT_TEMPCOPY_BUILD_OK),1)
$(error non-Git temporary-copy build requires NVBIT_TEMPCOPY_BUILD_OK=1)
endif

NVCC_CCBIN?=$(CXX)
NVCC=nvcc -ccbin=$(NVCC_CCBIN) -D_FORCE_INLINES
PTXAS=ptxas

DEBUG_TOOL?=0

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
$(error pkg-config could not resolve protobuf compile flags. Source ../pinned_build_env.sh before building; the generated protobuf 34.1 headers require the exact pinned C++ runtime headers.)
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

OBJECTS=$(SOURCES:.cu=.o)

# Host-only C++ translation units linked into tracer_tool.so. Each is
# kept as a separate .cc so the same translation unit also builds
# into a host-only unit-test binary under tests/, with no NVBit /
# CUDA / nvcc surface leaking into the test (mirrors the
# phase_classifier / memory_op_class test pattern).
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
	$(NVCC) -arch=$(ARCH) $(OPTFLAGS) $(OBJECTS) $(HOST_OBJECTS) $(TRACES_ENHANCED_OBJ) $(TRACES_ENHANCED_PROTOCOL_BUFFER_OBJ) $(LIBS) $(NVCC_PATH) -Xlinker --no-as-needed -lcuda -Xlinker --as-needed -lcudart_static $(PROTOBUF_LIBS) -shared -o $@

# allocation_registry.cc is host-only C++ (no CUDA / NVBit / nvcc).
# Build it with g++ + -fPIC so it can link into the shared tracer .so
# AND into the host-only unit-test binary below.
allocation_registry.o: allocation_registry.cc allocation_registry.h
	$(CXX) -std=c++20 -Wall -Wextra -fPIC $(OPTFLAGS) -c $< -o $@

# exclusion_aggregator.cc is host-only C++ (same constraints as
# allocation_registry.cc). Builds the typed-exclusion sidecar
# accumulator that recv_thread_fun feeds; serialized to
# traces/extra_info/{unsupported_memory_path,unmapped_memory_range}.json
# at trace finalization.
exclusion_aggregator.o: exclusion_aggregator.cc exclusion_aggregator.h
	$(CXX) -std=c++20 -Wall -Wextra -fPIC $(OPTFLAGS) -c $< -o $@

# Descriptor-relative phase-record publication helper. This host-only object
# owns the retained TRACES_FOLDER descriptor and is linked into tracer_tool.so.
phase_record_writer.o: phase_record_writer.cc phase_record_writer.h phase_record_path.h
	$(CXX) -std=c++20 -Wall -Wextra -fPIC $(OPTFLAGS) -c $< -o $@

# Generated protobuf headers the tracer .cu files consume via
# `#include "../traces_enhanced/pb_trace/include/instruction.pb.h"` (and
# trace.pb.h, address.pb.h). The headers live in the traces_enhanced
# subbuild but the .cu files depend on their wire layout: when
# `instruction.proto` changes, both `tracer_tool.o` and `inject_funcs.o`
# MUST be re-emitted before the .so is relinked. Without this dependency
# `make -C tools/nvbit_tracer/tracer_tool` would re-generate
# `instruction.pb.{h,cc}` (because traces_enhanced is a phony recursive
# build) but reuse stale `tracer_tool.o` against new `instruction.pb.o`,
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

tracer_tool.o: active_launch_gate.h module_origin_resolver.h phase_record_writer.h phase_record_path.h ../metric_extractor/tier2_sha256.h

inject_funcs.o: inject_funcs.cu common.h $(TRACES_ENHANCED_PB_HEADERS) | traces_enhanced
	$(NVCC) $(TRACER_INCLUDES) $(MAXRREGCOUNT_FLAG) -Xptxas -astoolspatch --keep-device-functions -arch=$(ARCH) -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

# MOD. Begin. ENHANCED TRACES
traces_enhanced:
	$(MAKE) -C $(TRACES_ENHANCED_DIR)
# MOD. End. ENHANCED TRACES

# Build the trace_printer executable
# First compile trace_printer's code into a shared library
trace_printer: traces_enhanced
	$(NVCC) -std=c++20 $(TRACER_INCLUDES) $(OPTFLAGS) -o $(TRACE_PRINTER) $(TRACES_ENHANCED_OBJ) $(TRACES_ENHANCED_PROTOCOL_BUFFER_OBJ) trace_printer.cc $(PROTOBUF_LIBS)

# phase_classifier_test: standalone host-only unit test for the three
# opcode-short predicates in phase_classifier.h. Compiles with g++ alone
# (no NVBit, no CUDA, no nvcc) so the test surface is reachable on hosts
# without a GPU toolchain.
TESTS_DIR=tests
PHASE_CLASSIFIER_TEST_BIN=$(TESTS_DIR)/phase_classifier_test
PHASE_CLASSIFIER_TEST_SRC=$(TESTS_DIR)/phase_classifier_test.cc

$(PHASE_CLASSIFIER_TEST_BIN): $(PHASE_CLASSIFIER_TEST_SRC) phase_classifier.h
	$(CXX) -std=c++20 -Wall -Wextra -O2 -o $@ $(PHASE_CLASSIFIER_TEST_SRC)

# memory_op_class_test: standalone host-only unit test for the
# `classify` helper in memory_op_class.h. Same g++-only build pattern as
# phase_classifier_test — no NVBit, no protobuf, no nvcc — so the
# classifier surface stays reachable on hosts without a GPU toolchain.
MEMORY_OP_CLASS_TEST_BIN=$(TESTS_DIR)/memory_op_class_test
MEMORY_OP_CLASS_TEST_SRC=$(TESTS_DIR)/memory_op_class_test.cc

$(MEMORY_OP_CLASS_TEST_BIN): $(MEMORY_OP_CLASS_TEST_SRC) memory_op_class.h
	$(CXX) -std=c++20 -Wall -Wextra -O2 -o $@ $(MEMORY_OP_CLASS_TEST_SRC)

# memory_op_access_kind_test: standalone host-only unit test for the
# `classify` helper in memory_op_access_kind.h. The access-kind
# classifier is orthogonal to the memory-op-class classifier: it tags
# the direction (load / store / atomic / async-copy) rather than the
# memory space. Same g++-only build pattern — no NVBit, no protobuf,
# no nvcc — so the producer-side write-policy classifier stays
# reachable on hosts without a GPU toolchain.
MEMORY_OP_ACCESS_KIND_TEST_BIN=$(TESTS_DIR)/memory_op_access_kind_test
MEMORY_OP_ACCESS_KIND_TEST_SRC=$(TESTS_DIR)/memory_op_access_kind_test.cc

$(MEMORY_OP_ACCESS_KIND_TEST_BIN): $(MEMORY_OP_ACCESS_KIND_TEST_SRC) memory_op_access_kind.h
	$(CXX) -std=c++20 -Wall -Wextra -O2 -o $@ $(MEMORY_OP_ACCESS_KIND_TEST_SRC)

# allocation_registry_test: standalone host-only unit test for the
# per-context allocation registry in allocation_registry.cc. Same g++-
# only pattern: no NVBit, no CUDA driver headers, no nvcc. The test
# casts pointer constants to the registry's `void *` context handle
# parameter so it is reachable on hosts without a GPU toolchain.
ALLOCATION_REGISTRY_TEST_BIN=$(TESTS_DIR)/allocation_registry_test
ALLOCATION_REGISTRY_TEST_SRC=$(TESTS_DIR)/allocation_registry_test.cc

$(ALLOCATION_REGISTRY_TEST_BIN): $(ALLOCATION_REGISTRY_TEST_SRC) allocation_registry.h allocation_registry.cc
	$(CXX) -std=c++20 -Wall -Wextra -O2 -o $@ $(ALLOCATION_REGISTRY_TEST_SRC) allocation_registry.cc

# exclusion_aggregator_test: standalone host-only unit test for the
# typed-exclusion accumulator. Same g++-only build pattern, no NVBit/
# CUDA dependency.
EXCLUSION_AGGREGATOR_TEST_BIN=$(TESTS_DIR)/exclusion_aggregator_test
EXCLUSION_AGGREGATOR_TEST_SRC=$(TESTS_DIR)/exclusion_aggregator_test.cc

$(EXCLUSION_AGGREGATOR_TEST_BIN): $(EXCLUSION_AGGREGATOR_TEST_SRC) exclusion_aggregator.h exclusion_aggregator.cc
	$(CXX) -std=c++20 -Wall -Wextra -O2 -o $@ $(EXCLUSION_AGGREGATOR_TEST_SRC) exclusion_aggregator.cc

# path_resolver_test: standalone host-only unit test for the
# `resolve_output_path` helper in path_resolver.h. Same g++-only
# build pattern: no NVBit, no CUDA, no nvcc. The helper itself is
# a header-only inline; the test simply includes the header.
PATH_RESOLVER_TEST_BIN=$(TESTS_DIR)/path_resolver_test
PATH_RESOLVER_TEST_SRC=$(TESTS_DIR)/path_resolver_test.cc

$(PATH_RESOLVER_TEST_BIN): $(PATH_RESOLVER_TEST_SRC) path_resolver.h
	$(CXX) -std=c++20 -Wall -Wextra -O2 -o $@ $(PATH_RESOLVER_TEST_SRC)

ACTIVE_LAUNCH_GATE_TEST_BIN=$(TESTS_DIR)/active_launch_gate_test
ACTIVE_LAUNCH_GATE_TEST_SRC=$(TESTS_DIR)/active_launch_gate_test.cc

$(ACTIVE_LAUNCH_GATE_TEST_BIN): $(ACTIVE_LAUNCH_GATE_TEST_SRC) active_launch_gate.h
	$(CXX) -std=c++20 -Wall -Wextra -O2 -o $@ $(ACTIVE_LAUNCH_GATE_TEST_SRC)

MODULE_ORIGIN_RESOLVER_TEST_BIN=$(TESTS_DIR)/module_origin_resolver_test
MODULE_ORIGIN_RESOLVER_TEST_SRC=$(TESTS_DIR)/module_origin_resolver_test.cc
ACTIVE_RANGE_SOURCE_TEST=$(TESTS_DIR)/active_range_source_test.py

$(MODULE_ORIGIN_RESOLVER_TEST_BIN): $(MODULE_ORIGIN_RESOLVER_TEST_SRC) module_origin_resolver.h ../metric_extractor/tier2_sha256.h
	$(CXX) -std=c++20 -Wall -Wextra -O2 -o $@ $(MODULE_ORIGIN_RESOLVER_TEST_SRC)

# phase_record_writer_test: standalone hostile-filesystem regression for the
# exact helper linked into the live tracer. Test-only hooks bind post-create
# link-count races, write/fsync/close failures, PID ownership, cleanup-error
# precedence, and fatal integration; no CUDA, NVBit, protobuf, or nvcc
# dependency is present.
PHASE_RECORD_WRITER_TEST_BIN=$(TESTS_DIR)/phase_record_writer_test
PHASE_RECORD_WRITER_TEST_SRC=$(TESTS_DIR)/phase_record_writer_test.cc

$(PHASE_RECORD_WRITER_TEST_BIN): $(PHASE_RECORD_WRITER_TEST_SRC) phase_record_writer.cc phase_record_writer.h phase_record_path.h
	$(CXX) -std=c++20 -Wall -Wextra -O2 -DSIMBENCH_PHASE_RECORD_WRITER_TESTING -o $@ $(PHASE_RECORD_WRITER_TEST_SRC) phase_record_writer.cc

unit-tests: $(PHASE_CLASSIFIER_TEST_BIN) $(MEMORY_OP_CLASS_TEST_BIN) $(MEMORY_OP_ACCESS_KIND_TEST_BIN) $(ALLOCATION_REGISTRY_TEST_BIN) $(EXCLUSION_AGGREGATOR_TEST_BIN) $(PATH_RESOLVER_TEST_BIN) $(ACTIVE_LAUNCH_GATE_TEST_BIN) $(MODULE_ORIGIN_RESOLVER_TEST_BIN) $(PHASE_RECORD_WRITER_TEST_BIN)
	./$(PHASE_CLASSIFIER_TEST_BIN)
	./$(MEMORY_OP_CLASS_TEST_BIN)
	./$(MEMORY_OP_ACCESS_KIND_TEST_BIN)
	./$(ALLOCATION_REGISTRY_TEST_BIN)
	./$(EXCLUSION_AGGREGATOR_TEST_BIN)
	./$(PATH_RESOLVER_TEST_BIN)
	./$(ACTIVE_LAUNCH_GATE_TEST_BIN)
	./$(MODULE_ORIGIN_RESOLVER_TEST_BIN)
	./$(PHASE_RECORD_WRITER_TEST_BIN)
	python3 $(ACTIVE_RANGE_SOURCE_TEST)

clean:
	rm -f *.so *.o trace_printer $(PHASE_CLASSIFIER_TEST_BIN) $(MEMORY_OP_CLASS_TEST_BIN) $(MEMORY_OP_ACCESS_KIND_TEST_BIN) $(ALLOCATION_REGISTRY_TEST_BIN) $(EXCLUSION_AGGREGATOR_TEST_BIN) $(PATH_RESOLVER_TEST_BIN) $(ACTIVE_LAUNCH_GATE_TEST_BIN) $(MODULE_ORIGIN_RESOLVER_TEST_BIN) $(PHASE_RECORD_WRITER_TEST_BIN)
	$(MAKE) clean -C $(TRACES_ENHANCED_DIR)

.PHONY: all traces_enhanced unit-tests clean
