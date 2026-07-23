# Host-side regression tests for the tracer's per-arch opcode maps.
#
# The opcode maps are header-only C++ data structures, so the tests can be
# compiled and run with nothing more than g++; no CUDA, no NVBit, no protobuf.
# This keeps the lockdown for opcode routing (e.g. the Blackwell SM120
# SR_SWINHI peer-DSMEM hygiene for CGAERRBAR / LDCU) reproducible from a
# clean host shell.
#
# Usage:
#   make -C tools/nvbit_tracer/isa_def test      # build + run all tests
#   make -C tools/nvbit_tracer/isa_def clean

CXX ?= g++
CXXFLAGS ?= -std=c++20 -Wall -Wextra -O0

TESTS := test_opcode_map

.PHONY: all test clean
all: test

test: $(TESTS)
	@set -e; for t in $(TESTS); do \
	  echo "==> ./$${t}"; \
	  ./$${t}; \
	done

test_opcode_map: test_opcode_map.cc blackwell_opcode.h trace_opcode.h operation_type.h
	$(CXX) $(CXXFLAGS) test_opcode_map.cc -o $@

clean:
	rm -f $(TESTS)
