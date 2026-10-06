# Shared Makefile include for SimBench-BW kernel subdirectories.
#
# Per-kernel makefiles under workloads/simbench_bw/<level>/<slug>/Makefile
# include this file to pick up the canonical tool paths and the four
# architecture flag sets that the suite supports:
#
#   SM90A  -- Hopper datacenter (H100): wgmma, cp.async.bulk.tensor, clusters,
#             DSMEM, mbarrier, elect.sync. NOT available on sm_120.
#   SM120  -- Blackwell workstation baseline (RTX PRO 6000 Blackwell cloud
#             target). HMMA FP16/BF16/TF32, QMMA FP8, sparse 2:4, clusters,
#             DSMEM, and TMA are probeable surfaces. FP4/FP6/NVFP4 rows use
#             explicit sm_120f/sm_120a feature probes and must not be inferred
#             from a successful baseline sm_120 compile.
#   SM120F -- Blackwell forward-compatible subset (CUDA 13.0+): adds dense
#             FP4/FP6/NVFP4 block-scaled MMA via warp-level
#             mma.sync.kind::{f8f6f4, mxf4, mxf4nvf4} with .block_scale.
#   SM120A -- Blackwell architecture-accelerated (non-forward-compatible):
#             adds sparse mxf4/mxf4nvf4, cvt.e2m1/cvt.e3m2, and sm_120-specific
#             SASS-level optimizations. Code compiled with SM120A is NOT
#             forward-compatible to future Blackwell dies.
#
# Tool overrides are allowed via environment variables. Defaults match the
# pinned toolchain recorded in workloads/simbench_bw/DEPENDENCIES.md when that
# file is authored.

NVCC      ?= nvcc
PTXAS     ?= ptxas
CUOBJDUMP ?= cuobjdump
NVDISASM  ?= nvdisasm

# -----------------------------------------------------------------------------
# Architecture flag sets
# -----------------------------------------------------------------------------

ARCH_SM86   = -gencode arch=compute_86,code=sm_86
ARCH_SM90A  = -gencode arch=compute_90a,code=sm_90a
ARCH_SM120  = -gencode arch=compute_120,code=sm_120
ARCH_SM120F = -gencode arch=compute_120f,code=sm_120f
ARCH_SM120A = -gencode arch=compute_120a,code=sm_120a

# Portable (cross-generation) flag set. A kernel declaring tier=P compiles
# under this flag set and produces correct results on both H100 and RTX PRO
# 6000 Blackwell. sm_120 is the safe portable baseline on Blackwell consumer
# dies; sm_120f is a superset that adds dense FP4/FP6/NVFP4 block-scaled MMA.
ARCH_PORTABLE = $(ARCH_SM90A) $(ARCH_SM120)

# -----------------------------------------------------------------------------
# Common compile flags
# -----------------------------------------------------------------------------

# C++ standard: kept at C++17 to match the current simbench workload
# conventions. Upgrade requires a per-suite-wide sweep.
NVCC_STD   = -std=c++17

# Register-usage reporting. Tracer + validator tooling cites the exact
# register count per kernel; -Xptxas=-v is the cheapest way to expose it.
NVCC_VERBOSE_PTXAS = -Xptxas=-v

# Line info for NVBit tracer PC-to-source mapping. Required for tracer
# cross-reference; harmless otherwise.
NVCC_LINEINFO = -lineinfo

NVCC_COMMON = $(NVCC_STD) $(NVCC_VERBOSE_PTXAS) $(NVCC_LINEINFO)

# -----------------------------------------------------------------------------
# SASS archival
# -----------------------------------------------------------------------------
#
# Per-kernel makefiles that need a SASS archive add a recipe of the form:
#
#   sass: $(TARGETS)
#   	@set -e; mkdir -p sass_archive; for t in $(TARGETS); do \
#   	    if [ -x ./$$t ]; then \
#   	        $(CUOBJDUMP) --dump-sass ./$$t > sass_archive/$$t.sass; \
#   	    fi; \
#   	done
#
# Keep `set -e` in archive recipes so disassembly failures fail the target
# instead of silently producing incomplete SASS archives.

# -----------------------------------------------------------------------------
# Arch-gate helpers
# -----------------------------------------------------------------------------
#
# A per-kernel Makefile declares arch-gate membership by setting ARCH_GATE
# to a space-separated list of tokens drawn from:
#
#   sm_86      -> emits ARCH_SM86  (Ampere consumer; RTX 3090 Ti local target)
#   sm_90a     -> emits ARCH_SM90A
#   sm_120     -> emits ARCH_SM120
#   sm_120f    -> emits ARCH_SM120F
#   sm_120a    -> emits ARCH_SM120A
#
# The list is deliberately unordered; NVCC_ARCH_FOR_GATE accumulates one
# -gencode clause per token so that a single nvcc invocation produces a
# fatbin containing each requested cubin. This matches the roster file,
# which stores `arch_gate` as an array of tokens. Cross-generation portable
# rows therefore declare `ARCH_GATE = sm_90a sm_120` (or a superset),
# Hopper-only rows declare `ARCH_GATE = sm_90a`, Blackwell-only rows
# declare `ARCH_GATE = sm_120a`, etc. Rows that need to run on the local
# 3090 Ti (sm_86) for cross-arch verification add `sm_86` to their gate;
# this is the local-cross-arch-loop additive widening (no Hopper / Blackwell
# semantics are removed by adding sm_86 to a row).
#
# Legacy convenience alias: `ARCH_GATE = portable` expands to the
# cross-generation pair `sm_90a sm_120`.

# Expand legacy alias.
ARCH_GATE_EXPANDED := $(patsubst portable,sm_90a sm_120,$(ARCH_GATE))

# Per-token lookup helper. Closes parentheses one-for-one with the $(if opens;
# the error branch intentionally avoids literal parens in its message so make
# does not mis-terminate the expansion.
NVCC_ARCH_TOKEN_FLAG = $(if $(filter sm_86,$(1)),$(ARCH_SM86),$(if $(filter sm_90a,$(1)),$(ARCH_SM90A),$(if $(filter sm_120,$(1)),$(ARCH_SM120),$(if $(filter sm_120f,$(1)),$(ARCH_SM120F),$(if $(filter sm_120a,$(1)),$(ARCH_SM120A),$(error ARCH_GATE token $(1) must be one of sm_86 sm_90a sm_120 sm_120f sm_120a; legacy alias portable is also accepted))))))

NVCC_ARCH_FOR_GATE := $(strip $(foreach tok,$(ARCH_GATE_EXPANDED),$(call NVCC_ARCH_TOKEN_FLAG,$(tok))))

# Per-kernel makefiles that need a single representative target flag for a
# single-arch-specific ptxas invocation can use NVCC_ARCH_FOR_GATE_HEAD,
# which is the flag set for the first token only.
NVCC_ARCH_FOR_GATE_HEAD := $(call NVCC_ARCH_TOKEN_FLAG,$(firstword $(ARCH_GATE_EXPANDED)))
