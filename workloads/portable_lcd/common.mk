# Shared Makefile include for portable_lcd kernel subdirectories.
#
# Per-kernel makefiles under workloads/portable_lcd/<slug>/Makefile include
# this file to pick up the canonical tool paths and the three architecture
# flag sets that the suite supports under the LCD discipline:
#
#   SM86   -- Ampere consumer (RTX 3090 Ti, GA102): the local smoke-test
#             host on this branch. CC 8.6, sm_80 baseline +
#             cooperative-groups + cp.async + mbarrier.
#   SM90   -- Hopper BASE arch (NOT sm_90a). The cubin slice that runs
#             cleanly on H100 (CC 9.0) without enabling architecture-
#             conditional Hopper features (wgmma, cp.async.bulk.tensor,
#             clusters, DSMEM, setmaxnreg). The H100 device-family is
#             reachable via this slice; sm_90a is forbidden.
#   SM120  -- Workstation Blackwell BASE arch (NOT sm_120a). The cubin
#             slice that runs cleanly on RTX PRO 6000 Blackwell (CC 12.0)
#             without enabling architecture-conditional Blackwell
#             features (FP4/FP6/FP8 MMA, tcgen05/TMEM, sparse mxf4).
#             sm_120a / sm_120f are forbidden.
#
# The single virtual arch `compute_80` is the LCD: every cubin slice is
# compiled from the same compute_80 PTX baseline. The three real-arch
# slices land in the produced fatbin via three -gencode clauses; a
# fourth -gencode clause embeds the compute_80 PTX itself for forward-
# compatibility JIT to any future arch >= sm_80 not covered by the
# embedded SASS.
#
# Architecture-conditional variants (sm_90a, sm_120a, sm_120f) are
# explicitly forbidden as build targets because they would enable
# feature surfaces outside the LCD discipline. The fatbin-slice verifier
# at tools/build/verify_lcd_arch_slices.sh enforces this empirically on
# every produced binary.
#
# Tool overrides are allowed via environment variables. Defaults match
# the pinned toolchain (CUDA 13.2 / NVBit 1.8 / g++ 15.x).

NVCC      ?= nvcc
PTXAS     ?= ptxas
CUOBJDUMP ?= cuobjdump
NVDISASM  ?= nvdisasm

# nvcc preflight: runs at make parse time so that any invocation
# (including `make clean`) on a host without a resolvable nvcc fails
# with a typed `nvcc_missing` diagnostic BEFORE any side effect,
# preserving previously-built runnables on disk and matching the
# negative-path expectation that no partial artifacts are produced.
# `command -v $(NVCC)` succeeds whether $(NVCC) is a bare name on
# PATH or an absolute / relative path to an executable.
ifeq ($(shell command -v $(NVCC) > /dev/null 2>&1 && echo present),)
$(error nvcc_missing)
endif

# -----------------------------------------------------------------------------
# Architecture flag sets (LCD-discipline; BASE archs only)
# -----------------------------------------------------------------------------

ARCH_SM86  = -gencode arch=compute_80,code=sm_86
ARCH_SM90  = -gencode arch=compute_80,code=sm_90
ARCH_SM120 = -gencode arch=compute_80,code=sm_120

# PTX fallback: embeds the compute_80 PTX baseline so the binary is
# JIT-compatible with any future arch >= sm_80 that the embedded SASS
# slices don't cover.
ARCH_PTX_FALLBACK = -gencode arch=compute_80,code=compute_80

# Full LCD flag set (all three base-arch slices + PTX fallback). Any
# kernel declaring ARCH_GATE = sm_86 sm_90 sm_120 expands to this set
# plus the PTX fallback. This is the standard portable_lcd target.
ARCH_LCD_FULL = $(ARCH_SM86) $(ARCH_SM90) $(ARCH_SM120) $(ARCH_PTX_FALLBACK)

# -----------------------------------------------------------------------------
# Common compile flags
# -----------------------------------------------------------------------------

# C++ standard: kept at C++17 to match workloads/simbench_bw/common.mk and
# workloads/cluster_microbench / workloads/d6_microbenchmarks conventions.
NVCC_STD = -std=c++17

# Register-usage reporting. -Xptxas=-v exposes per-kernel register count,
# which the tracer + validator tooling cites.
NVCC_VERBOSE_PTXAS = -Xptxas=-v

# Line info for NVBit tracer PC-to-source mapping. Required for tracer
# cross-reference; harmless otherwise.
NVCC_LINEINFO = -lineinfo

NVCC_COMMON = $(NVCC_STD) $(NVCC_VERBOSE_PTXAS) $(NVCC_LINEINFO)

# -----------------------------------------------------------------------------
# Arch-gate helpers
# -----------------------------------------------------------------------------
#
# A per-kernel Makefile declares arch-gate membership by setting
# ARCH_GATE to a space-separated list of tokens drawn from:
#
#   sm_86      -> emits ARCH_SM86
#   sm_90      -> emits ARCH_SM90  (NOT sm_90a; LCD discipline)
#   sm_120     -> emits ARCH_SM120 (NOT sm_120a / sm_120f; LCD discipline)
#
# The full LCD roster declares ARCH_GATE = sm_86 sm_90 sm_120, which
# expands to ARCH_LCD_FULL (all three base-arch slices + PTX fallback).
# Architecture-conditional tokens (sm_90a, sm_120a, sm_120f) trigger an
# error so the LCD subtree never accidentally produces them.
#
# Legacy alias: `ARCH_GATE = lcd` expands to `sm_86 sm_90 sm_120`.

ARCH_GATE_EXPANDED := $(patsubst lcd,sm_86 sm_90 sm_120,$(ARCH_GATE))

NVCC_ARCH_TOKEN_FLAG = $(if $(filter sm_86,$(1)),$(ARCH_SM86),$(if $(filter sm_90,$(1)),$(ARCH_SM90),$(if $(filter sm_120,$(1)),$(ARCH_SM120),$(if $(filter sm_90a sm_120a sm_120f,$(1)),$(error LCD discipline forbids architecture-conditional token $(1); use base arch sm_90 or sm_120 instead),$(error ARCH_GATE token $(1) must be one of sm_86 sm_90 sm_120; alias lcd is also accepted)))))

NVCC_ARCH_FOR_GATE_BASE := $(strip $(foreach tok,$(ARCH_GATE_EXPANDED),$(call NVCC_ARCH_TOKEN_FLAG,$(tok))))

# Append the PTX fallback to the per-token expansion so every produced
# binary carries a compute_80 PTX section, providing a forward-
# compatibility JIT path that the fatbin-slice verifier checks for.
NVCC_ARCH_FOR_GATE := $(NVCC_ARCH_FOR_GATE_BASE) $(ARCH_PTX_FALLBACK)
