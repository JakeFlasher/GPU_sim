# Read-only tombstone for the retired direct-NCU calibration binaries.
#
# The two CUDA sources are retained as historical evidence. This Makefile must
# not compile, capture, merge, clean, or otherwise mutate that evidence.

override MAKEFLAGS_INPUT_ORIGIN := $(origin MAKEFLAGS)
override MAKEFLAGS += -rR
.SUFFIXES:
override .DEFAULT_GOAL := help

override HISTORICAL_SOURCES := \
	problem_018_fused_rope_with_qk_norm_and_kv_cache_update.cu \
	problem_071_kv_cache_update_with_rope.cu

override RETIRED_TARGETS := \
	all \
	build \
	clean \
	capture \
	harvest \
	merge \
	problem_018_fused_rope_with_qk_norm_and_kv_cache_update \
	problem_071_kv_cache_update_with_rope

override AUTHORIZED_GOALS := help inventory list
override MAKECMDGOALS_INPUT_ORIGIN := $(origin MAKECMDGOALS)
ifeq ($(filter undefined default,$(MAKECMDGOALS_INPUT_ORIGIN)),)
$(error command-line MAKECMDGOALS is not an authorized historical inventory action)
endif
override REQUESTED_GOALS := $(strip $(MAKECMDGOALS))
override NON_INVENTORY_GOALS := $(filter-out $(AUTHORIZED_GOALS),$(REQUESTED_GOALS))
override SOURCE_REFUSAL_GOALS := $(foreach goal,$(NON_INVENTORY_GOALS),\
	$(if $(or \
		$(findstring %,$(goal)),\
		$(findstring *,$(goal)),\
		$(findstring ?,$(goal)),\
		$(findstring [,$(goal)),\
		$(findstring \,$(goal))),,\
		$(if $(or \
			$(wildcard $(patsubst %.o,%,$(goal)).cc),\
			$(wildcard $(patsubst %.o,%,$(goal)).cu)),\
			$(goal))))
override UNAUTHORIZED_GOALS := $(filter-out $(SOURCE_REFUSAL_GOALS),$(NON_INVENTORY_GOALS))

ifneq ($(UNAUTHORIZED_GOALS),)
$(error target set "$(UNAUTHORIZED_GOALS)" is not an authorized historical inventory action)
endif

ifneq ($(SOURCE_REFUSAL_GOALS),)
ifneq ($(MAKEFLAGS_INPUT_ORIGIN),file)
$(error command-line MAKEFLAGS is not an authorized historical inventory action)
endif
ifeq ($(findstring n,$(firstword $(MAKEFLAGS))),)
$(error source-shaped target set is not an authorized historical inventory action)
endif
ifneq ($(or $(findstring i,$(firstword $(MAKEFLAGS))),$(findstring q,$(firstword $(MAKEFLAGS))),$(findstring t,$(firstword $(MAKEFLAGS)))),)
$(error source-shaped target set is not an authorized historical inventory action under ignore, question, or touch modes)
endif
endif

.PHONY: help inventory list $(RETIRED_TARGETS)

help:
	@printf '%s\n' \
		'This direct-NCU workflow is retired and non-executable.' \
		'Authorized read-only actions: help inventory list'

inventory:
	@printf '%s\n' $(HISTORICAL_SOURCES)

list:
	@printf '%s\n' help inventory list

$(RETIRED_TARGETS):
	@printf '%s\n' 'target "$@" is retired and non-executable' >&2; exit 2

%.o: %.cc
	@printf '%s\n' 'target "$@" is not an authorized historical inventory action' >&2; exit 2

%: %.cc
	@printf '%s\n' 'target "$@" is not an authorized historical inventory action' >&2; exit 2

%.o: %.cu
	@printf '%s\n' 'target "$@" is retired and non-executable' >&2; exit 2

%: %.cu
	@printf '%s\n' 'target "$@" is retired and non-executable' >&2; exit 2

.DEFAULT:
	@printf '%s\n' 'target "$@" is not an authorized historical inventory action' >&2; exit 2
