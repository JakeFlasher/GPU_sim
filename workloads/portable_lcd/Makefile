# Top-level Makefile for the portable_lcd workload subtree.
#
# Each kernel lives at workloads/portable_lcd/<slug>/ with its own
# Makefile that includes ../common.mk and declares ARCH_GATE = lcd
# (alias for `sm_86 sm_90 sm_120`).
#
# `make` (no args) builds every kernel slug under this subtree.
# `make <slug>` builds a single kernel.
# `make clean` cleans every slug.
#
# The set of slugs is discovered automatically: every direct
# subdirectory of workloads/portable_lcd/ that contains a Makefile is
# treated as a kernel slug. New kernels are added by dropping a new
# slug directory into this subtree; no edits to this top-level
# Makefile are required.
#
# `tests/` is excluded from the slug set because it holds
# integration tests, not a kernel.

SLUGS := $(patsubst %/,%,$(filter-out tests/ cutlass/ flash_attn/,$(sort $(dir $(wildcard */Makefile)))))

.PHONY: all clean $(SLUGS) help

all: $(SLUGS)

help:
	@echo "portable_lcd top-level Makefile"
	@echo "  make            -- build every kernel slug"
	@echo "  make <slug>     -- build one slug"
	@echo "  make clean      -- clean every slug"
	@echo "  make help       -- this message"
	@echo ""
	@echo "Detected slugs: $(SLUGS)"

$(SLUGS):
	$(MAKE) -C $@ all

clean:
	@for s in $(SLUGS); do $(MAKE) -C $$s clean; done
