# SimBench-BW top-level Makefile.
#
# Per-kernel Makefiles live under l1/ l2/ l3/ l4/ <slug>/Makefile and
# include common.mk. This file aggregates suite-level workflows that
# don't belong inside a single kernel directory.
#
# Targets:
#
#   test                Convenience wrapper for the local pytest suite.
#                       (Equivalent to `python3 -m pytest tests/`.)
#
#   strict-verify       Run verify_coverage.py --strict (every check).
#                       This is what CI / pre-commit hooks should run.
#
#   verify-kernel-contracts
#                       Build + cuobjdump every row in the per-row
#                       contract spec (Round 8: AC-41 cubin-level
#                       enforcement). Asserts each row's cubin contains
#                       its declared SASS mnemonic (HGMMA / QGMMA /
#                       QMMA / QMMA.SP). Runs OUTSIDE pytest's hot path.
#                       Currently covers the instruction-path-bearing
#                       rows listed in verify_kernel_contracts.py.

PYTHON ?= python3

.PHONY: test strict-verify verify-kernel-contracts

test:
	$(PYTHON) -m pytest tests/ -q

strict-verify:
	$(PYTHON) verify_coverage.py --strict

verify-kernel-contracts:
	$(PYTHON) verify_kernel_contracts.py --strict
