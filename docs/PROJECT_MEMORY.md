# Unified GPU simulator continuation memory

This repository contains two explicit deployment lanes sharing one trace/issue/
memory/replay engine:

* **SM86 / RTX 3090 Ti** (`arch_id=sm_86`, `sku=rtx_3090_ti`): validated local
  compatibility lane. Its card-backed L1/L2/P-chase and observer candidates
  remain architecture-scoped and are selected only for SM86 traces.
* **SM120 / RTX 5090** (`arch_id=sm_120`, `sku=rtx_5090`): effective/conditional
  Blackwell lane. Unmeasured physical geometry remains typed as a research gap;
  effective L1/L2/TMA/IPC and the optional carded-DRAM prototype are explicit
  hypotheses, not recovered proprietary topology.

The common execution path is:

    NVBit/SASS + sidecars -> sim-trace -> sim-sm/sim-mem/sim-engine -> sim-run

Architecture identity is resolved from the architecture card, never inferred
from a bare capture directory. NCU observations are independent comparison
targets, never replay/model inputs. Issue, modeled-event, completion,
launch-envelope and typed-tail frontiers remain separate.

The original lane-specific continuation notes are retained in
`SM86_PROJECT_MEMORY.md` (when supplied by the SM86 branch) and
`SM120_PROJECT_MEMORY.md`. Use them as evidence records, not as cross-architecture
parameter defaults.
