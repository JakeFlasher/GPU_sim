# Simulator status

**Updated:** 2026-10-13
**Plan:** `/root/gpu_cache_memory_modeling_proposal/PLAN.md`
**Code:** `/root/GPU_simulator_rebase`
**Corpus:** `/root/sm120_rtx_5090_capture`

The complete real-trace corpus is consolidated at one root. The saved release
run covers 339 full and 339 fast roots, with 11,558 launches per mode. The
current adapter keeps capture discovery, native static-sidecar transport, replay
launch, NCU parsing, and post-replay comparison only.

The Rust path is conservative when SM120 issue, cache, or asynchronous completion
state is not directly observed. Raw async identities, the strict
`STRONG.SM`/`STRONG.GPU` route distinction, and explicit async timing-evidence
classes are core behavior; they are not claims of recovered proprietary topology. Use the proposal plan for the next
mechanism-local change. Do not rerun the corpus until the active cleanup/debug
stage is complete.
