# Simulator status — SM120 alpha

Updated 2026-10-14. The Rust simulator and Python replay tools build from the
`sim/` workspace. The complete real corpus is 339 roots / 11,558 launches per
mode; full replay precedes fast and complete-corpus runs use exactly 12 workers.

Implemented source behavior separates TMA submissions from legacy async-copy
commit state, routes LDGSTS explicitly, preserves barrier scope/precision
uncertainty, bounds diagnostics, and fails closed on unsupported directions or
malformed input. `completion_clock` is intentionally unavailable. Effective L1
and L2 scenarios remain explicit conditional configurations; no physical cache,
MSHR, scheduler, or engine claim is made.

Replay and post-replay generated JSONL use `.jsonl.zst` through
`tools/jsonl_io.py`. Raw capture sidecars and workload manifests are input data
and remain outside that generated-artifact rule. See the proposal `PLAN.md` and
compact reports for current evidence; superseded engineering history is in the
local adversarial dossier.
