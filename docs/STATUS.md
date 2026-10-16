# Simulator status — SM120 alpha

Updated 2026-10-16. The Rust simulator and Python replay tools build from the
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

The current effective-model slice also emits model-relative SM/SMSP IPC rows
(`01307b1c`, refined by `6a634f84`). The complete 339-root full/fast replay
completed after the change; IPC is an effective projection and does not claim
physical issue width or a clock crosswalk.

Q75 is the current active-IPC projection. Its pure-TMA arm applies only to
multi-transfer, payload-bearing, zero-LDGSTS launches outside the two-SM scope;
it uses an 8/5 effective lifetime arm and leaves all other launches on Q74.
The complete Q75 replay passed 339/339 roots and 11,558 launches per mode with
full integrated L1+L2 first, fast flat second, exactly 12 workers, and compressed
artifacts. Full active-IPC mean APE is 9.5537% (median 6.2304%, P90 17.0335%),
with unchanged maximum/family maxima and no paired-launch regression.
