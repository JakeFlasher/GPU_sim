# SM120 host runbook

The active RTX 5090 corpus is `/root/sm120_rtx_5090_capture`. Its `captures/`
and `profiles/` directories each contain 339 direct workload/kernel entries.
NCU is joined after replay and is never a simulator input. Full replay always
precedes the complementary fast frontier.

## Complete replay

```bash
cd /root/GPU_simulator_rebase
# Rust workspace/build root is /root/GPU_simulator_rebase/sim.
# Build before a new real-capture run:
(cd sim && cargo build -p sim-trace --release && cargo build -p sim-run --release)
python3 tools/sm120_corpus_replay.py \
  --capture-root /root/sm120_rtx_5090_capture \
  --output /root/sm120_replay_current \
  --mode both \
  --memory-mode l1-l2 \
  --fast-memory-mode flat \
  --cache-scope launch \
  --sm120-effective-l1 1024:none:50:512 \
  --ncu-shaped-metrics \
  --workers 12
```

The explicit `--workers 12` is mandatory for every complete-corpus
invocation. Full replay runs first and is required to use integrated `l1-l2`;
the complementary fast mode is flat-frontier-only. If effective parameters
are omitted, the binary resolves L1 `1024:none:50:512` and L2
`100663296:1:16` automatically. They are conditional effective scenarios,
not physical geometry claims; protocol identity is built into the default route.
`--cache-scope stream-l2` is an explicit persistent-L2 hypothesis; captures
with mixed device/stream identities fall back to launch-cold and report that
fallback. The driver
publishes `full.jsonl.zst` and `fast.jsonl.zst`; no new uncompressed replay
JSONL is produced.

The release binary is `/root/GPU_simulator_rebase/sim/target/release/sim-run`.
Use the RTX 5090 architecture card in `arch_profiles/cards/` and keep replay
results separate from raw captures/profiles.

## Independent comparison

```bash
python3 tools/sm120_corpus_compare.py \
  --full /root/sm120_replay_current/full.jsonl.zst \
  --fast /root/sm120_replay_current/fast.jsonl.zst \
  --profiles /root/sm120_rtx_5090_capture/profiles \
  --output /root/sm120_replay_current/ncu_join
```

`tools/jsonl_io.py` resolves compressed paths first and legacy uncompressed
siblings second, so logical `full.jsonl`/`fast.jsonl` names remain readable for
older archives.  The comparator writes `launch_comparisons.jsonl.zst`.
Preserve units, launch scope, and endpoint definitions.  Identity joins and
ordered fallback joins must be reported separately; an ordered association is
not an independently verified launch identity.  Do not fit NCU rows back into
replay parameters.

## Effective-L1 conditional diagnostic

The optional cache scenario is intentionally separate from the canonical flat
run. The former 12-root cache slice is archived historical development
material, not an active held-out guarantee; the complete 339-root corpus has
already been exposed. Use a newly predeclared real subset only for descriptive
mechanism work and report it as development-exposed.

```bash
sim/target/release/sim-run \
  --trace /root/sm120_rtx_5090_capture/captures/ROOT \
  --control /root/effective-cache-controls/ROOT.control_bits.json \
  --arch-card arch_profiles/cards/sm_120-rtx-5090.json \
  --sm120-alpha --mode l1 --sm120-effective-l1 1024:none:50:512
```

The `1024:none:50:512` values are the corrected full-ring effective
pointer-chase service scenario, not physical cache geometry.  The retired
`43/501` profile is historical and must not be used. Keep TMA/UTMA roots on
the explicit typed refusal path when native completion and mbarrier visibility
are unresolved. Join NCU rows only after replay and keep cycles and
microseconds in separate timing domains. Raw capture sidecars are retained as
capture evidence; the compressed-artifact rule applies to generated replay and
post-replay JSONL.
