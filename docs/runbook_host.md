# SM120 host runbook

The active RTX 5090 corpus is `/root/sm120_rtx_5090_capture`. Its `captures/`
and `profiles/` directories each contain 339 direct workload/kernel entries.
NCU is joined after replay and is never a simulator input. Full replay always
precedes the complementary fast frontier.

## Complete replay

```bash
cd /root/GPU_simulator_rebase
python3 tools/sm120_corpus_replay.py \
  --capture-root /root/sm120_rtx_5090_capture \
  --output /root/sm120_replay_current \
  --mode both
```

The release binary is `/root/GPU_simulator_rebase/sim/target/release/sim-run`.
Use the RTX 5090 architecture card in `arch_profiles/cards/` and keep replay
results separate from raw captures/profiles.

## Independent comparison

```bash
python3 tools/sm120_corpus_compare.py \
  --full /root/sm120_replay_current/full.jsonl \
  --fast /root/sm120_replay_current/fast.jsonl \
  --profiles /root/sm120_rtx_5090_capture/profiles \
  --output /root/sm120_replay_current/ncu_join
```

Preserve units, launch scope, and endpoint definitions. Do not fit NCU rows
back into replay parameters.

## Effective-L1 held-out diagnostic

The optional cache scenario is intentionally separate from the canonical flat
run. It must use a frozen real subset and an independently measured envelope:

```bash
sim/target/release/sim-run \
  --trace /root/sm120_rtx_5090_capture/captures/ROOT \
  --control /root/effective-cache-controls/ROOT.control_bits.json \
  --arch-card arch_profiles/cards/sm_120-rtx-5090.json \
  --sm120-alpha --mode l1 --sm120-effective-l1 1024:none:43:501
```

The `1024:none:43:501` values are an effective pointer-chase service
scenario, not physical cache geometry. Keep TMA/UTMA roots on the explicit
typed refusal path until native completion and mbarrier visibility are
resolved. Join NCU rows only after replay and keep cycles and microseconds in
separate timing domains.
