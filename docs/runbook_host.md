# SM120 host runbook

The active RTX 5090 corpus is `/root/sm120_rtx_5090_capture`. Its `captures/`
and `profiles/` directories contain 339 workload entries. NCU is joined after
replay and is never a simulator input. Full replay always precedes the fast
frontier.

## Complete replay

```bash
cd /root/GPU_simulator_rebase
(cd sim && cargo build -p sim-trace --release && cargo build -p sim-run --release)
python3 tools/sm120_corpus_replay.py \
  --capture-root /root/sm120_rtx_5090_capture \
  --output /root/sm120_replay_current \
  --mode both \
  --memory-mode l1-l2 \
  --fast-memory-mode flat \
  --cache-scope launch \
  --workers 12
```

Complete-corpus runs use twelve workers. The full lane uses integrated L1+L2;
the fast lane is flat-frontier-only. If effective parameters are omitted, the
binary resolves the current effective L1 and L2 scenarios. Explicit values are
conditional research scenarios, not physical geometry claims. The driver
writes `full.jsonl.zst` and `fast.jsonl.zst`.

The release binary is `/root/GPU_simulator_rebase/sim/target/release/sim-run`.
Keep replay results separate from raw captures and retain only compact reports
in the canonical result directory.

## Independent comparison

```bash
python3 tools/sm120_corpus_compare.py \
  --full /root/sm120_replay_current/full.jsonl.zst \
  --fast /root/sm120_replay_current/fast.jsonl.zst \
  --profiles /root/sm120_rtx_5090_capture/profiles \
  --output /root/sm120_replay_current/ncu_join
```

`tools/jsonl_io.py` resolves compressed paths first and accepts a legacy plain
sibling during migration. The comparator writes compressed comparison output.
Identity joins and ordered fallback joins are reported separately; an ordered
association is not an independently verified launch identity. Do not fit NCU
rows back into replay parameters.

## Conditional cache scenarios

An explicit L1 or L2 scenario can be used for a bounded real-trace study:

```bash
sim/target/release/sim-run \
  --trace /root/sm120_rtx_5090_capture/captures/ROOT \
  --control /root/effective-cache-controls/ROOT.control_bits.json \
  --arch-card arch_profiles/cards/sm_120-rtx-5090.json \
  --mode l1-l2 \
  --sm120-effective-l1 1024:none:50:512 \
  --sm120-effective-l2 100663296:1:16
```

These values describe an effective scenario. They are not claims about the
RTX 5090's private set/hash/associativity or engine service. Full SM120 replay
rejects flat and L1-only modes; use `--mode flat` only with the fast-frontier
path. Keep unresolved TMA completion typed rather than substituting a global
latency.
