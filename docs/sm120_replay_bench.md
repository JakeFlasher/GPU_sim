# SM120 replay driver

The canonical work root is `/root/sm120_rtx_5090_capture`; its `captures/` and
`profiles/` directories contain 339 workload entries. Full replay precedes the
fast frontier path. Use `tools/sm120_corpus_replay.py` for a complete run and
`tools/sm120_corpus_compare.py` for the independent NCU join.

Complete-corpus invocation requires `--workers 12`. Full mode uses integrated
`l1-l2`; fast mode is flat. The direct binary resolves effective L1
`1024:none:50:512` and effective L2 `100663296:1:16` when full SM120 replay
omits overrides. These are conditional scenarios, not physical geometry
claims. The protocol-specific phase-query route is built in and has no runtime
routing switch.

Generated replay and comparison streams use `.jsonl.zst` through
`tools/jsonl_io.py`; readers accept a legacy uncompressed sibling during
migration. Raw capture sidecars and `workload.jsonl` manifests are inputs and
remain unchanged. NCU values are post-replay comparison data only.

```bash
sim/target/release/sim-run \
  --trace CAPTURE --control CONTROL.json \
  --arch-card arch_profiles/cards/sm_120-rtx-5090.json \
  --mode l1-l2
```

Full SM120 replay rejects flat and L1-only memory modes. Use flat only through
the fast-frontier path. Explicit cache scenarios should be selected for a
bounded real-trace study and reported as effective assumptions. Unresolved
TMA completion remains typed unavailable rather than being replaced by a
universal latency.
