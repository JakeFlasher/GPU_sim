# SM120 replay driver

The canonical work root is `/root/sm120_rtx_5090_capture`; `captures/` and
`profiles/` each use 339 direct workload names. Full replay precedes the fast
frontier path.

Use `tools/sm120_corpus_replay.py` for a complete run and
`tools/sm120_corpus_compare.py` for the post-replay independent NCU join.
Keep execution output outside the raw corpus, then retain only the compact
result surfaces needed for the report. NCU and CUDA-event records are
independent observations and never replay inputs.

## Effective cache scenario

SM120 remains flat by default. For a bounded real-device cache experiment, the
direct runner accepts an explicit effective L1 envelope:

```bash
sim/target/release/sim-run \
  --trace CAPTURE --control CONTROL.json \
  --arch-card arch_profiles/cards/sm_120-rtx-5090.json \
  --sm120-alpha --mode l1 \
  --sm120-effective-l1 1024:none:43:501
```

The fields are `line-capacity:mshr-count|none:hit-cycles:miss-cycles`. They
are one measured scenario for a fully-associative effective model, not an
RTX 5090 set/hash/associativity claim. The option is rejected unless the
SM120 alpha lane and an L1 cache mode are selected; unresolved UTMA/TMA
completion remains a typed refusal rather than a generic global fallback.
