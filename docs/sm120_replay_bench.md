# SM120 replay driver

The canonical work root is `/root/sm120_rtx_5090_capture`; `captures/` and
`profiles/` each use 339 direct workload names. Full replay precedes the fast
frontier path.

Use `tools/sm120_corpus_replay.py` for a complete run and
`tools/sm120_corpus_compare.py` for the post-replay independent NCU join.
Keep execution output outside the raw corpus, then retain only the compact
result surfaces needed for the report. NCU and CUDA-event records are
independent observations and never replay inputs. Complete-corpus replay must
pass `--workers 12`; full mode is primary and fast mode is complementary.
Replay and comparison JSONL are streamed as `.jsonl.zst` through
`tools/jsonl_io.py`; readers accept a legacy uncompressed sibling when one is
found. Raw capture sidecars (`tma_trace.jsonl`, `sync_trace.jsonl`) and
`workload.jsonl` manifests are not replay artifacts and remain unchanged.

SM120 alpha uses the protocol-specific phase-query route by default; no
separate protocol-routing switch is required. Effective cache options remain
explicit conditional scenarios.

## Effective cache scenario

SM120 full replay is integrated `l1-l2` by default. It resolves effective L1
and L2 envelopes automatically; explicit values override those defaults for a
conditional scenario. Flat is reserved for the complementary fast frontier:

```bash
sim/target/release/sim-run \
  --trace CAPTURE --control CONTROL.json \
  --arch-card arch_profiles/cards/sm_120-rtx-5090.json \
  --sm120-alpha --mode l1-l2
```

The default fields are `1024:none:50:512` for L1 and `100663296:1:16` for
L2. They are effective scenarios, not RTX 5090 set/hash/associativity claims.
The L1 fields are `line-capacity:mshr-count|none:hit-cycles:miss-cycles`;
`50/512` supersedes the invalid
short-cycle `43/501` profile. Full SM120 replay rejects flat and L1-only memory
modes; these values are accepted as an explicit override of the integrated
L1/L2 path. Unresolved UTMA/TMA
completion remains a typed refusal rather than a generic global fallback.
