# PhaseFuse bounded scaling probe

This probe measures the public causal-frontier analyzer on deterministic
synthetic descriptors. It is **not** a native SM86 trace, cycle-accurate replay,
NCU validation, or a claim of O(N) behavior. The analyzer currently performs
canonicalization/global ordering, so sorted and shuffled inputs are reported
separately.

```bash
cargo build --release --manifest-path sim/Cargo.toml \
  -p sim-sm --bin phasefuse-scale
python benchmark.py --binary ../../../sim/target/release/phasefuse-scale \
  --repeats 3 --output /tmp/phasefuse_scale.jsonl
```

The child JSON follows [`phasefuse_scale.v1.schema.json`](../../../sim/run/schema/phasefuse_scale.v1.schema.json).
The Python wrapper records one
row per case/repeat, including wall/user/system time, peak RSS, timeout and exit
status, and emits a median summary alongside the raw JSONL. The matrix is
`N={1000,2000,4000,8000,16000,32000,64000}` × region length `{1,2,8,64}` ×
residency `{complete,unknown}` × order `{sorted,shuffled}`. A high refusal rate
is expected for unknown residency; terminal-state equality and reason-specific
refusals are the correctness criteria.
