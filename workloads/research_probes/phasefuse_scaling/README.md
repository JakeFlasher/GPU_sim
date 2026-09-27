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
The Python wrapper records one row per case/repeat, including bounded raw
stdout/stderr, wall/user/system time, peak RSS, timeout, exit status, and
contract failures. It emits a median summary alongside the raw JSONL and
returns non-zero if any repeat fails, the contract is malformed, the matrix is
incomplete, or the aggregate has no successful repeats. Child output uses
temporary files instead of pipes to avoid wait/pipe-buffer deadlock. The
matrix is `N={1000,2000,4000,8000,16000,32000,64000}` × region length
`{1,2,8,64}` × residency `{complete,unknown}` × order `{sorted,shuffled}`:
112 cases. The executable requires `N >= 4` so all four physical-warp
programs are non-empty. A high refusal rate is expected for unknown residency;
terminal-state equality and reason-specific refusals are the correctness
criteria. The summary records SHA-256 identities for the binary and source
inputs; hashes identify a capture but are not approval gates.

For a five-repeat bounded capture, write to a new directory rather than
overwriting an earlier result:

```bash
capture="results/captures/phasefuse_scaling_hardened_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$capture"
python benchmark.py \
  --binary ../../../sim/target/release/phasefuse-scale \
  --repeats 5 --timeout 30 --seed 7 \
  --output "../../../$capture/results.jsonl"
```
