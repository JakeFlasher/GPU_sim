# SM86 GCC/L1.5 scope candidate — 2026-10-15

## Scope

H6 adds a CTA-scoped line-domain view for the static `LDG.*.CONSTANT`
observer. It preserves both warp-qualified and CTA-qualified 128-B candidate
sets, making replication scope explicit without equating either candidate with
`gcc__cache_requests_type_constant.sum`. No cache/DRAM/timing state changes.

## Representative full replay

| Root | Warp-scoped candidate lines | CTA-scoped candidate lines | NCU GCC requests |
|---|---:|---:|---:|
| FP16 HMMA | 96 | 96 | 16 |
| FP32 dependency chain | 512 | 512 | 64 |
| local/register spill | 16 | 16 | 16 |
| RMSNorm BF16 | 1,024 | 512 | 384 |

CTA scope improves the local/register-spill relationship but does not identify
a universal GCC request rule: cache residency, line return granularity, and
warp/CTA aggregation differ by workload. NCU values were read only after
replay and were not used to select or fit the scope.

## Complete-corpus evidence

Full-first replay completed 356 roots (286 `ok`, 62 mixed-code controls, 8
replay/input failures), 12,392 launches in 35.194 s with 12 workers:
`../sm86_replay_results/full_corpus_h6_gcc_scope_20261015_v1/full.jsonl.zst`.
Across successful rows: 6,807,596 source-class instructions, 10,817,035
sectors, 7,142,629 per-instruction candidates, 1,039,257 warp candidates,
and 982,844 CTA candidates.

## Residuals and next arm

Exact-identity DRAM and IPC residuals remain unchanged from H5/H4 (40.91%,
18.26%, and 12.53% mean APE for DRAM reads, active cycles, and elapsed IPC).
H6 is therefore a scope-observation result, not an accuracy gain. The next
arm must target a bounded service/denominator relation with independent
mechanism evidence; no global GCC multiplier is promoted.
