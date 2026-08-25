# Completed milestone: sm_86 operation-specific write-completion latency

**Status:** complete 2026-08-19; schema-12 mechanism retained opt-in, all latencies rejected  
**Architecture:** `sm_86` / RTX 3090 Ti  
**Predecessor:**
[write-class completion feedback](sm86_write_class_completion_feedback_milestone.md)  
**Evidence:**
`results/captures/sm86_l1tex_admission_backpressure_20260818_v1/analysis/write_completion_latency_sweep.md`

## Goal and component

Test whether the remaining STORE visibility or RED RMW fence residual is a
fixed post-service tail. Schema 12 adds `write_completion_latency_cycles` to a
schema-11 candidate. The latency begins after service done, does not occupy the
service server, does not move queue release, and changes only the issuing warp's
MEMBAR completion frontier. A nonzero latency without schema-11 MEMBAR feedback
fails validation. Omitted/zero remains schema 11 and byte compatible.

`terminal_service_completion_time` remains the service frontier;
`terminal_write_completion_time` reports service plus the optional tail.

## Frozen experiment

STORE tests 0/512/1024/1536/2048 cycles. RED tests
0/1024/2048/3072/4096. Queue scope, capacity, hit/miss rates, width, cache order,
and admission timing are frozen. Five shallow unique-line cells select the
fence/GPC Pareto front; all Pareto candidates advance to 20 holdout cells.

| tail | latency | calibration fence MAE | calibration GPC MAE | Pareto |
|---|---:|---:|---:|---|
| STORE | 0 | 1,425.8 | 451.0 | yes |
| STORE | 512 | 1,323.4 | 474.8 | yes |
| STORE | 1,024 | 1,225.0 | 980.0 | yes |
| STORE | 1,536 | 1,327.4 | 1,492.0 | no |
| STORE | 2,048 | 1,558.2 | 2,004.0 | no |
| RED | 0 | 2,922.2 | 1,026.0 | yes |
| RED | 1,024 | 2,307.8 | 2,050.0 | yes |
| RED | 2,048 | 2,071.0 | 3,074.0 | yes |
| RED | 3,072 | 1,902.2 | 4,098.0 | yes |
| RED | 4,096 | 2,107.0 | 5,122.0 | no |

## Strict holdout

| tail | latency | fence MAE | GPC MAE | fence max | GPC max |
|---|---:|---:|---:|---:|---:|
| STORE | 0 | 496.4 | 1,124.8 | 2,456 | 3,452 |
| STORE | 512 | 519.7 | 1,363.5 | 1,944 | 3,452 |
| STORE | 1,024 | 792.9 | 1,758.9 | 1,432 | 3,452 |
| RED | 0 | 3,872.3 | 1,466.0 | 9,599 | 3,112 |
| RED | 1,024 | 3,279.8 | 2,014.1 | 8,575 | 3,820 |
| RED | 2,048 | 2,840.3 | 2,683.0 | 7,551 | 4,844 |
| RED | 3,072 | 2,512.8 | 3,534.7 | 6,527 | 5,868 |

STORE zero latency dominates every positive latency on holdout. RED exhibits a
monotonic trade: a longer tail improves fence fit only by degrading GPC fit. No
candidate improves both predecessor targets. A constant latency is therefore
not the missing completion semantics.

## Decision and successor

Promoted only as explicit research machinery:

- schema-12 non-occupying completion latency;
- typed rejection when MEMBAR completion is disabled;
- separate service and write-completion terminal diagnostics; and
- strict fence/GPC Pareto selection.

No STORE or RED latency becomes an architecture default. STORE retains schema
11 with zero extra tail. RED needs work-dependent RMW completion with bounded
parallelism or region/request structure, not another constant. The next
milestone should test RED request/region completion work while leaving admission
and STORE timing unchanged. Retirement and numerical GPC remain unavailable.

The sweep retains 730 deterministic completion pairs: seven full-matrix
candidates and three calibration-only rejects. This is retrospective mechanism
development over multi-pass NCU and unresolved-module NVBit evidence. No sudo or
new GPU capture was required.
