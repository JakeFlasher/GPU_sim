# SM86 L2 write-observer refinement — 2026-10-15

## Scope

This slice targets NCU-shaped L2 lookup metrics only. It does not alter L2 tag
state, DRAM demand, or replay timing.

## Change

Ordinary SM86 store sectors now use a conditional all-write-hit observer;
RED/reduction retains separate RMW semantics. Operation-qualified read/store/
atomic/reduction lookup lanes are summed with checked arithmetic and emitted as
L2 TEX-sector lookup hit/miss metrics.

The policy is supported by independent Ampere write-validate/write-allocate
research and retained real probe notes. NCU values from the scored corpus were
not used to select or fit it.

## Evidence

- Representative real full replay: FP32 FMA, FP16 HMMA and cp.async: **3/3
  OK**.
- cp.async selected NCU row: model L2 lookup sectors **256**, hits **128**,
  misses **128**; observed values were **256/128/128**.
- Full-corpus replay after the change: **356 roots**, **286 OK**, **62
  mixed-code control dispositions**, **8 replay/input failures**, **12,392
  successful launches**, **33.625 s** driver wall time, **12 workers**.
- Exact NCU identity rows: **107**; ordered/shape fallbacks remain unscored.

### Exact-identity residual comparison

| Metric | Baseline mean APE | Refined mean APE | Refined median | Refined P90 |
|---|---:|---:|---:|---:|
| L2 lookup-hit sectors | 54.52% | **35.04%** | 0% | 42.86% |
| L2 lookup-miss sectors | 84.20% | **27.00%** | 0% | 0% |
| L2 source-total sectors | 28.24% | 28.24% | 0% | 28.57% |
| DRAM read sectors | 40.91% | 40.91% | 40.25% | 70.08% |
| SM active IPC | 14.57% | 14.57% | 12.82% | 29.61% |
| SM elapsed IPC | 12.41% | 12.41% | 11.76% | 21.97% |

## Residuals and next hypothesis

Atomic/RED microbenchmarks still require operation-specific RMW observer
semantics; they are not folded into the ordinary-store arm. Local-memory sector
expansion and constant/GCC/DRAM demand remain major residual sources. This slice
is therefore conditional L2 metric coverage, not sub-10% closure.

Artifact: `../sm86_replay_results/full_corpus_refinement_l2_write_20261015_v1/full.jsonl.zst`.
