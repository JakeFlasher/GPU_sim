# SM86 dynamic-youngest priority and fixed-result queue audit — 2026-10-15

## Scheduler priority audit

Huerta's public SM86 scheduler orders a held greedy warp first, then the
youngest eligible warp by dynamic warp ID. H25 retains the trace's block-local
dynamic `warp_id` explicitly in `WarpReplay`/`WarpState`, uses it as the
deterministic tie-break for the existing admission/arrival order, and emits a
within-CTA inversion diagnostic. This does not change timing unless the trace
proves an order inversion.

Across all 12,392 full launches, the audit found **zero** arrival-vs-dynamic-ID
inversions. H25 therefore makes the source identity explicit without claiming
that an unobserved hardware scheduler ID was recovered.

## Fixed-latency result-queue investigation

An independent clock64 probe was rebuilt and run on the local RTX 3090 Ti from
the retained `fixed_latency_result_capacity` research source. Equal-work b7/b8/
b9 variants (8,192 generations, 11 trials) produced:

| Active warps | b7→b8 | b8→b9 |
|---:|---:|---:|
| 1 | ~0.000 cycles/generation | ~0.000 |
| 4 | ~−0.000 | ~−0.000 |
| 8 | +0.163 | +1.118 |

The W8 threshold reproduces the historical multi-warp shape, but W1/W4 are
null and a separate HADD2 collision/separated pair is strongly non-monotonic.
Execution-to-writeback release timing and physical queue capacity remain
unidentified. No result-queue timing arm is activated; the evidence is retained
as a mechanism-local negative/observer result rather than an invented delay.

## Complete-corpus evidence

Representative high-BAR roots completed 5/5 full replay. The complete
full-first replay then completed 356 roots (286 `ok`, 62 mixed-code controls,
8 replay/input failures), 12,392 launches in 40.456 s with 12 workers:

`../sm86_replay_results/full_corpus_h25_priority_audit_20261015_v1/full.jsonl.zst`

Post-replay report:
`../sm86_replay_results/metric_report_h25_priority_audit_20261015_v1/`.
The readiness schema is 29. H25 is bit-for-bit timing/traffic neutral versus
H24 (all L1/L2/DRAM/IPC/frontier deltas are zero); H24's small APE changes
therefore remain the current baseline.

## Verdict and next seam

Promote the explicit dynamic-ID audit and retain the fixed-result queue probe as
negative evidence. Do not enable an 8-entry/one-pop timing model without a
validated execution-to-writeback seam. Remaining accuracy work must target the
dominant eligible-not-selected/front-end/L1TEX dependency residuals with an
independent, mechanism-local source prior.
