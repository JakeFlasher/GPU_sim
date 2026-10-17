# SM86 constant-client reuse observers — H88 — 2026-10-16

## Implementation

Full replay now exposes bounded transport/reuse ratios for dynamic constant
metadata: address-bearing record coverage, references per unique bank-local
sector, and per-instruction sector requests per unique sector. These ratios
describe trace transport and reuse only; they never become L1/L2 tags, GCC
requests, DRAM sectors, timing, or NCU prediction inputs.

## Full-first evidence

* replay: `../sm86_replay_results/full_corpus_h88_constant_reuse_20261016_v1/`
* report: `../sm86_replay_results/metric_report_h88_constant_reuse_20261016_v1/`
* coverage: 286 `ok`, 62 controls, 8 replay errors; 12,392 full launches with
  twelve workers.

The observer is present on 286 full launches. Representative ratios include
9.25 references/unique-sector and 100% metadata coverage; incomplete legacy
records remain explicitly visible. GPC APE (12.9008%) and DRAM-read APE
(26.8371%) are bit-equivalent to H83, confirming no timing or service mutation.

## Boundary

H88 makes the unresolved GCC/DRAM interval more measurable but does not infer
request-to-sector or sector-to-DRAM ownership. The H85 independent matrix
continues to falsify a linear request-count rule; a dynamic source-client
completion witness remains required before P4 timing promotion.
