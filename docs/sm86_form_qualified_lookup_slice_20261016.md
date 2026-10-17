# SM86 H92 form-qualified L1 lookup selection — 2026-10-16

## Mechanism

H92 promotes a narrow, trace-only endpoint mixture for the SM86 NCU-shaped L1
lookup rows. It leaves L1/L2 state, MSHR ownership, request ordering, DRAM
service, and all cycle/IPC frontiers unchanged.

The selected global-load lookup endpoint is chosen only from static-SASS
`memory_scope` and the existing request-boundary audit:

* a launch containing only ordinary `LDG.*.CONSTANT` uses the existing
  packet-aware candidate (`baseline_hits - ordinary_same_request_merges`);
* an LDGSTS-only launch uses an all-miss endpoint for traced line/sector shapes
  other than the two-sectors-per-line async-copy control shape; the two-sector
  shape retains the baseline;
* a plain-LDG launch may use the packet candidate only when the modeled launch
  has no resident global hits and does expose ordinary same-request merges;
* mixed clients and LTC128B remain on the baseline until a per-request
  producer/client crosswalk exists.

The baseline hit/miss rows and a `selection_form_qualified` observer remain in
the replay artifact. No NCU value, workload name, or target error is used by
the selector.

## Evidence

Build and a four-root real full replay completed successfully. The complete
full-first replay then completed 356 roots (286 `ok`, 62 controls, 8 replay
errors), 12,392 full launches, exactly twelve simulator workers, and compressed
full/fast artifacts:

* replay: `../sm86_replay_results/full_corpus_h92_form_lookup_20261016_115517/`
* report: `../sm86_replay_results/metric_report_h92_form_lookup_20261016_v1/`

The replay summary records the exact release binary
`sim/target/release/sim-run`, `arch_profiles/cards/sm_86.json`,
`data/pchase/sm_86_static_card.json`, the complete
`../sm86_rtx_3090_capture/` corpus, generated control sidecars, and the
compressed full/fast output paths.

The H92 report has the same 91 available metric names, 1,647 unavailable names,
682 warnings, 71 scope-mismatch predictions, and 107 exact identity joins as
H88. A replay/report diff found changes in only the global-load L1 lookup-hit,
lookup-miss, and derived hit-rate metrics; every other scored metric is
bit-equivalent to H88.

### Accuracy delta

On the secondary scope-compatible exact joins:

| metric | H88 | H92 | delta |
|---|---:|---:|---:|
| L1 global lookup hit | 30.7137% | **18.0702%** | −12.6435 pp |
| L1 global lookup miss | 32.3914% | **18.8472%** | −13.5442 pp |
| L1 global hit rate | 30.7137% | **18.0702%** | −12.6435 pp |

Raw/all-row APE remains authoritative. H92 improves raw hit APE from 30.7137%
to 18.0702% and raw miss APE from 58.4438% to 45.0262%; the remaining raw
miss residual is dominated by the known LTC128B/scope-mismatched row and is
not hidden or reclassified as completion.

No other metric regressed. DRAM, L2, cycles, IPC, shared-memory, and launch
resource rows are unchanged from H88.

## Boundary and next evidence

This is an effective endpoint selection, not a claim that static aggregate
form labels identify a physical client. The next promotion boundary is a
same-binary NVBit/SASS per-request client/form crosswalk covering ordinary LDG,
CONSTANT/U16, LDGSTS, and LTC128B controls. Until that exists, do not extend
the selector to mixed launches or feed its output into cache/timing state.
