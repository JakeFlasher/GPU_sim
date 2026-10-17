# SM86 plan/source consistency audit — H63 — 2026-10-16

## Adversarial findings and fixes

The H30–H62 source/docs review found two concrete semantic defects:

1. H42 prose and the metric reducer described scope-compatible rows as
   “primary” even though scope-mismatch rows are excluded from the raw/all-row
   comparison. The reducer/report now says that rows remain compressed and are
   excluded from raw/all-row APE; its compared table is explicitly a secondary
   scope-compatible view. STATUS and the H42 compact report use the same labels.
2. The plan's DRAM section described a finite partition/bank controller as if it
   were already the active full implementation. The source currently uses
   `DramBw::Carded` aggregate read service (default effective-rate scenario),
   H30–H37 partition/route pressure as observer covariates, and the H19
   store/fence feedback candidate. The plan now marks finite partition/bank
   timing as a future evidence-gated arm and keeps dirty writeback,
coherence/visibility, and payload forwarding explicitly optional/typed
unavailable by default; typed trace events gate activity and flags configure
those optional arms.

The H57–H62 plan/status/memory entries were also reordered chronologically, and
H62 now names aggregate carded DRAM plus observer-only partition candidates.
No observer is promoted to a physical hash, queue capacity, row/bank identity,
completion timestamp, or proprietary topology claim. Additional source
consistency fixes rename modeled cache state/trace line-domain observers that
were called “physical,” document source-only GPGPU-Sim/Accel-Sim inspection,
and label the 825 GB/s value as an unlocked absolute-rate hypothesis mapped
onto the locked simulation clock.

The corpus runner now exposes an explicit `--no-memory-admission-feedback`
ablation, defaults the launch-scoped H19 candidate only for `cache_scope=launch`,
and records `memory_admission_feedback_scope_suppressed=true` for stream-L2
scope. This prevents a direct-run launch-scope validation error from turning a
stream-L2 replay into all-failure rows. Per-launch DRAM counters now retain the
configured service rate.

## Post-replay evidence

The corrected reducer was run against the retained real H41 full/fast artifacts
(no simulator replay and no NCU value used as an input):

- Output: `../sm86_replay_results/metric_report_h63_scope_label_audit_20261016_v1/`
- Input full/fast artifacts remain compressed `.jsonl.zst`.
- Full scope warnings: 125; explicit scope-mismatch exclusions: 67.
- Fast scope warnings/exclusions: 0/0.
- The full table is headed “scope-compatible compared metrics”; mismatch rows
  remain in compressed `metric_comparisons.jsonl.zst`.
- Raw/all-row H41 values remain the authoritative dashboard baseline; no APE
  value was converted into a completion or sub-10% claim.
- Machine-readable summary fields identify `comparison_view=scope_compatible_secondary`,
  `raw_all_row_ape_included=false`, and the scope-mismatch policy.
- A real stream-L2 full replay of `simbench_l1_vendor_sgemm_cublas` completed
  `ok` with the compressed artifact
  `../sm86_replay_results/h63_stream_l2_scope_20261016_v2/full.jsonl.zst`;
  the summary records candidate suppression and the launch row records the
  configured 825,000,000,000 B/s service rate.

## Checks

- `python3 tools/sm120_metric_report.py ...` completed on the real H41 corpus
  artifact and emitted compressed comparison/summary files.
- `python3 -m py_compile tools/sm120_corpus_replay.py tools/sm120_metric_report.py`
  passed.
- Release `sim-trace`/`sim-run` builds and
  `cargo check --manifest-path sim/Cargo.toml --workspace --all-features`
  passed.
- `git diff --check` passed.

Remaining raw/secondary residuals and typed tails are unchanged and tracked in
`docs/sm86_unresolved_gap_tracker_20261016.md`; this slice corrects the contract
language/report semantics and runner scope handling without claiming new model
accuracy. A single copy-paste consultation bundle containing the relevant
plans, compact reports, architecture cards, source excerpts, and current
machine-readable summaries is at
`docs/sm86_external_consult_prompt_20261016.txt`.
