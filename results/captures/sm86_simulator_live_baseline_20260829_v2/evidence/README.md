# SM86 evidence bundle

`raw_cohort.tar.zst` is a deterministic archive of the six-cell stage outputs.
It contains clean base-binary correctness manifests/logs/environment records,
NCU CSV/report metadata (including the independent repeat), NVBit root binding
sidecars/manifests/stats/per-CTA protobuf payloads, static SASS/control dumps,
and the two simulator replay outputs for each cell. The archive SHA-256 is recorded in
`../campaign_manifest.json`.

From the campaign directory, materialize the paths named by that manifest in a
clean checkout:

```bash
zstd -dc evidence/raw_cohort.tar.zst | tar -xf - -C evidence
```

The fresh-cohort validator uses the per-cell `replay_binding.json` sidecars to
bind replay bytes to the exact simulator binary, trace tree, control sidecar,
architecture card, and pchase card. It therefore does not rely on or rewrite
filesystem timestamps after archive extraction.

The archive is evidence retention, not simulator input. NCU values remain
comparison-only observations.

`../live_cohort_semantic_report.json` is the independent six-cell validator
result; it is intentionally kept outside the compressed raw archive so a
reviewer can inspect the pass/fail summary without extraction.

`../closure_report.json` is the bounded residual-selection record. It links
the canonical and repeated NCU CSVs, simulator twin hashes, typed readiness
causes, independent frontier gaps, explicit AC-3/AC-4 status, and the
no-go/next-discriminator decision. The live-cohort validator recomputes this
report from the current artifacts, so changing the report and its manifest
hash alone cannot make stale evidence pass. `host.json` and the
`host_preflight/` transcripts bind the host/tool commands to the campaign
input hashes.
