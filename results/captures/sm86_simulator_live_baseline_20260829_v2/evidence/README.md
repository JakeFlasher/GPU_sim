# SM86 evidence bundle

`raw_cohort.tar.zst` is a deterministic archive of the six-cell stage outputs.
It contains clean base-binary correctness manifests/logs/environment records,
NCU CSV/report metadata (including the independent repeat), NVBit root binding
sidecars/manifests/stats/per-CTA protobuf payloads, static SASS/control dumps,
and the two simulator replay outputs for each cell. The archive SHA-256 is recorded in
`../campaign_manifest.json`.

To materialize the paths named by that manifest in a clean checkout:

```bash
zstd -dc evidence/raw_cohort.tar.zst | tar -xf -
```

The archive is evidence retention, not simulator input. NCU values remain
comparison-only observations.

`../live_cohort_semantic_report.json` is the independent six-cell validator
result; it is intentionally kept outside the compressed raw archive so a
reviewer can inspect the pass/fail summary without extraction.

`../closure_report.json` is the bounded residual-selection record. It links
the canonical and repeated NCU CSVs, simulator twin hashes, typed readiness
causes, and the no-go/next-discriminator decision. `host.json` and the
`host_preflight/` transcripts bind the host/tool commands to the campaign
input hashes.
