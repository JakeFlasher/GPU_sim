# SM86 evidence bundle

`raw_cohort.tar.zst` is a deterministic archive of the six-cell stage outputs.
It contains correctness manifests/logs, NCU CSV/report metadata, NVBit root
manifests/stats/per-CTA protobuf payloads, static SASS/control dumps, and the
two simulator replay outputs for each cell. The archive SHA-256 is recorded in
`../campaign_manifest.json`.

To materialize the paths named by that manifest in a clean checkout:

```bash
zstd -dc evidence/raw_cohort.tar.zst | tar -xf -
```

The archive is evidence retention, not simulator input. NCU values remain
comparison-only observations.
