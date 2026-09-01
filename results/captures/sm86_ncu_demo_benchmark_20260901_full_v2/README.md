# SM86 NCU demo benchmark — final v2

This is the completed eight-row RTX 3090 Ti campaign captured from
`GPUSim_main` revision `41b871a5799799df51f5e85278206cc9dd5af022`.
It contains three independent NCU application-replay measurements per workload,
one exact NVBit trace per workload, one static-control sidecar, and two
byte-identical release-simulator replays.

The human-readable result is `final_report.md`; `final_report.json` retains raw
per-repeat values, coverage, formulas, typed availability, frontier tails, and
artifact identities. The simulator binary SHA-256 is
`394640988c9d5c55322a3d06b562ff87cb9d5c7a2ec3aa8562901c6d36e087ae`.

## Raw evidence

`evidence/raw_campaign.tar.zst` is a deterministic archive of the materialized
`evidence/raw` tree:

- SHA-256: `dc4a3f2176c3d342690eb625c3a8b34f1574f8f665eda578b95e555deab65b2d`
- Compressed bytes: `1128761`
- Decompressed tar bytes: `63027200`

From this directory, materialize it with:

```bash
zstd -dc evidence/raw_campaign.tar.zst | tar -xf - -C .
```

Then regenerate and validate the report from repository root:

```bash
.venv/bin/python workloads/research_probes/sm86_ncu_demo_benchmark/report.py \
  --campaign results/captures/sm86_ncu_demo_benchmark_20260901_full_v2/campaign_manifest.json \
  --validate-only
```

## Superseded live attempts

- `smoke_v1` was rejected when NCU's direct per-subunit L2 percentage produced
  an impossible 112.57%. Final cache rates instead derive launch-global weighted
  percentages from retained NCU hit/miss sector sums.
- `full_v1` was rejected because CUTLASS Conv2D emitted nonempty unmapped-range
  and cross-allocation NVBit sidecars. The final matrix uses the existing BF16
  embedding-gather workload at an explicit shortened timing shape; its 148,576
  trace records are conserved and all exclusion sidecars are empty.

NCU values remain comparison targets only. They are not simulator inputs and
were not used to alter architecture cards or per-workload simulator parameters.
