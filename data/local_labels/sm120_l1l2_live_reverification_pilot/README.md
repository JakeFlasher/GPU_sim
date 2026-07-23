# sm_120 L1/L2 live re-verification pilot

A **this-session `Verified-live-sm_120`** re-collection of the L1/L2 sector
hit-rate labels for three small portable_lcd cells, run live on the RTX 5060
Laptop GPU through `tools/local_labels/label_driver.sh` (NCU into
`data/local_labels/` only — the Rule-1 local-label exception).

**Purpose:** prove the live bi-label L1/L2 NCU pipeline works end-to-end in this
session and that the cold-cache methodology reproduces the previously-admitted
labels. This is a reproducibility/pipeline witness, **NOT** a new cohort — these
slugs already appear in the canonical `data/local_labels/label_admitted.json`.

**Live evidence (provenance per sidecar):**
- Device: NVIDIA GeForce RTX 5060 Laptop GPU (sm_120),
  `gpu_uuid_sha256=ef017d784a3b6f40a69523223cfa6f793072c5656e64dc3f1cd3ec2fefaee188`
- Driver 595.71.05, CUDA 13.2, NCU 2026.1.1.0 (build 37634170)
- Config: `warmup_count=0`, `cache_control=all` (true cold), `replay_mode=kernel`,
  N=5 repeats, `--clock-control none` (no clock locking; Rule 2).

**Result (`label_admitted.json`, schema `local_label_admitted.schema.json`):**

| cell | L2 sector hit % (this session / prior admitted) | L1TEX % | stddev_l2 |
|------|--------------------------------------------------|---------|-----------|
| silu_fwd_bf16_sm80 / timing_shape | 18.722 / 19.524 | 25.000 | 0.0039 |
| softmax_safe_bf16_sm80 / timing_shape | 18.264 / 18.0 | 50.000 | 0.0049 |
| layernorm_fwd_bf16_sm80 / timing_shape | 42.602 / 44.8 | 53.560 | 0.0097 |

All three reproduce the prior admitted L2/L1 within the 0.03 stddev gate,
confirming the cold-cache config + the live pipeline this session. The raw
`run_*.ncu-rep` reports are gitignored; only the parsed sidecars + the aggregate
are committed.

See the round summary for task4's full coverage reconciliation (74 admitted /
30 sm_120 / 9 frozen-roster matches — a shape-point naming drift between the
frozen roster and the collected cells).
