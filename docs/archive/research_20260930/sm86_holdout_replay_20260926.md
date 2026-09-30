# Eight-workload SM86 development replay — 2026-09-26

**Correction:** the filename says holdout, but these rows reuse the September-1
`sm86_ncu_demo_benchmark_20260901_full_v2` development corpus. They are not an
untouched holdout or new silicon captures. Raw values are retained in the
[CSV](sm86_holdout_replay_20260926.csv); no measured value was changed.

All eight replay instruction totals match the corresponding retained NCU totals.
Mean absolute GPC-frontier discrepancy is 3.46%. Every replay still reports
`incomplete_unmodeled_retirement_tails`; this is not elapsed-cycle accuracy.

## Counter-domain correction

`sim_l2_lookup_sectors` is the model's **TEX-source** lookup observer. The
retained NCU value was computed as `lts__t_sectors_lookup_hit.sum` plus
`lts__t_sectors_lookup_miss.sum`, **all sources**. The former 21.19% “L2 MAPE”
is withdrawn: unlike domains cannot establish cache accuracy. The CSV renames
that arithmetic column `l2_mismatched_domain_delta_pct` and records
`l2_comparison_status=not_comparable_all_sources_vs_tex`. RMSNorm's arithmetic
value is 7.64%, not the earlier prose table's 17.69%; SiLU's is 11.81%, not 11.78%.
`sim_dram_read_bytes` includes compulsory instruction demand; DRAM label columns
remain unscored here.

## Reproduction context and limitations

The recorded replay command was, for each workload name `W` in the CSV:

```bash
BASE=results/captures/sm86_ncu_demo_benchmark_20260901_full_v2/evidence/raw
timeout 180s sim/target/release/sim-run \
  --trace "$BASE/nvbit/$W" --control "$BASE/static/$W/$W.control_bits.json" \
  --arch-card arch_profiles/cards/sm_86.json \
  --pchase-card data/pchase/sm_86_static_card.json \
  --mode l1-l2 --cache-scope launch --carded-dram
```

NCU `rep-1/W/ncu.csv` was read after simulation, not supplied to replay.
The original run did not retain a complete binary/card/trace/control/output
content-identity bundle. It therefore remains a dated development observation,
not a claim of fresh reproducibility at every later revision. All data and
older capture trees remain intact. Use [STATUS](STATUS.md) for current claims.
