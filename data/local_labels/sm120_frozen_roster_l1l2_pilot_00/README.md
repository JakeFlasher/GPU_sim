# sm_120 frozen-roster L1/L2 collection pilot — the `_00` cells

Live NCU L1/L2 collection on the RTX 5060 (sm_120) for the 30 frozen-roster cells
with `_00`/`_01` shape-points (the new Blackwell cluster_*/conformer_* + related
slugs), via `tools/local_labels/label_driver.sh` (NCU into `data/local_labels/`
only). These cells were earlier mis-classified `un_arg_convention`; in fact their
binaries accept `--shape-point <id>` (a passthrough label), so they ARE
collectable — this pilot collected all 30.

**Config:** `warmup=0 cache-control=all` (cold), `--clock-control none`, N=5
repeats; driver 595.71.05, CUDA 13.2, NCU 2026.1.1.0.

**Result (`label_admitted.json` / `label_excluded.json`):** **4 admitted**
(stddev_l2 ≤ 0.03) + **26 repeatability-excluded** (stddev_l2 > 0.03). The
admitted are the high-L2 fused cells:
- flash_attn_bwd_bf16 / timing_00 (L2 97.41%)
- geglu_ffn_fused_bf16 / correctness_00, timing_00, trace_00 (L2 98.3–99.3%)

The 26 excluded confirm the sm_120 L2 label-stability limit at this config. The
raw `run_*.ncu-rep` reports are gitignored; the admitted cells' sidecars +
aggregates are committed; the excluded cells' raw sidecars are pruned (their
stddev is retained in `label_excluded.json`).

Consumed by `sm120_l1l2_frozen_grain_engineering_closure.json` (which partitions
all 54 roster cells into 16 admitted + 38 repeatability-excluded — see that
closure for the AC-3 terminal verdict).
