# Native CUPTI PC-state surface (2026-09-30)

## Collection

`tools/sm86_cupti_pc_campaign.py` fixes the NVIDIA sample's filename contract:
the sample prepends a context ID to `--file-name`, so each workload runs in a
dedicated directory with a basename-only output.  The wrapper explicitly sets
`SIMBENCH_ALPHA_NO_MANIFEST=1` and `NVBIT_ALPHA_NO_HASH=1`, does not invoke NCU,
and uses no score labels while collecting CUPTI data.

Two live collections completed on the RTX 3090 Ti/SM86 binaries:

| phase | warmups | sampled launches | workloads | PC samples | PC records | drops |
|---|---:|---:|---:|---:|---:|---:|
| cold | 0 | 8 | 15/15 | 470,506 | 21,161 | 0 |
| warm | 8 | 8 | 15/15 | 920,753 | 42,500 | 0 |

The exact per-workload values are retained in
`results/alpha/sm86_native_cupti_pc_all_20260930/summary.json` and
`.../sm86_native_cupti_pc_warm_20260930/summary.json`; the table is intentionally
not used as a score aggregate because samples are workload-dependent and the
campaign uses repeated launches.  Every workload had nonzero records and zero
drops.  The largest per-workload counts are visible in the retained summary.

The decoder now emits schema v2 summaries with:

* all-cycle scheduler-state populations;
* no-issue subsets kept separately (never added to all-cycle states);
* selected/not-selected retained but excluded from semantic pressure;
* function/PC/correlation coverage and per-buffer correlation IDs;
* unknown/unmodeled stall-state fractions.

NVIDIA documents that selected and not-selected are scheduler states rather
than true stalls, that no-issue metrics are a subset sampled only when no
instruction issued, and that PC stall samples are attributed to the consumer
instruction.  The implementation follows those distinctions.  See the
[Nsight Compute warp-stall definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
and [CUPTI PC-sampling utility contract](https://docs.nvidia.com/cupti/api/group__CUPTI__PCSAMPLING__UTILITY.html).

## Projection result

`tools/sm86_native_cupti_surface_projection.py` fits a fixed low-dimensional
(`four`) surface using replay scale plus CUPTI no-issue fraction,
long-scoreboard fraction, IMC-miss fraction, and aggregate pressure fraction.
The fit is family-disjoint and NCU values from the held-out family are loaded
only after prediction:

| phase | all rows | GPC MAPE | duration MAPE |
|---|---:|---:|---:|
| cold | 15/15 | **13.52%** | **14.53%** |
| warm | 15/15 | **23.46%** | **24.11%** |

This is materially better than the unconstrained CUPTI feature surface but
still worse than the promoted frontier/archetype route.  It remains a
non-promoted independent-observation committee member.  It demonstrates that
public scheduler-state observations can replace some invented latent terms,
but they do not reveal physical pipe/slot identity or cache ownership.

The paired cold/warm phase interpretation is retained in
`docs/sm86_native_cupti_phase_analysis_20260930.md`.

## Accidental diagnostic note

Before the wrapper was corrected, one bounded relative-path SiLU probe omitted
the no-manifest environment and printed the benchmark's built-in manifest path.
That artifact was not consumed by any model or campaign result.  All retained
native CUPTI campaigns use the explicit no-manifest/no-hash environment above.
No testing or verification workflow was run.
