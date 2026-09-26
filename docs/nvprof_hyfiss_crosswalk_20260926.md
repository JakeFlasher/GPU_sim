# NCU profiler crosswalk and HyFiSS/MICRO'24 positioning — 2026-09-26

## What was checked

The recommended `ConvolutedDog/nvprof-scripts` repository is a small NCU-report
helper, not an Accel-Sim implementation.  Its `nvprof.sh` builds a CUDA test
and runs `ncu --set full --target-processes application-only`; its report
reader exports occupancy, cache, memory-transaction, cycle, IPC/MIPS, and
kernel-duration columns to XLSX.  The helper was inspected at commit
`fa92d4d47c7c39627ecaafdd08be65a56100a48e`:

* [nvprof.sh](https://raw.githubusercontent.com/ConvolutedDog/nvprof-scripts/fa92d4d47c7c39627ecaafdd08be65a56100a48e/nvprof.sh)
* [read_ncu_report.py](https://raw.githubusercontent.com/ConvolutedDog/nvprof-scripts/fa92d4d47c7c39627ecaafdd08be65a56100a48e/read_ncu_report.py)
* [ncu_report_helper.py](https://raw.githubusercontent.com/ConvolutedDog/nvprof-scripts/fa92d4d47c7c39627ecaafdd08be65a56100a48e/ncu_report_helper.py)

The artifact hard-codes CUDA 11.8, GPU 0, a 128³ smoke workload, and an
absolute report path.  We therefore reuse its metric vocabulary and formulas,
not its execution harness.

## Crosswalk to this project

| Profiler family | Current project status |
| --- | --- |
| Launch occupancy limits / achieved active warps | Parsed in `NcuObservation`; native model-relative projections retained separately |
| L1/L2 hit rates | NCU observations plus native cache projections |
| Global load/store/ATOM/RED requests and sectors | Parsed and replayed; retained SM86 holdout has exact request/sector rows |
| Unified L1 requests | Rust parser now prefers GA102 `l1tex__t_requests.sum`, falling back to the complete 11-class decomposition; Python XLSX helper was fixed the same way |
| L2 TEX requests and read/write M-stage transactions | Parsed as independent NCU observations; simulator keeps lookup and DRAM/writeback domains separate |
| DRAM sectors/bytes and throughput | Parsed as external NCU observations, including GA102 DRAM read bytes/s and throughput percentage; simulator physical writeback remains typed unavailable |
| GPC/SM cycles, IPC, MIPS, duration | Parsed as external observations; simulator issue/event/completion frontiers remain independent |
| Mem Busy / Max Bandwidth / Mem Pipes Busy | Parsed as independent throughput observations; never summed or injected into replay |
| Stall taxonomy and FU execution distributions | HyFiSS/model-relative explanatory surface; not claimed as NCU-equivalent simulator counters |

NVIDIA documents that request, sector, and wavefront metrics are different
domains and that throughput percentages are derived peak-relative observables;
we preserve these domains rather than collapsing them into one score
([Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)).

## Concrete implementation completed

The Rust NCU parser now retains the previously missing profiler-only families:

* `gpu__time_duration.sum`;
* DRAM bytes/s and DRAM clock;
* GA102 `dram__bytes_read.sum.per_second` and
  `dram__throughput.avg.pct_of_peak_sustained_elapsed`;
* GPC clock and SM IPC/MIPS;
* Mem Busy, Max Bandwidth, and Mem Pipes Busy;
* L1 M-stage read/write sectors;
* canonical unified L1 request count, individual local/texture/surface request
  components, and L2 TEX request count; and
* optional raw theoretical-warp rows when the NCU section exports them.

The local XLSX exporter now exposes the same five throughput/clock columns and
uses the canonical `l1tex__t_requests.sum` counter with a complete 11-family
fallback. The Rust observation retains each local/texture/surface component
for auditability. Focused local-label validation passes **119 tests**.

## Hardware and claim boundary

`data/local_labels/ncu_metric_availability_probe_sm_86.json` records 35 of 43
requested profiler targets available on the RTX 3090 Ti; unavailable targets
are primarily per-operation DRAM-sector variants and occupancy-calculator
catalog entries. The original retained SM86 NCU holdout did not request every
new HyFiSS column, so missing values in those historical CSVs remain
capture-schema gaps rather than inferred zeros. A fresh three-workload subset
requesting the new families is retained in
`docs/sm86_profiler_crosswalk_holdout_20260926.csv`.

The direct GA102 DRAM-rate capture and parser round-trip are recorded in
[`sm86_dram_rate_capture_20260926.md`](sm86_dram_rate_capture_20260926.md).

HyFiSS is a useful prior-art methodology, not a direct SM86 ranking: its public
README describes 43 applications/1,784 kernels on GV100 and NCU-shaped output
([HyFiSS repository](https://github.com/ConvolutedDog/HyFiSS)); its MICRO'24
claims are scoped to that GV100 study.  No Accel-Sim/MGPUSim execution was
performed here.  Historical speed/error tables remain offline context only.

The accompanying pre-provided-results artifact is useful for adversarial
context but not for a new leaderboard. A bounded reconstruction of its workbook
cells gives GV100 arithmetic active-cycle relative-error means of 37.62% for
Accel-Sim (`n=1006`) and 47.96% for HyFiSS/OURS (`n=980` valid cells); the A100
HyFiSS sheet gives 48.17% (`n=1126`). A rough name-matched reconstruction of
Accel-Sim-log versus HyFiSS report wall times gives a geometric mean around
17x, while the paper's stated headline is 12.8x. These numbers differ in
coverage, missing-cell handling, host/workload scope, and denominator from the
paper's protocol, so they are retained only as an offline sanity check. The
artifact is [Zenodo 13150677](https://zenodo.org/records/13150677).

NCU is an independent silicon-observable validation source, not a complete
internal timing oracle.  NCU replay passes and periodic scheduler sampling can
have different domains and cache/pass effects; all such rows remain external
observations in this project.
