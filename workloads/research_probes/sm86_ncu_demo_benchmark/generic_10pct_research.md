# Primary-source basis for the 10% aggregate gate

Research date: 2026-09-01. The implementation target is the local RTX 3090 Ti
(`sm_86`/GA102). The gate is an acceptance rule for the retained cohorts, not a
claim that seven or eight kernels prove architecture-wide 10% accuracy.

## Pinned source study

The source trees and papers are retained under
`/home/jakeshea/GPU_sims_notes/upstream_research_20260901/`.

| Project | Pinned revision | Directly relevant evidence | Decision for this simulator |
| :--- | :--- | :--- | :--- |
| MGPU-Sim | `d8f4c0a7d7b75498a91525cde37f3c5d2d6b70b6` | Explicit component ports, scheduler/scoreboard ownership, memory-instruction child transactions, raw cache/MSHR/DRAM counters, and drained observation boundaries | Retain causal issue/memory events and count-first validation; do not copy AMD latency constants or its fixed-latency default DRAM path |
| Frontier | `a24cedabedcc7bd374073fd508dcf770c860ede5` | Total-ordered serving events, separate timing domains, robust hardware medians, and byte conservation | No low-level model is portable: Frontier has no SASS, warp, L1/L2, or DRAM simulation. Its exact measured-row lookup and in-sample training score are specifically prohibited for scored workloads |
| Huerta modern GPU model | `117f9dca3f46b1d85d2a1ec9ddac6b89d49399b3` | Ampere four-subcore issue, CGGTY selection, control bits, instruction front end, and memory-issue pipeline | These mechanisms are already represented in `sim/arch` and `sim/sm`; another frontend scalar is not the next justified change |
| Accel-Sim framework / performance model | `d930ad6d02c09bb56867132583735aba0389cff4` / `91880c53383d5a6a6742bfb1be2c5f34e39c7871` | SASS trace validation, sector/profiler semantics, finite L2 request/coalescer queues, broad count-level validation | Preserve lookup-vs-physical-fetch separation. Treat finite L2 queues as the next candidate only after an independent GA102 saturation probe |
| FlashGPU-Sim | `d20ce45e3399f3a1b5b0787bdbf7c2b544a69ff9` | Explicit asynchronous-copy request, in-flight, response, and completion state with byte conservation | Transfer the causal structure, not Hopper/Blackwell queue constants; the repository does not provide an independently validated GA102 card |
| Sim-FA | arXiv:2605.00555v3 | Exact addresses, finite MSHR/request tracking, slice hashing, and domain-specific asynchronous-pipeline validation | Supports address/count invariants and mechanism microbenchmarks. Its 5.7% result is FlashAttention/H800-specific and is not evidence for a generic GA102 constant |

Other inspected source trees include GPGPU-Sim, HyFiSS, PPT-GPU, and MacSim.
MGPU-Sim and Frontier mechanism memos, fetched PDFs, arXiv source, and extracted
paper text are retained beside the clones.

## Gate semantics

The 10% threshold applies independently to each aggregate metric family:

- terminal/GPC-cycle MAPE;
- runtime-equivalent MAPE;
- issue/SM-active-cycle MAPE;
- instruction-throughput MAPE;
- first-class cache hit/miss count normalized error; and
- traffic-counter MAPE.

Metric families are never averaged into one scalar. All rows and zero-hit
targets remain visible. The large cohort retains the existing 20% per-row and
per-counter tail guard, so a sub-10 aggregate cannot hide a catastrophic row.
The small prototype retains its approved aggregate non-regression contract and
requires universal raw issue/terminal frontier non-regression.

NCU values remain comparison-only targets. Seven hardware launches are reduced
by their median; the deterministic simulator runs once per workload. Cache
ratios remain secondary derivations from hit/miss counts.

## Why this change does not add a speculative timing mechanism

The existing candidate is already below 10% on every aggregate metric family.
The most promising unimplemented SOTA mechanism is finite per-slice L2 request
and miss capacity, followed by controller/bank-level GDDR6X scheduling and
typed writeback service. No retained independent GA102 probe currently
identifies the finite L2 capacity or partition hash. Copying an H100, B200, AMD,
or analytical-model constant would be unscientific and could improve a scored
cohort only by accident.

Therefore this milestone first tightens the executable acceptance contract and
replays the small and large cohorts without changing simulator predictions. A
future mechanism change must be selected by an independent queue-saturation,
stride/partition-conflict, or row-hit/row-miss microbenchmark, frozen before
the scored workloads are replayed, and accompanied by an ablation showing the
expected pressure-regime effect.

## Returning-atomic DRAM repeatability

The first fresh large campaign found one hardware-instability case rather than
a changed simulator prediction. FlashAttention backward retained identical
instruction, L1, and L2 counts across seven NCU launches, while
`dram__sectors_read.sum` took the values
`2940, 5052, 2940, 3656, 2936, 5252, 4812` (58.76% relative range). The older
independent atomic-completion campaign shows the same phenomenon: returning
ATOMG layouts can keep exact L1TEX/LTS request counts while DRAM-read sectors
vary by orders of magnitude across profiles. The kernel uses contended BF16
atomic CAS loops for dK/dV accumulation, so one seven-launch median is not a
stable per-row oracle for this counter.

The response is measurement-side only. A prospective second seven-launch NCU
batch is collected with the identical binary, metrics, replay/cache policy,
locked clocks, GPU UUID, and driver. Every sample from both batches is retained
and reduced together; neither simulator output nor the resulting accuracy is
an input to the follow-up trigger or merge. Timing and all other workload rows
continue to use the original seven launches, and simulator repetitions remain
one per workload.

## Primary papers

- Sun et al., *MGPUSim: Enabling Multi-GPU Performance Modeling and
  Optimization*, ISCA 2019, <https://doi.org/10.1145/3307650.3322230>.
- Khairy et al., *Accel-Sim: An Extensible Simulation Framework for Validated
  GPU Modeling*, ISCA 2020,
  <https://people.ece.ubc.ca/~aamodt/publications/papers/accelsim.isca2020.pdf>.
- Khairy et al., *Exploring Modern GPU Memory System Design Challenges through
  Accurate Modeling*, arXiv:1810.07269.
- Huerta et al., *Dissecting and Modeling the Architecture of Modern GPU
  Cores*, MICRO 2025, <https://doi.org/10.1145/3725843.3756041>.
- Zhou et al., *Sim-FA: A GPGPU Simulator Framework for Fine-Grained
  FlashAttention Pipeline Analysis*, arXiv:2605.00555.
- Pan et al., *Architecting the Next Generation of Asynchronous, Distributed
  GPUs for the AI Era*, arXiv:2608.22602.
- Feng et al., *Frontier: Towards Comprehensive and Accurate LLM Inference
  Simulation*, arXiv:2605.21312.
