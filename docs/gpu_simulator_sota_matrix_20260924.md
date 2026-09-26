> Dated context; current capabilities and open work are in [STATUS](STATUS.md).
> Historical completion claims and next-step prose are not active instructions.

# GPU simulator SOTA and capability matrix

Updated 2026-09-24. This inventory records public numerical results from
primary papers, official artifacts, and official project pages. It is not a
single-score leaderboard: hardware MAPE, cycle ratios, sampling error, KIPS,
wall-time speedup, and memory footprint are separate quantities.

Comparability labels:

* P0: potentially direct to this project (same SM86/RTX 3090 Ti and same NCU
  observable). No external published row currently meets P0.
* P1: same metric family, but different hardware/workload/configuration.
* P2: model-relative or sampling error, not silicon accuracy.
* P3: contextual only (functional/framework/old target/missing metadata).

## Aggregate numerical results: detailed and validated simulators

| System | Target and scope | Hardware-grounded result | Speed or other result | Relation to this project | Source |
|---|---|---|---|---|---|
| GPGPU-Sim 3.x | Fermi-era NVIDIA; 14 validated workloads | 35% reported error in the Accel-Sim landscape; broader Pascal study reports 22--105% by workload/ISA class | 3 KIPS in the landscape table | P3 historical baseline | [landscape](https://people.ece.ubc.ca/aamodt/papers/accelsim.isca2020.pdf), [SIGMETRICS](https://mkhairy.github.io/Docs/SigmetricsPaper.pdf) |
| Enhanced GPGPU-Sim / Volta memory model | TITAN V/QV100-like Volta; diverse kernels | 27% execution-cycle MAE, 0.96 correlation; old model 68% and 0.71 | Memory-counter errors reduced as much as 66x | P1 methodology only | [paper](https://arxiv.org/abs/1810.07269) |
| Accel-Sim 1.0 | Kepler/Pascal/Volta/Turing; 80 workloads/1,945 kernels plus 11,440 DeepBench kernels | SASS MAE 25%, 30%, 15%, 25% respectively; QV100 correlation about 1.00 | 12.5 KIPS trace-driven, 6 KIPS execution-driven | P1; no 3090 Ti row | [paper](https://mkhairy.github.io/Docs/Accel-Sim.pdf), [landscape](https://people.ece.ubc.ca/aamodt/papers/accelsim.isca2020.pdf) |
| Accel-Sim 2.0 | Modern SASS; H100 validation | Official claim: 13.4% mean absolute cycle error, 0.99 Pearson over 34,000+ kernels/22 suites | Official claim: about 2.2x faster than 1.x | P1; H100 is not RTX 3090 Ti | [official project](https://accel-sim.github.io/), [repository](https://github.com/accel-sim/accel-sim-framework) |
| MGPUSim | AMD GCN3 R9 Nano; 57 microbenchmarks plus multi-GPU workloads | 5.5% average execution-time difference; peak about 20% | 28 KIPS; 3.5x functional and 2.5x timing speedup with four host cores | P3; different vendor/ISA and NVIDIA path unfinished | [paper](https://people.bu.edu/joshi/files/mgpusim-isca2019.pdf), [repository](https://github.com/sarchlab/mgpusim) |
| NVArchSim / NVAS | Selective-fidelity hybrid model; 955 traces, mainly Volta plus Turing study | 17% overall MAE, 0.953 Pearson; Turing 17% MAE/0.992 Pearson | 40--250x faster than GPGPU-Sim on identical workloads | P1 methodology/speed reference | [HPCA paper](https://d1qx31qr3h6wln.cloudfront.net/publications/Need_for_Speed_Experiences_Building_a_Trustworthy_System-Level_GPU_Simulator.pdf) |
| Multi2Sim / Kepler | AMD/NVIDIA Kepler; 24 workloads | 19% reported error | 0.8 KIPS | P3 historical/contextual | [landscape](https://people.ece.ubc.ca/aamodt/papers/accelsim.isca2020.pdf), [Kepler paper](https://ieeexplore.ieee.org/document/7975298/) |
| gem5-APU / gem5 GPU | AMD CPU/GPU system model; 10 workloads | 42% reported error | No comparable KIPS in the cited table | P3 system-level prior art | [landscape](https://people.ece.ubc.ca/aamodt/papers/accelsim.isca2020.pdf), [gem5-gpu](https://arch.cs.ucdavis.edu/gpu/simulation/2014/02/15/gem5-gpu.html) |
| GpuTejas | Parallel trace-driven Tesla/Fermi/Kepler-style model; Rodinia | Parallel-vs-serial error <7.67%; Tesla M2070 used for validation | 17.33x over serial at 64 threads; 429x over GPGPU-Sim | P2 parallel-simulation precedent | [paper](https://www.cse.iitd.ac.in/~srsarangi/files/papers/gputejas.pdf), [project](https://www.cse.iitd.ernet.in/~srsarangi/tejasweb/gputejas/index.html) |
| PPT-GPU | Hybrid high-level trace/analytical NVIDIA model; Rodinia/Parboil | Earlier paper within 10% hardware; SC'21 MAPE <16%, correlation >0.98 | Earlier paper up to 450x faster than GPGPU-Sim; SC'21 orders faster | P1/P2; not detailed counter-equivalent timing | [SC'21](https://www.sc21.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap571.html), [2019](https://www.osti.gov/servlets/purl/1504654) |
| GATSim | Abstract timing with native/functional execution | About 4% error against a cycle-accurate simulator | >80 MIPS; about 400x faster | P2/P3; error is not independent silicon MAPE | [DATE paper](https://slam.ece.utexas.edu/pubs/date17.GATSim.pdf) |
| NaviSim | AMD RDNA; RX 5500 XT and Radeon Pro W6800, 10 workloads | 9.85% average execution-time error on RX 5500 XT | No comparable public speed value | P3 for SM86; retuning GCN parameters alone reached up to 318% error | [PACT paper](https://sarchlab.org/navisim_pact_2022.pdf) |
| HyFiSS | Hybrid fidelity/stall-aware GV100 model; 43 apps/1,784 kernels | MAPE described as comparable to Accel-Sim; absolute MAPE not public in artifact README | 12.8x faster than Accel-Sim; at least 3.2x lower storage | P1/P2; major stall-attribution precedent | [artifact](https://github.com/ConvolutedDog/HyFiSS), [abstract](https://www.researchgate.net/publication/386397923_HyFiSS_A_Hybrid_Fidelity_Stall-Aware_Simulator_for_GPGPUs) |
| Huerta modern GPU simulator | Enhanced Accel-Sim SM/sub-core model; RTX A6000 and Turing/Ampere | MICRO'25: 13.45% average MAPE, 20.58% reduction versus predecessor | No aggregate speed value in accuracy paper | P1; closest modern-NVIDIA architectural baseline, not 3090 Ti | [paper page](https://upcommons.upc.edu/entities/publication/e31cdd46-ce96-41ba-8ba0-1919bb7adeb4), [artifact](https://github.com/upc-arco/modern-gpu-simulator-micro-2025) |
| Huerta parallelization | OpenMP-parallelized Accel-Sim-derived model | Deterministic equality across thread counts | 5.8x average at 16 threads, up to 14x; >5-day runs reduced to <12 h | P2 host-parallelism result | [paper](https://arxiv.org/abs/2502.14691) |
| LPGSim | Lightweight trace/cycle-level model with compute coalescing and local clocks; three architectures | 21.4%, 23.3%, and 22.7% errors | 9.97x single-thread; 19.8x parallel; 197.4x total over stated SOTA baseline | P1/P2; target cards need recovery from full paper | [POMACS paper](https://doi.org/10.1145/3805642) |
| FlashGPU-Sim | Execution-driven cycle-level Hopper/Blackwell; SM90/SM120 | 5.24% cycle-level MAPE across 131 RTX 5090/H100/B200 configurations; H100 FA-2 6.68%, FA-3 8.12%, B200 FA-4 4.72% | 7.86x with 16 host threads; Triton frontend and async/TMA/mbarrier/DSM models | P1 modern-NVIDIA reference; different hardware/ISA and not SM86 | [paper](https://arxiv.org/abs/2609.15311), [repository](https://github.com/FlashGPU-Sim/FlashGPU-Sim) |
| Sim-FA | WarpGroup/tile asynchronous model; H800/H20 | H800 FA3 MAPE 5.7% (max 12.7%); H20 3.28% (max 5.4%); GEMM 5.49% | 7.4x geometric-mean faster than Accel-Sim 2.0 on 23 GEMMs; 5--23x lower host memory | P1/P2 specialized Hopper abstraction | [paper](https://arxiv.org/abs/2605.00555) |
| GPU-Tile-Sim / GTSim | Tile graph/dependency model; A100/H100 and preliminary Blackwell | A100/H100 MAPE range 1.22%--8.71%; attention 2.37%--6.50% | Primary abstract does not expose one speed number | P1/P2 high-level dependency model | [paper](https://arxiv.org/abs/2607.11262) |
| ACALSim + HPCSim | Parallel event-driven framework; A100-class GPU case study | A100 cycle ratio 0.72--1.22x; average deviation about 20% | >14x faster than identical-core SST; 41% lower memory; LLaMA-7B 17.7 min and LLaMA-13B 30.4 min | P1/P2 framework and event reference | [paper](https://arxiv.org/abs/2605.22936), [framework](https://github.com/twhpcedu/acalsim) |
| TrioSim | Single-GPU operator trace extrapolated to multi-GPU DNN training; high-level flow network | 2-GPU average error 2.91% data parallel, 4.54% tensor parallel, 6.82% pipeline parallel | Completes multiple training batches in seconds | P2 system-level extrapolation; not SM86 cycle/counter fidelity | [ISCA'25 paper](https://bu-icsg.github.io/publications/2025/TrioSim_ISCA_2025.pdf) |
| PARMA-DITAM GPGPU-Sim parallelization | H100-oriented parallel GPGPU-Sim compute bottleneck study | Maximum cycle error 3.22% | 3.58x average and 4.38x maximum wall speedup with 8 workers | P2 host-parallelism result; not selective-fidelity or SM86 validation | [paper](https://drops.dagstuhl.de/entities/document/10.4230/OASIcs.PARMA-DITAM.2026.6) |
| Vortex | Open RISC-V GPGPU with SimX, RTL, FPGA backends | No comparable SimX silicon-MAPE row located | 25.6 GFLOPS peak at 200 MHz on 32-core Stratix-10 FPGA | P3 accuracy; strongest open-architecture/configurability precedent | [project](https://github.com/vortexgpgpu/vortex), [paper](https://arxiv.org/abs/2110.10857) |
| This project: native SM86 replay | NVBit/enhanced-protobuf -> Rust SM/cache/memory/event replay; RTX 3090 Ti | Exact instruction count 197,120 vs 197,120 for one GEMM; GPC timing non-scoring: NCU 13,475--13,529 vs frontier 12,949 | Event replay 1.70x faster than dense on fresh GEMM; 3.4% replay-phase reduction after local optimization | P0 only for count row; P1 diagnostic timing/speed; no held-out silicon MAPE | [measurement boundary](sm86_measurement_boundary_20260924.md), [speed probe](sm86_speed_probe_20260923.md) |
| This project: simt_u32_v1 custom path | Fixed declarative SIMT ISA + dependency/resource pipeline | No silicon validation; model-relative | Two-group example retires at 45 cycles; barrier producer/consumer probe at 90 cycles | P3 hardware accuracy; useful configurable-architecture mechanism path | [custom pipeline](custom_program_pipeline_20260924.md) |

The lowest published HW-MAPE values are not directly interchangeable. Sim-FA's
5.7% is for a narrow Hopper FlashAttention family; MGPUSim's 5.5% is R9 Nano
execution time; Accel-Sim 2.0's 13.4% is H100; and FlashGPU-Sim's 5.24% is
an aggregate modern-GPU result across 131 configurations. This project's
current 0% row is an instruction-count mapping, not elapsed-time accuracy.

## Sampling and workload-reduction variants

These methods reduce detailed work. Their error is normally against full
simulation or a profile, so these rows are P2 rather than hardware-MAPE rows.

| Variant | Scope | Speedup | Error | Other result | Relation to this project | Source |
|---|---|---:|---:|---|---|---|
| PKA / Principal Kernel Selection | 147 workloads, three recent NVIDIA generations | 7.2x Rodinia average; 1,987x on seven MLPerf workloads | 12.6% Rodinia; 28.5% seven MLPerf | Silicon-verified selection methodology | P2; future large-workload option | [paper](https://mkhairy.github.io/Docs/PKA.pdf) |
| Sieve | Cactus and MLPerf on NVIDIA Ampere/Turing | About 1,000x | 1.2% average, 3.2% maximum; PKS 16.5%/60.4% | Profiling time 8x lower, up to 98x lower, than PKS | P2; workload prediction, not native replay MAPE | [paper](https://users.elis.ugent.be/~leeckhou/papers/ispass-2023.pdf) |
| Photon | FIR, ReLU, MM, VGG, ResNet on MGPUSim R9 Nano/MI100 | Up to 24.65x single kernel; 39.1x ResNet-152 | 6.83% average single-kernel; 10.7% ResNet-152 | ResNet-152 7.05 days -> 1.7 h | P2; AMD backend and sampling error | [paper](https://www.comp.nus.edu.sg/~tcarlson/pdfs/liu2023pafssmfgw.pdf) |
| STEM + ROOT | Rodinia, CASIO ML, HuggingFace/LLM | 3.00x Rodinia; 109.595x CASIO; 31,719.057x HuggingFace | 0.93%, 0.36%, 0.57% | H200 transfer from H100 sampling: 5.46%; 13 microarchitectural metrics | P2; scalable-workload option absent here | [paper](https://seonjinna.github.io/assets/pdf/MICRO25_stem_root.pdf) |
| Allegro | Seven ML workloads integrated with MacSim | Maximum 1,486.66x; average 983.96x at 5% bound | 0.057% geometric-mean total-time error | GPT-2 100-token context: real GPU 0.925 s; MacSim 20.88 h; GPGPU-Sim 3.52 d; MGPUSim 1.63 d | P2; repeated ML-kernel acceleration | [paper](https://seonjinna.github.io/assets/pdf/Allegro_mlarchsys24.pdf) |
| GCL-Sampler | 7,746 kernels; Turing/Ampere/Ada; HyFiSS integration | 258.94x P1; 203.97x P2; 203.64x P3 | 0.37%, 1.50%, 1.22% average cross-architecture error | NCU 2023.2 ground truth; nw 22 min -> 10 s with 0.5% error | P2; current cross-architecture sampling precedent | [paper](https://arxiv.org/abs/2603.00551) |

## Existing capabilities and current project gaps

Status labels: Yes means a bounded path exists; Partial means only a
diagnostic or restricted replay path exists; No means no implemented/validated
path; Model-relative means custom timing is not a hardware claim.

| Metric or capability | Current project status | Demonstrated elsewhere | Gap / priority |
|---|---|---|---|
| Native NVIDIA SASS trace ingestion | Yes, SM86 via NVBit/enhanced protobuf | Accel-Sim, NVAS, Huerta, FlashGPU-Sim, GCL | Keep; trace ingestion alone is not novelty |
| PTX/vISA execution-driven frontend | No | GPGPU-Sim, Accel-Sim PTX, gem5 GPU, Multi2Sim | P2 if execution-driven software matters |
| Value-dependent trace regeneration | No; fixed replay | NVAS functional regions; Accel-Sim dual path; execution-driven models | P1 for generality; keep explicit limitation |
| Elapsed-cycle prediction against NCU | Partial/non-scoring; frontier and tails incomplete | Accel-Sim, NVAS, Huerta, FlashGPU-Sim, Sim-FA, GTSim, HPCSim | **P0**: complete gpc cycles aggregation and holdout |
| Exact executed-instruction count | Yes for current trace contract; one GEMM 0% delta | Accel-Sim/NVAS counter correlation; GCL/Sieve counts | Broaden; do not call timing accuracy |
| L1/TEX access and hit rates | Partial; native metrics now expose per-origin/unified counters and typed flat-mode gaps | Accel-Sim, Huerta, HyFiSS, FlashGPU-Sim, GCL | P0/P1 stable cache-state replay and mapping |
| Decoded sector/line coalescing and alias topology | Yes; distinct sectors, 128-byte lines, and repeated ATOM/RED span aliases are retained for decoded traces | Accel-Sim/GPGPU-Sim memory-system counters and trace-driven variants expose related transaction domains | P1: topology is not a cache/protocol transaction count; validate broader address patterns |
| L2 access/hit/sector behavior | Partial; native TEX-source lookup/sector/hit rows in carded replay | Accel-Sim, Huerta, Sim-FA, GTSim, HyFiSS | P1 expose per-origin requests/hits/sectors |
| DRAM transactions/bandwidth/latency | Partial; ordinary and cold ATOM/RED RMW read demand/bytes plus dirty/terminal residency and an eventual writeback-demand bound are exposed; physical kernel-interval writes/bandwidth remain typed gaps | GPGPU-Sim, Accel-Sim, MGPUSim, NaviSim, HyFiSS, PPT-GPU | P1 for memory-bound claims |
| Achieved occupancy and IPC | Partial; native SM86 warp-lifetime integral now emits approximate achieved occupancy/IPC ratios | Detailed models; GCL validates occupancy and IPC | P1 stable achieved metrics with NCU semantics; current ratios exclude physical EXIT/launch tails |
| Warp scheduling and issue utilization | Partial; carded resources/frontiers plus model-causal readiness/pipeline counts; 17 NCU scheduler sample families are retained as independent observations | Accel-Sim, Huerta, HyFiSS, GPGPU-Sim | P1 for scheduler-design claims |
| Eligible-warp and scoreboard stall counters | NCU eligible/overlapping stall samples are parsed and retained in `ncu_observation.scheduler_stalls`; native output keeps them typed-unavailable and preserves exclusive causal readiness buckets | Nsight/Accel-Sim/HyFiSS expose scheduler-state families | P0/P1 if stall attribution is a claim; do not equate readiness cycles with NCU warp-cycle samples |
| Launch register/shared-memory resource metrics | Partial; validated trace-root metadata and a separate SM86 theoretical occupancy projection are emitted; achieved occupancy remains model-relative | NCU launch statistics, Accel-Sim, Huerta | P1 occupancy/resource studies still need independent holdouts |
| Driver-selected L1/shared carveout | Yes for external CUPTI join (`sharedMemoryExecuted`); explicit CUDA preference is optional and preserved separately | CUPTI/Nsight launch resource records, modern Accel-Sim resource metadata | P1; external resource capture must match launch identity |
| Branch-target divergence counters | Partial; per-lane successor observer matches NCU divergent/uniform target and CBU branch counts on the focused probe and retained GEMM/RMSNorm/SiLU holdout; per-thread target populations remain separate | NCU branch-target metrics, Accel-Sim | P1 for broader value-dependent CFG/control-flow coverage |
| Fine-grained stall categories | Partial diagnostic; no validated public taxonomy | HyFiSS structural/data/synchronization/control/idle stall reports | P1 if bottleneck explanation is a contribution |
| Per-warp/per-SM timeline | Partial trace/event diagnostics | NVAS, Accel-Sim, Huerta, HyFiSS, GCL | P1/P2 stable timeline output if causal analysis matters |
| Instruction mix / functional-unit utilization | Partial-to-strong for SM86: dynamic opcode/logical-pipeline projections now match FMA across the retained cohort and GEMM ALU/Tensor; ALU/LSU mapping remains approximate on RMSNorm/SiLU | Accel-Sim, HyFiSS, FlashGPU-Sim, GCL metrics | P1 for architecture studies |
| Tensor-core timing | Partial SM86 HMMA/IMMA | Accel-Sim Ampere/Hopper, Huerta, FlashGPU-Sim, Sim-FA, GTSim | P1 for modern AI; no TMA/WGMMA/mbarrier |
| Async global-to-shared destination work | Partial; cache-all LDGSTS destination groups/wavefronts are reported as a separate model-relative observer | Accel-Sim 2.0, FlashGPU-Sim, Sim-FA expose richer async-copy/TMA paths | P1 for SM86 async-copy timing; no complete coherence/visibility proof |
| TMA/WGMMA/mbarrier/warp specialization | No general native path; bounded custom async only | Accel-Sim 2.0, FlashGPU-Sim, Sim-FA, GTSim | P2 unless Hopper/Blackwell is target |
| General barriers, weak memory, inter-group atomics | No; bounded barriers/deterministic shared atomics | MGPUSim/NaviSim/HyFiSS broader synchronization | P2; bounded contract is valid but not CUDA-general |
| Multi-kernel streams/concurrent launches | Partial; launch/trace scope limited | Accel-Sim, NVAS, MGPUSim, gem5 GPU, ACALSim | P2 for end-to-end software-stack claims |
| Multi-GPU/PCIe/NVLink/fabric/coherence | No | MGPUSim, NVAS, gem5 GPU, MAccel-sim, ACALSim | P3/P2 depending question |
| CPU/driver/runtime/full-system | No | gem5-gpu, REASSEMBLE, NVAS, ACALSim/HPCSim | P3 current scope; P2 for universal claim |
| Power/energy | No | AccelWattch, GPUWattch, some PPT/HyFiSS extensions | P3 unless energy is a paper claim |
| Workload sampling / scalable LLM simulation | No | PKA, Sieve, Photon, STEM, Allegro, GCL | P2 if full LLM workloads are required |
| User-defined instruction semantics | Model-relative fixed simt_u32_v1 only | Vortex, Akita/MGPUSim components, VADL/OpenVADL | P1/P2: arbitrary ISA/ABI generation absent |
| User-defined scheduler/resource organization | Yes, bounded architecture specs/resource bindings | Vortex, MGPUSim/Akita, ACALSim, GATSim, GTSim | Strengthen with one consequential design study |
| Arbitrary vendor/ISA portability | No; SM86 plus fixed custom frontend | MGPUSim/NaviSim AMD, MacSim Intel/NVIDIA, Vortex RISC-V | P2/P3; no universal claim yet |
| Graphics/raster/ray tracing | No | Vortex and graphics-oriented frameworks | P3 outside current compute mission |
| Reproducible NCU/profiler binding | Yes, bounded host profile/trace/NCU join | Accel-Sim correlator/GPUVision, FlashGPU CSVs, GCL NCU | Maintain; expand from instruction row to holdout timing/counters |
| Host parallel scaling and peak RSS | Partial event-vs-dense and bounded wall/RSS probes | MGPUSim, GpuTejas, Huerta, LPGSim, ACALSim | P1 for speed claims: report threads, host, setup, RSS, equality |

## Directly comparable normalized project set

| Project metric | Value | Normalized quantity | Status |
|---|---:|---:|---|
| smsp instruction count vs instructions_issued | 197,120 vs 197,120 | Absolute count delta 0%; ratio 1.000 | Scored semantic mapping, one GEMM launch |
| NCU gpc cycles vs model GPC frontier | NCU 13,475--13,529; model 12,949 | Frontier/NCU ratio 0.957--0.961; diagnostic difference about 3.9--4.3% | Not a timing score: retirement/coherence/dirty-writeback tails incomplete |
| Event replay vs dense reference | 887.3 ms vs 1,511.5 ms | 1.70x model-relative speedup | Engineering result, not hardware speed |
| L1 preflight optimization | 782.036 -> 755.453 ms replay median | 1.035x / 3.4% replay reduction | Single fresh GEMM; not universal |
| Static-PC lookup optimization | 740.95 -> 732.34 ms replay median; 857.7 -> 844.0 ms wall median | 1.012x replay / 1.016x wall; event stream unchanged | Same host/trace before-after, not cross-host simulator speed |
| Physical-pipeline probe | NCU GA102 ADU 17,408; CBU 512; Uniform 8,192; FP64/XU 0 | Native dynamic physical assignment remains unavailable; logical FMA/Tensor/ALU/LSU rows remain separate | Schema/coverage improvement, not a physical-pipe accuracy score |
| Custom two-group pipeline | final cycle 45 | Model-relative cycle count | No silicon comparison |

The expanded native-metric cohort is recorded separately in
[`sm86_metric_comparison_20260924.csv`](sm86_metric_comparison_20260924.csv).
Its most useful cross-workload rows are:

| Workload | GPC frontier / NCU max | SM-active mean / NCU | Occupancy model / NCU | L2 TEX hit model / NCU | DRAM read sectors model / NCU |
|---|---:|---:|---:|---:|
| GEMM BF16 | 12,949 / 13,494 (-4.05%) | 11,030 / 11,402 (-3.26%) | 12.54% / 12.27% (+0.27 pp) | 95.27% / 95.11% (+0.16 pp) | 8,262 / 8,420 (-1.88%) |
| RMSNorm BF16 | 18,944 / 19,123 (-0.94%) | 13,039 / 13,063 (-0.18%) | 16.65% / 16.35% (+0.30 pp) | 66.15% / 68.86% (-2.71 pp) | 16,677 / 16,872 (-1.16%) |
| SiLU BF16 | 5,642 / 5,815 (-2.98%) | 3,010 / 3,616 (-16.75%) | 87.97% / 76.97% (+11.00 pp) | 50.00% / 55.04% (-5.04 pp) | 16,410 / 16,568 (-0.95%) |

The fresh CUPTI join also reports the driver-selected GEMM shared-memory
configuration as 32 KiB and the driver occupancy limit as 16 blocks/SM. These
are external launch-resource observations, not NCU-derived simulator inputs.
The small address-carry canary now matches NCU on instruction count, global
loads, ATOM sectors, RED sectors, and aggregate returning-write sectors; its
remaining DRAM-read mismatch is retained as a focused launch/flush-tail
diagnostic. The scaled read/ATOM/RED probe also charges cold RMW reads rather
than silently reporting zero RED demand.
The read/ATOM/RED domain sweep is available as
[`sm86_metric_domain_experiment_20260924.csv`](sm86_metric_domain_experiment_20260924.csv),
covering one-warp iterations before the 168-block scaled cases.

The broader [`sm86_full_metric_holdout_20260925.csv`](sm86_full_metric_holdout_20260925.csv)
keeps pipeline, global request/sector, shared-wavefront, L2, DRAM, branch,
and scheduler-target rows together across GEMM, RMSNorm, and SiLU. It is a
cross-workload observation table, not a fitted score.

These are model-relative diagnostic comparisons, not a fitted holdout score.
All three instruction counts are exact; the SiLU activity mismatch identifies
the next occupancy/launch-wave investigation rather than a correction target.
The static-PC optimization is an implementation result with byte-equivalent
event output; it does not alter the NCU comparison values.

## Conclusions

1. The strongest numerical competitors to a future SM86 result are not one
   simulator: Accel-Sim/Huerta/FlashGPU-Sim for detailed NVIDIA timing,
   NVAS/PPT-GPU/GATSim for selective or abstract timing, and
   Sieve/STEM/GCL-Sampler for workload-scale reduction.
2. The most consequential missing evidence here is not another functional
   opcode. It is held-out, launch-aligned elapsed-cycle and cache/memory
   counter validation on the 3090 Ti, with completed tail semantics.
3. The plausible contribution remains a fast, event-driven, configurable SM86
   timing path plus a bounded user-model frontend. Universal support would
   require a PTX/ABI or second-vendor frontend, memory/fabric model, and broad
   validation.
4. Sampling rows cannot support a claim that this simulator is more accurate
   than Accel-Sim; they measure how much detailed work can be skipped.
5. The next direct comparison should be one same-workload/same-NCU SM86
   elapsed-cycle row against an external NVIDIA simulator, not a cross-vendor
   MGPUSim number. Keep external rows P1--P3 and the local baseline matrix
   non-scoring until then.
6. The carveout “missing metric” is not safely solved by copying a preferred
   percentage: CUDA specifies that carveout as a hint, while CUPTI exposes the
   driver-selected `sharedMemoryExecuted` value. The project now supports that
   external resource-only join. For writeback, GPGPU-Sim documentation and the
   modern-memory-system study show why policy, sectoring, write allocation, and
   DRAM scheduling must be modeled together; a dirty-sector count alone cannot
   become an exact post-kernel completion timestamp.

## Primary-source index

* [GPGPU-Sim/Accel-Sim landscape](https://people.ece.ubc.ca/aamodt/papers/accelsim.isca2020.pdf)
* [GPGPU-Sim quantitative evaluation](https://mkhairy.github.io/Docs/SigmetricsPaper.pdf)
* [Accel-Sim 1.0](https://mkhairy.github.io/Docs/Accel-Sim.pdf)
* [Accel-Sim 2.0](https://accel-sim.github.io/)
* [MGPUSim](https://people.bu.edu/joshi/files/mgpusim-isca2019.pdf)
* [NVArchSim](https://d1qx31qr3h6wln.cloudfront.net/publications/Need_for_Speed_Experiences_Building_a_Trustworthy_System-Level_GPU_Simulator.pdf)
* [NaviSim](https://sarchlab.org/navisim_pact_2022.pdf)
* [PPT-GPU](https://www.sc21.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap571.html)
* [GATSim](https://slam.ece.utexas.edu/pubs/date17.GATSim.pdf)
* [GpuTejas](https://www.cse.iitd.ac.in/~srsarangi/files/papers/gputejas.pdf)
* [HyFiSS](https://github.com/ConvolutedDog/HyFiSS)
* [Huerta modern GPU](https://github.com/upc-arco/modern-gpu-simulator-micro-2025)
* [Huerta parallelization](https://arxiv.org/abs/2502.14691)
* [LPGSim](https://doi.org/10.1145/3805642)
* [FlashGPU-Sim paper](https://arxiv.org/abs/2609.15311) and [repository](https://github.com/FlashGPU-Sim/FlashGPU-Sim)
* [Sim-FA](https://arxiv.org/abs/2605.00555)
* [GPU-Tile-Sim](https://arxiv.org/abs/2607.11262)
* [ACALSim/HPCSim](https://arxiv.org/abs/2605.22936)
* [TrioSim ISCA'25](https://bu-icsg.github.io/publications/2025/TrioSim_ISCA_2025.pdf)
* [PARMA-DITAM'26](https://drops.dagstuhl.de/entities/document/10.4230/OASIcs.PARMA-DITAM.2026.6)
* [PKA](https://mkhairy.github.io/Docs/PKA.pdf)
* [Sieve](https://users.elis.ugent.be/~leeckhou/papers/ispass-2023.pdf)
* [Photon](https://www.comp.nus.edu.sg/~tcarlson/pdfs/liu2023pafssmfgw.pdf)
* [STEM + ROOT](https://seonjinna.github.io/assets/pdf/MICRO25_stem_root.pdf)
* [Allegro](https://seonjinna.github.io/assets/pdf/Allegro_mlarchsys24.pdf)
* [GCL-Sampler](https://arxiv.org/abs/2603.00551)
* [Vortex](https://github.com/vortexgpgpu/vortex)
* [CUDA carveout semantics](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)
* [CUPTI kernel resource activity](https://docs.nvidia.com/cupti/api/structCUpti__ActivityKernel9.html)
* [Enhanced GPGPU-Sim memory-system validation](https://arxiv.org/abs/1810.07269)
