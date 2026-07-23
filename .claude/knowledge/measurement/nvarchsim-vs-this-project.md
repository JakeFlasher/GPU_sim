---
Use when: positioning NVArchSim relative to this project's event-driven simulator (sim/); citing NVArchSim design principles, accuracy numbers, or fidelity/speed tradeoffs; correcting the NVArchSim publication-year confusion.
Primary raw source: "Need for Speed: Experiences Building a Trustworthy System-Level GPU Simulator", Villa, Lustig, Yan, Bolotin, Fu, Chatterjee, Jiang, Nellans (NVIDIA Research), HPCA 2021. DOI 10.1109/HPCA51647.2021.00077.
Local mirror: `references/papers/knowledge_base/active/2021__need_for_speed_experiences_building_a_trustworthy_system_level_gpu_simulator_nvarchsim.pdf`
Public PDF: https://d1qx31qr3h6wln.cloudfront.net/publications/HPCA_2021_NVArchSim.pdf
Supporting sources:
- `CLAUDE.md` - mission (re-founded 2026-07-02): build the open, Blackwell-capable event-driven simulator.
- `plans/micro2027_ncu_sim_execution_plan.md` §6 - the simulator design this card informs.
- `references/papers/notes/draft_new_gpu_simulator.md` - founding design note (NVArchSim-inspired).
Repo touchpoints:
- `sim/` - the event-driven simulator (Rust workspace).
- `.claude/skills/designing-gpu-simulator/` - the ACTIVE design skill routing this card.
Out of scope: NVArchSim source code (closed); NVIDIA internal NVAS workflow.
Version family: NVArchSim HPCA 2021. Verified from local mirror and public NVIDIA PDF metadata.
---

# NVArchSim: the Design Template This Project Instantiates Openly

**Status change (2026-07-02 re-founding):** the previous era treated NVArchSim as
prior art for a non-simulator project and forbade simulator revival. That rule is
retired. This project now **builds the open, Blackwell-capable analog** of
NVArchSim: an event-driven, trace-driven simulator (`sim/`, Rust) that predicts
NCU metric vectors on Ampere sm_86 + Blackwell sm_120, validated per-metric
against silicon.

## Factual Record

- Title: "Need for Speed: Experiences Building a Trustworthy System-Level GPU Simulator"
- Authors: Oreste Villa, Daniel Lustig, Zi Yan, Evgeny Bolotin, Yaosheng Fu, Niladrish Chatterjee, Nan Jiang, David W. Nellans
- Venue: HPCA 2021, Industry Track, pages 868-880
- DOI: `10.1109/HPCA51647.2021.00077`

Common correction: NVArchSim is HPCA **2021**, not HPCA 2012.

## Numbers to cite (verified July 2026)

- Hybrid trace- and execution-driven, "loosely cycle accurate", single-threaded,
  event-based; 40-250x faster than GPGPU-Sim 4.0.
- Silicon validation: **955 benchmarks vs GV100, cycle-count MAE 17%, Pearson
  0.953**; Turing port 17% / 0.992. This is the accuracy band NVIDIA's own
  internal tool reports - context for our ≤10% target / ≤15% floor (D5).
- Design principles adopted by `sim/` (execution plan §6): event queue with idle-
  cycle skipping; flow-controlled queues, no retry; variable fidelity per
  component; abstract bandwidth+latency interconnect channels (<1% accuracy
  impact); per-SM batching rather than per-instruction events.
- Philosophy for the paper: "accept and embrace that some degree of inaccuracy
  is inevitable" - trend accuracy across hundreds of workloads beats point
  accuracy on a handful; unphysical fudge factors reduce trustworthiness.

## How this project differs (paper positioning)

1. **Open** where NVAS is proprietary - the research community cannot run NVAS.
2. **Blackwell-capable** (GB206 + GB202, sm_120) where no open simulator has any
   validated config past sm_86.
3. **Output contract is the NCU metric vector** (cycles + sector hit rates +
   traffic + occupancy + SOL%) under `metric_registry.json` observation
   contracts, not cycle counts alone.
4. **Per-arch card calibration** (`arch_profiles/`) from microbenchmarks, with
   the targets-not-features rule (CLAUDE.md Rule 1) keeping NCU out of inputs.

## See Also

- `.claude/skills/designing-gpu-simulator/SKILL.md` - active design routing.
- `.claude/knowledge/simulator/nvasim-architecture.md` - engine/crate layout notes.
- `references/papers/notes/gpu-simulator-non-privileged-research.md` - simulator-
  validation contract methodology (salvaged from the wall_clock_sims era).

## Escalate to Raw When

- A new NVArchSim successor paper appears.
- A paper draft cites a specific NVArchSim number, figure, or method detail.
- Someone again cites NVArchSim as 2012; re-check the HPCA 2021 PDF.
