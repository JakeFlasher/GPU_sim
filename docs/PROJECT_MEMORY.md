# Project memory

Updated 2026-09-24. A compact continuation note for Codex and other agents;
root [AGENTS.md](../AGENTS.md) remains the project execution profile.

## Resume here

- [STATUS](STATUS.md): implemented paths, measured evidence, open boundaries.
- [Runbook](runbook_host.md): commands from the repository root.
- [Profile contracts](archprofile.md) and [custom programs](../arch_profiles/programs/README.md).
- [Roadmap](../plans/cross_architecture_gpu_simulator_research_plan_20260923.md):
  research choices, not an endless implementation queue.

## Durable decisions

- Software architectural simulation only. Native NVBit/SASS replay is anchored
  on SM86. Custom `simt_u32_v1` has a separate model-relative pipeline;
  `simt_its_v1` is a single-group functional example.
- NCU is an independent comparison target. No RTL, silicon internals, universal
  binary decoder, or arbitrary compiler/ISA generator is required.
- Fair numerical Accel-Sim/MGPUSim comparison remains **deferred**. Existing
  overlap probes are non-scoring. Do not resume M6 just because a historical
  plan calls it the next step.
- A count match is not timing accuracy. The 197,120-instruction mapping is
  demonstrated for one GEMM launch; GPC/SM cycle mapping, unmodeled tails,
  broader evaluation, and publication novelty remain open.
- The custom ISA supports shared load/store, lane-ordered atomic add,
  broadcast/reduction, and bounded barriers. It is not a general memory model.
  Matching barrier position/length/ID and concurrent residency are required.
- Scalar/profile issue-width overrides are mutually exclusive. The profile
  bridge accepts only its documented fields/defaults; native card identity and
  captured semantics do not change.

## Practical continuation facts

- Cargo package `sim-run` is in `sim/direct_run`; its library is
  `sim_direct_run`. Package `sim-run-campaign` is in `sim/run`; its library is
  `sim_run`. There is no package named `sim-direct-run`.
- Use `--manifest-path sim/Cargo.toml` from the repository root; use
  `--manifest-path Cargo.toml` or no manifest flag from `sim/`.
- L1/L1-L2 replay requires `--pchase-card`; flat replay does not.
- Machine-local captures and build outputs are ignored by Git. Dated reports
  identify their local paths; a fresh checkout does not contain those captures.
- The 2026-09-24 host-profile driver/UUID/clocks are dated observations, not a
  claim about the machine forever. Recheck hardware identity for new claims.
- Current changes were made on `GPUSim_main`; inspect the branch rather than
  assuming its name is `alpha`.
- `.claude/knowledge/` is not present in this checkout. Optional old card links
  must not cause repeated failed reads; use tracked docs, source, and
  `references/`. SOL/campaign material remains intact.

[Historical records](archive/README.md) retain older findings and measurements.
Do not interpret their workflow language or earlier blanket “complete” messages
as current scientific evidence. This note replaces stale continuation prose,
not Git history, session transcripts, or raw experiment data.
