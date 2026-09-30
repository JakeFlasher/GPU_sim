# Continuation memory

**Updated 2026-09-30.** Read this file, then [`STATUS`](STATUS.md), then only
the one focused archived report required for the next task. Historical reports
and plans are opt-in context; they are not queues or gates. `AGENTS.md` governs
execution.

## Scope and completion bar

- Repository: `/home/jakeshea/GPU_simulator_rebase`, branch `GPUSim_main`.
- Hardware anchor: RTX 3090 Ti, SM86, UUID
  `GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, driver 615.71.09.
- Locked operating point: graphics/SM 1350 MHz, memory 10501 MHz, P0.
- Native path: `NVBit/SASS -> protobuf -> sim-trace -> sim-sm/sim-mem/
  sim-engine -> sim-run`.
- Custom `simt_u32_v1` and functional `simt_its_v1` are not native SASS timing.
- NCU values are independent comparison targets, never same-workload features.
- Exact closure requires physical pipe/slot telemetry, unrestricted independent
  compiler/ABI/linker generation, cache owner/read-from/DRAM telemetry, and a
  common simulator completion endpoint. Metric projections do not close them.

## Current decision

The final public/open-RM seam audit is recorded in
[`sm86_final_external_seam_audit_20260930.md`](archive/research_20260930/sm86_final_external_seam_audit_20260930.md).
The long-running goal is **blocked** pending a new external endpoint; it is not
complete and no exact mechanism is being renamed as solved.

## Final performance deliverable

Both locked-clock full campaigns completed 15/15 workloads with seven NCU,
five native, and three replay repeats per workload:

| Campaign | NCU cache | Median NCU | Median GPC | Median native | Median replay |
|---|---|---:|---:|---:|---:|
| cold | `all` | 10,080 ns | 13,572 cycles | 0.009448 ms | 0.629196 s |
| warm | `none` | 8,704 ns | 11,709 cycles | 0.009435 ms | 0.644798 s |

Settings use explicit NCU `clock_control: none`; details and raw records are
under `results/alpha/full_3090ti_20260930_rm_cache_{cold,warm}_live/`.

## Durable evidence

- **Independent codegen:** current ABI-v8 no-template images execute vector,
  branch, structured-loop, descriptor-memory/cache-sensitive, and
  reduction/atomic families. The capability ledger is
  [`sm86_independent_codegen_capability_matrix_20260930.md`](archive/research_20260930/sm86_independent_codegen_capability_matrix_20260930.md).
  Unsupported classes remain tensor/texture/surface, dynamic shared memory,
  device functions, relocations, linker/debug/resource synthesis, and arbitrary
  SM86 forms.
- **Independent control:** branch, loop, cache-sensitive, and atomic bodies use
  corrected SM70/SM86 control-field synthesis without copying ptxas control
  words. This is finite operation-family coverage, not unrestricted scheduling.
- **Control layout correction:** SM70+/SM86 control bits are delay 0..3,
  yield 4, write barrier 5..7, read barrier 8..10, wait mask 11..16, reuse
  17..20. See [`sm86_control_word_decode_correction_20260930.md`](archive/research_20260930/sm86_control_word_decode_correction_20260930.md).
- **Static issue priors:** correct `NV20_SUBDEVICE_0` RM reads return legacy
  `fmla32=1`, `imla4=1` and v2 values `0,0,1,0,0,0,0,0,1`. These are static
  instruction-class priors only; throttle/internal KGR variants are unsupported.
- **Pipe boundary:** PMA raw fields, NCU PM sampling, open-RM topology and
  scheduler-contest campaigns expose aggregate/resource state only; no dynamic
  physical pipe, subunit, issue ordinal, or stable physical SMSP coordinate.
- **Cache boundary:** public flush controls, PTX `.wb/.cg/.wt`, PTE query,
  host WT visibility, pointer-chase, and cross-SM latency provide protocol and
  phase evidence only. PTE owner fields are disabled on production RM; line
  owner/read-from/dirty-sector/DRAM completion remain unknown.
- **Timing:** a 1,350,000-clock semantic timer body gives raw QMD/Driver/CUPTI/
  event/NCU medians 1,000,448/1,000,448/1,000,032/1,004,528/1,001,600 ns
  with external locks plus `--clock-control none`. Default NCU boost gives
  715,824 ns. Raw/Driver tail control words differ; this is semantic-body
  timing, not byte-identical SASS or simulator completion.

## Scientific invariants

- Keep issue, modeled-event, completion, launch-envelope, and typed-tail
  frontiers separate.
- Preserve NCU score-only status and related-cohort/OOD caveats.
- Preserve content/semantic identity caveats; timestamps do not prove freshness.
- Treat passing self-authored executions as implementation evidence, not
  independent silicon validation.
- Do not introduce test, hash, verification, audit, or approval workflows.

## Resume procedure

1. Read this file and `STATUS.md`.
2. Inspect current `git status` and the narrow source/artifact seam.
3. Do not rerun the full campaigns unless a new external endpoint changes the
   measurement contract; their final outputs already exist.
4. If a privileged/proprietary RM, firmware, profiler, or hardware trace becomes
   available, update the external seam audit and only then reconsider blocked
   status. Otherwise preserve the four boundaries explicitly.
