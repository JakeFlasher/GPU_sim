# Remaining-limitations completion audit

Updated 2026-09-28. This is a current-state audit, not a completion claim.
Historical tests establish implementation/refusal evidence only; they do not
establish native silicon behavior or an independent timing score. The current
alpha handoff did not rerun test suites, hashes, or verification gates.

| Requirement | Current mechanism/evidence | Status | Closure evidence still required |
| --- | --- | --- | --- |
| Native coherence/L1 invalidation | Generation-aware L1 observations, fill publication edge, requester-scoped `CCTL.IVALL`, global visibility events, qualifier-aware cacheable/volatile/proxy/atomic refusal, value/read-from ledger, explicit L1 event-path refusal, 190 `sim-mem` tests; independent SM86 message-passing and L1-control litmus (E26/E26b) | **Bounded T1 + hardware litmus evidence** | Payload/read-from values wired into native events, PTX HB/fence scopes, initial resident state, alias/proxy semantics beyond the declared qualifier, strict lockstep against batch-ahead, and native protocol completion |
| Resident dirty flush | Resident-hit writeback policy, L1/L2 overlap refusal, known-byte payload/cache/queue primitives, native capture-to-payload sidecar, generation-aware Rust admission/coalescing, real `sim-run --native-payload-sidecar` admission, L1→L2→DRAM ledger test | **Bounded T1 + admitted payload seam** | Native store-policy proof, payloads consumed by L1/L2 ownership/eviction, DRAM bytes/timing, host/stream flush boundary, and exact hardware-backed flush conservation |
| Physical-pipe reconstruction | Candidate domains, greedy earliest-completion scheduler, candidate issue/completion bounds and ambiguity visibility | **Model-only** | Independent dynamic route/subpipe witness `(function, PC, warp, ordinal, route, issue cycle)`; aggregate NCU metrics cannot substitute |
| Value-dependent SASS regeneration | Static SM86 artifact lane plus scalar `sm86_value_trace_v1` for predicates, numeric branches, initialized loads/stores, and input-varying values | **Finite functional lane** | NVBit/independent-executor differential on varied inputs with destination/load/atomic/fence values, masks, addresses, drops, and output bytes; static cubin remains value-independent |
| Arbitrary ISA/compiler generation | Bounded `ptx_subset_v2` custom compiler; pinned `ptxas` SM86 lane with cubin/SASS digests | **Explicit finite targets** | A complete independent open-ISA compiler/ISS lane for any broader claim; arbitrary NVIDIA generations/ABI remain unsupported and must refuse |
| Locked-clock timing validation | Operator reports successful `-pm 1`, `-lgc 1350`, and `-lmc 10501`; fresh identity check sees RTX 3090 Ti / SM86 / driver 615.71.09 at 1350/10501; corrected NCU probe returned valid counters but active samples switched 10501/P0→10251/P2 | **Readiness restored; every profiled NCU campaign rejected for drift** | Resolve driver/application-clock behavior, then require ≥20 exact active samples per profiled kernel, no drift, held-out repeats, variance report, and independent simulator comparison |
| Independent external-parser timing | Independent pinned Accel-Sim parser; canonical records/terminal/headers; three repeated counters `5024/96/1`; provenance/resource manifests; [`accelsim_literature_lower_bound_manifest_20260928.json`](accelsim_literature_lower_bound_manifest_20260928.json) | **T1 parser compatibility + literature lower-bound target** | Validate the manifest; report local NCU-backed accuracy against its per-domain error ceilings and host/toolchain-matched simulator speed against throughput/speedup floors; executable RTX3070 cycles are not a local NCU score |

## Non-negotiable refusal rules

- Never convert an optional Accel-Sim executable cycle counter or a local model
  timestamp into an NCU timing ratio. Published/collected Accel-Sim aggregates
  remain admissible lower-bound references when source, workload, GPU, metric,
  units, endpoint, statistic, and uncertainty are recorded. Unmatched rows are
  tagged `lower_bound_reference_unmatched`, not ground truth; only a declared
  metric-domain mapping can mark a floor attained.
- Never infer a dynamic physical pipe from opcode names or aggregate Nsight
  pipeline counters.
- Never clear dirty state without known payload bytes or an explicitly declared
  mask-only bounded experiment.
- Never publish an L1 fill before the generation-matched data-visible event.
- Never claim runtime values regenerated static SASS; static SASS is a compiler
  artifact, while value-dependent execution is a dynamic trace.
- Never run locked-clock scoring while the device identity or active clock
  window cannot be independently observed.

## Current external prerequisite

The device exposure prerequisite is now restored: `/dev/nvidia0`, `nvidiactl`,
UVM, and caps are present, and a fresh query identifies
`GPU-7cca3c43-0fc9-3d91-02a1-3c184a88363c`, compute capability 8.6, PCI
`00000000:01:00.0`, and driver 615.71.09. Read-only verification observed the
operator's target clocks (SM 1350 MHz, memory 10501 MHz), but the first fresh
under-load campaign was deliberately rejected: 150 samples contained 107
`1350/10501/P0` and 43 `1350/10251/P2` samples. The evidence is retained at
`/tmp/locked_clock_campaign_1790583918638322726.csv`; it is a clock-drift
diagnostic, not a timing score. Before any NCU timing cell, investigate why the
memory clock/P-state changes under load (including the driver's application
clock semantics), then repeat the active-window and held-out checks.
