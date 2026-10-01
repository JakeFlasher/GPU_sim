# SM86 public endpoint recheck — 2026-09-30

**Scope:** RTX 3090 Ti / GA102 / SM86, driver 615.71.09, current public
Nsight Compute and open NVIDIA kernel-module surfaces.

## Result

No new supported endpoint closes the physical-pipe or cache-completion
boundaries. The implementation must retain the existing unknown-preserving
contracts and must not promote aggregate counters or application-level
instrumentation into hardware receipts.

## Pipe and SMSP identity

The current NVIDIA open-driver request
[#1351](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1351)
explicitly asks for globally identifiable, time-resolved physical SM/SMSP
performance samples. Its description distinguishes aggregate `sm__*`/
`smsp__*` rollups and warp-ID sampling from a stable physical SM coordinate.
This independently confirms that the repository's PMA/NCU aggregate and traced
SM placement surfaces cannot reconstruct a dynamic
`(SM, SMSP, pipe, subunit, issue-slot)` tuple.

Nsight Compute's current profiling guide also defines
`sass__inst_executed_per_opcode_pipeline` as an **estimated** opcode-pipeline
count and states that dynamic assignment is not taken into account. The metric
can remain a comparison target, but it cannot be used as a per-instruction
route witness.

## Cache and completion

The recheck found no public user-space or open-RM interface exposing live
cache-line owner, read-from source, dirty-sector publication, or completed
DRAM acknowledgement for GA102. Existing flush controls, PTX operators, PTE
queries, latency probes, and host-visibility tests remain protocol/phase
observations. They do not justify mutating native replay state without the
explicit producer witness now required by `L1PayloadAdmissionEvent`.

## Independent compiler/ISA recheck

The current public CuAssembler README describes SM86 support as an early,
incomplete project and explicitly distinguishes the assembler from a compiler;
its stated future work includes broader instruction coverage, automatic control
codes, register counting, and control-flow support. The newer C++ port likewise
documents real SM80/SM86 operand-role and latency-table TODOs. OpenPTXas now
reports a GPU-verified SM120 stack, not an SM86 encoder/ABI path. These projects
therefore do not justify widening the repository's finite ABI-v8 capability
ledger into unrestricted SM86 generation.

## Modeling consequence

The next valid software work is limited to:

* candidate/interval projections with explicit unknown route and endpoint
  domains;
* strict producer event and residency contracts;
* differential tests that preserve atomicity and identity;
* independent NCU comparison reporting without consuming NCU values as model
  inputs.

No new full hardware campaign is justified by this recheck alone. Accel-Sim
literature aggregates remain contextual lower-bound evidence, not ground truth;
the NCU campaign remains the independent silicon comparison target.
