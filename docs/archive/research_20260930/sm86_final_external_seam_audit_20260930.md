# SM86 final external-seam audit

**Date:** 2026-09-30  
**Host:** RTX 3090 Ti / SM86, driver 615.71.09, locked 1350/10501 MHz.

## Public compiler/ISA seam

The current no-template ledger now executes straight-line vector, branch,
structured-loop, descriptor-memory, cache-sensitive, and reduction/atomic
families. Independent SM86 control synthesis is demonstrated for branch, loop,
descriptor-memory, cache-sensitive, and reduction/atomic bodies. The remaining
ABI/ISA surface is general instruction/modifier coverage, arbitrary scheduling,
resource/shared/texture/device-function ABI, relocations, and linker/debug
semantics. The public CuAssembler project itself describes SM86 support as an
ongoing/incomplete expansion rather than a complete compiler.

## Pipe/issue seam

Local PMA raw fields, Nsight source metrics, PM warp sampling, open-RM GR
queries, and the one-CTA/SM contention and operand-bank campaigns provide
aggregate/resource and control-word evidence only. The installed CLI has no
physical SM/SMSP coordinate plus dynamic route/issue ordinal. NVIDIA's public
tracker independently requests this missing interface. No current public
endpoint supplies the exact tuple.

## Cache seam

Correct-subdevice RM cache controls, public flush combinations, PTE information,
PTX cache operators, host WT visibility, pointer-chase latency, and cross-SM
latency all remain protocol/phase evidence. Correct PTE fields are disabled for
the production user client; internal cache controls are unsupported. Static
cache policy and patent disclosures do not expose line owner/read-from/dirty
sector/complete DRAM acknowledgement.

## Timing seam

The long semantic timer body now crosswalks raw QMD, Driver, CUPTI, CUDA events,
and NCU under explicit external locks plus `--clock-control none` within about
0.4%. Default NCU boost control produces a materially different span, so clock
policy is explicit. Raw/Driver text has tail control-word differences; the
crosswalk is semantic-body-level, not byte-identical SASS. Simulator completion
remains a separate frontier.

## Final performance deliverable

The two locked full campaigns are complete at 15/15 rows each:

* cold/cache-all: NCU 10,080 ns median, GPC 13,572, native 0.009448 ms,
  replay 0.629196 s;
* warm/cache-none: NCU 8,704 ns median, GPC 11,709, native 0.009435 ms,
  replay 0.644798 s.

## Audit conclusion

All locally reachable public and open-RM seams have been exercised or ruled out
with finite evidence. Exact closure now requires a new external endpoint:
proprietary RM/firmware telemetry, a supported physical SM/SMSP/pipe trace,
line-state/DRAM completion telemetry, or a synchronized simulator completion
instrument. The repository must not rename these boundaries as solved.

Public references consulted: [CuAssembler](https://github.com/cloudcores/CuAssembler),
[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html),
[NVIDIA physical-SM/SMSP request #1351](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1351),
and [NVIDIA open cache-operation source](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/615.71.09/src/nvidia/src/kernel/gpu/mem_sys/kern_mem_sys_ctrl.c).
