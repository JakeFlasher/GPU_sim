# Architecture

## Objective

Predict GPU execution behavior from program structure and dynamic NVBit traces
without feeding measured NCU target values into the simulator.

## Components

```text
dynamic_trace.pb
      |
      v
 sim-trace ------ static SASS/control data
      |                     |
      +----------+----------+
                 v
              sim-sm <---- arch profile
                 |
                 v
              sim-mem
                 |
                 v
             sim-engine
                 |
                 v
              sim-run ----> metric predictions
```

- `sim-engine` provides deterministic time-ordered events.
- `sim-trace` decodes the repository's enhanced NVBit protobuf.
- `sim-sm` models warp issue, dependencies, replay, and instruction progress.
- `sim-mem` models sectored L1, shared L2, MSHRs, and DRAM timing.
- `sim-arch` loads device-specific geometry and timing assumptions.
- `sim-run` from `sim/direct_run/` composes the libraries into the default
  direct application and emits predictions.
- `sim-run-campaign` in `sim/run/` retains the historical manifest/evidence
  interface and is invoked only for an explicit campaign task.

The implementation is event-driven: it should not introduce a global
per-cycle scan or one queued event per instruction when a local actor loop can
advance multiple instructions.

## Measurement boundary

NCU is run separately from NVBit. NCU results are observed values used by
`eval/`; NVBit traces and static program data are simulator inputs. A measured
target must never be copied into an architecture card or replay configuration.

## Architecture scope

The first complete path is `sm_86`. Types for `sm_120` are useful for
portability work, but unknown Blackwell behavior should remain an explicit
missing model rather than silently inheriting Ampere constants.

## Files and paths

Commands accept ordinary filesystem paths. Missing, malformed, or inconsistent
inputs return useful errors. Inputs do not need workflow signatures, ownership
tokens, cryptographic identities, special file descriptors, or publication
state.
