# SM86 constant-load path probe

This is an unscored architecture probe for the RTX 3090 Ti (`sm_86`). It is
kept separate from the seven-workload validation cohort so its measurements
cannot become per-workload labels or simulator inputs.

The kernel uses `__ldg` and the emitted SASS is inspected with
`cuobjdump --dump-sass`; the relevant instructions are
`LDG.E.CONSTANT`. Capture one active launch with the retained NVBit runtime,
emit its static control sidecar, and replay it with:

```bash
nvcc -O3 -arch=sm_86 -lineinfo -o constant_load_path constant_load_path.cu
cuobjdump --dump-sass constant_load_path
```

The final diagnostic runs used 16 blocks, 256 threads, 128 loop iterations,
and one warm-up. The hardware observations were collected independently with
NCU application replay (`--launch-skip 1 --launch-count 1 --cache-control all`
and `--clock-control none`). The low-pressure six-site form measured about
12.8k SM-active cycles in the final diagnostic run; the simulator's pressure
classifier selected the ordinary 11-cycle global-load arm and produced 12.6k
cycles. A four-load-per-iteration control measured about 17.8k hardware
SM-active cycles and 17.2k simulator cycles. These are sanity checks for the
path regime, not scored targets.

The classifier is based on architecture evidence, not workload names:

* NVIDIA's [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
  identifies `gcc` as the GPC Constant Cache (a L1.5 constant cache) and
  distinguishes its requests from the SM-local data cache.
* The released Huerta SM86 model in
  `GPU_sims_notes/upstream_research_20260901/modern-gpu-simulator-micro-2025`
  specifies an 11-cycle constant-cache structure and a 39-cycle L0/L1
  transport. The pressure-path effective operand visibility is therefore
  represented as `11 + 39` cycles, while ordinary constant streams retain the
  regular 11-cycle global-load arm.
* The five-entry Ampere memory dispatch window (four queue entries plus its
  dispatch latch) supplies the static stream-pressure boundary. Half-width
  (`U16`) constant operations at that boundary use the pressure path only when
  their compiler-emitted write barrier indicates a multi-cycle dependency.

The rule is intentionally conservative and remains an architecture hypothesis
until a broader independent constant-stream sweep is added. The scored
campaign records the resulting raw frontiers and retains all lower-bound and
GPC-domain diagnostics separately.
