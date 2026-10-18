# Portable LCD workload sources

This subtree contains portable CUDA/CUTLASS workload sources used to produce
explicitly labeled control traces. It is a workload input collection, not a
second simulator or an acceptance harness.

## Portability contract

The preserved kernels are built for the base slices `sm_86`, `sm_90`, and
`sm_120` without using architecture-conditional features. Device-family
intent is limited to the corresponding base architecture; no cross-generation
timing or feature claim is made. The roster is in `roster.toml`, and preserved
binary metadata is in `preserved_binaries_manifest.json`.

The source tree intentionally excludes FP4/FP6/NVFP4, architecture-specific
cluster/TMA extensions, and vendor-only timing claims from the portable lane.
The common make rules reject forbidden architecture tokens. These restrictions
keep portable controls distinguishable from native SM120 workloads.

## Build and run

Build one source directory with its local Makefile:

```bash
make -C workloads/portable_lcd/<slug>
workloads/portable_lcd/<slug>/<slug>_run --help
```

Each runner exposes a correctness shape and a timing shape, a seed, warmup/
repeat counts, and an output directory. Measurement manifests use the schemas
under `workloads/simbench_bw/schemas/`. The simulator consumes captured traces
and sidecars, not these source programs directly; see `docs/REPLAY.md`.

## Trace production

Use `tools/nvbit_tracer/README.md` and the pinned build environment to capture
real executions. Keep capture roots outside this Git checkout and pass them
explicitly to the architecture-neutral replay runner. NCU/CUDA observations
are joined after replay and never select simulator parameters.

## References

* `roster.toml` — source roster and architecture slices.
* `common.mk` — portable build guards.
* `verify_kernel_contracts.py` — source/manifest consistency checks for this
  workload collection.
* `docs/CONTRACT.md` and `docs/REPLAY.md` — simulator boundaries and replay.
