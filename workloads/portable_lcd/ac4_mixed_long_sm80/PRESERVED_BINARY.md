# Preserved pre-C1 campaign binary — do not rebuild, delete, or repin

`ac4_mixed_long_sm80_run` (ignored) is a protected, byte-exact campaign input:

    SHA-256 3a6bc3b721ce314308847c109a87aa373d71d3d1de8e7b9167bd0e910eb8d1ba

It was built before S4-C1 and contains an inert optional hook that looks up
`simbench_cupti_record_cuda_event_wall_time_ms` via dlsym at runtime. S4-C1
removed that hook from `ac4_mixed_long_sm80.cu` and deleted the collector
source, build target, and `.so`, so no post-C1 repository workflow can supply
the symbol; the binary times with CUDA events only and the hook never fires.

The tracked source therefore no longer reproduces this binary. Do not run
`make` here and do not regenerate `ac4_mixed_long_sm80_run`: active campaign
evidence pins this exact path and hash as `control_bits.input_sha256`,
`authority_reference_sha256`, and `executable_sha256`. Any rebuild or repin
occurs only after the Stage-6 operator campaign decision, under a separately
authorized rebuild/repin procedure.

The SHA-verified frozen authority
`data/local_labels/campaign/capture_roster_authority_sm86.json`
(`d02c817091b0a9f06c2106c0987c59a481910afcc3b78ab1de7a7d5bd5a6b4b4`)
binds this one binary to both runnable portable cells:

- `ac4_dram_streaming_dot_product`
- `ac4_global_atomic_histogram`

The generated 15-row portable manifest intentionally excludes this binary so
its two-cell identity cannot be flattened into the one-cell-per-row projection.
The protected-state battery carries its fixed pin independently.
