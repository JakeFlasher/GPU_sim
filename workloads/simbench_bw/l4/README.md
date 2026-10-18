# SimBench-BW L4 pipeline workloads

L4 rows are end-to-end workload pipelines composed from L2/L3 kernels. Each
row declares an intermediate checkpoint so component state can be compared in
addition to the final output.

## Rows and checkpoints

| Slug | Reference family | Boundary |
| --- | --- | --- |
| `moe_layer_residual_norm_pipeline` | grouped GEMM | router → grouped GEMM |
| `prefill_then_decode_pagedkv_pipeline` | paged attention | prefill → decode KV |
| `speculative_decode_draft_verify_pipeline` | attention | draft → verify |

Checkpoint schemas live under `workloads/simbench_bw/schemas/l4_checkpoints/`.
Reference helpers in `references/` construct both the final result and the
checkpoint from the same deterministic state. Tensor fields use canonical
byte-serialization digests; small integer fields retain values.

## Execution

The per-row runners support correctness and timing shapes, seed/warmup/repeat
controls, and optional checkpoint emission. Use the workload runner and its
schema validator for source-level checks. Native simulator acceptance remains
the real-trace full replay described in `docs/REPLAY.md`; a workload reference
or checkpoint is not a substitute for a captured GPU trace.
