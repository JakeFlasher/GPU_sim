# SM86 TF32 issue probe

This probe contains no global load, barrier, or reduction in its timed loop.
The dependent arm repeatedly feeds one accumulator tuple to
`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`; the independent arm
rotates eight accumulator tuples.  A per-lane numerical oracle checks all 32
accumulators after the loop, and the host rejects malformed iteration counts.

The independent arm is used only to establish a model-local initiation
interval.  The dependent arm includes dependency and loop overhead and is not
treated as a result-latency measurement.  NCU observations are independent
probe observations, not parameters fitted to a scored workload.

```bash
make -C workloads/research_probes/tf32_issue_probe
workloads/research_probes/tf32_issue_probe/tf32_issue_probe dependent 1024
workloads/research_probes/tf32_issue_probe/tf32_issue_probe independent 1024
```

The retained corrected record is
`docs/sm86_m5g1_tf32_pipeline_probe_v1.json`.
