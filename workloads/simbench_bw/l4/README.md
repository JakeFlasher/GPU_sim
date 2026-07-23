# SimBench-BW L4 Pipelines

L4 rows are the plan's "end-to-end pipeline" tier per AC-23 + AC-32:
multi-component CUDA pipelines composed from L2/L3 kernels, exposing
a canonical intermediate-state checkpoint at a plan-defined boundary so
kernel output can be compared component-level in addition to the
end-to-end final output.

## Rows

Three L4 pipeline rows are declared in `roster.toml`
(level = "L4"):

| slug | reference_family | boundary                              |
|------|-----------------|---------------------------------------|
| `moe_layer_residual_norm_pipeline`          | grouped_gemm     | router_to_grouped_gemm_boundary    |
| `prefill_then_decode_pagedkv_pipeline`      | paged_attention  | prefill_to_decode_kv_boundary      |
| `speculative_decode_draft_verify_pipeline`  | attention        | (per-slug; draft / verify boundary)|

## Per-slug checkpoint schemas

Each L4 row has a canonical checkpoint schema committed under
`workloads/simbench_bw/schemas/l4_checkpoints/<slug>.schema.json`. The
checkpoint is the canonical serialized intermediate state that a
compliant implementation must be able to emit at the pipeline
boundary, and is validated by
`workloads/simbench_bw/validate.py::validate_l4_checkpoint`.

Tensor-valued fields in the checkpoint are stored as SHA-256 digests
of their canonical byte serialization (BF16 / FP16 / FP8 / FP32 /
INT32 as per each field's declared `dtype`) so checkpoints stay small
enough to commit as golden files. Small integer-valued fields are
stored as full `values` arrays.

## Directory layout

```
l4/
├── README.md                                  (this file)
├── moe_layer_residual_norm_pipeline/          (Round 36: kernel + main + Makefile)
├── prefill_then_decode_pagedkv_pipeline/      (Round 36: kernel + main + Makefile)
└── speculative_decode_draft_verify_pipeline/  (Round 36: kernel + main + Makefile)
```

Round 30 (drift-recovery) lands the reference + checkpoint
emission for `moe_layer_residual_norm_pipeline` end-to-end in Python.
**Round 36 lands the kernel source trees for all three L4 pipelines**
end-to-end: each row builds via `make run-bin`, runs against its
declared `correctness_shapes[0]` axes, and matches its
`references/{grouped_gemm, paged_attention, attention}.py::build_reference`
output under the row's tolerance triplet.

## Reference + checkpoint helpers

Each L4 row's reference family exports two entry points that share
one boundary-state construction helper, so the end-to-end reference
and the canonical checkpoint provably derive from the same
deterministic state (Round 33 coherence contract per Codex Round 32-review):

- **`moe_layer_residual_norm_pipeline`** — in
  `references/grouped_gemm.py`.
  - `build_reference(slug, shape_point, seed)`: router + permuted
    grouped-GEMM + combine + residual + RMSNorm → final
    `[tokens, hidden]` output.
  - `build_l4_checkpoint(slug, shape_point, seed)`: canonical
    router-to-grouped-gemm boundary (expert_ids + permuted_tokens +
    expert_offsets + routing_weights).

- **`prefill_then_decode_pagedkv_pipeline`** — in
  `references/paged_attention.py`.
  - `build_reference(slug, shape_point, seed)`: shared
    `_prefill_then_decode_boundary_state()` constructs a paged KV
    cache + block table + RoPE cache; decode phase walks the block
    table to reconstruct K/V and runs scaled dot-product attention
    against a seeded decode query → final `[batch, q_heads, head_dim]`
    decode output.
  - `build_l4_checkpoint(slug, shape_point, seed)`: canonical
    prefill-to-decode KV-boundary (kv_block_table + kv_data_blocks +
    seq_lengths + rope_cos_sin_cache).

- **`speculative_decode_draft_verify_pipeline`** — in
  `references/attention.py`.
  - `build_reference(slug, shape_point, seed)`: shared
    `_speculative_decode_boundary_state()` builds draft logits,
    probabilities, tokens, and attention state; verifier-side
    computes target-model probabilities (seed + 900) + rejection-
    sampling acceptance factor → final
    `[batch, draft_length, vocab]` verifier-weighted probabilities.
  - `build_l4_checkpoint(slug, shape_point, seed)`: canonical
    draft-to-verifier boundary (draft_tokens + draft_logits +
    draft_probabilities + draft_attention_state_digest).

## Status (Round 33)

All three L4 pipelines have end-to-end references + canonical-schema
checkpoints + validated shape files. `run_suite.py --mode correctness
--kernel <slug> --component-strict --dry-run-cpu --allow-any-device`
emits + re-validates the canonical checkpoint for each row. The
correctness-path emits `suite_results.json` with
`status="unsupported"` / `reason="pending_timing"` to avoid
publishing placeholder native timing.

`run_suite.py --mode timing --kernel <slug> --component-strict`
(against a built kernel binary) emits `suite_results.json` with
`status="native"` + measured `timing_ms` + merged `component_health`
(Round 33 AC-32 integration). Full AC-32 closure still requires
reporting-platform promotion (hardware-gated on H100 + RTX PRO 6000
Blackwell); the kernel-source half of AC-32 is complete after Round
36.

Round 37 adds kernel-emitted checkpoints to every L4 binary. Each
`l4/<slug>/<slug>_run` now accepts `--emit-l4-checkpoint <path>`
and writes a schema-conforming JSON checkpoint (BF16 / FP32 /
INT32 digests computed by the binary itself via OpenSSL SHA-256).
`run_suite.py --component-strict` invokes the binary with that flag
and compares the emitted checkpoint against the Python reference's
`build_l4_checkpoint()` byte-for-byte; the Python reference is the
oracle, not the artifact source.
