# SimBench-BW correctness-reference families

Every roster row declares a `reference_family` that maps to a Python module here. One module covers one family across all kernels in that family; there is NO per-kernel bespoke Python reference.

| Family | # kernels | slugs |
|---|---:|---|
| `activation` | 2 | `geglu_ffn_fused_bf16`, `swiglu_ffn_fused_bf16` |
| `attention` | 7 | `flash_attn_bwd_bf16`, `flash_attn_bwd_fp8`, `flash_attn_fwd_bf16_causal`, `flash_attn_fwd_bf16_noncausal`, `flash_attn_fwd_fp8_causal`, `ragged_attention_prefill_bf16`, `speculative_decode_draft_verify_pipeline` |
| `audio` | 2 | `conformer_depthwise_glu_bf16`, `whisper_encoder_block_bf16` |
| `cluster_dsmem` | 4 | `cluster_remote_reduction`, `cluster_rmsnorm_dsmem_bf16`, `dsmem_pairwise_latency`, `dsmem_ring_bandwidth` |
| `control` | 2 | `vendor_fp8_matmul_cublaslt`, `vendor_gemm_cublas` |
| `conv` | 1 | `vae_conv_gn_silu_residual_bf16` |
| `diffusion` | 3 | `diffusion_cross_attn_video_bf16`, `diffusion_resnet_time_block_bf16`, `flux_adaln_zero_mod_bf16` |
| `embedding` | 2 | `embedding_bag_scatter_atomic`, `embedding_gather_vocab128k` |
| `grouped_gemm` | 5 | `moe_grouped_gemm_bf16`, `moe_grouped_gemm_fp8`, `moe_layer_residual_norm_pipeline`, `moe_router_topk_softmax`, `nvfp4_grouped_gemm_moe` |
| `mma_microbench` | 22 | `bf16_hmma_gemm_traceable`, `dram_stream_stride_mix`, `fp16_hmma_gemm_traceable`, `fp32_fma_dep_chain`, `fp4_e2m1_mma_dep_chain`, `fp6_e2m3_saturation`, `fp6_e3m2_mma_dep_chain`, `fp8_e4m3_qmma_dep_chain`, `fp8_e5m2_saturation`, `global_atomic_histogram`, `int8_imma_gemm_traceable`, `l2_thrash_pointer_chase`, `nvfp4_blockscale_gemm`, `register_spill_local_mem`, `shared_bank_conflict_stride`, `sparse_2to4_bf16_gemm`, `sparse_2to4_fp8_gemm`, `sparse_2to4_int8_gemm`, `tf32_hmma_gemm_traceable`, `warp_divergence_reconvergence`, `wgmma_bf16_saturation`, `wgmma_fp8_saturation` |
| `multimodal` | 1 | `clip_text_image_cross_attn_bf16` |
| `norm` | 4 | `groupnorm_fwd_bwd_bf16`, `layernorm_fwd_bwd_bf16`, `rmsnorm_bwd_bf16`, `rmsnorm_fwd_bf16` |
| `paged_attention` | 5 | `kv_cache_update_rope_bf16`, `paged_mla_decode_bf16`, `pagedkv_decode_gqa_bf16`, `pagedkv_decode_mqa_bf16`, `prefill_then_decode_pagedkv_pipeline` |
| `rope` | 3 | `qkv_rope_standard_bf16`, `qkv_rope_yarn_bf16`, `rope2d_mm_qkv_bf16` |
| `sampling` | 1 | `sampling_topk_temperature_bf16` |
| `ssm` | 4 | `chunk_gated_delta_rule_bf16`, `hyena_fft_conv_block_fp32`, `mamba2_discretize_chunk_bf16`, `mamba_selective_scan_bf16` |
| `tma` | 2 | `cp_async_shared_tile_copy`, `tma_tensor_copy_2d` |
| `tts` | 2 | `tts_duration_predictor_ln_bf16`, `tts_vocoder_upsample_silu_bf16` |
| `vision_encoder` | 2 | `sam_mask_decoder_cross_attn_bf16`, `vit_encoder_block_bf16` |

Total families: **19** (cap per plan: 20). Total rows covered: **74**.

## Dispatch contract

Each `<family>.py` exposes:

- `FAMILY_NAME: str` — canonical name matching the roster `reference_family` enum.
- `FAMILY_COVERS_SLUGS: tuple[str, ...]` — immutable list of slugs this family handles.
- `build_reference(slug: str, shape_point: dict, seed: int) -> Any` — returns the reference output.
- `compare(actual, reference, tolerance: dict) -> bool` — tolerance-aware comparison.

`run_suite.py --mode correctness` imports the family named by the row's `reference_family` and calls `build_reference` + `compare`. Any family module missing from `references/` is a hard failure even under `--dry-run-cpu`.