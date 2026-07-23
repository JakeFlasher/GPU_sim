# problem_006_multimodal_rope_position_calculation

| Field                       | Value                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------|
| Problem                     | Compute 3D position IDs for multi-modal rotary embeddings (text + image + video spans) from input_ids, per-image / per-video grid_thw, second_per_grid_ts, and attention_mask. |
| Reference spec              | `references/NVIDIA_Manuals/solbench_problems/problem_006_multimodal_rope_position_calculation.md` |
| Benchmark problem_dir       | `sol-execbench/data/benchmark/L2/006_multimodal_rope_position_calculation/`                 |
| Variant                     | `cuda_cpp`                                                                                  |
| Evidence                    | `Profiled-local-sm_120`                                                                          |
| Evidence (sm_86)            | `Profiled-local-sm_86`                                                                           |
| Local speedup vs reference  | 6.31 – 411.79× on sm_120 (diagnostic only; wide range reflects Python-overhead in reference)|
| SOL Score (online judge)    | <not yet submitted; Phase-2 deliverable, target band 0.3 – 0.5+>                            |
| Counter-free artifacts      | <not yet captured; Phase-1 framework deliverable>                                           |

## Approach

The reference implementation in `definition.json` is a deeply Python-bound
loop: for every batch it materialises `.tolist()`, repeatedly calls
`input_tokens.index(...)`, allocates `torch.arange` / `expand` / `stack` /
`cat` tensors per vision event, and concatenates them at the end. The
algorithmic work itself is tiny (a few thousand int64 writes per batch),
so the reference's wall-time is dominated by Python and many small kernel
launches.

This implementation collapses the work into a single CUDA launch with one
block per batch. Vision-event counts and global prefix offsets are
precomputed on the host with `at::` ops:

- `image_count[b]` = count of `vision_start_token_id` followed by
  `image_token_id` in batch `b`;
- `video_count[b]` analogous;
- `image_offset[b]`, `video_offset[b]` are exclusive prefix sums in
  batch order so the kernel can index `image_grid_thw` / `video_grid_thw`
  without a host roundtrip;
- `valid_lens[b] = sum(attention_mask[b])`.

The kernel (single thread per block, deliberately) then walks the batch's
sequence in lockstep with the reference: for each of `total_events` vision
markers, it finds the next `IMAGE_TOKEN_ID` or `VIDEO_TOKEN_ID` from `st`,
emits the text run `[st, ed)` with positions `[st_idx, st_idx + text_len)`,
and emits the post-spatial-merge `(llm_grid_t, llm_grid_h, llm_grid_w)`
3D positions with the time-axis offset by
`second_per_grid_t * tokens_per_second`. After the loop a final text run
fills any remaining valid positions and `[out_idx, S)` is filled with `1`
to mirror the reference's initial `torch.ones()`.

The global "no vision anywhere" branch (`total_num_images == 0 &&
total_num_videos == 0`) is handled entirely in host-side `at::` ops:
`pos1d = cumsum(mask, -1) - 1`, masked-fill to `1`, broadcast over the
leading `[3, ...]` dimension. No kernel launch.

Constants are pinned per `definition.json` axes (`spatial_merge_size=2`,
`image_token_id=151655`, `video_token_id=151656`,
`vision_start_token_id=151652`, `tokens_per_second=2.0`).

Local sm_120 result: 16/16 PASSED, `speedup_factor` 6.31 - 411.79× over
the reference. The wide speedup range reflects how much Python overhead
the reference incurs at different batch_size × seq_len shapes; the
kernel itself is roughly constant-time per batch position.

## Architecture-coupling caveats

The local 5060 (sm_120) compiles and smoke-runs this baseline; final SOL
scoring is on B200 (sm_100a). Differences worth noting for this pattern:

- HBM3e vs GDDR7 bandwidth: this kernel's memory footprint is tiny
  (~tens of kilobytes per batch), so it is latency-bound on both
  architectures rather than bandwidth-bound. Local wall-time is a
  reasonable proxy for B200 wall-time at this shape class.
- DSMEM and cluster-cooperative reduction (sm_100a-only) are out of
  scope for sm_120 builds; this kernel does not use them.
- L2 carveout: irrelevant at this footprint; the working set fits in
  L1 / shared on both architectures.
- TCGen05 / TMEM divergence: 5th-generation tensor cores and the
  dedicated TMEM register surface only exist on sm_100a; this kernel
  does not use tensor cores (it is purely int64 integer arithmetic),
  so the TCGen05 / TMEM split is not exercised here.
