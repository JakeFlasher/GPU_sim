# FlashAttention Deep Reference

Repository: <https://github.com/Dao-AILab/flash-attention>

PR case notes: `../prs/cutlass.md` (the only kept PR page; flash-attention PR page dropped during post-pivot trim — for FlashAttention PRs query GitHub directly)

Use this when the target is attention, online softmax, TMA/WGMMA attention, or
Tri Dao-style attention kernel prior art.

## Read Order

1. `csrc/flash_attn/src/flash_fwd_kernel.h`
2. `csrc/flash_attn/src/flash_bwd_kernel.h`
3. `csrc/flash_attn/src/softmax.h`
4. `csrc/flash_attn/src/kernel_traits.h`
5. `hopper/` and `flash_attn/cute/`
6. `benchmarks/benchmark_attn.py`
7. `benchmarks/bench_sm90.py`

## Search Patterns

```bash
rg -n "Flash_fwd_kernel|Flash_bwd_kernel|softmax|TMA|cute|wgmma|sm90|head_dim|causal" csrc flash_attn hopper benchmarks
```

## Candidate Use

- Use as baseline, prior art, or candidate seed only when the task allows it and
  license / attribution are handled.
- Record commit, copied files, online-softmax invariant, tile shape, mask
  semantics, and delta before mutating copied or adapted code.

