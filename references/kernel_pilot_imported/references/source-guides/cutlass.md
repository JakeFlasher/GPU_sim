# CUTLASS / CuTe Kernel Reference

Repository: <https://github.com/NVIDIA/cutlass>

PR case notes: `../prs/cutlass.md`

Use CUTLASS and CuTe as candidate stacks, baselines, or prior art for GEMM,
tensor-core tiling, TMA/WGMMA, epilogue fusion, and profiler-driven shape
exploration. If the user or baseline points to CUTLASS/CuTe, a standalone
candidate may use CUTLASS/CuTe directly.

## Required Companion References

Read these before using CUTLASS as prior art:

- `../ako4all/cutlass-cpp-kernel-reference.md`
- `../ako4all/cutlass-cpp/cutlass-cpp-overview.md`
- `../ako4all/nvidia-architecture-reference.md`
- `../ako4all/cuda-cpp/sm100-optimization-guide.md` + `../ako4all/cutlass-cpp/sm100-optimization-guide.md` for B200 (sm_100 cloud scoring target) and `sm120-optimization-guide.md` (RTX 5090 local proxy). sm_90 guide was dropped (Hopper not in solbench hardware path)

## Code Map

| Area | Paths to inspect |
| --- | --- |
| CUTLASS templates | `include/cutlass/` |
| CuTe C++ | `include/cute/` |
| Hopper examples | `examples/48_hopper_*` |
| Blackwell examples | `examples/50_blackwell_*`, `examples/60_*` |
| Profiler | `tools/profiler/` |
| Docs | `media/docs/` |
| Tests | `test/unit/` |
| CuTe DSL | `python/` |

## Search Patterns

```bash
rg -n "CollectiveBuilder|Mainloop|Epilogue|TileShape|ClusterShape|KernelSchedule|EpilogueSchedule|Tma|Wgmma|StreamK" include examples test tools
rg -n "block_scaled|fp8|fp4|int8|scale|visitor|EVT|epilogue" include examples test
rg -n "sm90|hopper|warp_specialized|persistent|stream_k|grouped" include examples test
```

## Baseline Extraction

- Use `tools/profiler/` to understand which tile/schedule family wins before
  designing a standalone candidate.
- Record tile shape, cluster shape, stage count, schedule, epilogue schedule,
  math instruction, scale layout, and alignment guard.
- Check unit tests for the shape and dtype combinations that catch partial-tile
  bugs.

## Candidate Translation

Reuse or translate, depending on the requested stack:

- tile-size intuition into a candidate tiling experiment
- epilogue fusion opportunity into the candidate epilogue
- scale layout and shape coverage into tests
- profiler result into candidate hypotheses

When copying/adapting CUTLASS or CuTe code, preserve license/notice context and
record exact files, commit, config, and delta in the source ledger and lineage.

## Useful Cross-Framework Priors

- TensorRT-LLM for NVIDIA-serving shape routing.
- SGLang and vLLM for concrete serving hot shapes.
- PyTorch for correctness and tolerance.
