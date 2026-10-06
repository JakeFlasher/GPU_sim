# Metric extractor (NVBit variant B)

This directory contains the compact per-launch NVBit metric extractor retained
for SOL/campaign experiments. It emits JSON Lines with instruction-class
histograms, global-sector efficiency, shared-memory bank conflicts, and branch
active-mask histograms. The optional
`METRIC_EXTRACTOR_TIER2_ADDRESS_DUMP=1` mode writes its bounded address
sidecar.

The wire format is documented in [JSONL_SCHEMA.md](JSONL_SCHEMA.md).

## Build

Build directly in the repository:

```bash
source tools/nvbit_tracer/pinned_build_env.sh  # optional compatibility env
make -C tools/nvbit_tracer/metric_extractor ARCH=sm_86
```

The current source supports `sm_86`, `sm_120`, and `sm_100a` build targets.

Clean generated files with:

```bash
make -C tools/nvbit_tracer/metric_extractor clean
```

## Injection

```bash
CUDA_INJECTION64_PATH="$PWD/tools/nvbit_tracer/metric_extractor/metric_extractor.so" \
  ./your_cuda_app
```

Do not inject this tool and `tracer_tool.so` into the same process. The
retained SOL integration is
[`nvbit_metrics.py`](../../sol_baselines_harness/surfaces/nvbit_metrics.py).
