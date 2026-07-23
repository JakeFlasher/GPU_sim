# Metric extractor (NVBit variant B)

This directory contains the counter-free per-launch metric extractor built on
the repository's vendored NVBit 1.8 SDK. It coexists with the full dynamic
trace producer in `../tracer_tool/`; the two tools share `../nvbit_release/`
and the opcode tables under `../isa_def/`.

The extractor emits one compact JSON-L record per accepted kernel launch:
instruction-class histograms, global-sector efficiency, shared-memory bank
conflicts, and branch active-mask histograms. Setting
`METRIC_EXTRACTOR_TIER2_ADDRESS_DUMP=1` additionally enables its bounded Tier-2
address sidecar. The wire contract is documented in [JSONL_SCHEMA.md](JSONL_SCHEMA.md).

## H3-preserved artifact

`metric_extractor.so` is a grandfathered, SHA-bound artifact. Its closed
authority is [`artifact_pin.json`](artifact_pin.json): the artifact was not
built under the current contract, H3 remains active, and repin/replacement
requires the outstanding operator decision. The embedded architecture is only
`sm_86`; do not use this preserved byte on `sm_100a` or `sm_120`.

Canonical Git worktrees refuse every extractor build, test-binary build, and
clean target even when a sentinel is supplied. `verify-libnvbit` is read-only
and remains available in the canonical checkout. Verification builds are
allowed only from a non-Git temporary copy after sourcing the exact pinned
environment and setting `NVBIT_TEMPCOPY_BUILD_OK=1`:

```bash
source tools/nvbit_tracer/pinned_build_env.sh
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cp -a tools/nvbit_tracer "$tmp/nvbit_tracer"
export NVBIT_TEMPCOPY_BUILD_OK=1
make -C "$tmp/nvbit_tracer/metric_extractor" clean
make -C "$tmp/nvbit_tracer/metric_extractor" ARCH=sm_86

# Host-only regressions
make -C "$tmp/nvbit_tracer/metric_extractor" run-test-classifier
make -C "$tmp/nvbit_tracer/metric_extractor" run-test-saturating-counter
make -C "$tmp/nvbit_tracer/metric_extractor" run-test-tier2-sha256
make -C "$tmp/nvbit_tracer/metric_extractor" run-test-tier2-marker-metadata
```

The current replacement contract permits verification targets `sm_86` and
`sm_100a`; it rejects `all` and `sm_120`. A successful temporary-copy probe is
evidence about feasibility only and never updates `artifact_pin.json`; do not
copy a temporary build product back into the repository.

## Injection

```bash
CUDA_INJECTION64_PATH="$PWD/tools/nvbit_tracer/metric_extractor/metric_extractor.so" \
  ./your_cuda_app
```

The production integration is
[`../../sol_baselines_harness/surfaces/nvbit_metrics.py`](../../sol_baselines_harness/surfaces/nvbit_metrics.py).
Do not inject this tool and `tracer_tool.so` into the same process.
