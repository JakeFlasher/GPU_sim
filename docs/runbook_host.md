# Host NVBit build and runtime runbook

This runbook covers the certified `sm_86` development host and the split
toolchain required when CUDA 13.3 is the system default but the vendored NVBit
tracer must be built with CUDA 13.2.

## Certified host matrix

| Surface | Pinned/observed value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 Ti, `sm_86` |
| Driver | 610.43.02 |
| Authorized Task-17 driver successor | 610.43.03 (explicit authority dated 2026-08-03; capture verification pending) |
| Default CUDA compiler | 13.3 under `/opt/cuda` |
| Pinned NVBit build CUDA | 13.2 under `/usr/local/cuda-13.2` |
| NVBit | 1.8 vendored core |
| Project Python | 3.12.12 in `.venv` |
| Protobuf compiler | pinned protoc 34.1 under `.toolchain/pb` |

NVBit 1.8 carries a CUDA 13.2 header/API snapshot; this repository's custom
tracer is also built with nvcc 13.2 because nvcc 13.3 fails on its pinned
protobuf/Abseil headers. That build pin does not require target workloads to be
compiled with CUDA 13.2: CUDA 13.3 remains the system default, and a CUDA
13.3-built sm_86 target passed the live trace contract on this host. NVBit's
published `driver <=575.xx` line has not been formally corrected, so driver
610.43.02 is locally verified outside the published envelope, not claimed as
official NVBit support. Driver/KMD 610.43.03 is an explicitly authorized
Task-17 successor authority only; it does not rewrite historical 610.43.02
evidence or imply capture verification before the successor gate passes.

## Readiness ladder

Do not use “ready” without naming one of these predicates. They are separate
conjunctions, not a promotion chain: `Pn == true` never proves `Pn+1`.

| Predicate | Required evidence | Current state |
|---|---|---|
| P1 `offline-source-ready(bytes)` | Default Rust/Python/SimBench gates plus the capture-focused battery are green for the same bytes; exactly the live host probe is deselected; frozen-v1, durable-v2, and matrix identities close. | **Established only for `500cd28264e04f1437b01f88c44e0bd9e54bb05f` / tree `b95aff848455d5096eff3ad68941ec1d27f67a5a`.** Race-free producer hashes were independently anchored and replayed with current-source matching under byte-scoped P1 anchor SHA-256 `49e7d24fa60280a64e3cc5fc89fcb28e9efc23826a73503b275961a776f27863`. Rust formatting and workspace tests passed; Python reported 6766 passed, 62 skipped, 47 deselected, and 533 subtests passed; SimBench reported `ok`; the focused battery reported 1040 passed / 1 deselected; and marker collection selected only the named hardware probe. This does not transfer to later bytes. |
| P2 `clean-S-ready(S)` | P1 at clean `HEAD == S`; operator-clean-S live-v1 exists at `inputs/r28_<S8>` with all 122 frozen runnable identities reverified and start/authority/final HEAD equal S; the successor matrix is emitted from it at S with the active registry. | **False.** No current clean-S output set is admitted. |
| P3 `live-canary-ready(host,S)` | P2; live host probe passes; capture-admitted preflight is fresh within 900 seconds; on-device injection smoke is nonempty and contract-valid; three P0 samples prove persistence enabled and 1350/10501 MHz clocks. | **False / typed pending external.** An artifact or collected probe without a passing verdict is not credit. |
| P4 `campaign-ready` | P3 immediately before dispatch; campaign lock free; no storage-fault receipt; sufficient retained/staging budget. Supplemental capture also requires the pinned live canary and closed launch-driver/exporter-authority amendments. | **False.** Supplemental dispatch remains prohibited. |

There is no current hardware credit beyond the historical 1/122 canonical
credit. Clock lock and restore cross the operator privilege boundary; Claude or
campaign automation must never run these commands. The operator commands are:

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1350
sudo nvidia-smi -lmc 10501
# Mandatory restore after the session:
sudo nvidia-smi -rgc
sudo nvidia-smi -rmc
```

Recompute the focused capture portion of P1 with the exact bounded battery:

```bash
.venv/bin/python -m pytest -q \
  tools/local_labels/test_campaign_preflight.py \
  tools/local_labels/test_sm86_capture_matrix.py \
  tools/local_labels/test_sm86_capture_chunk.py \
  tools/local_labels/test_parse_ncu_launch_identity.py \
  tools/local_labels/test_sol_workload_capture_chunk.py \
  tools/local_labels/test_sol_workload_capture_campaign.py
```

The result must report exactly one deselected test. A separate `--collect-only
-q -m hardware_evidence` run over the same six files must identify exactly
`tools/local_labels/test_campaign_preflight.py::test_environment_probes_ok_on_reference_host`.
The receipt above establishes those predicates only for its recorded commit and
tree; rerun both commands for any later byte set. Also run the repository-default
commands from `CLAUDE.md`; a passing focused subset cannot override a failing
default gate.
P1 is byte-scoped; record the tested tree identity. P2 must separately record S
and prove a clean tree. P3 and P4 require fresh live evidence and are never
inferred from P1/P2.

The authority path is frozen v1 → durable v2 → clean-S live v1 → noncanonical
successor matrix. Every transition revalidates its predecessor; frozen evidence
is never rewritten.

Check the live host before a capture:

```bash
nvidia-smi
nvcc --version
/usr/local/cuda-13.2/bin/nvcc --version
.venv/bin/python --version
```

Any mismatch in the pinned build tools is a build refusal, not an invitation to
weaken the version checks.

## Compiler-environment hazard

`/etc/profile.d/cuda.sh` exports `NVCC_CCBIN=/usr/bin/g++-15`. The certified
campaign uses g++-14, and capture preflight treats ambient compiler overrides
as hazardous. Always enter capture commands through:

```bash
tools/local_labels/run_sm86_capture_clean_env.sh COMMAND [ARG ...]
```

The wrapper removes `NVCC_CCBIN`, `NVCC_PREPEND_FLAGS`, and
`NVCC_APPEND_FLAGS` before the campaign constructs its compiler environment.
Do not replace the wrapper with a shell alias or hand-edited profile.

## Pinned NVBit/protobuf build

The checked-in `.pc` metadata was created in an older checkout. The Makefiles
now relocate it through `pkg-config --define-prefix`, and the sourced
environment refuses a version or checkout-prefix mismatch:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
mamba create -y -p "$REPO_ROOT/.toolchain/pb" \
  -c conda-forge libprotobuf=7.34.1  # one time per checkout
cd "$REPO_ROOT"
source tools/nvbit_tracer/pinned_build_env.sh
pkg-config --define-prefix --variable=prefix protobuf
```

The printed prefix must resolve under this repository's `.toolchain/pb`, not
`/home/jakeshea/non_root_profiler/`.

Rebuild the trace wire library and tracer on `sm_86`. Clean first; the single
final build recursively regenerates every protobuf binding with the pinned
compiler:

```bash
source tools/nvbit_tracer/pinned_build_env.sh
make -C tools/nvbit_tracer clean
make -C tools/nvbit_tracer ARCH=sm_86
make -C workloads/simbench_bw nvbit-pb-python
```

Rebuild active metric-extractor instrumentation from source:

```bash
make -C tools/nvbit_tracer/metric_extractor clean
make -C tools/nvbit_tracer/metric_extractor \
  ARCH=sm_86 NVCC_CCBIN=/usr/bin/g++-14
make -C tools/nvbit_tracer/metric_extractor run-test-classifier
make -C tools/nvbit_tracer/metric_extractor run-test-saturating-counter
make -C tools/nvbit_tracer/metric_extractor run-test-tier2-sha256
make -C tools/nvbit_tracer/metric_extractor run-test-tier2-marker-metadata
cuobjdump --list-elf tools/nvbit_tracer/metric_extractor/metric_extractor.so
```

The final listing must contain the intended host architecture. A missing,
stale, or wrong-architecture `.so` is a hard stop. Generated `.so`, `.o`, and
host-test executables are not source artifacts and should not be committed.

At runtime, the tracer needs the pinned protobuf libraries. Sourcing the build
environment supplies them; a fresh CUDA 13.3 workload shell may instead add
only `$PWD/.toolchain/pb/lib` to `LD_LIBRARY_PATH`:

```bash
source tools/nvbit_tracer/pinned_build_env.sh
ldd tools/nvbit_tracer/tracer_tool/tracer_tool.so
```

An `ldd` pass alone does not prove injection. The campaign preflight must also
produce a non-empty, contract-valid smoke trace.
