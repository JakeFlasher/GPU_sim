> **SUBTREE-HISTORICAL:** this file preserves the SOL-ExecBench campaign
> context. Its rules and workflow descriptions are historical. Root
> `AGENTS.md` controls all work; use the current campaign skill and runbooks
> for executable commands.

# CLAUDE.md — Project Context

## Project Overview

This repo builds a **counter-free GPU analytics framework** for profiling
complete *.cu CUTLASS C++ / CuTe C++ / CUDA C++ solutions of NVIDIA's
**SOL-ExecBench** (arXiv:2603.19173) — 235 GPU kernel optimization
problems across four categories (L1=94, L2=82, Quant=33, FIB=26)
targeting the NVIDIA B200 (sm_100 / sm_100a). Solutions are scored by
the SOL Score:

    S(Tk) = (Tb − T_SOL) / ((Tk − T_SOL) + (Tb − T_SOL))   ∈ [0, 1]

S = 0.5 matches the internal scoring baseline; S = 1.0 reaches the
analytically derived hardware Speed-of-Light bound from `SOLAR/`. The
paper's agent baseline achieved overall median S = 0.732 (L1=0.688,
L2=0.761, Quant=0.757, FIB=0.789).

### Two-phase mission

**Phase 1 — local framework smoke (this machine, RTX 3090 Ti sm_86; the RTX 5060 Laptop sm_120 sibling defers to a separate future session).**
Hand-port the 235 official SOL-ExecBench Python references into *.cu
CUTLASS C++ / CuTe C++ / CUDA C++ template solutions (one *.cu per
problem) and build a counter-free analytics framework over them. The
framework drives the native `sol-execbench` evaluator (installed via
`uv sync` of the in-tree `sol-execbench/` source tree) as a host
subprocess — it keeps producing the correctness verdict + diagnostic
speedup vs. PyTorch reference — and layers five counter-free analytic
surfaces on top: NVBit dynamic SASS / address traces, CUPTI Activity
wall-clock attribution, Nsight Systems with NVTX ranges, static
introspection via `cuobjdump --dump-sass` / `nvdisasm` / `ptxas -v`,
and `compute-sanitizer` (memcheck / racecheck / synccheck / initcheck).
**Local performance is not a goal.** The Phase 1 success bar is "the
framework runs end-to-end on a representative *.cu pilot, emits all
five analytic artifacts cleanly, and the same invocation surface is
what will be shipped to the B200 provider."

The RTX 5060 Laptop sm_120 host that produced the pre-2026-05-25
landed manifests under `tools/sol_baselines_harness/.harness_traces/`
is no longer attached. Those manifests are retained as a frozen
evidence trail (legacy compatibility guards only) so the additive-not-
surgical discipline can be verified on every new arch addition; they
are NOT inputs, dependencies, or prerequisites for the live sm_86 work
on the current RTX 3090 Ti host.

**Phase 2 — migrate the framework to a B200 provider.** The B200
provider is also the host class for the official SOL-ExecBench online
judge, so a kernel that passes correctness on the provider has a
defensible path to a real leaderboard SOL Score. Re-run the counter-
free profiling across all 235 *.cu hand-ports on actual sm_100a hardware
to harvest the TCGen05 / TMEM / TMA / cluster / DSMEM / NVFP4 / FP8 /
L2 / HBM3e behavior that the local sm_86 host (and the future sm_120
cloud host, when one is allocated for the Tier-2 cache-predictor
cross-arch calibration) cannot reveal. The B200
profiling intel is the deliverable — downstream optimization of the
*.cu kernels is **out of scope for this project**.

### Acceptance contract for *.cu template solutions

This project does **not** supersede or optimize any kernels. A *.cu
template solution is acceptable when:

1. **Correctness passes** on every workload in the problem's
   `workload.jsonl` under the native `sol-execbench` evaluator.
2. **No reward-hack flag** from the evaluator.
3. After Phase-2 online-judge submission, the SOL Score lands in
   roughly the **`0.3 – 0.5+` range** for most template solutions.
   This is a **target / reporting expectation, not a rejection
   threshold** — template solutions are functional ports, not tuned
   kernels, and some (e.g. NVFP4 / FP8 quantization ports) will land
   below 0.3 on the leaderboard. Such pilots still count as accepted
   profiling inputs as long as criteria 1–2 hold. **The local-proxy
   `local_sol_score` column is diagnostic only** and is NOT
   substitutable for the online-judge SOL Score. (A historical
   variant of the same column name carrying an arch suffix is
   retained only in archived output; current report generation emits
   the canonical `local_sol_score` key from
   `tools/sol_baselines_harness/report.py`.)

Speed against the PyTorch reference (`speedup_factor`) is recorded as
diagnostic data only and is **NOT** a landing gate. A *.cu with
`speedup_factor < 1.0` still lands. The legacy `deferred_slow` status
is retired. The `deferred.json` ledger now records only:

- `deferred_correctness` — at least one workload failed correctness.
- `deferred_reward_hack` — evaluator detected an evaluator-state
  exploit. Should be empty by construction; we never ship those.

The 235 *.cu hand-ports live under `baselines/{L1,L2,Quant,FIB}/<slug>/
{cutlass,cuda_cpp}/`, the same layout and `solution.json` contract as
the 8 existing landed pilots.

## Critical Rules

1. **No NCU / Nsight Compute in the deployed framework path or on
   B200.** The framework's runtime path (the five counter-free analytic
   surfaces driven by `tools/sol_baselines_harness/`) and any B200
   invocation MUST NOT depend on `ncu` / `nv-nsight-cu-cli` or
   `.ncu-rep` / `.ncu-repz` artifacts. Tools that depend on NCU access
   do not transfer to our non-root cloud B200 surface, so the deployed
   framework disciplines itself from the start.

   **Calibration-only carve-out (operator-authorized 2026-05-16):**
   NCU is permitted as an *offline ground-truth source* for the
   Tier-2 cache-prediction calibration on sm_86 and sm_120 only. The
   calibration workflow runs NCU once per calibration kernel to
   harvest L1 / L2 **read-only** sector hit-rate metrics
   (`l1tex__t_sector_op_read_hit_rate.pct`,
   `lts__t_sector_op_read_hit_rate.pct`, plus the underlying
   `…_sectors_op_read_hit.sum` / `…_sectors_op_read.sum` counters).
   The combined `…_sector_hit_rate.pct` family conflates store
   traffic under L2's write-back / no-write-allocate policy and
   biases labels — do not use it as the calibration target. Labels
   feed the cache predictor at
   `tools/sol_baselines_harness/metrics/cache_predictor.py`, and ships
   a calibrated model + per-arch ledger that does NOT need NCU at
   deployment. The deployed framework path and every B200 invocation
   remain NCU-free.

   **Tiered accuracy gate (operator-set 2026-05-16):** ±5 % MAPE per
   launch on the *predictor-friendly* cohort (LayerNorm, RMSNorm,
   SiLU, Softmax, well-behaved GEMM, and equivalents); ±15 % MAPE per
   launch on the *predictor-hostile* cohort (carveout-sensitive,
   persisting-window, prefetch-hint, warm-cold-pair, spill-to-local
   kernels, and equivalents); typed-fail-closed (`predictor_typed_
   unavailable` with a reasons array) on cells the model cannot
   honestly score. Both L1 and L2 hit-rate predictions are scored
   independently. Predictor-friendly / hostile partition is contract-
   pinned per-cell, not chosen post-hoc — see the cross-study
   precedent at `/home/jakeshea/3090ti_5060laptop_cross_study`.

2. **No clock-policy mutation on any device.** Do not invoke
   `nvidia-smi --lock-gpu-clocks`, `--lock-memory-clocks`, persistence,
   power, MIG, or compute-mode mutation flags from our code or scripts.
   Do not edit host driver state. The `sol-execbench` CLI exposes
   `--lock-clocks`, but clock policy belongs to the leaderboard
   scoring service, not to us. Even on the local 3090 Ti we treat
   clock-locking as unavailable, to keep our dev-loop aligned with
   the non-root cloud B200 surface.

3. **No privileged CUPTI surfaces.** No Events, Metrics, Range Profiling,
   PM Sampling, or PerfWorks APIs. **CUPTI Activity is allowed** for
   dev-loop wall-clock attribution (flush via `cuptiActivityFlushAll`).
   Note that the local `sol-execbench/` checkout defaults to CUPTI-based
   timing in `core/bench/timing.py` while the published paper §5.1
   describes CUDA-event timing — when reasoning about Tk, anchor each
   claim to one authority (paper / leaderboard service / this checkout)
   and call out which.

4. **No host-driver work.** Host package install via `apt` / `dnf` /
   the provider's package manager is fine. NVIDIA driver install,
   DKMS, MOK, `modprobe nvidia`, kernel module parameter toggles, and
   reboot workflows are out of scope.

5. **Submission surface: CUDA C++ / CUTLASS C++ / CuTe C++ only.** Do
   not submit Python, Triton, CuTe **DSL** (Python), cuTile Python,
   `cudnn_frontend` Python, or `torch.compile` solutions. CuTe is
   allowed only as the C++ template surface (`cutlass::cute`) compiled
   into a CUDA / C++ extension. Submission `spec.languages` ∈
   `["cutlass"]` or `["cuda_cpp"]`. Entry points bind through PyTorch
   (`PYBIND11_MODULE`, default `spec.binding: "torch"`) and launch work
   on `at::cuda::getCurrentCUDAStream()`. cuBLAS / cuDNN are **host**
   APIs — create and cache handles on the host, set the handle's stream
   to the current PyTorch stream, set math / dtype modes explicitly,
   and **never** call them from `__global__` or `__device__` code.
   CUTLASS device-side templates are fine and are launched from the
   host entry-point.

6. **No evaluator-state exploits.** Every `run()` call must compute
   outputs from the current inputs. Do not cache outputs or
   intermediates across correctness / timing using `data_ptr`, raw
   tensor addresses, shape-only keys, globals, static device buffers,
   or one-time state that changes semantics after the first invocation.
   The harness shifts input pointers by 256 B every iteration via a
   fresh allocator; address-keyed caching silently goes stale and fails
   correctness.

7. **Default-stream discipline.** Submitted kernels may use only the
   current PyTorch CUDA stream (`at::cuda::getCurrentCUDAStream()`).
   Do not create user CUDA streams, dispatch to background host
   threads, enqueue unsynchronized work, use `torch.jit.fork`, or rely
   on null-stream side effects. CUDA Graphs (`cudaGraphInstantiate`,
   `torch.cuda.CUDAGraph`) are disallowed unless a per-problem note
   proves current input/output pointers and current-stream behavior
   survive the harness's stream-injection detector.

8. **Precision is contractual.** Match the problem `dtype` exactly for
   inputs and outputs. Wider accumulation is fine (FP32 accumulator
   for BF16 / FP16 / FP8 inputs is expected). Do not precision-downgrade
   compute to pass tolerance (e.g. FP16 compute upcast to FP32 output).
   For FP8 (e4m3, e5m2) and NVFP4 (e2m1) problems, the kernel must
   document scale layout, block size (typically 16- or 32-element
   blocks), and format interpretation in a header comment.

9. **`IISWC_conference_latex/` is operator-only.** Do **not** read,
   edit, write, list, glob, grep, or otherwise touch any path under
   `IISWC_conference_latex/` from Claude — and do **not** delegate
   such access through Agent / subagent invocations, Explore /
   general-purpose / Plan agents, or any other tool that would pull
   the folder's contents into context indirectly. The IISWC 2026 #159
   paper draft lives in a separate author-driven review workflow.
   Treat the directory as a black box even when the operator asks
   you to "look at the paper" — surface this constraint and decline.
   This rule overrides any read-only tool default that would otherwise
   pull file contents into context. The pattern-atlas, methodology
   notes, and any other technical material live entirely outside this
   folder.

## Workflow

- **Source ranking for unknowns:** repo source / local NVIDIA docs in
  `references/NVIDIA_Manuals/` / the published paper first; then
  `/ask-codex` (`gpt-5.5:xhigh`) only after the local search misses or
  conflicts. Persist resolved findings to
  `~/.claude/projects/<project>/memory/`. **Memory is host-local** and
  does NOT travel between the laptop and cloud B200 — re-derive on the
  receiving host.
- **Use the official frameworks; do not re-implement.** SOL-ExecBench
  evaluation runs the native CLI directly:
  `sol-execbench <problem_dir> --solution solution.json --json`. The
  upstream package is installed via `uv sync` of the in-tree
  `sol-execbench/` source tree. T_SOL bounds come from `SOLAR/`
  (NVlabs SOLAR: roofline + Orojenesis).
- **Python environment is `uv`-only; do NOT use conda for this
  project.** The canonical interpreter is the **repo-root**
  `.venv/bin/python` (i.e. `<your-clone-of-solbench_baseline>/.venv/bin/python`),
  produced by `uv sync --all-packages --all-groups --python 3.12`
  from the repo root. This is a uv workspace; the root `pyproject.toml` declares
  `sol-execbench/`, `SOLAR/`, and the vendored patched `torchview/`
  as workspace members and resolves them to their in-tree editable
  installs. Always run the harness / tests / `sol-execbench` CLI
  through this venv (either activate it via
  `source .venv/bin/activate` or invoke `uv run …` from the repo
  root). A conda-`base` shell on this host may shadow workspace
  packages with installs from a different clone, which silently
  desynchronises the running evaluator from the vendored source
  the cache key hashes. The subprocess builder in
  `tools/sol_baselines_harness/surfaces/subprocess_env.py` copies only
  its curated environment allow-list and explicitly scrubs profiling
  variables between surfaces. Confirm the env is correct with
  `python -c "import sol_execbench, solar, torchview; print(sol_execbench.__file__, solar.__file__, torchview.__file__)"` —
  every path must resolve to the in-tree workspace packages for this
  checkout.

## Counter-Free Analytics Framework

The framework drives the native `sol-execbench` evaluator (installed
via `uv sync` of the in-tree `sol-execbench/` source tree) and layers
five counter-free analytic surfaces over every *.cu under `baselines/`.
Every surface is non-root, non-NCU, non-clock-mutating, and **designed
to migrate** to the B200 provider used for Phase 2 — "designed to
migrate", not "migrates intact": each surface depends on toolchain
availability and provider policy (preinstalled `nsys` /
`compute-sanitizer` binaries, CUPTI library versions, NVBit
driver-injection allowlist, writable scratch space, kernel-mod and
ptrace policy) that can differ from what the local RTX 3090 Ti provides.
Surfaces that are blocked on the chosen B200 provider must record
`status: blocked` in the per-surface manifest (see "Counter-free
artifact manifest" below) — they do NOT silently demote the *.cu to
unprofiled.

| # | Surface | Role | Invocation hook |
|---|---|---|---|
| 1 | **NVBit** dynamic per-warp metric histograms (instruction mix, sector efficiency, bank conflicts, branch active-mask) + opt-in Tier-2 address-trace sidecar | Per-launch metric extractor emits JSONL launch records; Tier-2 address dump emits an ordered compact sector-event sidecar when `METRIC_EXTRACTOR_TIER2_ADDRESS_DUMP=1` (env-gate at `nvbit_tracer/metric_extractor/metric_extractor.cu:376-403`). Producer is `nvbit_tracer/metric_extractor/metric_extractor.so` (matches the `b200_preflight.py` `nvbit_metrics_tool` required probe and the `nvbit_metrics` surface plugin at `tools/sol_baselines_harness/surfaces/nvbit_metrics.py`). | `CUDA_INJECTION64_PATH=nvbit_tracer/metric_extractor/metric_extractor.so` env on the evaluator subprocess |
| 2 | **CUPTI Activity** (non-counter) | Wall-clock attribution per kernel launch and memcpy phase. Must be flushed via `cuptiActivityFlushAll` before reading. **Collision risk:** the upstream `sol-execbench/src/sol_execbench/core/bench/timing.py:426` already uses CUPTI Activity for its own timing path. Prefer extending / harvesting that path over loading a second CUPTI consumer in the same process. When a separate harness invocation is needed, run it as a matched separate eval to avoid double-subscription. | Extension of `core/bench/timing.py` or a matched separate eval invocation |
| 3 | **Nsight Systems** (with **optional** NVTX ranges) | Application-level correlation. `nsys profile --trace=cuda,nvtx` captures the evaluator subprocess; NVTX-tagged ranges inside the *.cu solution are **best-effort and optional** — submitted kernels must not depend on NVTX being available (judge image may exclude it). When NVTX is desired for richer correlation, ship an instrumented profiling variant (e.g. `kernel_nvtx.cu`) distinct from the canonical submitted *.cu. | `nsys profile --trace=cuda,nvtx --output=…` |
| 4 | **Static introspection**: `cuobjdump --dump-sass --dump-ptx`, `cuobjdump --list-elf`, `cuobjdump --print-gpu-summary`, `nvdisasm`, `ptxas -v` | Build-time SASS / PTX / register-pressure / spill reports plus per-`.so` cubin arch listing (`--list-elf` proves `sm_100a` is actually present after the B200 build). No runtime dependency. | Post-build hook against `torch.utils.cpp_extension`'s emitted `.so` |
| 5 | **`compute-sanitizer`** (memcheck / racecheck / synccheck / initcheck) | Catches correctness bugs the SOL-ExecBench randomized correctness trial would surface. Non-root. | `compute-sanitizer --tool=… python eval_driver.py …` |

`cuda-gdb` is allowed for one-off diagnostics when launch / ptrace
permissions allow, but is not part of the standard framework pipeline.

### Artifact layout

The framework writes one artifact bundle per `(slug, language)` pair:

```
tools/sol_baselines_harness/.harness_traces/
└── <slug>__<lang>/
    ├── trace.json              # existing sol-execbench evaluator trace (rows[])
    ├── nvbit_metrics/          # per-launch JSONL records + optional Tier-2 address-trace sidecar
    ├── cupti/activity.jsonl    # CUPTI Activity records, flushed per launch
    ├── nsys/profile.nsys-rep   # Nsight Systems report (NVTX ranges visible)
    ├── static/                 # cuobjdump SASS + ptxas -v stats per .so
    └── sanitizer/              # compute-sanitizer report per tool
```

The existing `tools/sol_baselines_harness/run.py` keeps producing the
top-level `trace.json` (SOL Score / speedup / correctness fields). The
four additional surfaces are layered on top by Phase-1 work — they do
NOT change the existing `trace.json` schema.

### Counter-free artifact manifest (`manifest.json`)

Every `.harness_traces/<slug>__<lang>/` bundle carries a `manifest.json`
that records per-surface availability. The framework does NOT assume
every surface emits on every host — provider-side toolchain blocks
(NVBit injection denied by container policy, `nsys` binary absent,
CUPTI library mismatch, etc.) are recorded explicitly instead of
silently degrading to an unprofiled state.

Schema:

```json
{
  "schema_version": 1,
  "slug": "problem_NNN_<slug>",
  "language": "cuda_cpp | cutlass",
  "host": {
    "kind": "local_sm_8_6 | local_sm_120 | b200_sm_100a",
    "device_name": "<from nvidia-smi>",
    "compute_capability": "<sm_XXX>",
    "driver_version": "<from nvidia-smi>",
    "cuda_runtime_version": "<from cudaRuntimeGetVersion>",
    "captured_at_utc": "<ISO-8601>"
  },
  "surfaces": {
    "nvbit_metrics":  { "status": "ok|unavailable|blocked|failed|skipped", "tool_version": "...", "command": "...", "exit_code": 0, "artifact_path": "nvbit_metrics/", "waiver_reason": null },
    "cupti_activity": { "status": "...", "tool_version": "...", "command": "...", "exit_code": 0, "artifact_path": "cupti/activity.jsonl", "waiver_reason": null },
    "nsys":           { "status": "...", "tool_version": "...", "command": "...", "exit_code": 0, "artifact_path": "nsys/profile.nsys-rep", "waiver_reason": null },
    "static":         { "status": "...", "tool_version": "...", "command": "...", "exit_code": 0, "artifact_path": "static/", "waiver_reason": null },
    "sanitizer":      { "status": "...", "tool_version": "...", "command": "...", "exit_code": 0, "artifact_path": "sanitizer/", "waiver_reason": null }
  },
  "evidence_label": "Profiled-local-sm_86 | Profiled-local-sm_120 | Profiled-B200-sm_100a | Profiled-local-complete | Profiled-B200-complete"
}
```

`status` semantics:

| status | meaning |
|---|---|
| `ok` | Surface emitted a clean artifact (exit 0; artifact non-empty). |
| `unavailable` | Tool not present on the host (e.g. `nsys` binary missing). `waiver_reason` records what was looked up and not found. |
| `blocked` | Tool present but provider policy / kernel / container security blocks invocation (e.g. NVBit injection denied). `waiver_reason` records the policy origin. |
| `failed` | Tool invoked but exited non-zero or produced an empty / unparseable artifact. `waiver_reason` records the failure mode. |
| `skipped` | Explicitly waived for this baseline. Rare; needs a `waiver_reason` referencing CLAUDE.md / a per-problem note. |

### Phase-1 → Phase-2 preflight gate (`b200_ready_manifest.json`)

Before fanning the framework out to all 235 *.cu profiling runs on a
B200 provider, the operator MUST clear the preflight gate. The
preflight is implemented at
`tools/sol_baselines_harness/b200_preflight.py` and is read-only
end-to-end (no clock-policy mutation, no NCU, no host-driver work).

Invocation:

```sh
python tools/sol_baselines_harness/b200_preflight.py \
    --output tools/sol_baselines_harness/b200_ready_manifest.json
```

The manifest captures the host toolchain inventory plus per-probe
status, with the same `ok | unavailable | blocked | failed | skipped`
vocabulary the per-surface manifest uses. Schema:

```json
{
  "schema_version": 1,
  "captured_at_utc": "<ISO-8601>",
  "host": { "kernel": "...", "arch": "...", "python_version": "...", "hostname": "..." },
  "required_probes": ["nvcc", "driver", "cuda_runtime", "device", "cutlass", "sm_100a_compile_proof"],
  "recommended_probes": ["nsys", "compute_sanitizer", "cupti", "nvbit_tracer", "storage"],
  "checks": { "<probe>": { "status": "...", ... } },
  "blockers": ["<probe_name>", ...],
  "ready_for_phase_2": true
}
```

Probes:

| Category | Probe | What it checks |
|---|---|---|
| Required | `nvcc` | `nvcc --version`, major/minor recorded. |
| Required | `driver` | GPU driver version via `nvidia-smi --query-gpu=driver_version`. |
| Required | `cuda_runtime` | Runtime version (currently inferred from nvcc; authoritative reading is `cudaRuntimeGetVersion()` via a host stub compile). |
| Required | `device` | At least one GPU visible; compute capability recorded. For Phase-2 the expected CC starts with `10.` (sm_100 / sm_100a). |
| Required | `cutlass` | `cutlass/include/cutlass/version.h` parses; major / minor / patch recorded. |
| Required | `sm_100a_compile_proof` | `nvcc -gencode arch=compute_100a,code=sm_100a -c stub.cu` accepts a one-line stub. Proves the toolchain codegens for B200 even from a sm_86 / sm_120 host. |
| Recommended | `nsys` | `nsys --version` (Surface 3). |
| Recommended | `compute_sanitizer` | `compute-sanitizer --version` (Surface 5). |
| Recommended | `cupti` | `libcupti.so` resolves on the search path (Surface 2). |
| Required | `nvbit_metrics_tool` | `nvbit_tracer/metric_extractor/metric_extractor.so` exists AND its `cuobjdump --list-elf` cubin list covers the host arch (Surface 1; escalates to `failed` on a B200 host when `sm_100a` is missing from the cubin list). |
| Recommended | `storage` | Free disk on the artifact path; warns under 20 GiB. |

Exit codes: `0` = ready (no required blocker); `1` = manifest emitted
but at least one required probe failed; `2` = manifest emission
itself failed.

Recommended probes that report `unavailable` / `failed` do NOT block
Phase 2 — they translate into per-surface `status: unavailable`
manifests for those slugs (see "Counter-free artifact manifest"
above). The point of the preflight is to **make provider-side
toolchain gaps explicit before kicking off 235 profiling runs**, not
to demand a uniformly-equipped host.

### Scoring path — three distinct authorities; keep them straight

- **Paper §5.1**: CUDA events, 10 warmup + 50 timed × 3 trials,
  L2-cache clear (256 MB), subprocess isolation, 300 s timeout, B200
  clocks locked at 1500 MHz by `nvidia-smi`. The published description.
- **Public leaderboard / scoring service**: definitive Tk for the SOL
  Score. Owns clock policy. We submit; we do not run it.
- **Local `sol-execbench/` checkout (the code in this repo)**: CLI
  `--timeout` defaults to 600 s; `core/bench/timing.py:426` defaults
  to a CUPTI-based timer; L2-cache clear sizes itself off the actual
  device's L2 despite a stale 256 MB comment. Use for local
  sanity-check timing; never quote its number as "the SOL Score."

None of the five analytic surfaces above replace the official SOL
scoring — they sit beside it.

## Architecture Portability Discipline

The current local dev host is RTX 3090 Ti **sm_86** (Ampere, GA102 —
6 MB L2 / 6 slices, 192 KB L1/SM, 128 B line). The 5060 Laptop
**sm_120** (Blackwell consumer, GB206 — 32 MiB L2, ~128 KB unified
L1/SM, 8 GiB GDDR7) is the cross-arch sibling host that produced the
Round-11 + R19-R23 calibration artifact family; the 5060 host is no
longer attached, so its artifacts are preserved as frozen evidence
(read-only input to the active plan at `.humanize/plans/crossarch-
predictor-validity-restore-sm86-primary-20260528-052258.md`) and as
the cross-arch validity reference for the sm_86 work on the live 3090
Ti host. The cross-arch discipline — that predictor validity is
established SEQUENTIALLY across separate sessions (sm_86 fully on the
live host this loop, then sm_120 in a later session) before shipping
to B200, never on both arches in one session — is preserved by
treating sm_120 as audit history of one calibration side, not as a
current execution target. B200 scoring is
**sm_100 / sm_100a** (Blackwell datacenter, GB100). Treat all three
as different targets. A local pass on sm_120 (or sm_86 historically)
proves compile and smoke-run viability only. It does **not** validate:

- TCGen05 / TMEM codegen
- TMA descriptor shapes and async-copy semantics
- Cluster behavior and DSMEM (distributed shared memory)
- Register allocation, occupancy, and spill pressure
- L2 partitioning and HBM3e bandwidth (B200: 192 GB / 8 TB/s vs the
  3090 Ti's GDDR6X / the 5060's GDDR7)
- CUTLASS schedule / stage selection
- NVFP4 block-scaling and FP8 e4m3 / e5m2 layout paths
- B200's 5th-gen tensor cores

The B200 problem packager
(`sol-execbench/src/sol_execbench/driver/problem_packager.py`)
auto-injects `-gencode arch=compute_100a,code=sm_100a` for
`target_hardware: ["B200"]`. The `LOCAL` path auto-detects the local
GPU. Local NVBit-producer builds use `make ARCH=sm_86` on the live
3090 Ti host (formerly `make ARCH=sm_120` when the 5060 Laptop was
the active dev surface; the sm_120 cubin path remains supported for
cross-arch verification builds). Build twice when in doubt.

### Vocabulary: host-identity tokens

The framework uses three distinct host-identity tokens; mixing them
up loses information. Pin them with these examples:

- `host.kind` — the value emitted by
  `surfaces/host_probe.py:_classify_host_kind` from the live
  compute capability. Live 3090 Ti host: `local_sm_8_6`
  (digit-underscored — outlier Ampere shape preserved for the
  landed-manifest contract; use the digit-underscored spelling).
  Cross-arch sibling 5060 host (historical, no
  longer attached): `local_sm_120` (glued-digit Blackwell-consumer
  shape; the host.kind token is preserved in the vocabulary
  because the framework still consumes sm_120 evidence artifacts
  cross-arch). B200 datacenter: `b200_sm_100a` (special-cased
  prefix).
- `arch_tag` — the value derived by
  `arch_resolver.arch_tag_from_host_kind` for use in evidence
  labels and manifest filenames. Examples: `sm_120` (5060),
  `sm_100a` (B200), `sm_86` (Ampere). Note that the host.kind /
  arch_tag asymmetry for Ampere (`local_sm_8_6` → `sm_86`) is
  load-bearing — the digit-underscored host.kind survives in
  landed manifests while the arch_tag glues the digits for the
  evidence-label suffix.
- `host_class` — the TOML identifier used by
  `tools/sol_baselines_harness/calibration/cache_prediction/
  host_resource_availability_v1{,_<host_class>}.toml` sibling files.
  The mapping lives in `host_resource_availability.py:60-64`
  (`_HOST_CLASS_TO_MANIFEST_FILENAME`). Examples:
  `local_sm_86_3090_ti_24gib` is the bare
  `host_resource_availability_v1.toml` (no suffix — the sm_86 host is
  the historical baseline); `local_sm_120_rtx_5060_laptop_8gib` is
  the suffixed `host_resource_availability_v1_sm120_rtx_5060_laptop_8gib.toml`.
  Note the Ampere host_class glues the digits (`sm_86`) while the
  host.kind underscores them (`sm_8_6`); the asymmetry is preserved
  for the landed-manifest contract.

### Evidence labels (use in commit messages and PR notes)

The local labels are parameterised by arch — substitute the actual local
arch (`sm_86` on the current RTX 3090 Ti host, or `sm_120` on the
historical RTX 5060 Laptop cross-arch sibling) into the placeholder
`<arch>` below.

- `Compile-checked-<arch>` — local nvcc accepts; no run. Today
  `Compile-checked-sm_86` on the 3090 Ti host; frozen
  `Compile-checked-sm_120` rows from the 5060 Laptop sibling remain
  read-only evidence.
- `Ran-local-<arch>` — local run passes correctness on every workload
  under the native `sol-execbench` evaluator. Local speedup vs the
  PyTorch reference is diagnostic only and is **not** required for
  this label. The framework's five counter-free analytic surfaces may
  or may not have executed yet.
- `Profiled-local-<arch>` — `Ran-local-<arch>` PLUS `manifest.json`
  emitted with **no surface in `failed` state**. Each surface is
  either `ok`, `unavailable`, `blocked`, or `skipped` with a
  `waiver_reason`. A blocked NVBit injection on the local host
  (uncommon) is acceptable and does not demote the label.
- `Profiled-local-complete` — stricter variant: `manifest.json`
  emitted with **all five surfaces in `ok` state**. Useful as a
  sanity check that the local host can in fact produce the full
  artifact set; not a requirement for Phase 2 entry.
- `Profiled-B200-sm_100a` — `manifest.json` emitted on the B200
  provider with no surface in `failed` state. The Phase-2 deliverable
  label. Provider-side `blocked` / `unavailable` surfaces are
  recorded in the manifest with `waiver_reason` and DO NOT demote
  this label — they are the empirical evidence of provider
  constraints that subsequent work has to design around.
- `Profiled-B200-complete` — stricter variant: all five surfaces `ok`
  on the B200 provider. Aspirational; achievability depends on the
  provider's toolchain availability.
- `Scored-B200-sm_100a` — submitted to the official online judge and
  a SOL Score returned. Target reporting band: 0.3 – 0.5+ for
  template solutions, but scores below that are still recorded and
  the pilot still counts as an accepted profiling input.
- `Not-B200-validated` — explicitly flag local-only (sm_86 or sm_120)
  results.

## Key Paths

- `SOL-Execbench.pdf` — benchmark paper.
- `AVO.pdf` — agent-optimizer companion paper (NVIDIA, overlapping
  author set with SOL-ExecBench). Reference context only; not a
  performance bar for this project.
- `sol-execbench/` — official NVIDIA evaluation framework. Working
  examples at `examples/{cutlass,cuda_cpp,cute_dsl,cudnn,cublas,
  cutile,pytorch,triton}/` — `cutlass/gemm`, `cuda_cpp/flux_rope`,
  and `cuda_cpp/rmsnorm` are the canonical seed templates for the
  235 hand-ports; the Python-side examples (`pytorch/`, `triton/`,
  `cute_dsl/`, `cutile/`, `cudnn/`) supply the algorithmic blueprints
  that get re-expressed in CUTLASS C++ / CUDA C++. Schema docs at
  `docs/{definition,solution,workload,trace}.md`. The vendored
  `sol-execbench/` package pins CUTLASS v4.4.1 via its dependency set.
- `SOLAR/` — NVlabs SOL bound deriver. Install via `bash install.sh`.
- `cutlass/` — full CUTLASS source tree (local copy is v4.4.2 per
  `cutlass/include/cutlass/version.h`).
  **CUTLASS 3.8** was the first release with major SM100 / TCGen05 /
  TMEM / NVFP4 support — do NOT pin `CUTLASS_3_7`. Local builds pick
  up CUTLASS via `CUTLASS_DIR=$PWD/cutlass`.
- `references/NVIDIA_Manuals/solbench_problems/` — 235 markdown
  problem specs (`problem_NNN_<slug>.md`), each with shape table,
  reference PyTorch `run()`, per-workload SOL + baseline +
  reference-implementation latencies, and a leaderboard 1st-place
  latency when one exists. The PyTorch `run()` here is the behavioral
  contract a *.cu hand-port must reproduce.
- `references/NVIDIA_Manuals/` — NVIDIA reference doc mirror (CUDA,
  PTX, CUTLASS, CuTe, cuTile, Blackwell, Ampere, NCU metric
  definitions). Crawled / refreshed via
  `references/NVIDIA_Manuals/scripts/` (incl. `sol_execbench_crawler.py`).
- `nvbit_tracer/metric_extractor/` — non-root NVBit injector + metric
  extractor; Surface 1 of the five counter-free analytic surfaces.
  Emits per-warp metric histograms as per-launch JSONL plus an opt-in
  Tier-2 address-trace sidecar (env-gated at
  `metric_extractor.cu:376-403`). Builds for sm_86 locally and
  sm_100a on the B200 provider. The vendored `nvbit_tracer/nvbit_release/`
  (NVBit 1.8 SDK) and `nvbit_tracer/isa_def/` (opcode tables) are
  preserved as shared dependencies. The legacy `nvbit_tracer/tracer_tool/`
  Accel-Sim trace dumper was removed in the Scope-B cleanup (2026-05-16).
- `tools/sol_baselines_harness/` — drives the native `sol-execbench`
  CLI as a host subprocess. Produces `trace.json` per (slug, language)
  and is the layering point for the five additional counter-free
  analytic surfaces.
- `tools/sol_baselines_scaffold/` — emits `solution.json` + parent
  `README.md` boilerplate for each new hand-port; lints the surface
  against the forbidden-pattern catalog.
- `scripts/kb/` — knowledge-base routing toolchain.
- `baselines/` — home for the 235 *.cu template solutions, organized as
  `baselines/{L1,L2,Quant,FIB}/<problem_slug>/{cutlass,cuda_cpp}/`
  (category → problem slug → implementation surface). The four category
  roots (`L1/`, `L2/`, `Quant/`, `FIB/`) are pre-created; per-problem
  subdirectories are created on demand. The problem slug is the full
  markdown stem from `references/NVIDIA_Manuals/solbench_problems/`
  (e.g. `problem_004_gemm_n128_k2048`). A problem may carry either or
  both language variants side-by-side. Each problem has its own
  `README.md` recording reference spec, benchmark `<problem_dir>`,
  variant choice, evidence label, SOL Score (when available),
  speedup-vs-reference (diagnostic), and architecture-coupling caveats.
  Full convention in `baselines/README.md`.
- `IISWC_conference_latex/` — **operator-only black box, see Rule 9.**

## Submission Format (summary; full schema in `sol-execbench/docs/solution.md`)

- `spec.languages` ∈ `["cutlass"]` or `["cuda_cpp"]` (Python DSLs are
  excluded by Rule 5).
- `spec.target_hardware`: **`["B200"]`** for cloud scoring,
  **`["LOCAL"]`** (UPPERCASE) for the local 3090 Ti compile-check, or
  both. Lowercase `"local"` fails Pydantic validation
  (`solution.py:57`).
- `spec.entry_point` = `"file.cu::function_name"` (or `.cpp` for pure
  CUTLASS C++).
- `spec.destination_passing_style: true` (default; outputs as the last
  positional args, written in-place, no return).
- `spec.dependencies`: e.g. `["torch"]`, `["torch", "cutlass"]`. The
  docs list `"CUTLASS_3_7"` (`docs/solution.md:111`) but
  `driver/templates/build_ext.py:30` unconditionally injects
  `CUTLASS_DIR` and ignores dependency-specific header injection. Do
  not rely on this; rely on explicit compile flags below.
- `spec.binding`: defaults to `"torch"` for C++/CUDA; can be omitted.
- `spec.compile_options`: **be explicit in every solution.** Do not
  rely on defaults silently. The B200 packager auto-injects
  `-gencode arch=compute_100a,code=sm_100a` if no arch flag is set;
  `LOCAL` detects the compile machine's GPU.
- `sources`: array of `{path, content}`. Entry-point file must appear.

CUDA C++ entry signature (by-value; PyTorch tensors are reference-
counted, so by-value preserves in-place semantics):

```cpp
void run(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    // launch kernel, write into C in-place — no return
}
```

## Leaderboard Submission

Real SOL Scores come from **manually uploading each solution through
the official SOL-ExecBench online judge**
(<https://research.nvidia.com/benchmarks/sol-execbench>). There is no
programmatic submission API on our side and no CI integration to the
leaderboard. **Do not spend time** wiring up an upload script, polling
the leaderboard for results, scraping submission status, or
reverse-engineering the judge's protocol — that step is human-driven by
design.

Per-baseline flow (Phase 2 deliverable):

1. Write the *.cu hand-port and `solution.json` under
   `baselines/<category>/<slug>/<language>/`.
2. Local correctness check on the local host (sm_86 on the 3090 Ti
   today; sm_120 if running on a future RTX 5060 host) via the native
   `sol-execbench` CLI
   (`sol-execbench <problem_dir> --solution solution.json --json`)
   with `--target_hardware: ["LOCAL"]`. Catches correctness errors
   cheaply; does NOT validate the sm_100a code path. Local
   `speedup_factor < 1.0` is acceptable — it is NOT a landing gate.
3. Run the five counter-free analytic surfaces locally on at least one
   representative *.cu to confirm the framework end-to-end pipeline.
4. On the B200 provider, re-run correctness + the five analytic surfaces
   for every *.cu in `baselines/`. Capture the artifact bundles per
   `(slug, language)`.
5. Operator manually uploads `solution.json` files to the online judge
   UI. The acceptance band for these template solutions is **SOL Score
   0.3 – 0.5+** — we are not chasing the leaderboard top, we are
   harvesting profiling intel.
6. Returned SOL Score, leaderboard latency, and `Scored-B200-sm_100a`
   evidence label get recorded in
   `baselines/<category>/<slug>/README.md`.
