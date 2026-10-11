# `tools/sol_baselines_harness/`

> **SM120 target note (2026-10-06):** the active CC 12.0 live target is the
> NVIDIA GeForce RTX 5090. Historical `5060`
> fields and artifacts remain backward-compatible labels for the older laptop
> lane; they must not be used as a RTX 5090 roofline or card. The dedicated
> cloud capture/replay workspace is `/root/sm120_rtx_5090_capture`.


Drive *.cu template solutions through the official `sol-execbench`
evaluator, emit per-workload report rows, score against the SOL Score
formula (when local arch config is available), aggregate across all
landed pilots, lint the surface, and record the deferred-ledger of
pilots that fail correctness. The harness is also the layering point
for the six counter-free analytic surfaces described in
[`../../AGENTS.md`](../../AGENTS.md) and the retained campaign skill.

> **Current contract.** The project's
> landing contract is **correctness-only** — `speedup_factor` against
> the PyTorch reference is diagnostic, not gating. The legacy
> `deferred_slow` status retired. `report.py` now gates cache replay,
> fresh-run stamping, evidence-label assignment, and category-floor
> membership on `all_passed` only. The `meets_speed_gate` row field
> survives as a diagnostic column in `report.json` / `report.md`.
> SOL Score 0.3 – 0.5+ from the online judge is the reporting band
> for template solutions (not a rejection threshold).

The module is intentionally split so each concern is independently
testable:

| File | Role |
|------|------|
| `run.py` | Drive one baseline through the upstream CLI; emit one report row per workload. |
| `native_dispatch.py` | Build and run the native `sol-execbench` invocation as a host subprocess. |
| `native_env.py` | Compute the ten-component native host-env fingerprint (Python / Torch / nvcc / runtime CUDA / CUTLASS tree / cuDNN / sol-execbench package / NVCC host compiler / C++ compiler / C compiler); raise `DegradedNativeEnvError` when any component reads `"unknown"`. |
| `parse_trace.py` | Read JSONL `Trace` lines from the upstream evaluator via the live Pydantic schema (no schema duplication). |
| `normalise_benchmark.py` | Stage a Pydantic-compatible copy of the downloaded benchmark directory (empty-string optional fields → `None`). |
| `score.py` | SOL Score wrapper. When `SOLAR/configs/arch/5060.yaml` is present AND the slug has a builder in `solar_analysis.py:_REGISTRY`, the canonical `local_sol_score_5060` column is the median of `compute_score(t_k_5060, t_b_5060, t_sol_5060)` across the baseline's workloads, with `t_sol_5060` from SOLAR Stage 4. Falls back to `"unknown_t_sol"` when the YAML is absent OR the slug has no registered builder. |
| `cache.py` | Cache-key derivation (`solution_source_hash` + `sol_execbench_source_hash` + `native_env_hash` + `environment_libs` + `build_mode` + `surface_set`); PASSED-only replay. Strict-equality match on every component; unknown or malformed digest identities are rejected before reuse. |
| `lint.py` | Forbidden-surface scan over landed sources, harness production code, and `deferred.json` hygiene gate. |
| `report.py` | Aggregate JSON + Markdown report across every landed baseline; lint pre-flight gate. |
| `deferred.json` | Ledger of pilots that failed correctness; schema documented inline. |

## Prerequisites

Three things must exist before running the harness:

1. **Benchmark data** (one-time, ~few GB):

   ```sh
   ./tools/sol_execbench/scripts/download_data.sh
   ```

   Subsets land at
   `tools/sol_execbench/data/benchmark/{L1,L2,Quant,FlashInfer-Bench}/<NNN_name>/`.

2. **Native `sol-execbench` install** (one-time):

   ```sh
   uv sync --project tools/sol_execbench
   ```

   The command installs the upstream package and CLI. Its declared CUTLASS
   dependency remains independently versioned; the active harness build path
   overrides header search with the pinned
   `third_party/cutlass/` v4.6.1 tree via `CUTLASS_DIR`.

3. **Host CUDA toolchain** (host-side, one-time):

   The harness drives a host-resident `nvcc` / `cuobjdump` / `nvdisasm`
   / `ptxas` for the static surface and the JIT build path. Driver +
   CUDA toolkit setup is operator-driven by design (CLAUDE.md Rule 4).

The SOLAR local-arch config (`SOLAR/configs/arch/5060.yaml`) is
**optional**:

- **Absent** — every emitted `local_sol_score_5060` cell reads the
  sentinel `"unknown_t_sol"`; report generation still completes.
- **Present** (the current repo state) — `local_sol_score_5060` is a
  numeric median for slugs covered by a builder in
  `solar_analysis.py:_REGISTRY` (one entry per landed pilot today),
  and the sentinel for slugs without a registered builder. The
  diagnostic `upper_bound_score_t_sol_zero` column is always numeric
  whenever the YAML is present, regardless of whether a builder is
  registered.

## Invocation

### Single-baseline run

```sh
python tools/sol_baselines_harness/run.py \
    --baseline baselines/FIB/problem_001_fused_add_rmsnorm_h2048/cuda_cpp
```

Output: JSON with `rows[]` (one entry per workload), `expected_rows`,
`returncode`, `failure_reasons`. Exit codes are triage information:

| Exit | Meaning |
|---|---|
| `0` | Every workload PASSED. |
| `1` | At least one workload not PASSED, but the harness itself ran cleanly (this is the deferred-ledger signal). |
| `2` | Harness/environment broke (upstream CLI non-zero, row-count mismatch, unparseable JSONL, etc.). |

### Aggregate report across every landed baseline

```sh
python tools/sol_baselines_harness/report.py
```

Walks every `baselines/<cat>/<slug>/<lang>/` directory. For each
baseline, the candidate trace is `tools/sol_baselines_harness/.harness_traces/<slug>__<lang>.json`
(the report's own deterministic stamp location). If a caller-
supplied `--trace-dir` is also present, legacy `round-*-trace*.json`
files there are scanned as a fallback. A candidate replays only
when its stored `CacheKey` matches the freshly derived key (see
[Cache-key semantics](#cache-key-semantics)), every row is `PASSED`,
AND every row's `speedup_factor > 1.0`. Otherwise the baseline is
re-run through `run_baseline`, a fresh trace is written to
`tools/sol_baselines_harness/.harness_traces/<slug>__<lang>.json`,
and the cache index is stamped for the next invocation. Outputs:

- `tools/sol_baselines_harness/report.json` — machine-readable.
- `tools/sol_baselines_harness/report.md` — human-readable.

The report is **lint-gated and write-protected**: when the
forbidden-surface scan or `deferred.json` hygiene check fails, the
lint failure is detected BEFORE any output file is written, the
detailed reasons are printed to stderr, and the program exits 2.
No stale `report.json` / `report.md` survives a dirty lint run.

### Dry-run

```sh
python tools/sol_baselines_harness/run.py \
    --baseline baselines/.../cuda_cpp \
    --print-cmd
```

Prints the constructed native `sol-execbench <problem_dir> --solution
... --json` invocation and exits 0 without running it. Useful when
the host toolchain isn't ready yet or the baseline isn't ready for
live execution.

## Per-workload row schema

Each row in the harness output corresponds to one entry in the
problem's `workload.jsonl`:

```json
{
  "slug": "problem_001_fused_add_rmsnorm_h2048",
  "language": "cuda_cpp",
  "workload_index": 0,
  "status": "PASSED",
  "latency_ms": 0.0123,
  "reference_latency_ms": 0.0456,
  "speedup_factor": 3.71,
  "hardware": "5060Laptop",
  "under_contention": false,
  "definition": "001_fused_add_rmsnorm_h2048"
}
```

Possible `status` values (mirrors the upstream `EvaluationStatus` enum):

- `PASSED` — correctness and performance trial completed.
- `INCORRECT_NUMERICAL` — value mismatch beyond the per-workload tolerance.
- `INCORRECT_OUTPUT_SHAPE` / `INCORRECT_OUTPUT_DTYPE` — structural mismatch.
- `COMPILE_ERROR` / `RUNTIME_ERROR` — kernel failed to build / launch.
- `TIMEOUT` — workload exceeded the upstream CLI's 600 s default.
- `REWARD_HACK` — evaluator detected an evaluator-state exploit
  (e.g. `data_ptr`-keyed cache, monkey-patched timer).

`speedup_factor` is `reference_latency_ms / latency_ms`; a pilot lands
when every workload reports `PASSED` AND `speedup_factor > 1.0`.

`under_contention=true` rows come from `--workers > 1` runs; those rows
are accurate for status but NOT for latency (multiple workers contend
on the same GPU). The default `--workers 1` keeps every row
contention-free.

## Aggregate report schema (`report.json`)

```json
{
  "landed_pilots": [
    {
      "category": "FIB",
      "slug": "problem_001_fused_add_rmsnorm_h2048",
      "language": "cuda_cpp",
      "trace": "/<repo>/tools/sol_baselines_harness/.harness_traces/problem_001_fused_add_rmsnorm_h2048__cuda_cpp.json",
      "workload_count": 7,
      "all_passed": true,
      "meets_speed_gate": true,
      "slowest_speedup": 4.2868,
      "local_sol_score_5060": 0.8840,
      "upper_bound_score_t_sol_zero": 0.8687,
      "evidence_label": "Ran-local-sm_120"
    }
  ],
  "deferred_pilots": [
    {
      "slug": "problem_049_group_limited_topk_routing",
      "language": "cuda_cpp",
      "status": "deferred_correctness",
      "evidence": ".humanize/rlcr/<loop>/round-N-pilot-trace.json",
      "harness_commit": "7284281"
    }
  ],
  "category_floor": {
    "required": ["FIB", "L1", "L2", "Quant"],
    "landed": ["FIB", "L1", "L2", "Quant"],
    "complete": true
  },
  "lint": {
    "forbidden_surface_hits": [],
    "deferred_ledger_errors": [],
    "clean": true
  },
  "local_proxy_disclaimer": "local proxy score; computed via ...",
  "local_arch_yaml_present": true
}
```

The example mirrors the current `report.json` shape for the
8-landed-pilot corpus: `meets_speed_gate` sits between `all_passed`
and `slowest_speedup`, the canonical `local_sol_score_5060` is
numeric for slugs covered by a registered SOLAR Stage 4 builder, the
`trace` path is the report's own stamp under `.harness_traces/`, and
the four category-floor slots are landed. When the local-arch YAML
is absent or a slug lacks a registered builder, the canonical column
falls back to `"unknown_t_sol"` and the diagnostic column may still
emit a numeric value (or `"unknown_t_sol"`, when YAML is absent).

`report.md` is the same payload rendered as a human-readable Markdown
table; the local-vs-leaderboard disclaimer is the first non-title
paragraph.

## Cache-key semantics

The aggregate report (`report.py`) maintains a per-trace cache index
at `--cache-file` (default
`tools/sol_baselines_harness/.harness_cache.json`). A candidate trace
is replayed only when ALL of:

1. The trace path appears in the cache index.
2. The stored `CacheKey` for that path matches the freshly derived
   key for the baseline, component-by-component.
3. Every row in the trace's `rows[]` is `PASSED`
   (`is_passed_trace`).

If any of those fails, `report.py` re-runs the baseline through
`run_baseline`, writes a fresh trace to
`tools/sol_baselines_harness/.harness_traces/<slug>__<language>.json`,
and stamps the new key into the cache index before continuing.

The key tuple is:

```
(solution_source_hash, sol_execbench_source_hash,
 native_env_hash, environment_libs, build_mode, surface_set)
```

| Component | Source |
|---|---|
| `solution_source_hash` | SHA-256 over the canonical-order solution content (name + definition + languages + entry_point + binding + dependencies + `compile_options.{cflags, cuda_cflags, ld_flags}` + per-source `(path, content)`). `compile_options` sub-keys are iterated in sorted order, but element order within each list is preserved verbatim because flag order can be semantically significant. Mirrors the upstream `Solution._compute_hash` plus the `compile_options` coverage (a build-flag-only edit must bust the cache). |
| `sol_execbench_source_hash` | `git rev-parse HEAD:tools/sol_execbench` from the repo root — the Git tree hash of the vendored `tools/sol_execbench/` subdirectory at the current HEAD. Changes only when the evaluator source tree changes; a top-level commit that only touches `tools/sol_baselines_harness/` or `baselines/` leaves this hash invariant, so cached PASS traces remain replay-eligible across harness-only commits. Falls back to `"unknown"` outside a git tree (deliberately busts the cache). |
| `native_env_hash` | SHA-256 over a canonical-order JSON of ten host-environment components (`python_version`, `torch_version`, `nvcc_version`, `cuda_runtime_version`, `cutlass_tree_hash`, `cudnn_version`, `sol_execbench_package_version`, `nvcc_ccbin`, `cxx_compiler`, `c_compiler`). Strict identity: any component reading `"unknown"` raises `DegradedNativeEnvError` at derive time so a degraded fingerprint is never silently reused. The raw component dict is persisted in the cache record so debugging can inspect which component differed without re-probing. |
| `environment_libs` | Sorted tuple of `(name, version)` pairs from the upstream evaluation's `environment.libs` field (e.g. `("torch", "2.4.0")`). Extracted from the trace's recorded `environment_libs` field. |
| `build_mode` | `"judge"` or `"profiling"`. Separates the canonical judge build (byte-identical to the landed `solution.json`) from the profiling build (staged solution with `-DSOLBENCH_PROFILING` always and `-DSOLBENCH_HAS_NVTX` conditional on host probes). |
| `surface_set` | Sorted tuple of surface names that participated in the run. A partial-surface run (e.g. `("static",)`) never collides with a full-set run (`("cupti_activity", "nsys", "nvbit_metrics", "sanitizer", "static")`) because the bundle the report consumes differs. |

A source edit busts the cache via the first component; a host
toolchain bump (torch, nvcc, cuDNN) busts it via the third; an
evaluator library bump (captured in the trace's recorded env) busts
it via the fourth; switching judge ↔ profiling or partial ↔ full
surface sets busts it via the fifth / sixth. The SHA-256 of the
JSON-canonicalised tuple is also available (`CacheKey.as_sha256`)
for use cases that want a single string identifier.

**Replay is PASSED-only.** A non-PASSED row in the cached trace
forces a re-run so the deferred-ledger writer sees fresh evidence;
silent replay of a deferred row would let stale evidence rot.

## Deferred ledger schema (`deferred.json`)

A landed `solution.json` belongs under `baselines/`. A pilot that
fails one or more workloads belongs in
`tools/sol_baselines_harness/deferred.json` instead.

```json
{
  "schema": {
    "version": 3,
    "fields": {
      "slug": "problem_NNN_<slug>",
      "language": "cuda_cpp | cutlass",
      "status": "deferred_correctness | deferred_reward_hack",
      "justification": "<sentence(s) referencing the failing workload(s)>",
      "evidence": "<trace.json path or named round artifact>",
      "harness_commit": "<7-40 hex char git SHA>"
    },
    "valid_statuses": [
      "deferred_correctness",
      "deferred_reward_hack"
    ],
    "retired_statuses": [
      "deferred_slow"
    ]
  },
  "entries": [ "..." ]
}
```

Status semantics:

| Status | Meaning |
|---|---|
| `deferred_correctness` | At least one workload returned a non-`PASSED` status other than `REWARD_HACK` (e.g. `INCORRECT_NUMERICAL`, `INCORRECT_OUTPUT_SHAPE`, `COMPILE_ERROR`, `RUNTIME_ERROR`, `TIMEOUT`). |
| `deferred_reward_hack` | Evaluator detected an evaluator-state exploit. Should be empty here — we don't ship those. |
| `deferred_slow` | **Retired.** Slow-but-correct kernels now land under `baselines/` per the correctness-only contract (re-baseline 2026-05-13). |

Lint enforces:

- `harness_commit` must be a real git SHA (7-40 hex chars).
  Placeholders like `"round-2-pending"` are rejected.
- `status` must be one of the declared `valid_statuses`.
- All six fields above must be present on every entry.

## Evidence labels

Per CLAUDE.md, every parent `baselines/<cat>/<slug>/README.md` carries
one of:

| Label | Meaning |
|---|---|
| `Compile-checked-sm_120` | Local `nvcc` accepts; no run. |
| `Ran-local-sm_120` | Local 5060 run PASSED correctness on every workload AND every workload's `speedup_factor > 1.0`. The harness row's `meets_speed_gate` field is the canonical signal — a PASSED-but-slow run keeps the `Compile-checked-sm_120` label. |
| `Scored-B200-sm_100a` | Submitted to the leaderboard and a SOL Score returned. **Not produced by this harness this round** — leaderboard upload is operator-driven; see CLAUDE.md `## Leaderboard Submission`. |
| `Not-B200-validated` | Explicit flag for sm_120-only artefacts that cannot validate B200 codegen (TCGen05 / TMEM / TMA / cluster behaviour / NVFP4 / FP8 layout / etc.). |

The aggregate report emits `Ran-local-sm_120` only when both
`all_passed` (correctness) AND the per-row speed gate
(`speedup_factor > 1.0` on every workload) hold, and
`Compile-checked-sm_120` otherwise. `all_passed` and
`meets_speed_gate` ship as separate diagnostic fields on every
report row so a PASSED-but-slow run is visible in the trace
without being mis-labeled as a landed pilot.

## Subset-name mapping

Internal category → downloader subset name (mirrors
`baselines_scaffold/generate.py`):

| Internal | Downloader |
|---|---|
| `L1` | `L1` |
| `L2` | `L2` |
| `Quant` | `Quant` |
| `FIB` | `FlashInfer-Bench` |

The harness's `_resolve_problem_dir` consumes this map: a baseline at
`baselines/FIB/problem_001_fused_add_rmsnorm_h2048/cuda_cpp/` resolves
its `solution.json.definition` field against
`tools/sol_execbench/data/benchmark/FlashInfer-Bench/001_fused_add_rmsnorm_h2048/`.

## Local-vs-leaderboard disclaimer

Three distinct authorities define a kernel's measurable performance —
keep them straight:

1. **Public leaderboard / scoring service** owns canonical `Tk` for
   the SOL Score; runs the paper §5.1 methodology on B200 @ 1.5 GHz,
   and owns clock policy.
2. **Paper §5.1** describes CUDA events, 10 warmup + 50 timed × 3
   trials, 256 MB L2 clear, subprocess isolation, 300 s timeout,
   clocks locked at 1.5 GHz via `nvidia-smi`. This is the published
   description.
3. **Local `tools/sol_execbench/` checkout** (this harness) has CLI
   `--timeout` defaulting to 600 s; `core/bench/timing.py` defaults
   to a CUPTI-based timer; L2-cache clear sizes itself off the actual
   device's L2. Use for local sanity-check timing — **never quote its
   number as "the SOL Score."**

The aggregate report emits two distinct score columns:

- `local_sol_score_5060` — the canonical local SOL Score column. The
  per-workload SOLAR Stage 4 derivation is wired via
  `tools/sol_baselines_harness/solar_analysis.py:_REGISTRY` (one builder
  per landed slug; populated across Rounds 8 → 18 for the current
  8-pilot corpus). For each row in a registered baseline, `t_sol_5060`
  comes from `solar_analysis.t_sol_for_baseline_workload()`, which
  drives `SOLAR/solar/perf/perf_model.py:EinsumGraphPerfModel.predict`
  against a per-workload analysis derived from the benchmark
  definition. The column is the median of
  `compute_score(t_k_5060, t_b_5060, t_sol_5060)` across rows.
  Honestly falls back to `"unknown_t_sol"` when the local-arch YAML is
  absent OR the slug has no registered builder (or when any single row
  is missing a Stage 4 estimate — partial coverage never mixes with
  the sentinel in the same row's score).
- `upper_bound_score_t_sol_zero` — a `t_sol = 0.0` diagnostic
  baseline emitted as a numeric median whenever the local-arch YAML
  is present. Computed as the median of `compute_score(t_k, t_b, 0.0)`
  per row, which collapses the upstream formula to `Tb / (Tk + Tb)`.
  For `Tk < Tb` (the speedup case) the value is strictly > 0.5 and
  trends toward 1.0 as speedup grows. Useful for sanity-checking the
  speedup signal but **NOT a SOL Score**.

  **Field-name caveat**: the name predates the discovery that the
  SOL Score formula is monotonic-INCREASING in
  `t_sol` for speedup rows. The partial derivative is
  `dS/dT_SOL = (T_b - T_k) / v²` where
  `v = (T_k - T_SOL) + (T_b - T_SOL)`; for `T_k < T_b` this is
  positive, so a real SOLAR-derived `t_sol > 0` produces a *larger*
  score than the `t_sol = 0` substitution. Therefore this value is
  a **lower** bound on the canonical column for speedup rows, NOT an
  upper bound. Every landed pilot in the committed `report.json`
  confirms the relationship empirically — e.g.
  `problem_001_fused_add_rmsnorm_h2048` has `local_sol_score_5060 =
  0.8840` vs `upper_bound_score_t_sol_zero = 0.8687`. The JSON key
  is retained for backward compatibility (cache index records,
  downstream consumers) and is queued for a future rename round.

The two-column split is intentional: the canonical column is
guaranteed to never carry a placeholder number under a SOLAR-derived
label, while the diagnostic column makes the speedup signal
visible in a clearly-named field. Even when the SOLAR config is
present, the canonical local SOL Score is **NOT comparable to the
leaderboard SOL Score**:

- Local/cloud `t_k` for the active SM120 route is taken on the RTX 5090, not B200 (sm_100a); retained 5060 rows remain historical.
- Local `t_b` is our PyTorch reference, not the benchmark-internal
  scoring baseline `T_b` (which is a paper-specific calibration we
  don't have).
- Local `t_sol` would be the 5060 roofline, not the B200 roofline.

The Markdown report's first paragraph carries this disclaimer
verbatim. When the local SOLAR config is absent, both score columns
fall back to the sentinel `"unknown_t_sol"`.

## Architecture-portability discipline

The active RTX 5090 is **sm_120**; B200 scoring is **sm_100a**. Historical 5060 artifacts retain their original labels. A
local `PASSED` on sm_120 proves compile and smoke-run viability
only — it does NOT validate TCGen05 / TMEM codegen, TMA descriptor
shapes, cluster + DSMEM behaviour, register pressure, L2 partitioning
+ HBM3e bandwidth, CUTLASS schedule selection, NVFP4 / FP8 layout
paths, or B200's 5th-gen tensor cores. Every landed
`baselines/<cat>/<slug>/README.md` carries an `Architecture-coupling
caveats` section that calls out the specific surfaces requiring
re-validation on B200.

## Fail-closed semantics

`run.py` exits non-zero on any of:

- Upstream `sol-execbench` CLI exited non-zero.
- Parsed row count ≠ `workload.jsonl` line count (under-run or over-run).
- Zero parsed rows when at least one workload was expected.
- Unrecoverable JSONL parse errors and zero validated `Trace` rows.

Non-`PASSED` rows themselves are **not** treated as `run.py` harness
failure — they are the harness's primary signal. The exit-1 path
communicates "at least one workload not PASSED, harness itself ran
fine"; the deferred-ledger writer is the right place to record that.
The harness's own gates (exit 2) signal environment problems, not
kernel problems.

`report.py` adds two report-level gates:

- **Lint write-protection.** When the forbidden-surface scan finds
  any hit OR `deferred.json` has any hygiene error, the lint failure
  is detected BEFORE `report.json` / `report.md` are written and the
  program exits 2. Existing on-disk report files are NOT overwritten
  by a dirty run.
- **Cache-keyed replay.** A trace under `--trace-dir` is replayed
  only when its stored `CacheKey` exactly matches the current key
  for the baseline (strict equality on every component; no asymmetric
  fallback) AND every row is `PASSED`. Source edits, host toolchain
  bumps (torch / nvcc / cuDNN / runtime CUDA / CUTLASS tree),
  `tools/sol_execbench/` subdirectory tree-hash changes, and evaluator
  library bumps all bust the cache deterministically — no stale PASS
  can be laundered through a slug-substring trace match.

## Forbidden-surface lint coverage

The lint scan in `lint.py` re-uses the scaffold validator's pattern
set (`baselines_scaffold.validate._FORBIDDEN_PATTERNS`) so the
landed-source surface and the scaffold-time surface stay in lock-step.
Covered patterns (each tagged with the CLAUDE.md rule they enforce):

| Pattern | Rule |
|---|---|
| `ncu` / `nv-nsight-cu-cli` invocation | Rule 1 (no NCU) |
| `nvidia-smi --lock-gpu-clocks` / `--lock-memory-clocks` / `--lock-clocks` | Rule 2 (no clock mutation) |
| `cudaStreamCreate{,WithFlags,WithPriority}` | Rule 7 (default-stream discipline) |
| `torch.jit.fork` | Rule 7 |
| `cache.*data_ptr(` (case-insensitive) | Rule 6 (no evaluator-state exploits) |
| `Event.elapsed_time =` (monkey-patch) | Rule 6 |
| `do_bench = lambda` (monkey-patch) | Rule 6 |
| `cudaGraphInstantiate` / `torch.cuda.CUDAGraph` | Rule 7 (requires co-located per-problem note) |

CUDA-Graph hits are forgiven when a `stream-injection` or `CUDA Graphs … note` comment is co-located in the same file (the same escape hatch the scaffold validator uses).
