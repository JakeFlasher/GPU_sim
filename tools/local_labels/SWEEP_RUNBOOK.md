# Sweep Runbook (`sweep_repeatability_matrix.py`)

This runbook documents the operator workflow for running the
parameterized NCU repeatability sweep harness on local hardware
(RTX 3090 Ti sm_86, RTX 5060 laptop sm_120) and admitting cells into
`data/local_labels/label_admitted.json`.

The harness wraps `tools/local_labels/label_driver.sh` in nested
loops over the matrix axes (`warmup_count` × `cache_control` ×
`repeats`) for an operator-chosen candidate cell set, aggregates the
per-cell sidecars via `tools/local_labels/repeatability_aggregator.py`,
and emits a verdict-table (CSV + JSON) plus a terminal verdict
(`admitting_config_found` | `no_admitting_config_found`).

The runbook itself is exercised by
`tools/local_labels/test_sweep_runbook.py` per
`BL-20260504-runbook-command-verification`. A regression that breaks
the documented command shape lights up the integration test.

## Source of Truth (read this BEFORE Step 1)

This runbook follows the hybrid source-of-truth +
verification-target pattern documented in
`BL-20260504-runbook-self-reference`
(see `docs/lessons.md`):

- **Source of truth**: the runbook procedure is read from the
  working branch's HEAD (`5060_dev` HEAD), NOT from a pinned
  target SHA. The branch HEAD is the authoritative procedure.
- **Verification target**: this runbook has NO separately
  pinned verification-target SHA. The verification surface is
  the sweep harness `tools/local_labels/sweep_repeatability_matrix.py`
  and the per-cell label driver `tools/local_labels/label_driver.sh`
  — both stable across the loop session, so any commit on
  `5060_dev` containing both surfaces is a valid verification
  target.

You can re-read this runbook from branch HEAD without changing
your worktree:

```bash
git show 5060_dev:tools/local_labels/SWEEP_RUNBOOK.md | less
```

The reason for the no-source-of-truth-pinning approach:
pinning a target SHA in a runbook that lives at that same SHA
produces a self-referential staleness trap. The BitLesson cited
above records the trap and the fix.

## Critical Rules (binding)

- All NCU invocation MUST flow through `tools/local_labels/label_driver.sh`,
  which writes only into `data/local_labels/`. Per CLAUDE.md
  Critical Rule 1, the local-label exception bounds NCU artifacts to
  this quarantine.
- The harness's only output prefix is `data/local_labels/sweeps/`.
  An `--out-root` outside this sub-prefix is rejected at parse time.
- The `cache_control` axis maps to NCU's read-only
  `--cache-control all|none` flag. NEVER use `nvidia-smi` to lock
  clocks, change persistence, power, MIG, or compute mode (CLAUDE.md
  Critical Rule 2).
- The `stddev_threshold = 0.03` admission gate in
  `repeatability_aggregator.py` MUST NOT be loosened. The
  source-text scanner asserts the literal `0.03` survives.

## Command syntax

The harness exposes two mutually-exclusive modes (`--dry-run` /
`--live`) and a fixed set of axis flags. The full CLI surface:

```
python3 tools/local_labels/sweep_repeatability_matrix.py \
    {--dry-run | --live} \
    --candidates <ARCH>/<SLUG>/<SHAPE_POINT>[,...] \
    --binaries <ARCH>/<SLUG>/<SHAPE_POINT>=<PATH>[,...] \
    --measurement-manifests <ARCH>/<SLUG>/<SHAPE_POINT>=<PATH>[,...] \
    --predictor-trace-jsons <ARCH>/<SLUG>/<SHAPE_POINT>=<PATH>[,...] \
    --warmup-counts <N>[,<N>...] \
    --cache-controls all,none \
    --repeats <N> \
    --seed <N> \
    [--out-root data/local_labels/sweeps/<run-id>] \
    [--ncu /usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu] \
    [--driver tools/local_labels/label_driver.sh] \
    [--aggregator tools/local_labels/repeatability_aggregator.py]
```

`--measurement-manifests` AND `--predictor-trace-jsons` are
both REQUIRED in `--live` mode and each must cover every
candidate cell named in `--candidates`. The harness forwards
each cell's manifest path to its per-launch `label_driver.sh`
invocation as `--measurement-manifest <path>`, where the driver
reads the manifest's `kernel_signature_sha256` as the operator-
pinned source-of-truth (failing closed before NCU on any
unavailability or mismatch). The harness ALSO runs the
pre-flight provenance audit
(`tools/local_labels/provenance_audit.py`) per cell using the
cell's `--predictor-trace-jsons` entry; any audit failure
aborts the sweep with exit 2 BEFORE any NCU shim is spawned.
`--dry-run` mode ignores the predictor-trace-jsons flag.

### Per-flag reference

- `--dry-run`: Print the planned per-config invocations to stdout
  without invoking `label_driver.sh` or NCU. Use this to audit the
  matrix BEFORE committing wall-clock budget.
- `--live`: Execute the matrix end-to-end. Per (cell, config, repeat)
  the harness invokes `label_driver.sh`, which invokes NCU.
- `--candidates`: Comma-separated list of candidate cells, each named
  as `<arch>/<slug>/<shape_point>`.
- `--binaries`: Comma-separated `<cell>=<path>` map. Every cell named
  in `--candidates` must have a binary path. Use the binary at
  `workloads/simbench_bw/results/<slug>/<sku>/sm_<N>/<binary>` from
  the simbench_bw build tree, or your own per-cell binary.
- `--measurement-manifests`: Comma-separated `<cell>=<path>` map of
  operator-supplied measurement manifests, one per candidate cell.
  REQUIRED in `--live` mode; the harness rejects a live invocation
  whose manifest map does not cover every candidate
  (`sweep_repeatability_matrix: --measurement-manifests is REQUIRED
  in --live mode (operator-pinned kernel_signature_sha256 source-of-truth)`).
  Each manifest is a JSON file carrying the
  operator-pinned `kernel_signature_sha256` (and optionally
  `binary_sha256`). The harness forwards the path to the per-launch
  `label_driver.sh` as `--measurement-manifest <path>`; the driver
  reads the manifest BEFORE NCU collection and fails closed on any
  unavailability, malformed JSON, missing
  `kernel_signature_sha256`, or `binary_sha256` mismatch with the
  profiled binary. See "Cross-Host sm_120 Pivot Workflow (AC-5.2)"
  below for an example manifest body.
- `--predictor-trace-jsons`: Comma-separated `<cell>=<path>` map of
  predictor-side per-shape trace JSON sidecars, one per candidate
  cell. REQUIRED in `--live` mode (mirrors `--measurement-manifests`).
  The harness runs `tools/local_labels/provenance_audit.py` per cell
  BEFORE any NCU shim is spawned; the audit cross-checks the seven-
  field comparison-runner join key (slug, shape_point, binary_sha256,
  source_commit, kernel_signature_sha256, arch, ncu_metric_version)
  between the predictor JSON and the operator-staged measurement
  manifest. Any non-PASS verdict (or partial coverage of candidate
  cells) aborts the sweep with exit 2 BEFORE NCU runs. The
  predictor-side trace JSON sidecars live at
  `workloads/simbench_bw/results/<slug>/<sku>/<arch>/trace_<shape_point>.json`
  for microbench cells and the matching path under
  `workloads/simbench_bw/results/<slug>_sm80/...` for activation
  cells.
- `--warmup-counts`: Comma-separated warmup counts. Recommended
  starting set: `0,1,5`.
- `--cache-controls`: Comma-separated NCU `--cache-control` modes.
  Allowed values: `all`, `none`. Recommended starting set:
  `all,none` to compare the flush-between-replays behavior against
  the cache-retention behavior.
- `--repeats`: Number of repeats per (cell, config) tuple. The
  contract requires N >= 5 for admission; the recommended starting
  value is `5`.
- `--seed`: Deterministic ordering seed. Two invocations with the
  same `--seed` and the same matrix produce a byte-identical
  verdict-table CSV/JSON.
- `--out-root`: Where the harness writes its output sub-tree. Must
  be under `data/local_labels/sweeps/`; defaults to that prefix.
- `--ncu`, `--driver`, `--aggregator`: Path overrides for the
  Nsight Compute CLI binary, the label driver, and the repeatability
  aggregator. The defaults resolve to the binaries on PATH and the
  in-tree scripts.

## Artifact paths

Every sweep run lands a self-contained sub-tree under
`data/local_labels/sweeps/`:

```
data/local_labels/sweeps/sweep_<UTC-timestamp>_<digest>/
    verdict-table.csv          # one row per (cell, warmup, cache_control)
    verdict-table.json         # same rows as a list of dicts
    verdict.json               # terminal verdict + admitting config
    cell=<arch>__<slug>__<shape>/
        cfg=warmup<N>_cache_<mode>/
            <arch>/<slug>/<shape>/
                label_<repeat>_<launch>.json            # per-launch sidecar
                label_run_manifest_<repeat>_<launch>.json
                run_<repeat>_<launch>.ncu-rep
            _aggregator_out/
                label_repeatability.json
                label_admitted.json
                label_excluded.json
```

The top-level `verdict.json` carries:

```
{
  "verdict": "admitting_config_found" | "no_admitting_config_found",
  "config": {"warmup_count": <N>, "cache_control": "<mode>", "repeats": <N>}
            | null,
  "evidence": "<one-sentence rationale referencing the stddev threshold>"
}
```

## Dry-run interpretation

Run the harness with `--dry-run` first. Stdout enumerates every
planned `label_driver.sh` invocation in the form:

```
bash tools/local_labels/label_driver.sh --device sm_86 \
    --slug <slug> --shape-point <shape> \
    --repeat <repeat> --launch 0 --binary <path> \
    --warmup <N> --cache-control <mode> ...
```

Use the dry-run output to verify:

- Every (cell × warmup_count × cache_control × repeat) tuple appears
  exactly once.
- Every cell has a `--binary` path the operator expects.
- The `--cache-control` values match the operator's intended axis
  set (`all`, `none`, or both).

A missing `--binaries` entry surfaces as a `# missing --binaries
entry for ... ; skipping` comment in the dry-run output. Any cell
with that comment will be excluded by the live run; fix the
`--binaries` map before committing wall-clock budget.

## Required evidence capture

After a live sweep, the operator MUST commit the following artifacts
per `BL-20260508-post-commit-recapture-discipline`:

- `data/local_labels/sweeps/sweep_<timestamp>_<digest>/verdict-table.csv`
- `data/local_labels/sweeps/sweep_<timestamp>_<digest>/verdict-table.json`
- `data/local_labels/sweeps/sweep_<timestamp>_<digest>/verdict.json`
- The per-launch sidecars and `label_run_manifest_*.json` files under
  `cell=.../cfg=.../<arch>/<slug>/<shape>/` for the admitting config
  (the config the operator selects for label admission).
- The aggregator's per-config `_aggregator_out/label_admitted.json`
  for the admitting config.
- The merged `data/local_labels/label_admitted.json` produced by
  re-running the aggregator with the operator's chosen
  per-config sidecar set as input.

The round summary records:

- The exact NCU command shape used for each admitted cell (paste
  one verbatim sample from the `verdict-table.csv` row's
  cache_control / warmup / repeats).
- The git SHA of the binary the cell ran against (from the
  `label_run_manifest_*.json` files).
- The wall-clock budget consumed (rough timing across the matrix).
- Per-arch architecture-coupling discipline metadata: device name
  (`NVIDIA GeForce RTX 3090 Ti` or the 5060 laptop's name), driver
  version, CUDA toolkit version, observed clocks / throttle state.

## Operator workflow

> **Pointer to the canonical cohort path:** the **Cache-Predictor
> First-Geomean Cohort: Full Sweep Matrix** section below pins
> the canonical 13-cell + full-matrix invocation that the
> first-headline-geomean round depends on. Treat this generic
> operator workflow as the rehearsal/ad-hoc shape for sm_86 cells
> outside the cohort; for the cohort itself, follow the cohort
> section verbatim.

1. Pick a candidate cell set (4–6 cells per arch for ad-hoc
   rehearsal; the cache-predictor cohort uses 13 cells — see the
   cohort section below). Recommended for rehearsal: include the
   cells in `data/local_labels/label_excluded.json` from prior
   runs so a stable config admits them.
2. Locate the binaries: under
   `workloads/simbench_bw/results/<slug>/<sku>/sm_<N>/<binary>`.
3. Dry-run the matrix:
   ```
   python3 tools/local_labels/sweep_repeatability_matrix.py \
       --dry-run \
       --candidates sm_86/<slug>/<shape>[,...] \
       --binaries sm_86/<slug>/<shape>=<path>[,...] \
       --warmup-counts 0,1,5 \
       --cache-controls all,none \
       --repeats 5 \
       --seed 0
   ```
4. Audit the dry-run output. Confirm every cell-config tuple is
   present and every binary path is valid.
5. Stage one operator-supplied measurement-manifest JSON per
   candidate cell AND name the matching predictor-side per-shape
   trace JSON sidecar for the `--predictor-trace-jsons` map.
   Each manifest carries the pinned `kernel_signature_sha256`
   (and optionally `binary_sha256`); the predictor JSON is the
   trace-time sidecar at
   `workloads/simbench_bw/results/<slug>/<sku>/<arch>/trace_<shape_point>.json`.
   See the "Cross-Host sm_120 Pivot Workflow (AC-5.2)" section
   below for the manifest body format; the same shape applies on
   the 3090 Ti for sm_86 cells.
6. Run the matrix live (note that BOTH
   `--measurement-manifests` AND `--predictor-trace-jsons` are
   REQUIRED — the harness rejects `--live` without either flag,
   and runs the pre-flight provenance audit per cell BEFORE any
   NCU shim is spawned):
   ```
   python3 tools/local_labels/sweep_repeatability_matrix.py \
       --live \
       --candidates sm_86/<slug>/<shape>[,...] \
       --binaries sm_86/<slug>/<shape>=<path>[,...] \
       --measurement-manifests sm_86/<slug>/<shape>=<manifest_path>[,...] \
       --predictor-trace-jsons sm_86/<slug>/<shape>=<predictor_json_path>[,...] \
       --warmup-counts 0,1,5 \
       --cache-controls all,none \
       --repeats 5 \
       --seed 0 \
       --ncu /usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu
   ```
7. Read `verdict.json`. If
   `verdict == "admitting_config_found"`, the named `config`
   admits at least one cell from the candidate set under the
   `stddev_threshold = 0.03` gate. Confirm the chosen config
   admits enough cells (≥5 per arch for the contract's smoke-test
   floor) by reading the corresponding rows of `verdict-table.csv`.
8. If
   `verdict == "no_admitting_config_found"`, expand the candidate
   cell set or revisit the matrix axes; the round cannot close
   AC-5 closure on tooling alone.
9. Re-run the aggregator on the chosen config's per-config sidecars
   to populate `data/local_labels/label_admitted.json`:
   ```
   python3 tools/local_labels/repeatability_aggregator.py \
       --root data/local_labels/sweeps/sweep_<timestamp>_<digest>/cell=.../cfg=warmup<N>_cache_<mode>/ \
       --out data/local_labels/
   ```
10. Commit the artifacts named in "Required evidence capture" above.
11. Record the round summary's evidence-status declaration: per
    CLAUDE.md "Two-Device Local Development Model", a sweep on the
    receiving device admits to `Verified-live-sm_86` (3090 Ti) or
    `Verified-live-sm_120` (5060 laptop). `Cross-arch-verified`
    requires both arches' live evidence and is one-arch-at-a-time.

## Cache-Predictor First-Geomean Cohort: Full Sweep Matrix

This section pins the canonical sweep shape for the
cache-predictor first-headline-geomean cohort on the local
RTX 3090 Ti (sm_86). The cohort is the **13 cells** that close
the predictor↔label join end-to-end:

  - 8 memory-microbench cells, all at `shape_point = trace_00`:
    `working_set_capacity_sweep`, `l2_thrash_pointer_chase`,
    `dram_stream_stride_mix`, `carveout_sensitivity_probe`,
    `warm_cold_pair_relaunch`, `l2_persisting_window_sweep`,
    `l2_prefetch_hint_sweep`, `register_spill_local_mem`.
  - 5 activation cells:
    `layernorm_fwd_bf16_sm80/timing_shape`,
    `rmsnorm_fwd_bf16_sm80/timing_shape`,
    `silu_fwd_bf16_sm80/correctness_shape`,
    `silu_fwd_bf16_sm80/timing_shape`,
    `softmax_safe_bf16_sm80/timing_shape`.

### Canonical sweep matrix (full)

The cohort sweep uses the FULL matrix without exception:

```
warmup_count   in {0, 1, 5, 10}    (4 values)
cache_control  in {all, none}      (2 values)
repeats        = 5
```

That is 4 × 2 × 5 = 40 NCU invocations per cell. Across all 13
cells the matrix runs **520 NCU invocations** total. At
~30 s/invocation including warmup and aggregation overhead, the
operator wall-clock budget is **~4.3 hrs** for a single host
session. Do not narrow the matrix below this shape: the cohort
characterizes the predictor across both warmup-sensitive and
cache-control-sensitive axes, and a narrower matrix risks
admission false-negatives on otherwise-stable cells. (If
operator wall-clock is genuinely scarce, queue the sweep for a
session that can dedicate the full budget; do NOT trim the
matrix in flight.)

### Mandatory pre-flight provenance audit

The provenance audit at
`tools/local_labels/provenance_audit.py` is REQUIRED before the
live sweep launches. The audit cross-checks the seven join-key
fields between the predictor-side per-shape trace JSON sidecar
and the operator-staged measurement manifest (slug, shape_point,
binary_sha256, source_commit, kernel_signature_sha256, arch,
ncu_metric_version). The harness wires the audit into its
`--live` preflight gate via the `--predictor-trace-jsons` flag;
omitting the flag in `--live` mode is rejected with exit 2
BEFORE any NCU shim is spawned.

### Canonical full-matrix invocation

Stage one operator-supplied measurement-manifest JSON per
candidate cell AND name the corresponding predictor-side per-
shape trace JSON sidecar in the `--predictor-trace-jsons` map.
Then invoke:

```
python3 tools/local_labels/sweep_repeatability_matrix.py \
    --live \
    --candidates \
        sm_86/working_set_capacity_sweep/trace_00,\
        sm_86/l2_thrash_pointer_chase/trace_00,\
        sm_86/dram_stream_stride_mix/trace_00,\
        sm_86/carveout_sensitivity_probe/trace_00,\
        sm_86/warm_cold_pair_relaunch/trace_00,\
        sm_86/l2_persisting_window_sweep/trace_00,\
        sm_86/l2_prefetch_hint_sweep/trace_00,\
        sm_86/register_spill_local_mem/trace_00,\
        sm_86/layernorm_fwd_bf16_sm80/timing_shape,\
        sm_86/rmsnorm_fwd_bf16_sm80/timing_shape,\
        sm_86/silu_fwd_bf16_sm80/correctness_shape,\
        sm_86/silu_fwd_bf16_sm80/timing_shape,\
        sm_86/softmax_safe_bf16_sm80/timing_shape \
    --binaries <one cell=path entry per candidate> \
    --measurement-manifests <one cell=path entry per candidate> \
    --predictor-trace-jsons <one cell=path entry per candidate> \
    --warmup-counts 0,1,5,10 \
    --cache-controls all,none \
    --repeats 5 \
    --seed 0 \
    --ncu /usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu
```

`--measurement-manifests` and `--predictor-trace-jsons` MUST
each cover every one of the 13 candidate cells. The harness
fails closed with exit 2 if either map is missing or partial.

`stddev_threshold = 0.03` is unchanged; do NOT loosen
admission. If fewer than 5 cells admit despite the full 520-
invocation matrix, the round closes at the documented negative-
evidence path: the comparison runner emits
`result_kind = "typed_unavailable"` with
`typed_reason = "headline_geomean_admitted_cell_count_below_threshold"`.
Cite that verdict honestly in the round summary rather than
loosening the gate or retrying with a wider candidate set.

### Pre-flight invocation order (binding)

1. Stage all 13 measurement manifests + 13 predictor-trace JSONs
   on disk.
2. Run `tools/local_labels/validate_measurement_manifest.py`
   over every staged manifest (catches schema drift before
   commit).
3. Run
   `tools/local_labels/provenance_audit.py --predictor-trace-json
   <path> --manifest <path> --arch sm_86` for each cell
   (catches join-key drift before NCU launch). The harness's
   `--live` invocation runs this audit again as a final gate;
   running it manually first surfaces drift instantly without
   queueing the harness.
4. Only after both validators print PASS for all 13 cells, run
   the canonical full-matrix invocation above.

## Cross-Host sm_120 Pivot Workflow (AC-5.2)

This section is for the operator pivoting from the 3090 Ti (sm_86)
host BACK to the RTX 5060 laptop (sm_120) host to close AC-5.2.
The current/pre-pivot host is the RTX 3090 Ti (compute capability
`8.6`, sm_86); sm_120 binaries cannot execute there. The receiving
host is the RTX 5060 laptop (expected compute capability `12.0`,
sm_120); the live sm_120 sweep runs on that host. CLAUDE.md
"Two-Device Local Development Model" is the binding architectural
authority:

> "Codex review is one-arch-at-a-time. A round verified live on
> one arch is NOT verified on the other."

### Pre-pivot state (must be true on the 3090 Ti before transfer)

- AC-5.1 closed: `data/local_labels/label_admitted.json` carries
  ≥5 admitted cells with `arch == "sm_86"`. Verify via the bounded
  operator self-check tool:

  ```
  python3 tools/local_labels/check_admission_floor.py --arch sm_86 --floor 5
  ```

  Expected output:

  ```
  sm_86=5/5 (admitted)
  ```
- All seven Round 1 + Round 2 corrective fixes are committed.
- Architectural-impossibility probe is committed at
  `data/local_labels/sweeps/clock_throttle_evidence/host_arch_probe_*.json`.

### Cross-host transfer

Per CLAUDE.md "Session Handoff" + project `HANDOFF.md`:

- Either `git push` then `git pull` on the 5060 laptop (preferred:
  preserves immutable history; `.humanize/` and other gitignored
  loop state will be missing on the receiving end), OR
- raw `scp -r` (preserves `.humanize/` loop state and built
  binaries; raw transfer carries stale host artifacts that MUST
  be rebuilt).

Tracked artifacts that survive both transports: the 5 sm_86
admitted cells in `data/local_labels/label_admitted.json`, all
the schemas, runbooks, and tooling. Untracked artifacts that
survive only raw scp: the per-launch sidecars under
`data/local_labels/sm_86/.../label_*_*.json` (kept as historical
sm_86 evidence; not regenerated on the 5060). Built binaries are
NOT portable — rebuild on the receiving host.

### Post-transfer verification (run on the RTX 5060 laptop)

```
nvidia-smi --query-gpu=compute_cap,name --format=csv,noheader
```

The expected output names the 5060 laptop's compute capability
12.0 token (the 5060 laptop reports `12.0, NVIDIA GeForce RTX 5060
Laptop GPU` or equivalent). If the output reports compute capability
8.6, the pivot has not happened — recheck the host.

### Host-transition artifact handling while Task7/H3 is open

Per CLAUDE.md "Session Handoff" + `HANDOFF.md`'s "Verify CUDA /
NVBit / NVML environment + rebuild any built artifacts" section:

- Per-slug binaries under `workloads/simbench_bw/l1/<slug>/<slug>`
  and `workloads/portable_lcd/<slug>/<slug>_run`.
- The NVBit tracer (`tools/nvbit_tracer/tracer_tool/tracer_tool.so`) is a
  preserved artifact. Verify or byte-exactly restore its pinned bytes; do not
  rebuild, clean, replace, or repin it in the canonical checkout.
- Provision an absent exact protobuf prefix only with
  `bash tools/nvbit_tracer/pinned_build_env.sh --ensure`, then source
  `tools/nvbit_tracer/pinned_build_env.sh` for verify/export-only behavior.
  If a build feasibility check is required, use a non-Git temporary copy with
  `NVBIT_TEMPCOPY_BUILD_OK=1`; never use that sentinel in a Git worktree.

### sm_120 candidate cell set

Recommended starting set (4-6 cells, including the prior-failed
cell from M1's loop):

- `sm_120/fp32_fma_dep_chain/trace_00` (REQUIRED — the prior
  failed cell from M1's loop; admitting this is the most
  informative outcome).
- `sm_120/layernorm_fwd_bf16_sm80/timing_shape`
- `sm_120/rmsnorm_fwd_bf16_sm80/timing_shape`
- `sm_120/silu_fwd_bf16_sm80/timing_shape`
- `sm_120/softmax_safe_bf16_sm80/timing_shape`
- (optional) `sm_120/silu_fwd_bf16_sm80/correctness_shape`

If the matrix's first sweep does not admit ≥5 sm_120 cells,
expand the set with additional same-family slugs from
`workloads/simbench_bw/roster.toml` (status=active rows) or
`workloads/portable_lcd/roster.toml`.

### Per-cell measurement-manifest map

Every candidate cell MUST name a measurement-manifest path the
sweep harness forwards to `label_driver.sh --measurement-manifest`.
The manifest format (operator-supplied; the on-disk simbench_bw
`measurement_manifest.json` does NOT carry
`kernel_signature_sha256` natively, so the operator stages a
purpose-built manifest):

```json
{
    "schema_version": 2,
    "slug": "<slug>",
    "shape_point": "<shape>",
    "binary_sha256": "<hex sha256 of the binary the sweep will profile>",
    "kernel_signature_sha256": "<hex sha256 of the kernel signature>"
}
```

`binary_sha256` MUST equal `sha256sum <binary>` exactly; the
driver verifies this and fails closed if it disagrees.
`kernel_signature_sha256` is the operator-pinned source-of-truth
for the join key.

#### Pre-flight validation (recommended)

After staging all per-cell manifests but BEFORE running the
live sweep, verify each manifest is well-formed via the bounded
operator-facing validator:

```
python3 tools/local_labels/validate_measurement_manifest.py \
    --manifest <manifest-1> --manifest <manifest-2> ... \
    [--binary <cell>=<path>,...]
```

Expected output on success:

```
validate_measurement_manifest: 5 manifest(s) valid
```

The validator checks each manifest's schema (required fields,
`schema_version=2`, 64-char lowercase hex on both sha256 fields)
and, when `--binary <cell>=<path>` is set, computes each
binary's sha256 via Python `hashlib` and verifies the matching
manifest's `binary_sha256` field. Documented exit codes: 0 = all
valid, 1 = schema or sha256 mismatch, 2 = io error (manifest /
binary file not found, malformed JSON).

The pre-flight is functionally redundant with the
`label_driver.sh` fail-closed path that fires at live-sweep
time — but the driver-side fail only fires AFTER NCU is
invoked, so the validator surfaces the error instantly without
burning sweep wall-clock budget.

### sm_120 sweep invocation (live mode)

`--measurement-manifests` AND `--predictor-trace-jsons` are
both REQUIRED in `--live` mode; the harness runs the
pre-flight provenance audit per cell BEFORE any NCU shim is
spawned. Stage one predictor-side per-shape trace JSON sidecar
per candidate cell (under
`workloads/simbench_bw/results/<slug>/<sku>/sm_120-laptop/trace_<shape_point>.json`)
in addition to the measurement manifest.

```
python3 tools/local_labels/sweep_repeatability_matrix.py \
    --live \
    --candidates sm_120/fp32_fma_dep_chain/trace_00,sm_120/<slug2>/<shape2>,... \
    --binaries sm_120/fp32_fma_dep_chain/trace_00=<path>,sm_120/<slug2>/<shape2>=<path>,... \
    --measurement-manifests sm_120/fp32_fma_dep_chain/trace_00=<manifest_path>,sm_120/<slug2>/<shape2>=<manifest_path>,... \
    --predictor-trace-jsons sm_120/fp32_fma_dep_chain/trace_00=<predictor_json_path>,sm_120/<slug2>/<shape2>=<predictor_json_path>,... \
    --warmup-counts 0,1,5 \
    --cache-controls all,none \
    --repeats 5 \
    --seed 0 \
    --ncu /usr/local/cuda-13.2/nsight-compute-2026.1.0/ncu
```

`stddev_threshold=0.03` is unchanged; do NOT loosen admission.

### Hardware-metadata capture (record in the round summary)

The capture step is a bounded operator-facing CLI:

```
python3 tools/local_labels/capture_observed_state.py --expected-arch sm_120
```

Expected output:

```
capture_observed_state: device=NVIDIA GeForce RTX 5060 Laptop GPU compute_cap=12.0 (matches sm_120)
capture_observed_state: wrote data/local_labels/sweeps/clock_throttle_evidence/observed_state_<UTC>.json
capture_observed_state: wrote data/local_labels/sweeps/clock_throttle_evidence/observed_state_<UTC>.csv
```

The tool runs the read-only nvidia-smi `--query-gpu=name,
driver_version,compute_cap,clocks.sm,clocks.mem,
clocks_throttle_reasons.active --format=csv,noheader` query
verbatim, parses the CSV output, also probes
`nvcc --version` for the CUDA toolkit `release X.Y` substring
(captures the literal sentinel `"unavailable"` if nvcc is
absent, fails, or returns unparseable output — toolkit absence
does NOT gate capture), captures `source_commit` from
`git rev-parse HEAD` per
`BL-20260508-post-commit-recapture-discipline`, and writes both
the JSON sidecar (structured fields including the new
`cuda_toolkit_version` field) and the CSV sidecar (verbatim
nvidia-smi line) under
`data/local_labels/sweeps/clock_throttle_evidence/`. The writes
use a tmp + rename atomic pattern so a concurrent reader never
sees a half-written file with the final name.

Documented exit codes: 0 = captured (and `--expected-arch` match
if set), 1 = `--expected-arch` mismatch (operator is on the
wrong host; no captures written), 2 = nvidia-smi unavailable /
non-zero / parse error.

### Evidence merge

After the sm_120 sweep admits ≥5 cells, MERGE the new sm_120
records into `data/local_labels/label_admitted.json` WITHOUT
dropping the existing sm_86 records, AND validate the merged
file against the strict schema, in one bounded CLI invocation:

```
python3 tools/local_labels/merge_admitted_cells.py \
    --existing data/local_labels/label_admitted.json \
    --new-from-sweep data/local_labels/<sweep_dir>/label_admitted.json
```

Expected output:

```
merge_admitted_cells: merged sm_86=5 sm_120=5 total=10 (schema-valid)
merge_admitted_cells: wrote data/local_labels/label_admitted.json
```

The tool concatenates the two `cells` lists (existing first,
then new), strict-validates the merged result against
`workloads/simbench_bw/schemas/local_label_admitted.schema.json`,
and writes the merged file in place at
`data/local_labels/label_admitted.json`. It rejects with a typed
`merge_admitted_cells:` diagnostic and rc=2 on:

- a missing `--existing` or `--new-from-sweep` path,
- malformed input JSON,
- a duplicate `(slug, shape_point, arch)` tuple across the two
  inputs (no silent override of an existing cell),
- a merged result that fails strict-schema validation.

The tool is the executable replacement for the legacy inline
`python3 -` heredoc merge / validate recipes the runbook used to
carry in earlier versions.

### Cross-arch M1 runner integration check

Run the comparison runner against the merged file with same-key
predictor bundles for sm_120 cells; the runner is expected to
admit when the merged file carries `evidence_kind: live` for the
sm_120 leg AND to route `evidence_kind: synthetic` sm_120 cells
to the typed-unavailable arm. The receiving RTX 5060 laptop
verifies this contract with the two strict-v3 wire-format
regression tests that AC-11 split out of the original
single-test variant:

```
pytest -q \
    workloads/simbench_bw/tests/test_comparison_runner.py::test_synthetic_v2_strict_admitted_label_file_with_live_evidence_admits_sm120 \
    workloads/simbench_bw/tests/test_comparison_runner.py::test_synthetic_v2_strict_admitted_label_file_with_synthetic_evidence_routes_to_typed_unavailable_sm120
```

Each variant locks one half of the live-only floor contract:

- `..._with_live_evidence_admits_sm120` is the v3 admission
  wire-format guard. It writes a strict-schema admitted-cell
  file with `evidence_kind: "live"` on every sm_120 cell, runs
  the comparison runner CLI, and asserts the result is
  `admitted` with `headline_cells_per_arch["sm_120"] >= 5`.
- `..._with_synthetic_evidence_routes_to_typed_unavailable_sm120`
  is the synthetic-only typed-unavailable guard. It writes the
  same strict-schema file but with
  `evidence_kind: "synthetic"` on every sm_120 cell and asserts
  the runner emits
  `result_kind == "typed_unavailable"` with
  `typed_reason == "headline_geomean_admitted_cell_count_below_threshold"`.

Passing both nodeids on the receiving RTX 5060 laptop is the
producer/consumer wire-format pre-flight before any live sm_120
sweep work proceeds. The single-test variant the earlier runbook
named has been removed; only the two split nodeids above are
authoritative as of the AC-11 closure round.

### Final closure

When the sm_120 sweep lands AND the M1 runner integration passes
with `headline_cells_per_arch["sm_120"] >= 5`, the round summary
declares evidence status `Cross-arch-verified` per CLAUDE.md.

The operator can verify the closure state with a single command
on either host. Note the mandatory `--all-source-commits` flag:
the merged corpus carries sm_86 cells at the 3090 Ti's pre-sweep
HEAD AND sm_120 cells at the laptop's pre-sweep HEAD; current
`git rev-parse HEAD` on either host matches at most one of these
commits, so the helper's default (filter live cells to current
HEAD) would drop cells from the OFF-HOST sweep commit and report
a false-negative `cross_arch=below_floor` even when both arch
slices meet the floor. The cumulative-corpus mode admits the
merged file's full per-arch counts:

```
python3 tools/local_labels/check_admission_floor.py --cross-arch --floor 5 --all-source-commits
```

Exit 0 means both arches' admitted-cell counts meet the floor
AND the merged file strict-validates against
`workloads/simbench_bw/schemas/local_label_admitted.schema.json`.
Exit 1 means at least one arch is below the floor; exit 2 means
the file failed schema validation or was unreadable. The tool is
the executable acceptance check that replaces the legacy inline
`python3 -c` admission-count recipes the runbook used to carry
in earlier versions.

Per-arch variants are also supported (each with the same
`--all-source-commits` rationale: the merged corpus crosses two
sweep commits; on the off-host the per-arch check would
otherwise drop the off-host arch's cells):

```
python3 tools/local_labels/check_admission_floor.py --arch sm_86 --floor 5 --all-source-commits
python3 tools/local_labels/check_admission_floor.py --arch sm_120 --floor 5 --all-source-commits
```

(Operators who know the exact pre-sweep HEAD SHA per arch can
substitute `--source-commit <SHA>` for `--all-source-commits` to
mirror `comparison_runner`'s 7-tuple join more precisely; the
cumulative-corpus mode is the simpler default for the
post-merge acceptance check.)
