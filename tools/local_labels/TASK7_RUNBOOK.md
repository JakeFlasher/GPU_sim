# task7 Operator Runbook — sm_86 Local-Label Hardware Acceptance

This runbook walks the operator on the **RTX 3090 Ti (sm_86)**
through `task7`: the AC-2/AC-3 hardware acceptance for local
labels on Ampere. It is the sm_86 analog of the R10/R11 work
that was completed on the RTX 5060 laptop (sm_120) for
`workloads/simbench_bw/l1/fp32_fma_dep_chain/`.

The runbook is the operator-facing companion to the carry-over
`task7` row recorded in `tools/BLOCKED_ON_3090TI.md` (and in the
active session's gitignored `goal-tracker.md` under
`.humanize/rlcr/<session-id>/`).

## When to use this runbook

Run this verification when the working directory has been
pivoted from the 5060 laptop to the 3090 Ti, the operator
wants to close `task7`, and the device has a working CUDA
toolkit + NCU installation per CLAUDE.md Rule 1's local-label
exception. The expected outcome is either an admitted cell at
`data/local_labels/sm_86/<slug>/<shape_point>/` or an excluded
cell with a typed reason — both are valid AC-2/AC-3 evidence
(R10/R11 precedent on sm_120: a `label_repeatability_failed`
verdict IS the AC-3 evidence because the repeatability gate
firing correctly is what the criterion requires).

## Source of Truth (read this BEFORE Step 1)

This runbook follows the hybrid source-of-truth +
verification-target pattern documented in
`BL-20260504-runbook-self-reference`
(see `docs/lessons.md`):

- **Source of truth**: the runbook procedure is read from the
  working branch's HEAD (`5060_dev` HEAD), NOT from a pinned
  `git checkout` target SHA. The branch HEAD is the
  authoritative procedure.
- **Verification target**: this runbook has NO separately
  pinned verification-target SHA. The verification surface is
  the local-label driver and the `workloads/portable_lcd/`
  kernels — both stable since the prior 5060 session completed
  the 15-kernel portable_lcd artifact generation on sm_120, so
  any commit on `5060_dev` containing both surfaces is a valid
  verification target. (The sister
  `tools/nvbit_tracer/CROSS_ARCH_RUNBOOK.md` IS an example of
  the non-degenerate hybrid pattern: it pins a verification
  target SHA distinct from the source-of-truth read.)

You can re-read this runbook from branch HEAD without changing
your worktree:

```bash
git show 5060_dev:tools/local_labels/TASK7_RUNBOOK.md | less
```

The reason for the no-source-of-truth-pinning approach: pinning
a target SHA in a runbook that lives at that same SHA produces
a self-referential staleness trap. The trap appeared during
the 5060 session's R7-R9 attempt to land
`CROSS_ARCH_RUNBOOK.md` and the fix is recorded in the
BitLesson cited above.

## Prerequisites

- Local hardware:
  - RTX 3090 Ti (CC 8.6, sm_86) — Ampere desktop card.
- CUDA toolkit (the current development chain uses CUDA 13.2
  on the 5060 session; a similar version on the 3090 Ti is
  expected).
- gcc ≥ 9.0 (project uses C++17).
- NCU installation (Nsight Compute) — required by
  `tools/local_labels/label_driver.sh`. Per CLAUDE.md Rule 1:
  NCU may be invoked on local devices (RTX 3090 Ti sm_86 and
  RTX 5060 laptop sm_120) ONLY into the quarantine path
  `data/local_labels/`. The driver code that performs the
  local invocation lives ONLY under `tools/local_labels/`.
  No other path in the repository may invoke NCU. The
  quarantine boundary is enforced by the static guard at
  `workloads/simbench_bw/tests/test_no_root_or_ncu_static_guard.py`.
- Working NVIDIA driver visible to `nvidia-smi`.
- The portable_lcd kernel binaries from Round 13 (or a fresh
  build if the working directory is clean). See "Pick a
  portable_lcd kernel" below.

## Pre-flight checks

Capture pre-sweep evidence into a log so the round summary
can reference durable artifacts:

```bash
nvidia-smi > /tmp/task7_preflight.log 2>&1
echo "Exit: $?" >> /tmp/task7_preflight.log
nvcc --version >> /tmp/task7_preflight.log 2>&1
ncu --version >> /tmp/task7_preflight.log 2>&1
```

Verify:

- `nvidia-smi` exit 0 and the GPU shows up as
  `NVIDIA GeForce RTX 3090 Ti` (or similar Ampere model)
  with compute capability 8.6.
- `nvcc --version` reports a CUDA toolkit that supports
  sm_86 compilation.
- `ncu --version` returns the actual `Version <release>` line
  (NCU 2026.1.x on the current chain). If NCU is missing,
  install it before continuing — `tools/local_labels/label_driver.sh`
  requires it.

Capture the GPU UUID for provenance (used by the driver to
populate `gpu_uuid_sha256`):

```bash
nvidia-smi --query-gpu=uuid --format=csv,noheader >> /tmp/task7_preflight.log
```

## Pick a portable_lcd kernel

Round 13 built and traced 15 active portable_lcd kernels on
sm_120. Any of them is a valid task7 target on sm_86 because
the kernels compile to both architectures (the Makefile uses
`-arch=all` or per-arch slug naming).

Recommended kernel: `gemm_hmma_bf16_sm80`. This is a
straightforward bfloat16 GEMM that exercises the L1/TEX and
L2 cache hierarchies and does not depend on Hopper/Blackwell
features. The expected NCU metrics are well-defined.

Locate the binary:

```bash
ls workloads/portable_lcd/gemm_hmma_bf16_sm80/gemm_hmma_bf16_sm80_run
```

The Makefile defines `RUN = $(TARGET)_run` and builds the
binary directly in the kernel directory root, NOT under
`results/<slug>/`. If the binary is missing (artifacts are
gitignored per repo policy), rebuild via:

```bash
make -C workloads/portable_lcd/gemm_hmma_bf16_sm80
```

The Makefile's default target `all` depends on `run-bin`,
which builds `<slug>_run`. The Makefile does NOT execute the
binary — running it (which emits the NVBit dynamic trace,
threadblock JSONs, and the schema-v2 manifest under
`<output_dir>/<slug>/`) is a separate step that task7 does
not need: `label_driver.sh` runs the binary itself under NCU,
so the operator only needs the binary to exist and be
executable.

After build, re-verify the binary path:

```bash
test -x workloads/portable_lcd/gemm_hmma_bf16_sm80/gemm_hmma_bf16_sm80_run && echo "OK"
```

For the shape mode, use one of the two modes the portable_lcd
runner exposes
(`workloads/portable_lcd/runner_common.h:62-83`):

- `--correctness_shape` — small fixed shape used for
  numerical-correctness checks. Recommended smoke choice for
  task7: small enough that the driver completes quickly, large
  enough that NCU's L1/TEX and L2 sector hit-rate metrics are
  well-defined.
- `--timing_shape` — larger shape used for timing
  characterization. Acceptable but not required for task7
  smoke.

The `--shape-point` driver flag value MUST be the literal
string `correctness_shape` (or `timing_shape`) when targeting
the portable_lcd runner. Generic `--shape-point trace_00`
tokens are simbench-style and will fail the driver's
shape-binding guard
(`tools/local_labels/label_driver.sh:172-235`) before NCU is
invoked.

## Run label_driver.sh × N=5

Run the driver 5 times with distinct `--repeat` indices. Each
invocation produces one per-launch sidecar at
`data/local_labels/sm_86/<slug>/<shape_point>/label_<repeat>_<launch>.json`.

The driver requires forwarded workload args after `--` so that
the profiled invocation truthfully binds the requested
shape_point. For `--shape-point correctness_shape` (or
`timing_shape`) the forwarded args MUST contain
`--correctness_shape` (or `--timing_shape`) exactly once and
must NOT carry any generic `--shape-point` token. The driver
auto-appends `--warmup <count>` for these portable_lcd shape
modes when no warmup flag is forwarded
(`tools/local_labels/label_driver.sh:267-342`).

```bash
SLUG=gemm_hmma_bf16_sm80
SHAPE=correctness_shape
BIN=workloads/portable_lcd/${SLUG}/${SLUG}_run
WORKLOAD_OUT=results/local_labels_run/${SLUG}

for REPEAT in 0 1 2 3 4; do
    tools/local_labels/label_driver.sh \
        --device sm_86 \
        --slug ${SLUG} \
        --shape-point ${SHAPE} \
        --repeat ${REPEAT} \
        --launch 0 \
        --binary ${BIN} \
        --warmup 1 \
        -- --correctness_shape --reps 1 --output_dir ${WORKLOAD_OUT}
done
```

A few notes on the loop:

- The literal `--` token is REQUIRED. The driver scans args
  before the separator as driver flags, and args after the
  separator as forwarded workload args. Without `--`, the
  driver fails fast with `forwarded application args do not
  unambiguously bind shape_point=<SHAPE>` and prints the hint
  `did you forget the '--' separator before app args?`.
- `--reps 1` is the smallest legal value (the runner rejects
  `--reps 0` with `repeat_iters_must_be_positive` per
  `runner_common.h:86-89`). The repeatability gate's N>=5
  comes from running the driver 5 times — NOT from setting
  `--reps 5` inside one invocation.
- `${WORKLOAD_OUT}` is the runner's `--output_dir` (where the
  runner writes its own manifest + correctness-mode artifacts).
  It is intentionally separate from the local_labels
  quarantine path (`data/local_labels/sm_86/...`) where
  `label_driver.sh` writes the per-launch sidecars.

After the loop, expect 5 sidecars:

```bash
ls data/local_labels/sm_86/${SLUG}/${SHAPE}/
# Expected output:
# label_0_0.json  label_1_0.json  label_2_0.json  label_3_0.json  label_4_0.json
```

If any iteration fails, the driver exits non-zero and prints a
typed diagnostic to stderr; no sidecar is emitted for that
iteration. Investigate the error before proceeding. Common
causes: NCU permission denied (running NCU may require
elevated privileges on some host configurations — consult the
NCU documentation), missing GPU visibility (re-run pre-flight
to confirm `nvidia-smi` exit 0), stale binary path, or
forwarded args that fail the shape-binding guard (rerun with
the documented `-- --correctness_shape --reps 1 --output_dir
...` shape).

## Validate via import_label.py

Each per-launch sidecar must admit individually:

```bash
for f in data/local_labels/sm_86/${SLUG}/${SHAPE}/label_*_0.json; do
    python tools/local_labels/import_label.py "$f"
done
```

Expected output: each invocation prints `OK` (or the
equivalent admission line). Any rejection indicates a
schema/provenance issue with the sidecar; investigate before
running the aggregator.

## Run repeatability_aggregator.py

The aggregator walks the quarantine root, validates each
sidecar via `import_label.py`, groups by cell `(device, slug,
shape_point)`, computes per-target stddev across repeats, and
writes 3 index files:

```bash
python tools/local_labels/repeatability_aggregator.py \
    --root data/local_labels
```

Expected output files:

- `data/local_labels/label_repeatability.json` — per-cell stats
  (n_repeats, per-target mean and stddev, binary_sha256
  consistency flag).
- `data/local_labels/label_admitted.json` — cells that pass the
  3pp stddev threshold (or whatever override was passed via
  `--stddev-threshold`).
- `data/local_labels/label_excluded.json` — cells that fail
  the threshold, with `typed_reason` set to one of:
  - `label_repeatability_failed` — per-target stddev exceeds
    threshold.
  - `label_repeatability_missing` — fewer than the required
    repeats (default 5).
  - `label_provenance_inconsistent` — sidecars in the same
    cell disagree on `binary_sha256`.

## Interpret the verdict

**Cell admitted in `label_admitted.json`**: AC-2/AC-3 hardware
acceptance succeeded. The cell's per-target hit rates with
0.95 CIs are the live label data.

**Cell excluded in `label_excluded.json` with
`label_repeatability_failed`**: this IS the AC-3 evidence.
The repeatability gate fired correctly on real hardware. The
plan's AC-3 sub-criterion is "produce repeatability evidence
that admits or excludes the cell with a typed reason"; an
exclusion with a typed reason is a valid AC-3 outcome (R10/R11
precedent on sm_120: a noisy compute-bound microbench yielded
`label_repeatability_failed` with L2 stddev exceeding 3pp,
and that verdict was the AC-3 closure).

**Cell excluded with `label_repeatability_missing`**: indicates
fewer than 5 admissible sidecars in the cell. Re-run any
failed iterations from the loop above; each must produce a
sidecar that admits via `import_label.py`. Then re-run the
aggregator.

**Cell excluded with `label_provenance_inconsistent`**:
indicates `binary_sha256` mismatch across the cell's sidecars.
This means the binary changed mid-loop. Restart the loop with
a fresh build of the portable_lcd kernel and re-run all 5
iterations against the same binary.

## Persist evidence

The round summary that closes task7 should record:

- Device: GPU model, compute capability, driver version, NVML
  UUID.
- Tooling: CUDA toolkit version, NCU version (from
  `safe_ncu_version`), NVBit build version.
- Workload: portable_lcd kernel slug, shape point, launch
  index, binary SHA-256.
- Verdict: `label_admitted` or `label_excluded` + typed_reason.
- Stddev: per-target stddev across the 5 repeats.
- Captured logs: `/tmp/task7_preflight.log` and the 5 per-launch
  sidecar paths.
- Cross-reference: link to the R10/R11 sm_120 round summaries
  as the analog being mirrored.

## What this runbook does NOT do

- **`task11b-task11d-cross-arch-followup`** — that is the
  NVBit tracer cross-arch verification (rebuild
  `tracer_tool.so` for sm_86, rerun the live simbench sweep,
  verify `allocation_id` behavior, etc.). See
  `tools/nvbit_tracer/CROSS_ARCH_RUNBOOK.md` for that
  procedure.
- **AC-5+ implementation** — gated on AC-4 cross-arch
  closure (Codex's directive #3). Do not start AC-5 work on
  the 3090 Ti session before `task7` AND
  `task11b-task11d-cross-arch-followup` have closed and earned
  `Cross-arch-verified` status per CLAUDE.md "Two-Device Local
  Development Model" Rule 3.
- **Cloud / server GPU labeling** — explicitly forbidden by
  CLAUDE.md Rule 1. NCU runs on the 3090 Ti and 5060 laptop
  ONLY.

## References

- `CLAUDE.md` — project critical rules; Rule 1 local-label
  exception governing NCU usage on local devices.
- `tools/local_labels/README.md` — driver-level reference for
  the schema, the denylisted field names, the typed exclusion
  reasons, and the repeatability threshold default.
- `tools/local_labels/label_driver.sh` — the entry point
  invoked above; produces one per-launch sidecar per
  invocation.
- `tools/local_labels/import_label.py` — per-sidecar
  admission validator.
- `tools/local_labels/repeatability_aggregator.py` — N>=5
  cell-level verdict producer.
- `tools/nvbit_tracer/CROSS_ARCH_RUNBOOK.md` — sister runbook
  for `task11b-task11d-cross-arch-followup` (the NVBit tracer
  cross-arch side); the operator typically runs both runbooks
  in sequence on the 3090 Ti session.
- The prior 5060 session's `round-10-summary.md` and
  `round-11-summary.md` (under
  `.humanize/rlcr/<prior-session-id>/`) — sm_120 precedent (the
  work on `workloads/simbench_bw/l1/fp32_fma_dep_chain/` whose
  sm_86 analog is task7). Loop-state summaries are gitignored
  per the `.humanize/` policy; consult them from the originating
  worktree if needed.
- `plans/nonroot_gpu_cache_observability_plan.md` — AC-2 (Local
  labels) and AC-3 (Repeatability) sub-criteria.
- `docs/lessons.md` —
  `BL-20260504-runbook-self-reference` documents the
  no-SHA-pinning pattern this runbook uses.
