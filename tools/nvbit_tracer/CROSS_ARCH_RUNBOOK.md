# NVBit Tracer Cross-Arch Verification Runbook

This runbook captures the deterministic plan for verifying that the
NVBit tracer producer + consumer surfaces work correctly across both
local devices (RTX 3090 Ti sm_86 and RTX 5060 Laptop sm_120). It is
the operator-facing companion to the carry-over
`task11b-task11d-cross-arch-followup` row recorded in
`tools/BLOCKED_ON_3090TI.md` (and in the active session's gitignored
`goal-tracker.md` under `.humanize/rlcr/<session-id>/`).

The runbook is designed for the case where one session lands new
producer or consumer changes on one architecture and the next session
on the other architecture replays the host + offline tests, the live
sweep, and confirms the producer/consumer surfaces stay correctly
decoupled per CLAUDE.md's Two-Device Local Development Model.

## When to use this runbook

Run this verification on the OTHER architecture immediately after a
round that lands producer or consumer changes on EITHER architecture
and earns `Verified-live-<one_arch>`. The goal is to promote the
round to `Cross-arch-verified` per CLAUDE.md "Two-Device Local
Development Model" Rule 3.

The most recent example: Round 5 + Round 6 of the 5060_dev session
landed producer-side per-lane resolution + consumer-side validator
allocation/range identity + `cross_allocation_lane_span` taxonomy +
file-shape importer fix on sm_120. That work earned
`Verified-live-sm_120` only via the persisted live evidence at
parent commit `1781304`. The 3090 Ti session running this runbook
on the runbook-containing commit (see "Target Commit" below)
confirms cross-arch parity and unlocks AC-4 closure.

## Source of Truth (read this BEFORE Step 1)

This runbook follows the hybrid source-of-truth +
verification-target pattern documented in
`BL-20260504-runbook-self-reference` (see
`docs/lessons.md`).

**This runbook has TWO SHAs to keep separate:** the **verification
target** (where the operator runs the verification) and the
**source of truth** (where the operator reads the runbook). They
are different and must remain distinct. Conflating them caused
two consecutive review stalls (R7 review + R8 review) before
the Round 9 fix that introduced this section.

### Always read this runbook from `5060_dev` HEAD

Read the **latest** version of this runbook from `5060_dev` branch
HEAD (or whichever commit you initially clone). You can re-read it
without changing your worktree:

```bash
git show 5060_dev:tools/nvbit_tracer/CROSS_ARCH_RUNBOOK.md | less
```

The latest version is the authoritative source of truth for the
verification procedure. **Do NOT re-read the runbook from your
worktree after `git checkout` — earlier commits on `5060_dev`
contain stale runbook text from prior rounds.**

### Worktree-staleness warning

The verification target SHA pinned below (`f6b99da`) contains the
correct verification SURFACE (host unit tests, Python tests,
producer side, the `entries=<N>` CLI behavior), but the runbook
TEXT in the worktree at `f6b99da` is the Round 8 version that
incorrectly pins `50cf61a`. **DO NOT follow the worktree's
runbook text after checkout.** Continue using the latest version
you read pre-checkout (or via `git show 5060_dev:...`) as the
authoritative procedure.

The pattern this section enforces is recorded as
`BL-20260504-runbook-self-reference` in `docs/lessons.md`.
The historical R7-R9 trap that motivated the policy is
documented in a sibling "Trap History" section below for
operators who want the background; the operator-facing
verification procedure does not depend on it.

## Prerequisites

- Local hardware:
  - RTX 3090 Ti (CC 8.6, sm_86) — Ampere desktop card.
  - OR RTX 5060 Laptop (CC 12.0, sm_120) — Blackwell laptop card.
- CUDA toolkit (must support both sm_86 and sm_120 compilation; the
  current development chain uses CUDA 13.2 on the 5060 session).
- gcc ≥ 9.0 (project uses C++17).
- The pinned C++ protobuf toolchain under `.toolchain/pb` (protoc 34.1,
  libprotobuf 34.1.0, and its Abseil closure). Provision an absent prefix only
  with `bash tools/nvbit_tracer/pinned_build_env.sh --ensure`; ordinary
  sourcing validates and exports without repairing a mismatch. Python bindings under
  `workloads/simbench_bw/_pb_generated` are tracked repository authorities,
  not canonical-checkout build outputs.
- jsonschema (`pip install jsonschema` if not already present).
- Working NVIDIA driver visible to `nvidia-smi`. Laptop dGPU may be
  in P8 deep-idle when the runbook starts; the pre-sweep `nvidia-smi`
  probe (step 4 below) wakes it.

## Target Commit

The cross-arch verification target is:

```
f6b99da fix cross-arch runbook target-SHA contradiction
```

This is Codex's R8-review-directed verification target. It is the
first commit on `5060_dev` whose verification SURFACE matches what
this runbook documents (host unit tests, Python tests, producer
side, `entries=<N>` CLI behavior). Note that the runbook TEXT at
`f6b99da` itself is the Round 8 version pinning `50cf61a` — that
text is STALE; do not follow it. The Source-of-Truth section above
describes how to re-read the latest runbook without changing your
worktree.

Any later commit on `5060_dev` that preserves the verification
surface is also a valid target; the surface is identical at every
commit from `1781304` (Round 6 close) onwards because no producer
or live-test changes have landed since.

### Commit Lineage Reference

These are the historical milestones on `5060_dev` relevant to this
runbook. Only the LAST (`f6b99da` for now; any later fixing commit
also valid) is the verification target. The earlier ones are
historical context — DO NOT use them as `git checkout` targets:

| Commit | Round | Role |
|--------|-------|------|
| `1781304` | R6 close | Round 5 + Round 6 producer/consumer surface complete; live sm_120 evidence captured at this commit (`round-6-live-sweep.log`); runbook DOES NOT exist at this commit |
| `50cf61a` | R7 close | Runbook file introduced at this commit + `entries=<N>` CLI behavior added; runbook target-SHA initially miswired (still pointed at `1781304`) |
| `f6b99da` | R8 close | Runbook target-SHA partially fixed (`1781304` → `50cf61a`); recursion trap remained because `50cf61a`'s runbook text is the Round 7 version that points at `1781304` |
| _this commit and later_ | R9 close+ | Source-of-Truth section breaks the recursion trap; verification target now stably points at `f6b99da` (or later) |

Confirm `git rev-parse HEAD` matches `f6b99da` (or your chosen
later valid target) before running the verification. If the
commit does not match, the expected numerics in this runbook may
not apply.

## Verification Steps

### Step 1 — Sync to the target commit

```bash
git fetch origin            # if a remote is configured
git checkout f6b99da        # or any later commit on 5060_dev
git status                  # must be clean
```

**Reminder from the Source of Truth section above:** the runbook
TEXT in your worktree at `f6b99da` is the Round 8 version pinning
`50cf61a`. That is STALE. Continue using the LATEST runbook
version (the one you read pre-checkout, or via
`git show 5060_dev:tools/nvbit_tracer/CROSS_ARCH_RUNBOOK.md`) as
the authoritative procedure for the rest of these steps.

If the working tree is from an scp from another local device (no
remote credentials), confirm with `git rev-parse HEAD` that the
SHA matches `f6b99da` or a later commit on `5060_dev`. Confirm the
runbook FILE is present in the worktree (its TEXT will be stale,
which is fine for the verification surface):

```bash
ls tools/nvbit_tracer/CROSS_ARCH_RUNBOOK.md
```

If the file is missing, you are on a pre-`50cf61a` commit; advance
to a Round-9-or-later commit before continuing.

### Step 2 — Verify the preserved tracer contract

```bash
bash tools/nvbit_tracer/pinned_build_env.sh --ensure
source tools/nvbit_tracer/pinned_build_env.sh
make -C tools/nvbit_tracer check-pinned-env
```

Task7/H3 remains open. Canonical Git worktrees refuse every build, clean,
regeneration, replacement, and repin target. A build feasibility check is
permitted only in a non-Git temporary copy with
`NVBIT_TEMPCOPY_BUILD_OK=1`; the sentinel never authorizes a canonical
worktree mutation. Verify the preserved artifact with:

```bash
cuobjdump --list-elf tools/nvbit_tracer/tracer_tool/tracer_tool.so | grep cubin
```

### Step 3 — Run pre-existing host unit binaries

```bash
tools/nvbit_tracer/tracer_tool/tests/phase_classifier_test
tools/nvbit_tracer/tracer_tool/tests/memory_op_class_test
tools/nvbit_tracer/tracer_tool/tests/allocation_registry_test
tools/nvbit_tracer/tracer_tool/tests/exclusion_aggregator_test
```

Expected output (all PASS):

- `phase_classifier_test: PASS`
- `memory_op_class_test: PASS`
- `allocation_registry_test: 27 tests passed`
- `exclusion_aggregator_test: 13 tests passed`

A different count is a regression. Investigate before continuing.

### Step 4 — Run Python offline tests

```bash
pytest tools scripts/kb -q
```

The target list is intentionally `tools` (not the four
subdirectories `tools/nvbit_tracer tools/cloud tools/local_labels
scripts/kb` it once enumerated): a top-level scan picks up every
`test_*.py` under `tools/` including the C-AC-B
`tools/test_runbook_policy.py` and
`tools/test_runbook_command_verification.py` test files. Pytest's
file-based discovery still skips non-test directories under
`tools/` (e.g., `tools/cloud/mock_inputs/`).

Expected output (target commit `f6b99da`):

```
398 passed, 62 subtests passed in <~1.5s>
```

The same `398` count held at every historical commit on this
branch through R9. **The count grows as later rounds land new
tests** (Loop 2 R1 alone added ~40 tests via the C-AC-B runbook
policy + command-verification work, lifting the baseline to
~582+). The expected discipline is that the count NEVER drops
below the baseline at the previous round's HEAD; it may rise by
any amount as new tests land.

### Step 5 — Run the live sweep with persisted log

Capture pre and post `nvidia-smi` probes plus the full pytest run
into a single log file inside the loop directory so the evidence is
durable across sessions:

```bash
SESSION_DIR=.humanize/rlcr/<session-id>
SWEEP_LOG=$SESSION_DIR/round-N-cross-arch-<arch>-sweep.log
{
  echo "=== CROSS-ARCH SWEEP <arch> on commit $(git rev-parse HEAD) ==="
  echo "=== PRE-SWEEP nvidia-smi ==="
  nvidia-smi --query-gpu=name,driver_version,compute_cap,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv
  echo "PRE-SWEEP nvidia-smi exit: $?"
  echo
  echo "=== Live pytest sweep ==="
  SIMBENCH_REQUIRE_LIVE_GPU=1 pytest workloads/simbench_bw/tests -q 2>&1
  echo "Pytest exit: $?"
  echo
  echo "=== POST-SWEEP nvidia-smi ==="
  nvidia-smi --query-gpu=name,driver_version,compute_cap,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv
  echo "POST-SWEEP nvidia-smi exit: $?"
} > "$SWEEP_LOG" 2>&1
tail -30 "$SWEEP_LOG"
```

Expected:

- Pre-sweep `nvidia-smi` exit 0 (P0/P4 active state acceptable).
- `679 passed, 2 skipped` on the live pytest sweep.
- Post-sweep `nvidia-smi` exit 0 (state may shift down to P4/P5 due
  to dGPU power management, but exit must be 0).

### Step 6 — Confirm the three flagged arch-coupling sites

These are the live tests whose outputs are expected to differ
slightly across architectures. They must continue to pass on each
arch with the per-arch-acceptable values.

#### 6a. Compression distribution

`workloads/simbench_bw/tests/test_nvbit_trace_contract_live.py`
`test_live_trace_compression_formats_observed_and_reconstruct`
prints the observed `compression_format` distribution for the live
trace at the end of the test. Expected values (from prior captures):

- sm_86 baseline (3090 Ti, master commit `d16b375` Round 24): `{base_stride: 33, base_delta: 11, list_all: 0}`.
- sm_120 baseline (5060 Laptop, captured at historical commit `1781304` Round 6 in `round-6-live-sweep.log`; identical at the verification target `f6b99da` and at any later `5060_dev` commit because no producer changes have landed since): `{base_stride: 35, base_delta: 11, list_all: 0}`.

Both shapes admit the test (it only requires base_stride and
base_delta both non-empty; the producer's `enable_compress=1` default
exercises both formats). A drift to a fundamentally different shape
(e.g., `list_all` becoming non-zero, or one of base_stride /
base_delta becoming zero) signals a producer regression.

#### 6b. cp.async SASS witness

`workloads/simbench_bw/tests/test_phase_injection_smoke.py`
`test_cp_async_kernel_sass_witness` confirms `kernel_cp_async_no_sync`
contains the `cp.async` SASS family on both arches. The kernel is
hard-architecture-portable (LDGSTS on sm_86; LDGSTS on sm_120 with a
slightly different operand shape but the test matches on the SASS
opcode head). Expected: PASSED on both arches.

#### 6c. ASYNC_COPY classifier sidecar

The producer-side `memory_op_class_classifier` routes LDGSTS through
`unsupported_memory_path` typed exclusion (the `ASYNC_COPY` op class
is in `is_excluded_op_class`). The live test
`test_live_trace_unsupported_memory_path_sidecar_admits` verifies
the sidecar is present with at least one ASYNC_COPY entry. Expected
on both arches: PASSED.

### Step 7 — Confirm the wire-extension allowlist

The producer can emit three typed reasons; the consumer must admit
all three on both arches. Running the live sweep above should not
encounter any `unknown` typed reason on either arch.

Producer-side typed reasons (`tools/nvbit_tracer/tracer_tool/exclusion_aggregator.cc`):

1. `unsupported_memory_path` (file: `unsupported_memory_path.json`)
2. `unmapped_memory_range` (file: `unmapped_memory_range.json`)
3. `cross_allocation_lane_span` (file: `cross_allocation_lane_span.json`)

Consumer-side allowlist (`tools/nvbit_tracer/import_nvbit_exclusion.py:_SCHEMA_FILES`):

```python
_SCHEMA_FILES: dict[str, str] = {
    "unsupported_memory_path": "nvbit_unsupported_memory_path.schema.json",
    "unmapped_memory_range": "nvbit_unmapped_memory_range.schema.json",
    "cross_allocation_lane_span":
        "nvbit_cross_allocation_lane_span.schema.json",
}
```

Producer-side `MemoryOpClass` enum values (`instruction.proto`):

```
MEMORY_OP_CLASS_UNSPECIFIED = 0
MEMORY_OP_CLASS_GLOBAL = 1
MEMORY_OP_CLASS_SHARED = 2
MEMORY_OP_CLASS_LOCAL = 3
MEMORY_OP_CLASS_TEXTURE = 4
MEMORY_OP_CLASS_SURFACE = 5
MEMORY_OP_CLASS_ATOMIC = 6
MEMORY_OP_CLASS_TMA = 7
MEMORY_OP_CLASS_ASYNC_COPY = 8
MEMORY_OP_CLASS_CLUSTER_DSMEM = 9
```

Consumer-side `VALID_OP_CLASSES` (`import_nvbit_exclusion.py`):

```python
VALID_OP_CLASSES: tuple[str, ...] = (
    "GLOBAL", "SHARED", "LOCAL", "TEXTURE", "SURFACE",
    "ATOMIC", "TMA", "ASYNC_COPY", "CLUSTER_DSMEM",
    "UNSPECIFIED",
)
```

Producer-side `address_format` enum values
(`tools/nvbit_tracer/tracer_tool/tracer_tool.cu`):

```cpp
enum class address_format { list_all = 0, base_stride = 1, base_delta = 2 };
```

Consumer-side recognized values
(`tools/nvbit_tracer/import_trace_contract.py`):

```python
_ADDRESS_FORMAT_LIST_ALL = 0
_ADDRESS_FORMAT_BASE_STRIDE = 1
_ADDRESS_FORMAT_BASE_DELTA = 2
```

If a future producer change widens any of these enums, the consumer
side must be updated in lockstep BEFORE the live sweep runs (per
CLAUDE.md's wire-extension policy).

### Step 8 — Spot-check producer-shaped sidecars

The producer aggregator emits each typed-reason sidecar as a JSON
array. Validate one with the importer CLI:

```bash
TRACES_DIR=$(ls -td /tmp/pytest-of-*/pytest-*/nvbit_trace_contract_live*/traces 2>/dev/null | head -1)
python tools/nvbit_tracer/import_nvbit_exclusion.py \
  "$TRACES_DIR/extra_info/unsupported_memory_path.json"
```

Expected: `ok: <path> typed_reason=unsupported_memory_path entries=<N>`
where N >= 1 (the live test emits at least one ASYNC_COPY entry).

If `unmapped_memory_range.json` and `cross_allocation_lane_span.json`
exist (the producer always emits all three files even when one is
empty), validate them too:

```bash
python tools/nvbit_tracer/import_nvbit_exclusion.py \
  "$TRACES_DIR/extra_info/unmapped_memory_range.json"
python tools/nvbit_tracer/import_nvbit_exclusion.py \
  "$TRACES_DIR/extra_info/cross_allocation_lane_span.json"
```

For `simbench_bw`'s single-allocation microbench, `unmapped_memory_range.json`
and `cross_allocation_lane_span.json` are typically empty:
`ok: <path> is empty (entries=0)`.

### Step 9 — Update the goal tracker

After successful verification, edit
`.humanize/rlcr/<session-id>/goal-tracker.md`:

1. Add a new **Plan Evolution Log** row:
   ```
   | <round> | Cross-arch verification: ran the runbook on <arch> at commit <sha>;
     all unit tests, Python tests, and the live sweep PASSED with the expected
     numerics. The three flagged arch-coupling sites confirmed (compression
     distribution within per-arch acceptable shape, cp_async SASS witness
     PASSED, ASYNC_COPY sidecar admits). Captured log:
     <path-to-sweep-log>. | <reason: closes the cross-arch follow-up that
     <prior round>+ queued> | AC-4 (Cross-arch-verified) |
   ```

2. Find the most recent **Completed and Verified** AC-4 row that
   carried `Verified-live-<one_arch>` only, and update its
   `Verified Round` cell to add `, R<round> Cross-arch-verified`.
   Update the row's status line to add **`Cross-arch-verified`**.

3. Mark the corresponding `task11b-task11d-cross-arch-followup`
   Active Task as `~~completed R<round>~~` with a brief evidence
   note.

4. Move any then-resolved "sm_86 cross-arch follow-up" row in
   **Blocking Side Issues** (if present) to a `~~RESOLVED~~`
   prefix with the round number.

### Step 10 — Commit and push

The runbook itself is checked in (lives at
`tools/nvbit_tracer/CROSS_ARCH_RUNBOOK.md`); the per-session sweep
log lives under `.humanize/` (gitignored). Commit only production
files. Loop-state files (contract, summary, tracker, log) stay
local; the loop machinery preserves them per session.

If the verification round itself produced any production-file edits
(e.g., a polish commit while the verification was running), commit
those separately with the standard
"Co-Authored-By: Claude Opus 4.7 (1M context)" footer.

## Failure-Mode Debugging

### If host unit tests fail

The host C++ tests are arch-independent (they don't link any CUDA
runtime; they exercise pure C++ logic). A regression here usually
means the build path is broken or a test was left in a bad state.
Inspect:

- `g++ --version` and `nvcc --version` — must match the dev session.
- Re-source `tools/nvbit_tracer/pinned_build_env.sh` and rerun the read-only
  pin/artifact checks from Step 2. Do not clean or rebuild the canonical tree.
- The exact test that failed; line-by-line `assert` output points
  to the broken invariant.

### If Python tests report a protobuf dependency failure

```bash
git restore --source=HEAD --worktree -- workloads/simbench_bw/_pb_generated
python -m pip install -r requirements.txt
```

Missing tracked binding bytes are repository corruption and require the first
byte-exact restore. A missing/incompatible Python runtime is a separate host
prerequisite and requires the exact manifest pin. Do not generate replacement
bindings in the canonical checkout.

### If the live sweep fails

Inspect the `round-<round>-cross-arch-<arch>-sweep.log` carefully:

- `nvidia-smi` exit non-zero pre-sweep: the laptop dGPU is in deep
  idle. Wait 10 seconds and re-probe; on a desktop card, check
  whether the driver was unloaded.
- A specific test failure with `allocation_id_missing` or
  `allocation_id_unknown`: the producer regressed; cross-check the
  most recent producer-side commits.
- A distribution drift in
  `test_live_trace_compression_formats_observed_and_reconstruct`
  (e.g., `list_all` becoming non-zero on either arch): the producer's
  `enable_compress` path may have regressed.
- A failure in
  `test_phase_injection_patch_surface_guard.py` for new patch-surface
  files: a code edit introduced an `AC-N` literal in code; rewrite
  the comment in domain language.

### If a new typed reason or enum value appears on the OTHER arch

This is the asymmetric arch-coupling failure mode CLAUDE.md's
Wire-Extension Policy guards against. The producer is allowed to emit
new enum values (additive-only); the consumer must be updated to
admit them in lockstep. If you see `typed_reason_unknown` or a
`compression_format` value other than 0/1/2, the next mainline round
must:

1. Add the new value to the consumer-side allowlist
   (`_SCHEMA_FILES` for typed reasons; `VALID_OP_CLASSES` for op
   classes; the validator's `_ADDRESS_FORMAT_*` constants for
   compression formats).
2. Add the corresponding JSON Schema file or enum widening if it's
   a typed-reason or op-class addition.
3. Re-run the cross-arch verification.

## Cross-Arch Verification History

The most recent cross-arch verifications:

- Round 2 (this loop, current 5060 session): reproduced live sm_120
  sweep with persisted evidence at `round-2-live-sweep.log`. Promoted
  prior-completed AC-4 slices to `Cross-arch-verified` (since R24 of
  the prior loop had landed `Verified-live-sm_86`).
- Round 6 (this loop, current 5060 session): closed the Round 5
  consumer-side review gaps (`allocation_id_missing` validator gate,
  producer-shaped `validate_path`, allocation-sidecar exact-shape).
  Status `Verified-live-sm_120` at commit `1781304`; sm_86 follow-up
  still queued.
- Round 7 (this loop, current 5060 session): introduced this runbook
  and the Round 7 CLI ergonomics polish (`entries=<N>` on the
  `import_nvbit_exclusion` success line) at commit `50cf61a`. No
  live sweep this round — Round 7 delta over `1781304` is markdown +
  Python CLI polish only, so the Round 6 sm_120 live evidence
  remained valid.
- Round 8 (this loop, current 5060 session): partial markdown-only
  fix to the runbook target SHA. Round 7 initially pinned `1781304`
  (the commit one before this runbook was added) for the operator
  checkout, which would have removed the runbook from the worktree
  at the verification step. Round 8 changed the pin to `50cf61a` in
  the working-tree runbook. **Codex's Round 8 review identified
  this as an incomplete fix:** `git checkout 50cf61a` lands in a
  worktree where the runbook itself is the un-fixed Round 7 version
  still pinning `1781304`. Round 8's fix moved one rung further
  down the recursion ladder without breaking it.
- Round 9 (this loop, current 5060 session): **broke the recursion
  trap** with two changes: (a) repinned the verification target to
  `f6b99da` (Codex's R8-review-named target) — the verification
  surface there is correct even though the worktree's runbook text
  is still the Round 8 version; (b) added the "Source of Truth"
  subsection at the top of this runbook explaining that the
  worktree runbook at the target SHA is stale and that operators
  must keep using the runbook version they read pre-checkout (or
  `git show 5060_dev:tools/nvbit_tracer/CROSS_ARCH_RUNBOOK.md`)
  as the authoritative procedure. The trap pattern is recorded as
  `BL-20260504-runbook-self-reference` in `docs/lessons.md`.

## See Also

- `CLAUDE.md` "Two-Device Local Development Model" — the authoritative
  source for `Verified-live-<arch>` / `Cross-arch-verified` semantics.
- `tools/BLOCKED_ON_3090TI.md` — committed sm_86 issue ledger
  (carries the operator-facing entry for
  `task11b-task11d-cross-arch-followup`).
- `.humanize/rlcr/<session-id>/goal-tracker.md` — active loop's
  gitignored Active Tasks and Blocking Side Issues for the AC-4
  cross-arch follow-up. Substitute the running session id when
  reading from a worktree (e.g.
  `ls -td .humanize/rlcr/*/ | head -1`).
- `tools/nvbit_tracer/README.md` — general tracer build / runtime
  documentation.
- `tools/nvbit_tracer/import_trace_contract.py` and
  `tools/nvbit_tracer/import_nvbit_exclusion.py` — the offline
  validation surfaces this runbook exercises.
