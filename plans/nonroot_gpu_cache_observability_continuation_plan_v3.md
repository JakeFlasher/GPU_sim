# Continuation Plan — Counter-Free GPU Cache Observability (sm_86 Cross-Arch Loop) — v3

## Plan Lineage

This plan supersedes
`plans/nonroot_gpu_cache_observability_continuation_plan_v2.md`.
It preserves v2's tracks (A–E) and acceptance criteria
(C-AC-* and original-plan AC-N) and adds:

- A Carried-Progress table reflecting what closed in the v2
  RLCR loop (Rounds 0–19, sessions
  `2026-05-06_19-07-40`).
- A Newly Encountered Blockers section documenting two
  loop-process issues that surfaced during v2 (the Codex
  sandbox device-masking issue and the toolchain
  framework drift to conda).
- A Mitigations section that codifies the recovery path
  (`HUMANIZE_CODEX_BYPASS_SANDBOX=true`, native
  `g++-14` from OS package manager, persistent shell-rc
  exports).
- A Restarted-Loop Discipline section setting Round-0
  expectations for the next RLCR session.

CLAUDE.md Rules 1–5 remain binding. The Two-Device Local
Development Model evidence statuses
(`Offline-verified`, `Verified-live-sm_86`,
`Verified-live-sm_120`, `Cross-arch-verified`) gate every
claim of completion.

## Carried Progress (entry to v3 loop)

The v2 RLCR loop ran 20 rounds (Round 0 through Round 19)
on the 3090 Ti host. Codex's Round 19 full-alignment
review terminated the loop with
`Mainline Progress Verdict: STALLED` and final marker
`STOP`. Session id: `2026-05-06_19-07-40`. Final commit
on the working branch (`5060_3090ti_cross_dev`):
`7b4aef6` (HANDOFF.md restoration to native gcc-14;
post-loop housekeeping).

| AC | v3 entry status | Evidence |
|----|----------------|----------|
| AC-1 | Closed in original plan; out of scope. | — |
| AC-2 / AC-3 (C-AC-2A3A) | **Active / pending** | No live sm_86 evidence captured in v2; ledger entry intact at `tools/BLOCKED_ON_3090TI.md::task7`. |
| AC-4 (C-AC-4A) | **Active / pending** | A host-local `tracer_tool.so` exists, but it is gitignored and was last rebuilt during loop-process recovery before commit `7b4aef6` landed; it must be re-verified or rebuilt at HEAD with native `g++-14` in the v3 Round 0. No live sm_86 trace contract sweep captured. |
| C-AC-A2 | **Closed (Offline-verified)** | Round 3 + Round 3 review. `tools/test_blocked_on_3090ti_hygiene.py` reports `22 passed, 80 subtests passed`; bidirectional ledger schema landed. |
| C-AC-B / C-AC-C / C-AC-D / C-AC-E | Offline-verified (carried from v2 entry). | — |
| C-AC-F2 / C-AC-G2 / C-AC-H2 | **Active / pending** | Schema-prose lockstep refined in Rounds 13–18 (HRD typed-reason taxonomy 16 → 17, slowdown schema honest about producer-side enforcement); no live sm_86 evidence captured. |
| C-AC-I2 | **Active / pending** | Closure gate at `tools/test_c_ac_i2_closure_gate.py` passes locally with GPU visible (`5 passed, 2 subtests`); the v2 Round 19 work-shell broad sweep recorded `1992 passed, 2 skipped, 678 subtests`, but that evidence was captured under the prior conda toolchain. No post-`7b4aef6` native `g++-14` broad sweep is recorded; v3 Round 0 must re-capture it. Codex's review shell could not reproduce the GPU-visible result in v2, so AC remains formally pending. |
| AC-6 / AC-7 / AC-8 / AC-10 / AC-11 / AC-14 (sm_86) | **Active / pending** | Codex sandbox prevented live verification; producer/schema/parity surfaces stable on disk. |
| AC-15 / AC-16 (sm_86 producer-side) | Offline-verified (re-tagged from v2 entry); **Active / pending** for `Verified-live-sm_86` promotion. | — |
| AC-18 (workload portion) | **Closed (Offline-verified)** | Round 0 + Round 0 review. Both workload READMEs document the live CLI shapes; `tools/test_runbook_command_verification.py` admits the workload subset. |

## v2-Loop Work Log Summary

The v2 loop's mainline progress was concentrated in two
phases:

1. **Rounds 0–3 substantive**: AC-18 workload READMEs
   (Round 0); toolchain conda-fallback added to HANDOFF.md
   (Round 1, since reverted); plan-token scanner +
   bidirectional Track A ledger landing (Rounds 2–3).
2. **Rounds 4–19 host-only cleanup arc**: scoped
   schema-prose drift fixes, HRD typed-reason taxonomy
   17-member lockstep across producer / schema / parity
   test / HRD-specific test docstrings (Rounds 13–16),
   HRD bundle-status fast-path consistency (Round 17),
   slowdown schema enforcement-layer honesty (Round 18),
   and `label_driver.sh` rejected-arch
   single-emission cleanup (Round 19).

Codex's Round 19 STALLED verdict explicitly named the
host-only cleanup arc as the symptom: technically valid
work that did not move any live-evidence acceptance
criterion. The verdict mandates that the v3 loop pivot
to substantive Track B capture work and stop clearing
queued cleanup as the round mainline objective.

## Newly Encountered Blockers (v2 loop process)

These blockers are loop-process issues surfaced by v2;
they are NOT acceptance-criterion blockers and they do not
modify the original plan's deliverables. They DO modify
the v3 loop's setup discipline. P1 and P2 are real
blockers (with mitigations); P3 is an informational
note about a Stop-hook race condition that did not
corrupt any tracked state.

### Blocker P1 — Codex review-shell sandbox masks `/dev/nvidia*`

The Stop hook spawns Codex with `--full-auto`
(deprecated alias for `--sandbox workspace-write`).
This sandbox masks `/dev/nvidia*` and reports
`nvidia-smi rc=9` in Codex's review shell, even when the
parent shell that launched Claude Code can see the GPU
(`rc=0`, `/dev/nvidia0`, `/dev/nvidiactl` present).

**Consequence in v2:** Codex's reviews repeatedly
reported the same "GPU/runtime visibility unavailable"
blocking side issue from Round 1 review through Round 19
review, even when the work shell's broad sweep was
clean. The blocker was real for Codex but absent from
the operator's perspective — a one-sided blocker that
the v2 plan did not anticipate.

**Mitigation:** the Stop hook supports an opt-out via
env var. The hook is NOT a tracked project file — it
lives in the user-global humanize plugin at
`~/humanize/hooks/loop-codex-stop-hook.sh` (`/home/<user>/humanize/`),
installed outside the repo. At commit `7b4aef6` of the
humanize plugin its lines 1189–1191 read:

```bash
CODEX_AUTO_FLAG="--full-auto"
if [[ "${HUMANIZE_CODEX_BYPASS_SANDBOX:-}" == "true" ]] || [[ "${HUMANIZE_CODEX_BYPASS_SANDBOX:-}" == "1" ]]; then
    CODEX_AUTO_FLAG="--dangerously-bypass-approvals-and-sandbox"
fi
```

Setting `HUMANIZE_CODEX_BYPASS_SANDBOX=true` in the
shell rc that launches Claude Code is expected to
switch Codex to `--dangerously-bypass-approvals-and-sandbox`,
which preserves device access — but only if the
installed Stop hook actually contains the branch above.
The flag is honest about its risk: it removes Codex's
sandbox AND approval rails, so Codex inherits the
parent's full filesystem and execution permissions, not
just `$PROJECT_ROOT`. The v3 loop MUST (a) set the env
var before Round 0 and (b) Round 0 setup MUST
fail-closed if the active hook cannot be verified
(`grep -q HUMANIZE_CODEX_BYPASS_SANDBOX
~/humanize/hooks/loop-codex-stop-hook.sh` returns
non-zero or the hook file does not exist).

### Blocker P2 — Toolchain framework drift to conda

A "Receiving-host gcc fallback (Round 1)" section was
added to `HANDOFF.md` during the v2 loop, recommending a
non-root conda-forge env at
`~/miniforge3/envs/cuda13-toolchain/` to provide
`g++-14` when the receiving host shipped only
`gcc-15` / `gcc-16`. The fallback got treated as the
canonical setup on the 3090 Ti host, leaving the 5060
laptop and the 3090 Ti on different toolchain stacks.

**Consequence in v2:** the cross-arch parity discipline
required identical toolchains; the conda env created a
silent divergence between machines. Tests that rebuild
the tracer via `subprocess.run(make, ...)` failed
on the 3090 Ti when `g++-14` was not on `$PATH` and
`NVCC_CCBIN` was not exported. Each time the codex
shell or a fresh pytest subprocess spawned, it inherited
a different env and the build behavior depended on which
shell rc was sourced.

**Mitigation:** restored to native system `g++-14`:

- Arch (3090 Ti): `yay -S gcc14` from AUR. Provides
  `/usr/bin/g++-14` (GCC 14.3.1).
- Debian/Ubuntu (5060 laptop, presumed): `sudo apt
  install -y g++-14 gcc-14`.
- Persistent shell-rc export, identical on both hosts:
  - bash/zsh: `export NVCC_CCBIN=g++-14`
  - fish: `set -x NVCC_CCBIN g++-14`
- Conda envs removed (`mamba env remove -n
  cuda13-toolchain`).
- `HANDOFF.md` rewritten in commit `7b4aef6` to
  document this discipline and the per-OS install
  recipes; the conda-fallback prose is gone.

### Blocker P3 — Stop-hook compaction race (informational)

The user invoked `/compact` while a Stop hook was
executing during Round 19, which produced two
overlapping codex review runs (the in-flight hook plus
my own debug invocation) and one overwritten cache log.
No tracked file was corrupted; the canonical
`round-19-review-result.md` survived intact. The race
is a Stop-hook resilience issue that is explicitly
deferred per v2 ("RLCR Stop-hook resilience: Not in
original plan; out of scope per user decision"). v3
preserves this deferral.

## Acceptance Criteria (preserved from v2 by reference)

The v2 acceptance-criteria set is preserved here BY
REFERENCE (not by literal inclusion) — v3 does NOT add
or remove acceptance criteria, and v3 does NOT
re-paste v2's Track A through Track E text. Both v2
and v3 must remain on disk together so v3 can be read
alongside its predecessor. The v3 deltas are scoped
to: the Carried-Progress table, the Newly Encountered
Blockers section, the Mitigations section, the
Restarted-Loop Discipline checklist, and the Carried
Queued Side Issues / BitLessons sections below.

The v2 tracks (A–E) and the v2 negative-test schema
remain authoritative for what counts as a passing
sub-criterion. The v3 loop's first three rounds should
focus on Track B mainline objectives — specifically
C-AC-4A (the gating AC for the rest of Track B) — to
move the loop's center of mass back to live evidence
capture per Codex's STALLED verdict.

Refer to
`plans/nonroot_gpu_cache_observability_continuation_plan_v2.md`
sections "Acceptance Criteria" → "Track A — sm_120-blocked
/ sm_86-blocked dual ledger" through "Track E — AC-18
closure (workload READMEs + fake-* command verification)"
for the full text. This v3 file does not duplicate that
text.

## Restarted-Loop Discipline (mandatory before Round 0)

The v3 loop's Round-0 contract MUST verify ALL of the
following preconditions BEFORE writing any code:

1. **Native toolchain pin** is in place:
   - `which g++-14` returns `/usr/bin/g++-14` (NOT a
     conda env path).
   - `echo $NVCC_CCBIN` returns `g++-14` from a
     fresh subshell.
   - `mamba env list` shows only `base` (no
     `cuda13-toolchain` or any project-specific
     conda env).
2. **Codex sandbox bypass** is enabled AND the
   active Stop hook actually supports it:
   - `echo $HUMANIZE_CODEX_BYPASS_SANDBOX` returns
     `true` from a fresh subshell.
   - The active Stop hook path is recorded
     (default `~/humanize/hooks/loop-codex-stop-hook.sh`)
     and `grep -q HUMANIZE_CODEX_BYPASS_SANDBOX
     ~/humanize/hooks/loop-codex-stop-hook.sh` returns
     rc=0; the surrounding `CODEX_AUTO_FLAG="--dangerously-bypass-approvals-and-sandbox"`
     branch must be present. If the hook is installed
     elsewhere (e.g., a fork or pinned version), record
     the actual path and verify the same way.
3. **GPU visibility** confirmed in BOTH the work shell
   AND a Codex-equivalent shell:
   - `nvidia-smi --query-gpu=name --format=csv,noheader`
     returns `NVIDIA GeForce RTX 3090 Ti` with `rc=0`.
   - `/dev/nvidia0`, `/dev/nvidiactl`,
     `/dev/nvidia-uvm` all present.
   - `python3 -m pytest tools/test_c_ac_i2_closure_gate.py
     -q` returns `5 passed, 2 subtests passed` (the
     visibility-check tests admit; without GPU they
     fail with `3 failed, 4 passed`).
4. **Broad sweep clean** with native toolchain:
   - `python3 -m pytest tools/ workloads/simbench_bw/tests/
     -q` returns `1992+ passed, ~2 skipped, 678+
     subtests`. The exact count may rise if the v2
     loop landed new tests; it must not fall.
5. **Codex review subshell honors the bypass**:
   - During Round 0 review, Codex's review-shell
     `nvidia-smi` reports `rc=0` (NOT `rc=9`), and
     the broad sweep matches the work-shell count.
6. **HANDOFF.md is current** at commit `7b4aef6` or
   later: contains the native gcc-14 install recipes,
   no conda fallback prose.

If any precondition fails, Round 0's contract
documents the gap and resolves it BEFORE the round's
mainline work. Do not proceed with Track B capture
work while preconditions are red.

## v3-Loop First Mainline Targets (recommended Round 0)

Per Codex's Round 19 STALLED verdict, the v3 loop must
not spend its first round on host-only cleanup. The
recommended Round-0 mainline objective is **C-AC-4A**
(NVBit trace contract sm_86 live evidence), because:

- It is the gating AC for Track B (most of Track B
  depends on a working sm_86 tracer).
- It is offline-deferrable for parts that do not need
  live capture (the helper-section resolver static
  unit tests, the live-witness fixtures), but the live
  sweep itself is the actual deliverable.
- Round 0 must rebuild the tracer with native `g++-14`
  after commit `7b4aef6` and verify the cubin set with
  `cuobjdump --list-elf
  tools/nvbit_tracer/tracer_tool/tracer_tool.so | grep cubin`
  (expect sm_75 / sm_80 / sm_86 / sm_87 / sm_88 / sm_120
  given `-arch=all`); ignored host-local build artifacts
  from earlier rounds must NOT be treated as fresh HEAD
  evidence.
- Sweep evidence persisted into
  `.humanize/rlcr/<v3-session-id>/round-N-cross-arch-sm_86-sweep.log`
  per `tools/nvbit_tracer/CROSS_ARCH_RUNBOOK.md` Step 5.

After C-AC-4A closes, Track B's dependent ACs (C-AC-F2,
C-AC-G2, C-AC-H2, AC-6, AC-7, AC-8, AC-10, AC-11, AC-14)
can land in dependency order. Track D (AC-15 / AC-16
producer-side live evidence) is optional in this loop
and only after Track B's producer surfaces are stable.

## Carried Items from v2 (out of scope, preserved)

| Item | Rationale | When to reconsider |
|------|-----------|--------------------|
| AC-12 server collection bootstrap | Server work out of project scope until BOTH local devices pass `Cross-arch-verified`. | After v3 closes Track B. |
| AC-16 server-architecture NVML | Same scope rationale as AC-12. | Same trigger. |
| AC-18 cloud `tools/cloud/RUNBOOK.md` | Same scope rationale as AC-12. | Same trigger. |
| RLCR Stop-hook resilience | Not in original plan; out of scope per user decision. | A future plan that explicitly scopes RLCR-process resilience. |

## Carried Queued Side Issues from v2

These remain queued (not blocking) and must NOT replace
the v3 Round-0 Track B objective. Each carries forward
its v2 revisit trigger; if the trigger fires during a
v3 round, the issue gets prioritized into that round's
contract on its own merits.

| Issue | Revisit Trigger |
|-------|-----------------|
| Event-based async-free retirement for stream-ordered allocator hooks | When a producer-side allocator-hook regression surfaces in a future Track B round. |
| Per-source NVBit drop counters beyond residual expected-record mismatch | When C-AC-H2 sm_86 round runs against the live trace budget. |
| Workload-side `wall_time_ms` emission for `dram_stream_stride_mix` | When AC-16 sm_86 round runs the timing capture. |

## Carried BitLessons from v2

The v2 loop added `BL-20260506-schema-validator-probe-driven-tightening`
in Round 10 (multi-round schema-prose / probe-driven tightening pattern).
It lives in `.humanize/bitlesson.md` (gitignored; preserve across raw scp
per CLAUDE.md, but does NOT survive `git fetch` alone). The four prior
BitLessons (`BL-20260504-runbook-self-reference`,
`BL-20260505-tools-subdir-module-name-collision`,
`BL-20260505-cross-arch-sass-mnemonic-divergence`,
`BL-20260504-runbook-command-verification`) are also carried.

## Out-of-scope for the v3 loop (carried from v2; do not displace mainline)

- Server-architecture NVML / cloud bootstrap.
- Simulator revival (CLAUDE.md Rule 5 binding).
- Driver-root operations (CLAUDE.md Rules 2-4 binding).
- NCU on cloud/server GPUs (CLAUDE.md Rule 1; local
  exception remains for `tools/local_labels/`).
- Any reorganization of `.humanize/rlcr/` from a previous
  loop (the v2 directory at `.humanize/rlcr/2026-05-06_19-07-40/`
  is host-local historical reference; v3 starts a fresh
  session id).

## Risks (updated)

- **Track B live capture surfaces arch-coupling** — the
  cache-modifier SASS regex
  (`BL-20260505-cross-arch-sass-mnemonic-divergence`)
  already admits both Ampere and Blackwell forms; new
  arch-coupling findings should land as BitLesson
  entries with producer/verifier lockstep changes.
- **Codex's `--dangerously-bypass-approvals-and-sandbox`
  removes safety rails** — Codex can write/execute
  anywhere with the parent's permissions. For our
  read-mostly research-repo review use case this is
  bounded but it is honestly named.
- **First-round host-shell coupling** — the GPU device
  permissions on the 3090 Ti host depend on the user
  session being in the `video` group. If GPU access is
  lost mid-loop (e.g., session restart, suspend/wake
  cycle), Round-0 preconditions must be re-verified
  before continuing.
- **Cross-arch session pivot** — when this loop
  eventually hands off to a 5060 laptop session for
  the `Cross-arch-verified` promotions of Track B, the
  receiving session must follow `HANDOFF.md` first
  actions including the AUR-vs-apt branch in step 3
  ("Toolchain framework discipline").

## Naming Convention

Preserved from v1/v2: continuation-specific criteria
use prefix `C-AC-`; references to the original plan's
criteria keep their unprefixed `AC-N` form.
