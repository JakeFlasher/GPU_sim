# BitLesson Knowledge Base

This file is project-specific. Keep entries precise and reusable for future rounds.

## Entry Template (Strict)

Use this exact field order for every entry:

```markdown
## Lesson: <unique-id>
Lesson ID: <BL-YYYYMMDD-short-name>
Scope: <component/subsystem/files>
Problem Description: <specific failure mode with trigger conditions>
Root Cause: <direct technical cause>
Solution: <exact fix that resolved the problem>
Constraints: <limits, assumptions, non-goals>
Validation Evidence: <tests/commands/logs/PR evidence>
Source Rounds: <round numbers where problem appeared and was solved>
```

## Entries

## Lesson: vacuous-pass-over-zero-items
Lesson ID: BL-20260708-vacuous-trace-validator
Scope: any fail-closed verdict computed by looping/counting over a set of items — tools/local_labels/campaign_preflight.py (injection smoke over threadblock records); tools/local_labels/sol_reconstitution.py (runnability over workload traces); tools/nvbit_tracer/import_trace_contract.py (validate_trace_dir); the future sim/trace reader completeness check.
Problem Description: A pass/ok verdict computed by iterating a collection is VACUOUS when the collection is empty — the loop simply never runs, so "all passed" is trivially true over zero items. Two live instances: (1) `validate_trace_dir` returns ok=True on a present-but-EMPTY `threadblocks/` (zero `.pb` files), and gating on it + a nonzero admitted count from the SEPARATE `stats.csv` still false-PASSes; (2) `classify_cli_outcome` returned RUNNABLE for a benchmark with 0 counted workloads (missing/differently-named `workload.jsonl`) because the short-trace guard `len(traces) < expected_workloads` is `N < 0` = False and "all passed" holds over an empty non_passed set ("N/0 workloads passed").
Root Cause: the "all items passed" predicate does not require items >= 1; and when the item count comes from a SEPARATE source than the verdict (stats.csv vs threadblocks; workload.jsonl vs CLI traces), the two can disagree and a degenerate/empty case slips through.
Solution: require the count of items the verdict actually ranged over to be >= 1 BEFORE admitting — `threadblocks_pb_count >= 1` (same glob the validator walks), and `expected_workloads >= 1` (else EVAL_INCOMPLETE). Compute the count from the SAME source the verdict uses, gate it in the verdict (not merely record it), and mirror it in the schema where applicable.
Constraints: the count must be over the exact items the verdict inspects; a count from a decoupled aggregate file is not a substitute.
Validation Evidence: `injection_smoke_verdict` + `test_injection_smoke_empty_threadblocks_blocks` + `test_schema_rejects_ok_injection_with_zero_threadblocks` (R1); `classify_cli_outcome` + `test_zero_expected_workloads_never_runnable` (R2). Adversarially surfaced + re-verified in workflows wsj35awey/wdut377h5 (R1) and wf_247bb1e8 (R2).
Source Rounds: 1, 2

## Lesson: catch-all-collapses-typed-failures
Lesson ID: BL-20260708-catchall-mistypes-failures
Scope: any classifier mapping an execution outcome to a typed reason — tools/local_labels/sol_shrink_apply.py (classify_shrink_result); tools/local_labels/build_capture_roster_authority.py (shrink→exclusion-reason routing); the exclusion ledger and the future AC-9 admissibility gates.
Problem Description: A catch-all `else` (or "not runnable and not X → Y" fall-through) silently mis-types every failure not explicitly enumerated into ONE bucket. Concretely: `classify_shrink_result` returned `unavailable::oom_shape_gated` (a CAPACITY verdict) for ANY non-runnable reduced status other than `host_capacity_oom` — so a reduced-shape `correctness_failed` (a kernel defect) and a `compile_failed` were both mislabeled capacity deferrals and routed to `host_capacity_deferred`, corrupting the exclusion ledger (2 cells claimed capacity-gated when actually just wrong).
Root Cause: (1) terminal cases were not each enumerated to their TRUE typed reason; (2) the downstream guard checked the LABEL (`classification == unavailable`) instead of the STRUCTURED condition that label requires (the reduced run actually being capacity/shape-gated).
Solution: enumerate every terminal outcome to its true typed reason (correctness→correctness_failed; compile/eval→calibration_inconclusive; only OOM/no-safe-axis→unavailable); AND make the fail-closed guard verify the SPECIFIC structured precondition (`reduced_run_status ∈ {host_capacity_oom, no_safe_axis}` required before `host_capacity_deferred`), so a mislabeled verdict fails closed to the weaker reason rather than an unproven strong one.
Constraints: the structured precondition must be a first-class recorded field (not re-parsed from a detail string); the guard belongs at the consumer (roster), not only the producer.
Validation Evidence: `test_reduced_correctness_fail_is_correctness_not_capacity`, `test_host_capacity_shrink_correctness_fail_is_correctness_not_deferred`, `test_unavailable_label_without_capacity_status_fails_closed`, `test_no_host_capacity_deferred_without_capacity_gated_shrink`. Surfaced by Codex R3 review.
Source Rounds: 4

## Lesson: tracer-proto-change-is-a-multi-surface-lockstep-rebuild
Lesson ID: BL-20260708-tracer-proto-lockstep-rebuild
Scope: any change to tools/nvbit_tracer/traces_enhanced/dynamic_trace/*.proto (the trace wire schema) and rebuilding tools/nvbit_tracer/tracer_tool/tracer_tool.so.
Problem Description: A one-field proto change fans out across FIVE committed surfaces that must move in the SAME commit, and the tracer rebuild has two non-obvious traps. (1) Surfaces: the .proto; the committed C++ bindings (pb_trace/src+include, regen `make -C traces_enhanced` with the PINNED protoc); the committed Rust bindings (`sim/trace/regen_pb.sh`) AND the FNV pin table in `sim/trace/tests/proto_schema_pin.rs`; the Rust struct-LITERAL construction sites (reader.rs/lanes.rs/tests) which fail to compile with `missing field` until the new field is added; the tracked Python bindings (`workloads/simbench_bw/_pb_generated/*_pb2.py`), which are committed authorities restored byte-exactly from Git and never regenerated in a canonical worktree. (2) Rebuild trap A: the tracer .o compile needs the PINNED protobuf 34.1 RUNTIME headers or it fails with `SetHasBitForRepeated was not declared` / a cross-version-runtime `#error` — the committed .pb.h already use 34.x features, but the Makefile's `INCLUDES=-I$(NVBIT_PATH)` omits the protobuf include, and the host default is an older protobuf. (3) Rebuild trap B: passing `INCLUDES=...` on the make command line PROPAGATES to the `traces_enhanced` sub-make and CLOBBERS its own `-I. -I./pb_trace/include`, breaking proto regen with `*.pb.h: No such file`.
Root Cause: the trace wire schema is consumed by C++/Rust/Python producers+validators simultaneously; and the pinned-toolchain protobuf is wired via env (PKG_CONFIG_PATH/LD_LIBRARY_PATH) but the compiler HEADER search path for the pinned protobuf is not set by the Makefile.
Solution: after editing the .proto — regenerate the C++/Rust/Python outputs only in an explicitly authorized non-Git temporary copy with the pinned environment, then land the reviewed byte-exact outputs and repin the Rust FNV hashes (run `cargo test -p sim-trace --test proto_schema_pin`, paste the printed `on-disk` hash); canonical Git worktrees restore tracked Python bindings from Git and never run generation. Add the field to every Rust struct literal. The repaired Makefiles now use private `TRACER_INCLUDES` / `TRACE_PROTO_INCLUDES` variables and obtain protobuf CFLAGS from relocated pkg-config metadata, while `pinned_build_env.sh` exports pinned `PROTOC`; rebuild through the top-level `make -C tools/nvbit_tracer` only in that authorized temporary copy. For historical commits before that repair, `CPLUS_INCLUDE_PATH="$PB_PREFIX/include"` plus explicit pinned `PROTOC` was the non-propagating workaround. Never pass a propagating `INCLUDES=` override. Verify the nvbit_tracer pytest, Rust workspace, tracer C++ unit-tests, and live preflight.
Constraints: additive only (proto3 `optional`/enum-append, no schema_version bump); a new cross-message field that would create a circular proto import (e.g. reusing MemoryOpClass from instruction.proto inside address.proto, which instruction.proto imports) must instead define a dedicated local enum.
Validation Evidence: commit 14f86c5a (address-carrying contract, atomic leg); the two failed rebuilds this round diagnosed traps A and B.
Source Rounds: 7

## Lesson: guard-must-be-on-the-consumer-path-not-only-a-helper
Lesson ID: BL-20260708-guard-on-consumer-path
Scope: any fail-closed guard/validator added to a multi-entry subsystem — the sim/trace reader (TraceDir::read_threadblocks_in_launch_order vs sim/sm replay::extract_per_sm_blocks); import guards; admissibility gates with more than one read/consume entry point.
Problem Description: A newly-added identity/validation guard was placed in ONE reader entry point (`TraceDir::read_threadblocks_in_launch_order`) and unit-tested there (helper passes), but the SIMULATOR — the actual consumer — reached the data through a DIFFERENT entry (`replay::extract_per_sm_blocks` → `read_threadblock` + `visit_memory_access`) that never called the guard. So a malformed global/atomic/LDGSTS-source record could still reach replay classification even though the helper + its unit tests were green. The round summary claimed "the sim never consumes a malformed record" — false for the real path.
Root Cause: the guard was written + tested as a helper and wired into the most obvious reader method, but the subsystem has MULTIPLE consume paths; the one the downstream code actually takes was not routed through it. Unit tests on the helper prove the guard works, not that the consumer invokes it.
Solution: enforce the guard on the path the consumer ACTUALLY takes (call `check_allocation_identity` per record inside `extract_per_sm_blocks` using the same registry), and write the regression at the CONSUMER level (drive `extract_per_sm_blocks`/`replay_kernel` on a malformed input and assert it fails before classification) — not only a helper-level unit test. Grep every read/consume entry point for the guard call.
Constraints: the regression must exercise the consumer entry (extract/replay), not just the helper; keep the legacy/absent-sidecar fallback intact so valid + old traces still pass.
Validation Evidence: `replay_rejects_malformed_global_atomic`, `replay_rejects_malformed_ldgsts_source` (sim-sm, drive extract_per_sm_blocks) + `sidecar_parser_enforces_python_parity`; commit 8328f8e2. Surfaced by Codex R9 review.
Source Rounds: 9, 10
