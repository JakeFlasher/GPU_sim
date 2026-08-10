# Evaluator prototype

The offline P2 evaluator consumes the frozen exact 7+1 metric projection and
strict `sim_result.v2` artifacts. Seven secondary metrics are scorable;
`exec_cycles_gpc_max` remains `required_unimplemented` on
`aggregation_semantics_study` and is never numerically scored.

Matched labels require authority independent of the label document. No
production matched-label/crosswalk authority is installed, so the production
loader fails closed with `LABEL_ADMISSION_AUTHORITY_UNAVAILABLE` and does not
open a label payload. The only active scoring lane uses checked-in, code-pinned
synthetic fixtures under `eval/tests/fixtures/evaluator_mvp/`. Those fixtures
confer no hardware, accuracy, or publication credit and do not weaken the
refusing `eval.production_evaluator` boundary. The evaluator build identity is a
separate code-pinned synthetic input; the scoring process never invokes Git and
has no ambient provenance fallback.

Run the offline canary from the repository root:

```bash
.venv/bin/python -m eval.score \
  evaluator-mvp-canary-v1 \
  eval/tests/fixtures/evaluator_mvp/sim_result.v2.json \
  eval/tests/fixtures/evaluator_mvp/matched_labels.v1.json \
  eval/tests/fixtures/evaluator_mvp/synthetic_matched_label_authority.v1.json \
  eval/tests/fixtures/evaluator_mvp/synthetic_evaluator_build_identity.v1.json
```

Success is one canonical sealed `accuracy_report.v1` line on stdout with empty
stderr. A typed CLI/input failure is one canonical error line on stderr with
empty stdout. No real label path is accepted by this fixture-only entry point.

## Current-simulator runtime recording

`eval.bench` records external runtime for the repository-owned release
`sim/target/release/sim-run` only. It has no binary override, never builds inside
the timing boundary, does not import or interpret labels, and never queries,
invokes, or produces a baseline. The caller-supplied input is opened only as a
strict `sim_run_manifest.v2` candidate and is rejected if that contract fails.
Build separately before recording:

```bash
cargo build --release --manifest-path sim/Cargo.toml -p sim-run
.venv/bin/python -m eval.bench record \
  /absolute/path/to/run_manifest.v2.json \
  /absolute/path/to/new-runtime-record.v1.json
```

The release inode, manifest inode, and manifest-parent directory are retained
before any repeat. `sim-run` receives the manifest and evidence base only through
one canonical inherited-descriptor argument; path components are opened without
following symlinks, and relative launch evidence stays anchored beneath the
retained base.

The fixed protocol retains one summary-excluded warm-up and five timed repeats.
If the sample coefficient of variation is greater than 5%, it retains four more
repeats; a final value over 5% is `RUNTIME_NOISY`. Timing begins immediately
before spawning the retained executable and ends at exact-child `wait4` return.
Each sample binds external wall time, child user/system CPU, peak RSS, observed
load average, termination, and bounded stdout/stderr artifacts. Successful result
bytes must be canonical `sim_result.v2`, bind the retained binary and canonical
manifest, and remain byte-identical across every warm-up and timed repeat.

Each successful result also binds the selected-roster replay count from the exact
aggregate instruction metric; the checked replay invariant equates issued
instructions and processed records, so throughput never uses whole-bundle counts
when a manifest selects only part of a trace bundle.

Publication is exclusive: an existing record or sample directory is never
replaced. `load_runtime_record` accepts only records bound to the current
checkout's fixed release path and revalidates the record seal, manifest,
executable, all sample bytes, derived statistics, repeat/noise decision, and every
retained successful result variant. The active v1 lane is explicitly exploratory.
Dirty builds are `DIRTY_BUILD_UNCLAIMABLE`; clean builds remain
`WORKLOAD_TIER_AUTHORITY_UNAVAILABLE` because admitted T-mid/T-big workload
authority is not installed. The checked-in address-carry canary is fixture-only
and cannot support a runtime or speed-up claim. Production accuracy links and
read-only baseline joins remain unavailable until their independent authorities
land.

## Target-free CPU profiling

`eval.profile` records a user-space Linux `perf` profile of the fixed repository
release `sim-run`; it never accepts target, scoring, comparison-database, runtime
record, or same-cell tuning inputs. Build the symbolized release-codegen binary
first, then publish to a new generation path:

```bash
cargo build --release --manifest-path sim/Cargo.toml -p sim-run
.venv/bin/python -m eval.profile record \
  /absolute/path/to/run_manifest.v2.json \
  /absolute/path/to/new-profile-generation \
  /absolute/path/to/anchored-clean-source-gate-generation \
  64-hex-external-content-manifest-anchor
```

The fixed protocol forces the documented `PERF_CONFIG=/dev/null` configuration-
isolation mechanism in the closed environment for every child. It first requires
`perf config --list` to succeed with byte-empty stdout and stderr, then obtains the
version with the supported `perf --version` grammar, runs a directly supervised
determinism reference, executes `perf record` with `cycles:u` and DWARF call
graphs, and finally emits a symbolized canonical-text `perf report`. No unsupported
global `--no-config` option is used. The producer has exact-child
`waitid`/`wait4` authority for the direct reference and each retained `perf`
wrapper. It does **not** claim an independently attached PID for the sim-run
descendant created by `perf`: that subject is bound to the retained release inode
through a parent `/proc/<pid>/fd/<fd>` capability, and successful termination is
accepted only through zero-exit perf-wrapper completion plus a valid,
byte-identical simulator result and authenticated perf data; the perf session
process group is terminated before the exact wrapper is reaped, without
inventing separate PID authority for the descendant. The raw `perf.data`
inode remains retained while report generation consumes that exact inode through
the same capability pattern; path replacement or restore is input drift.

The direct and profiled `sim_result.v2` bytes must be identical; variable host
phase diagnostics live only in `sim_profile_record.v1`. An independently
published clean-source gate generation and its externally supplied content-
manifest SHA-256 are mandatory. The producer revalidates that authority before,
during, and immediately before publication, retains and hashes `eval/profile.py`
and its schema, and rejects a dirty or stale release result whose build commit
does not equal the clean source commit. The profiler, release binary, manifest,
result build/commit, trace, cards, configuration, Linux kernel, CPU, and affinity
identities are hash-bound, and all invocations use one closed audited environment
that rejects secret-bearing names. The record and validator bind the exact
`PERF_CONFIG=/dev/null` assignment, the empty effective-config probe, each supported
argv, and the common environment hash. Ambient `HOME`, `XDG_CONFIG_HOME`, or caller
`PERF_CONFIG` values are never inherited. No config-enabled, software-event,
unsymbolized, pathname-substitution, or subject-PID fallback is permitted.

A successful output is one exclusive, no-replace, read-only directory containing
raw `perf.data`, `perf.report.txt`, exact simulator result bytes, retained perf
diagnostics, and the sealed profile record. `load_profile_record` recomputes the
closed artifact inventory and hashes and revalidates current profiler, binary,
and manifest identities. Setup and execution failures remain typed as
unavailable, permission-denied, unsupported, failed, timeout, overflow,
nondeterminism, or input drift. The profiler never invokes real hardware during
the fake-perf test suite.
