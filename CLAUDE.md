# GPU simulator project context

## Mission

Build a fast, event-driven, trace-driven simulator of modern NVIDIA GPUs in
Rust. Bring up Ampere `sm_86` first, then Blackwell `sm_120`.

The intended dataflow is:

`NVBit tracer_tool` -> `traces_enhanced` protobuf -> `sim/trace` -> `sim-sm`
-> `sim-mem` -> `sim-engine` -> NCU-shaped output from `sim-run`.

NVBit/SASS traces and architecture cards are simulator inputs. NCU silicon
measurements are validation targets and must not be leaked into features for
the same scored workload.

## Research workflow

- Work on the simulator, capture tooling, experiments, and evaluation directly.
- NCU, NVBit, CUDA, Rust, Python, GPU execution, and user-authorized `sudo`
  commands are available. Use them when they are the appropriate tools.
- This is an alpha/staging development workspace. Make in-scope local changes
  and validate them boldly; generic production-risk anxiety is not a reason to
  quarantine the native tools or postpone a runnable experiment. Destructive,
  irreversible, external, or scope-expanding actions still require the usual
  confirmation boundary.
- Prefer a small focused test or experiment over repository-wide ceremony.
- Start with a leading root-cause or implementation hypothesis and inspect the
  smallest set of files, traces, and logs that can discriminate it. If a couple
  of focused probes produce no new evidence, stop broadening the search and
  either implement the best supported fix or report the best current diagnosis.
- If the user interrupts a broad investigation or asks for intuition, switch
  immediately to the best falsifiable diagnosis and one discriminating check.
- Establish observability early (progress, event/cycle counts, queue/warp
  state, errors, and useful metrics) and run the smallest representative
  workload as soon as the changed path is runnable. A real execution result is
  often more informative than a large suite of self-authored tests.
- Use existing tests and a small targeted regression test when a contract needs
  protection; do not generate speculative test pyramids. Treat tests as one
  evidence lane, alongside executable runs, invariants, differential checks,
  and measurements.
- Fix failures locally and continue. Do not start audit/reviewer swarms.
- Do not create hash gates, immutable receipts, authority/admission chains,
  frozen candidates, no-execution states, production-disable switches, Fable
  packaging, RLCR loops, or blocking lifecycle hooks.
- Do not spawn subagents unless the user explicitly asks for parallel agents.

## Optional retained capabilities

The full SOL-ExecBench and historical measurement-campaign stack is retained
for future use. Invoke `.claude/skills/run-sol-campaign/SKILL.md` when a user
requests that work. Its files may use manifests, checksums, receipts, schemas,
and historical evidence as local data formats, but they do not impose a review,
freeze, admission, or wait workflow on the simulator or any unrelated task.

`AGENTS.md` is the canonical Codex execution prompt. Apply it before this
domain-context file: it defines the outcome-first loop, bounded investigation,
early executable validation, and approval boundaries. Keep this file focused on
simulator context rather than duplicating that behavior contract.

## Useful starting points

- `docs/README.md` — documentation map
- `docs/ARCHITECTURE.md` — project architecture
- `plans/micro2027_ncu_sim_execution_plan.md` — simulator execution plan
- `plans/sm86_comprehensive_live_capture_campaign_plan.md` — capture plan
- `sim/` — Rust simulator workspace
- `tools/nvbit_tracer/` — NVBit tracer
- `tools/local_labels/` — NCU capture and label tooling
- `.claude/skills/run-sol-campaign/` — opt-in SOL/campaign capability

Ordinary checks, selected according to the change, include:

```bash
(cd sim && cargo fmt --all -- --check)
(cd sim && cargo test --all-targets)
.venv/bin/python -m pytest -q <focused paths>
```

These checks validate implementation behavior. They are not approval gates and
must not be expanded into audit protocols.
