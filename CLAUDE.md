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
- Prefer a small focused test or experiment over repository-wide ceremony.
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

See `AGENTS.md` for the binding research-first workflow policy.

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
