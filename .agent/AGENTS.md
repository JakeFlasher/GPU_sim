# `.agent/` local Codex profile

Codex loads this file after the repository root `../AGENTS.md`. The root file
is canonical; this file is intentionally additive and must not introduce a
competing workflow or weaken higher-priority instructions.

## Local emphasis

- Treat this as an alpha/staging engineering workspace. For an implementation
  request, work in the smallest useful slice, preserve unrelated changes, and
  use the available native GPU/toolchain commands instead of waiting for a
  production-style approval loop.
- Keep the root loop visible in decisions: goal and acceptance bar → leading
  hypothesis → narrow execution seam → minimal patch → bounded observability →
  representative executable run → only the checks that materially support the
  bar → stop and report.
- Do not broaden a search after a small bounded number of probes (normally no
  more than two) yields no new evidence. If the user interrupts or asks for
  intuition, give the best falsifiable diagnosis and one discriminating check.
- Prefer a real smoke run and focused regression checks over a speculative test
  pyramid. Do not spawn subagents or reviewer swarms unless the user explicitly
  requests parallel work. Use an external review only when explicitly
  requested or authorized, and keep it bounded and actionable.
- Routine local reads, edits, native execution, and non-destructive validation
  are authorized for requested changes. Ask before external writes,
  destructive/irreversible actions, material privileged side effects, or scope
  expansion.
- Preserve the simulator evidence boundary: NVBit/SASS traces and architecture
  cards are inputs; NCU silicon measurements are independent targets and must
  never become hidden features, tuning inputs, implementation inputs, or
  training data for the scored workload.
- Preserve independent issue, modeled-event, completion, launch-envelope, and
  typed-tail frontiers as separate evidence. Never turn them into a summed
  scalar selection score; use a mechanism-local or explicit partial-order rule.
- Treat mtime freshness as insufficient for retained replays; bind simulator,
  card, trace/control, and output bytes to a content-identity record.
- Pin host-specific GPU name/UUID/compute capability/driver/index instead of
  accepting a self-declared hardware identity.
- At handoff, include the causal mechanism for a nontrivial fix, commands and
  observed results, uncertainty/caveats, and the next concrete action if any.
