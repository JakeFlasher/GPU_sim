# Simulator documentation map

- **Active branch:** RTX 3090 Ti / SM86 full replay. Read
  [`../plans/3090ti_full_replay_modeling_plan.md`](../plans/3090ti_full_replay_modeling_plan.md)
  first, then [`PROJECT_MEMORY.md`](PROJECT_MEMORY.md) and [`STATUS.md`](STATUS.md).
- **Active corpus:** `../sm86_rtx_3090_capture/` (356 roots; 354 contract-clean
  and 2 typed-invalid dispositions). Complete replay uses twelve workers and
  writes new generated JSONL as `.jsonl.zst`.

- [`PROJECT_MEMORY.md`](PROJECT_MEMORY.md) — concise continuation memory.
- [`STATUS.md`](STATUS.md) — current source and replay state.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — crate and event architecture.
- [`COMPONENTS.md`](COMPONENTS.md) — component map.
- [`runbook_host.md`](runbook_host.md) — complete-corpus commands.
- [`sm120_replay_bench.md`](sm120_replay_bench.md) — historical replay/NCU
  adapter notes; the active SM86 adapter/runner is specified in the branch plan.

The active source and the canonical real-trace corpus are authoritative. Old
campaign narratives and generated scratch reports are not part of the runtime
workflow. The SM120 documents and tools remain historical/shared implementation
context unless explicitly invoked by an SM86 refactor.
