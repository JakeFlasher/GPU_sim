# Retained campaign inventories

Files in this directory are point-in-time inventories consumed by or useful to
the opt-in SOL/campaign compatibility stack. They describe historical bytes;
they do not authorize execution, publication, commits, phases, or agent work.

The large datasets themselves remain under `data/` and
`tools/sol_execbench/data/`. A missing or stale inventory affects only a
specific historical reproduction that explicitly consumes it. It must never
freeze unrelated simulator, capture, evaluation, or research work.

Do not regenerate these inventories as a routine repository audit. For a new
experiment, record ordinary metadata appropriate to the scientific question.
