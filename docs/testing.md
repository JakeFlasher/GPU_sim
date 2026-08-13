# Testing

Tests answer whether code behaves as intended. They do not authorize later
work.

Use the smallest relevant check while iterating:

```bash
cd sim
cargo test -p sim-engine
cargo test -p sim-trace
cargo test -p sim-sm
cargo test -p sim-mem
cargo test -p sim-run

# Optional retained campaign runner and evidence adapter
cargo test -p sim-run-campaign
cargo test -p sim-sm --features campaign --test launch_evidence
```

Python examples:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q eval/tests/test_direct_score.py
.venv/bin/python -m pytest -q arch_profiles/tests
.venv/bin/python -m pytest -q tools/nvbit_tracer/test_capture_process.py
```

Before a broad handoff, run the suites touched by the change. Hardware tests
may use the available GPUs, NCU, NVBit, clocks, and administrative commands.
A skipped hardware test is simply not executed; it neither passes nor blocks
unrelated implementation.

The full SOL/campaign suites are intentionally excluded from bare `pytest`.
Invoke them by path through `.claude/skills/run-sol-campaign/SKILL.md` when that
capability is in scope.

Do not create audit-only test suites that inspect Git state, prompt text,
reviewer counts, hashes, receipt chains, or workflow status.
