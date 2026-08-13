# Label-tool scope

The direct NCU entry point is `collect_ncu.py`; `metrics.py` and `ncu_csv.py`
provide small reusable parsers. Other files retain the complete historical
campaign implementation and tests for opt-in use through `run-sol-campaign`.

Legacy names such as authority, admission, receipt, seal, audit, freeze,
protected path, and gate describe compatibility formats or old workflows.
They never authorize tools, block unrelated work, require review agents, or
freeze the repository. Root `AGENTS.md` controls.
