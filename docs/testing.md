# Test boundaries

The default test command covers the simulator, trace and label pipeline,
calibration references, and lightweight SimBench contracts:

```bash
.venv/bin/python -m pytest -q
```

Tests marked `campaign_evidence` require the immutable external capture archive
and are run through a gate that fails when the archive is unavailable:

```bash
tools/dev/campaign_evidence_gate.sh
```

Tests marked `hardware_evidence` require machine-local GPU traces or control-bit
sidecars. They are excluded from the default CPU gate and must never pass merely
because evidence is absent. Their hard-failing runner is completed in cleanup
Stage 5.

The heavy SOL-ExecBench/FlashInfer evaluation bundle has its own environment
under `tools/sol_execbench/` and is not collected by the default repository
suite.
