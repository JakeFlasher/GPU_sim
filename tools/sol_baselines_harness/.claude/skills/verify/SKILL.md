---
name: verify-sol-calibration-retirement
description: Runtime verification for fail-closed SOL legacy calibration CLI retirement without hardware or frozen-evidence mutation.
---

# Verify SOL calibration retirement

Run direct scripts from a temporary unrelated working directory with the repository `.venv/bin/python -I -B`.

1. Invoke `--help` for the retired sm86 fitted-sidecar, variant snapshot, sm120 associativity sweep, and sm120 fitted-sidecar scripts; require exit 0 and `usage:`.
2. Invoke each action with only temporary input/output paths; require `legacy_ncu_calibration_retired_targets_not_features` and no created output.
3. For the variant action, set `PATH` to a nonexistent directory to prove rejection occurs before historical Git lookup.
4. Hash tracked snapshot/sidecar/report inputs before and after and require byte identity.
5. Exercise `resolve_historical_variant_audit_sha256` through the canonical `sol_baselines_harness` package with `PYTHONPATH=<repo>/tools`; require every expected SHA to match and a synthetic wrong expected SHA to fail closed.

Never use hardware, NCU, sudo, or a tracked evidence path as an action output.
