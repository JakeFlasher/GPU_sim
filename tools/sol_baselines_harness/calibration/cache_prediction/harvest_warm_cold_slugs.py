"""Direct-NCU-harvest driver for the two warm/cold-promoted SOL
problem slugs (problem_018 / problem_071). Operator-facing helper
intended to run ONCE on the local sm_86 3090 Ti to lift the two
slugs from the intermediate `corpus_expansion_pending /
ncu_labels_missing` diagnostic state into the final
`measured / predictor_hostile / strict < 0.10 rel-err` state.

Per `.humanize/predictor-regressions/warm_cold_protocol_pin.md`
section 5: the binary at `baselines/L1_direct_ncu_binaries/<slug>`
is the deployed baseline compiled as a standalone calibration
executable (not a cohort-B surrogate probe). `ncu --launch-skip 5
--launch-count 1` lands on the 6th invocation of the single target
kernel (BL-20260521-ncu-launch-index-single-target-probes).

Invocation:

    PYTHONPATH=scripts .venv/bin/python \\
        scripts/baselines_harness/calibration/cache_prediction/\\
        harvest_warm_cold_slugs.py

The script:
  1. Discovers landed `*.cu` sources under `baselines/L1/`.
  2. Filters to just the two warm/cold-promoted slugs.
  3. Builds sweep specs against
     `baselines/L1_direct_ncu_binaries/<slug>` (the direct-harvest
     binaries the sibling Makefile produces).
  4. Runs the calibration sweep (NCU read-only sector hit-rate
     harvest, kernel-replay mode, no clock-control mutation).
  5. Loads the existing `label_ledger_v1_full_partition.toml`,
     strips the two placeholder `[[unavailable]]` rows for these
     slugs, merges the two freshly-harvested `[[entries]]` rows,
     and re-serializes the ledger via the strict TOML serializer.

The script is idempotent: re-running it overwrites the two
[[entries]] rows with the freshly-harvested values (binary_sha256
+ raw_csv_sha256 + provenance are recomputed from the actual
binary on disk).
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[4]

    # Late import so the module can be imported without dragging the
    # heavy calibration package's transitive deps when only the
    # docstring is read.
    sys.path.insert(0, str(repo_root / "scripts"))
    from baselines_harness.calibration.cache_prediction.calibration_sweep import (
        run_calibration_sweep,
        load_ledger_from_toml,
        serialize_ledger_to_toml,
        LabelLedger,
    )
    from baselines_harness.calibration.cache_prediction.run_ncu_harvest_batch import (
        discover_landed_baselines,
        build_sweep_specs_from_landed_sources,
    )

    target_slugs = (
        "problem_018_fused_rope_with_qk_norm_and_kv_cache_update",
        "problem_071_kv_cache_update_with_rope",
    )

    landed = [
        source for source in discover_landed_baselines(repo_root)
        if source.slug in target_slugs
    ]
    if len(landed) != len(target_slugs):
        landed_slugs = {s.slug for s in landed}
        missing = sorted(set(target_slugs) - landed_slugs)
        print(
            f"ERROR: landed-baseline discovery missed slugs: "
            f"{missing}; expected both warm/cold-promoted slugs "
            f"to have hand-ports under baselines/L1/<slug>/cuda_cpp/",
            file=sys.stderr,
        )
        return 2

    probe_binary_root = repo_root / "baselines" / "L1_direct_ncu_binaries"
    specs = build_sweep_specs_from_landed_sources(
        landed, probe_binary_root=probe_binary_root,
    )
    missing_binaries = [
        s.cell_name for s in specs if s.binary_path is None
    ]
    if missing_binaries:
        print(
            f"ERROR: direct-harvest binaries missing for: "
            f"{missing_binaries}. Build them with `make ARCH=sm_86` "
            f"under {probe_binary_root}",
            file=sys.stderr,
        )
        return 2

    print(
        f"Running NCU harvest for {len(specs)} slug(s):",
        file=sys.stderr,
    )
    for s in specs:
        print(f"  {s.cell_name} -> {s.binary_path}", file=sys.stderr)

    new_ledger: LabelLedger = run_calibration_sweep(
        specs,
        ncu_metric_version="ncu-2026.1",
        host_arch="sm_86",
        host_descriptor_hash="local_sm_86_3090ti",
        ncu_version="2026.1.0",
        driver_version="595.71.05",
    )

    print(
        f"Harvest result: {len(new_ledger.entries)} entries, "
        f"{len(new_ledger.unavailable)} unavailable rows",
        file=sys.stderr,
    )
    for e in new_ledger.entries:
        label = e.label
        print(
            f"  ENTRY {e.cell_name}: "
            f"l1_hit_rate={label.l1_global_load_hit_rate:.4f} "
            f"l2_hit_rate={label.l2_read_hit_rate:.4f}",
            file=sys.stderr,
        )
    for u in new_ledger.unavailable:
        print(
            f"  UNAVAILABLE {u.cell_name}: {u.unavailable_reason}",
            file=sys.stderr,
        )

    existing_ledger_path = (
        repo_root
        / "scripts/baselines_harness/calibration/cache_prediction/"
        / "label_ledger_v1_full_partition.toml"
    )
    existing_ledger = load_ledger_from_toml(existing_ledger_path)

    # Strip placeholder `[[unavailable]]` rows AND any pre-existing
    # `[[entries]]` rows for the target slugs before merging — the new
    # ledger carries fresh rows that supersede whichever bucket the
    # slugs sat in previously. Without the `[[entries]]` strip, the
    # merge would emit duplicate cell names and the strict
    # serializer would reject. The previous bucket (entries vs
    # unavailable) is intentionally not preserved: the re-harvest
    # represents a fresh observation against the current binary
    # bytes, and the new ledger row supersedes the old one.
    stripped_unavailable = tuple(
        u for u in existing_ledger.unavailable
        if u.cell_name not in target_slugs
    )
    stripped_entries = tuple(
        e for e in existing_ledger.entries
        if e.cell_name not in target_slugs
    )
    cleaned_existing = LabelLedger(
        schema_version=existing_ledger.schema_version,
        captured_at_utc=existing_ledger.captured_at_utc,
        host_arch=existing_ledger.host_arch,
        host_descriptor_hash=existing_ledger.host_descriptor_hash,
        ncu_version=existing_ledger.ncu_version,
        driver_version=existing_ledger.driver_version,
        entries=stripped_entries,
        unavailable=stripped_unavailable,
    )

    # Manual merge: the strict merger in run_ncu_harvest_batch
    # rejects ANY new cell_name overlap (entries-vs-entries OR
    # entries-vs-unavailable) but we just stripped the
    # unavailable conflicts above. Use the same primitive shape.
    merged_entries = list(cleaned_existing.entries) + list(new_ledger.entries)
    merged_unavailable = list(cleaned_existing.unavailable) + list(new_ledger.unavailable)

    final_ledger = LabelLedger(
        schema_version=cleaned_existing.schema_version,
        captured_at_utc=cleaned_existing.captured_at_utc,
        host_arch=cleaned_existing.host_arch,
        host_descriptor_hash=cleaned_existing.host_descriptor_hash,
        ncu_version=cleaned_existing.ncu_version,
        driver_version=cleaned_existing.driver_version,
        entries=tuple(merged_entries),
        unavailable=tuple(merged_unavailable),
    )

    serialize_ledger_to_toml(final_ledger, existing_ledger_path)
    print(
        f"Merged ledger written to {existing_ledger_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
