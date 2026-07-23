"""Emit
``data/local_labels/audit_evidence/ac6_label_stability_diagnostic/label_stability_table.json``.

Per (round_number, cell_id) this records the per-cell facts
that the scanner test pins against fresh CSV reads:

- ``min_stddev_l2_sector`` over rows where ``n_sidecars > 0``.
- ``best_warmup_count`` / ``best_cache_control`` for the
  minimum.
- ``admitted_config_count`` (rows where ``admitted == true``).
- ``parse_failure_count`` (rows where ``n_sidecars == 0``).
- ``parse_failure_count_metric_out_of_range`` (the subset whose
  ``detail`` substring contains
  ``ncu_csv_metric_out_of_range``).
- ``parsed_config_count`` (rows where ``n_sidecars > 0``).
- ``mean_l2_sector_mean_across_parsed_configs`` (average over
  the parsed-config subset).

The ROUND_CLOSE_VERDICTS table is hand-pinned because each
recovery round's sweep dir is named by the orchestrator at
sweep time; pinning the table here keeps the per-round
identity stable for the scanner. Add a new entry when a new
round closes.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT_DIR = (
    _REPO_ROOT
    / "data"
    / "local_labels"
    / "audit_evidence"
    / "ac6_label_stability_diagnostic"
)
_ARTIFACT_PATH = _ARTIFACT_DIR / "label_stability_table.json"


_ROUND_CLOSE_VERDICTS: tuple[dict[str, Any], ...] = (
    {
        "round_number": 24,
        "verdict_table_relpath": (
            "data/local_labels/sweeps/"
            "sweep_20260513T040005Z_88a98f984c6b/"
            "sweep_20260513T040006Z_b8d531b868fa/"
            "verdict-table.csv"
        ),
        "repeats": 5,
        "comment": (
            "First headline regen on the expanded 13-cell "
            "cohort that includes 4 admitted microbench cells "
            "(l2_persisting, l2_thrash, register_spill, "
            "warm_cold)."
        ),
    },
    {
        "round_number": 25,
        "verdict_table_relpath": (
            "data/local_labels/sweeps/"
            "sweep_20260513T083533Z_c423f775011f/"
            "sweep_20260513T083533Z_b8d531b868fa/"
            "verdict-table.csv"
        ),
        "repeats": 5,
        "comment": (
            "Post-dram-stream-kernel-rehabilitation close. "
            "Only 2 of 8 microbench cells cleared the "
            "repeatability gate (l2_thrash, register_spill); "
            "6 of 8 exceeded the stddev gate across every "
            "warmup/cache config."
        ),
    },
    {
        "round_number": 26,
        "verdict_table_relpath": (
            "data/local_labels/sweeps/"
            "sweep_20260513T094819Z_9eea75dc57dd/"
            "sweep_20260513T094820Z_567b8e0a38e8/"
            "verdict-table.csv"
        ),
        "repeats": 10,
        "comment": (
            "Microbench label-side recovery attempt under "
            "controlled host conditions and doubled "
            "--repeats. Only 1 of 8 microbench cells cleared "
            "the repeatability gate (l2_thrash). dram_stream "
            "produced 6 of 8 ncu_csv_metric_out_of_range "
            "parse failures."
        ),
    },
    {
        "round_number": 27,
        "verdict_table_relpath": (
            "data/local_labels/sweeps/"
            "sweep_20260513T111518Z_6ba4090e29d0/"
            "sweep_20260513T111519Z_b8d531b868fa/"
            "verdict-table.csv"
        ),
        "repeats": 5,
        "comment": (
            "Drift-recovery round. Full 13-cell sweep at "
            "--repeats 5 after a bounded 4-cell probe gate "
            "(register_spill, carveout, working_set, "
            "l2_persisting). The probe admitted "
            "register_spill at 5 configs (min stddev 0.003 "
            "at warmup=5/cache=none) and working_set at 1 "
            "config (min 0.009 at warmup=1/cache=all); "
            "carveout and l2_persisting did not admit. The "
            "full sweep confirmed register_spill admission "
            "(min stddev 0.002 at warmup=10/cache=none) and "
            "the prior round's l2_thrash admission, but "
            "working_set's label-side admission did not "
            "reproduce. The post-rewrite dram_stream kernel "
            "continues to produce ncu_csv_metric_out_of_range "
            "parse failures."
        ),
    },
    {
        "round_number": 29,
        "verdict_table_relpath": (
            "data/local_labels/sweeps/"
            "sweep_20260513T144243Z_37e5cfb34f77/"
            "sweep_20260513T144244Z_3623f99d7515/"
            "verdict-table.csv"
        ),
        "repeats": 5,
        "comment": (
            "Bounded 4-cell probe (register_spill, "
            "carveout, working_set, l2_persisting) after the "
            "working_set + carveout kernel rewrites landed "
            "their first scalar bundles. The probe admitted "
            "carveout at 1 config (min stddev 0.0085 at "
            "warmup=0/cache=all) for the first time across "
            "the loop's recovery rounds, and reconfirmed the "
            "prior-round register_spill admission (min "
            "stddev 0.0069 at warmup=10/cache=none, plus four "
            "additional admitting configs). working_set "
            "missed the 0.03 stddev threshold at every "
            "config (best 0.041 at warmup=0/cache=all); "
            "l2_persisting carried forward its prior "
            "label-side regression (best 0.075 at "
            "warmup=10/cache=all). The post-rewrite "
            "working_set + carveout kernels each emit one "
            "ncu_csv_metric_out_of_range parse failure on the "
            "warmup=5/cache=none corner; the gate (>=3 "
            "additional cells admitted beyond l2_thrash + "
            "register_spill) was NOT met (only carveout "
            "newly admitted), so the conditional full "
            "13-cell sweep was skipped per the R29 contract."
        ),
    },
    {
        "round_number": 30,
        "verdict_table_relpath": (
            "data/local_labels/sweeps/"
            "sweep_20260513T155513Z_758be398302e/"
            "sweep_20260513T155513Z_19c537e3ae1c/"
            "verdict-table.csv"
        ),
        "repeats": 5,
        "comment": (
            "Bounded 1-cell working_set probe with a new "
            "L2-cache-wash preamble prepended to the kernel "
            "(768 lines/warp x 64 warps = 49152 lines = "
            "6 MiB = exactly the sm_86 L2 capacity) intended "
            "to absorb inter-launch L2 carryover. The wash "
            "preamble FALSIFIED the R30 hypothesis: best L2 "
            "stddev rose to 0.1357 at warmup=1/cache=all "
            "(3.3x worse than R29's 0.041); 3 of 8 configs "
            "produced ncu_csv_metric_out_of_range for "
            "lts__t_sector_hit_rate.pct (values 134.93, "
            "145.3, 151.48 -- all above NCU's [0,105] "
            "saturation clamp window). The best-config mean "
            "L2 hit rate was 0.882 at warmup=1/cache=all, "
            "but the scanner-verified mean across all "
            "parsed configs is 0.681912 (the structured "
            "per_cell.mean_l2_sector_mean_across_parsed_configs "
            "field is the canonical source-of-truth -- this "
            "value differs from the single-best-config "
            "0.882 because the across-parsed-config mean "
            "averages over the 5 admitted-by-parser configs "
            "whose hit rates range from 0.5214 to 0.8817). "
            "Hypothesised cause: the wash + main 3-stage "
            "pattern's combined footprint (6.5 MiB) exceeds "
            "L2 (6 MiB), so end-of-launch L2 residency is "
            "sensitive to exact LRU eviction order across "
            "launches. R31 candidate remediation: smaller "
            "wash (1-2 MiB) or main-pattern resizing so the "
            "combined footprint fits in L2."
        ),
    },
    {
        "round_number": 31,
        "verdict_table_relpath": (
            "data/local_labels/sweeps/"
            "sweep_20260513T163441Z_f40dd5010e42/"
            "sweep_20260513T163441Z_19c537e3ae1c/"
            "verdict-table.csv"
        ),
        "repeats": 5,
        "comment": (
            "Bounded 1-cell working_set probe AFTER the R30 "
            "wash preamble was REVERTED back to the R29 "
            "layered-reuse baseline. The reverted kernel "
            "produced best L2 stddev 0.0560 at "
            "warmup=1/cache=all (within the R31-contract's "
            "[0.030, 0.060] R29-baseline window; the R29 "
            "originally-observed 0.0410 was at the favorable "
            "tail of the across-retrace noise distribution). "
            "The rollback is CONFIRMED: the R30 wash regression "
            "(0.1357 best) is removed and the baseline is "
            "re-established. Three of 8 configs still produce "
            "ncu_csv_metric_out_of_range parse failures "
            "(values 144.68, 121.68, 125.95 at the same "
            "warmup=0/all, warmup=0/none, warmup=5/all "
            "corners as R30), establishing that those NCU "
            "OOR failures are an intrinsic property of the "
            "working_set kernel's NCU-metric pipeline "
            "interaction, NOT caused by the wash. Mean L2 "
            "across parsed configs is 0.6805 (similar to "
            "R30's 0.6819), so the hit-rate level is stable "
            "across the kernel reversion; the variance "
            "distribution is what changes. The admitted-"
            "microbench-cell count stays at 3 (l2_thrash + "
            "register_spill + carveout). The next round picks "
            "up kernel-pattern-lever remediation per "
            "BL-20260513-l2-wash-preamble-at-l2-capacity-"
            "amplifies-label-variance Solution paths (a)/"
            "(b)/(c)."
        ),
    },
    {
        "round_number": 34,
        "verdict_table_relpath": (
            "data/local_labels/sweeps/"
            "sweep_20260513T174903Z_d092d6a89335/"
            "sweep_20260513T174903Z_19c537e3ae1c/"
            "verdict-table.csv"
        ),
        "repeats": 5,
        "comment": (
            "Bounded 1-cell working_set probe AFTER a kernel-"
            "pattern resize landed: the kernel's Stage 3 shrunk "
            "from 4096 distinct lines to 256 lines (kernel "
            "intra-launch L2 footprint drops from ~680 KiB to "
            "~210 KiB, ~3% of sm_86 L2 capacity). The resize is "
            "the kernel-pattern-remediation arm of "
            "BL-20260513-l2-wash-preamble-at-l2-capacity-"
            "amplifies-label-variance Solution path (a). "
            "OUTCOME: working_set ADMITTED for the first time "
            "across the loop's recovery rounds, at warmup=1/"
            "cache=all with stddev_l2_sector 0.0136 and "
            "mean_l2_sector 0.892 (the admitted-microbench "
            "count rises from 3 to 4; the admitted-cell set "
            "is now l2_thrash_pointer_chase + register_spill_"
            "local_mem + carveout_sensitivity_probe + working_"
            "set_capacity_sweep). Second-best config: "
            "warmup=10/cache=all (stddev 0.0352, near-admit). "
            "Two of 8 configs produced ncu_csv_metric_out_of_"
            "range for lts__t_sector_hit_rate.pct "
            "(warmup=5/all 137.82, warmup=10/none 123.27); "
            "the count is down from R31's 3 OOR failures "
            "(the kernel-pattern resize appears to partly "
            "resolve the NCU-pipeline saturation tail). The "
            "NVBit-instrumented kernel runtime dropped from "
            "R31's 106s to R34's 28s, and the actual_slowdown_"
            "ratio dropped from 491x to 176x (below the 200x "
            "budget; no slowdown sidecar emitted)."
        ),
    },
    {
        "round_number": 35,
        "verdict_table_relpath": (
            "data/local_labels/sweeps/"
            "sweep_20260513T183046Z_c8e4ff16aa03/"
            "sweep_20260513T183046Z_6fbaaad5cd22/"
            "verdict-table.csv"
        ),
        "repeats": 5,
        "comment": (
            "Bounded 1-cell l2_persisting_window_sweep probe "
            "AFTER the kernel was rewritten from a per-thread "
            "stride-1 streaming loop to the warp-coalesced "
            "layered-reuse pattern proven on working_set R34 "
            "(per-warp anchor + Stage 1 4-cycle per-warp "
            "distinct + Stage 2 256-shared-line sweep). The "
            "kernel intra-launch L2 footprint is 192 KiB "
            "(~3.2% of sm_86 L2 capacity), comparable to "
            "working_set R34's ~210 KiB. OUTCOME: documented-"
            "debug-finding (no admit). Three iterations were "
            "attempted within the round: (i) the kernel form "
            "committed here (Stage 1 + shared Stage 2) "
            "measured best stddev_l2_sector 0.0585 at "
            "warmup=5/cache=all on its first probe and 0.1362 "
            "at warmup=1/cache=all on a re-probe of the same "
            "kernel state (the committed verdict-table is "
            "the second probe); (ii) Stage 1 only, 16 cycles "
            "per warp, no shared phase, measured best stddev "
            "0.0602 at warmup=10/cache=none; (iii) Stage 1 "
            "16-cycle with an active-warp gate "
            "(warp_id < 64) collapsed the HRD profile to "
            "bundle_underconfigured=true and violated the "
            "R35 blocking side issue (must preserve "
            "bundle_underconfigured=false). All three "
            "iterations stayed above the 0.045 partial-"
            "success threshold. The label-side variance is "
            "run-to-run unstable for the same kernel binary "
            "(0.0585 vs 0.1362 across consecutive probes of "
            "the committed Variant 1 kernel), confirming "
            "the variance source is the hardware-advisory "
            "persistence-window hint set host-side over "
            "d_in: cudaAccessPropertyPersisting biases L2 "
            "replacement against eviction inside the window, "
            "but the actual eviction order is hardware-"
            "determined and varies launch-to-launch. The "
            "kernel-pattern change alone cannot eliminate "
            "this variance because the persistence window "
            "is part of the cell's identity. The admitted-"
            "microbench-cell count stays at 4 (l2_thrash + "
            "register_spill + carveout + working_set). Next-"
            "round options: (1) host-side persistence-config "
            "modification (e.g. hitRatio < 1.0 or a smaller "
            "window) which changes the cell's "
            "persistence-window-exerciser semantic and needs "
            "user buy-in; (2) shape config edit "
            "(trace_l2_accuracy_xl warps from 256 to 64) to "
            "match the working_set round-34 parallelism "
            "exactly; (3) pivot the fifth-cell admission "
            "target from l2_persisting to "
            "dram_stream_stride_mix (carry-forward NCU CSV "
            "out-of-range is the remaining blocker there, "
            "but the kernel pattern is more deterministic)."
        ),
    },
    {
        "round_number": 36,
        "verdict_table_relpath": (
            "data/local_labels/sweeps/"
            "sweep_20260514T045356Z_6ac73b4bf47e/"
            "sweep_20260514T045356Z_566d6ee03b26/"
            "verdict-table.csv"
        ),
        "repeats": 5,
        "comment": (
            "Bounded 1-cell dram_stream_stride_mix probe "
            "AFTER the R36 round repaired the post-round-35 "
            "provenance/build-commit drift (the chore-evidence "
            "follow-up commit had moved HEAD one step past "
            "the round-35 close, putting all sidecar / HRD "
            "MPI / R27 predictor_build_commit values "
            "outside the {HEAD, HEAD^} accept window). The "
            "round attempted the fifth-cell pivot Codex's "
            "round-35 review recommended: rewrite the "
            "dram_stream kernel with the warp-coalesced "
            "layered-reuse template proven on working_set "
            "round-34. Two kernel variants were attempted: "
            "(i) two-stage layered-reuse (64-cycle Stage 1 + "
            "16-cycle Stage 2 with shared distinct lines), "
            "and (ii) the round-25 main loop with a "
            "warp-coalesced identity-output replacing the "
            "global identity loop. Both variants produced "
            "bundle_underconfigured=true on the trace_l2_"
            "accuracy_xl shape (warps=8, accesses=2048, "
            "stride_classes=2), violating the round contract's "
            "blocking side issue. The 8-warp parallelism is "
            "intrinsically too narrow to populate enough HRD "
            "bins under the warp-coalesced template; the "
            "working_set round-34 template was sized for 64 "
            "warps. Both variants were reverted before commit; "
            "the kernel sources are restored to the round-25 "
            "baseline and retraced at current HEAD. The "
            "committed verdict-table is the round-25-baseline "
            "probe at current HEAD: 5 of 8 configs NCU OOR "
            "(lts__t_sector_hit_rate.pct values 144.96, "
            "147.16, 141.7, 169.24, 204.89), 3 parsed configs "
            "best stddev_l2_sector 0.168189 at warmup=1/"
            "cache=all (mean 0.740). The OOR pattern matches "
            "rounds 25-27 within a small drift. The admitted-"
            "microbench-cell count stays at 4 (l2_thrash + "
            "register_spill + carveout + working_set). "
            "Documented-debug-finding outcome: the next round "
            "must either (a) widen the dram_stream "
            "trace_l2_accuracy_xl shape to a larger warps "
            "count compatible with the working_set round-34 "
            "template, or (b) consider a shape-config edit "
            "for the cell, or (c) pivot the fifth-cell "
            "admission target to a remaining microbench "
            "candidate (warm_cold_pair_relaunch or "
            "l2_prefetch_hint_sweep)."
        ),
    },
    {
        "round_number": 37,
        "verdict_table_relpath": (
            "data/local_labels/sweeps/"
            "sweep_20260514T061534Z_7592b405043e/"
            "sweep_20260514T061534Z_8cd60de80bfd/"
            "verdict-table.csv"
        ),
        "repeats": 5,
        "comment": (
            "Bounded 1-cell warm_cold_pair_relaunch probe "
            "AFTER the live provenance audit accept-window "
            "repair landed (predictor source_commit now "
            "accepts {HEAD, HEAD^}, mirroring the focused "
            "nested-provenance and regeneration-delta "
            "scanner discipline). The round attempted the "
            "fifth-cell admission target Codex's round-36 "
            "review named: rewrite warm_cold_pair_relaunch "
            "with the warp-coalesced layered-reuse pattern "
            "(per-warp 32-line window cycling, single "
            "warp-coalesced identity-output) in lockstep "
            "with a host-side d_out shrink to warps * 32 "
            "and a (warps, 32) reference shape in "
            "references/mma_microbench.py. The retraced "
            "kernel landed bundle_underconfigured=true on "
            "the trace_l2_accuracy_xl shape (warps=8, "
            "allocation_bytes=524288, accesses=512), with "
            "the allocation rd=[2,2] bin at 28 obs and two "
            "region-layer bins at 2 and 5 obs (all below "
            "the MINIMUM_BIN_SAMPLE_COUNT=50 floor). The "
            "8-warp parallelism reproduces the round-36 "
            "dram_stream Variant 1 lesson: the working_set "
            "round-34 template is sized for 64 warps and "
            "does not transfer to 8-warp trace shapes. "
            "Kernel + main.cu + mma_microbench.py reference "
            "were reverted before commit; the cell source "
            "is the prior baseline. The committed "
            "verdict-table is the baseline retraced and "
            "probed at current HEAD: best parsed "
            "stddev_l2_sector=0.044826 at warmup=5/"
            "cache=none (mean=0.848980), just below the "
            "0.045 partial-success threshold but still "
            "above the 0.03 admission floor; 2 of 8 "
            "configs failed NCU CSV parse with "
            "lts__t_sector_hit_rate.pct values 473.87 and "
            "354.59 (saturation window [0, 105]). The "
            "admitted-microbench-cell count stays at 4 of "
            "8; the documented-debug-finding outcome "
            "matches the round-35 (l2_persisting) and "
            "round-36 (dram_stream) pattern. The remaining "
            "unattempted cell is l2_prefetch_hint_sweep."
        ),
    },
)


def _read_rows(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"emit_label_stability_artifact: missing verdict-table.csv "
            f"at {csv_path}"
        )
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def _facts_for_cell(rows: list[dict]) -> dict:
    """Compute the per-cell record from the rows of a single
    cell. Returns the dict shape the scanner pins against the
    same recomputation in
    ``test_label_stability_scanner.py``."""
    parsed = [r for r in rows if int(r.get("n_sidecars", "0") or "0") > 0]
    failures = [r for r in rows if int(r.get("n_sidecars", "0") or "0") == 0]
    metric_oor = [
        r for r in failures
        if "ncu_csv_metric_out_of_range" in (r.get("detail") or "")
    ]
    admitted = [
        r for r in rows
        if str(r.get("admitted", "")).strip().lower() == "true"
    ]
    if parsed:
        best = min(parsed, key=lambda r: float(r["stddev_l2_sector"]))
        min_stddev = round(float(best["stddev_l2_sector"]), 6)
        best_warmup = int(best["warmup_count"])
        best_cache = str(best["cache_control"])
        mean_l2 = round(
            sum(float(r["mean_l2_sector"]) for r in parsed) / len(parsed),
            6,
        )
    else:
        min_stddev = None
        best_warmup = None
        best_cache = None
        mean_l2 = None
    return {
        "min_stddev_l2_sector": min_stddev,
        "best_warmup_count": best_warmup,
        "best_cache_control": best_cache,
        "admitted_config_count": len(admitted),
        "parse_failure_count": len(failures),
        "parse_failure_count_metric_out_of_range": len(metric_oor),
        "parsed_config_count": len(parsed),
        "mean_l2_sector_mean_across_parsed_configs": mean_l2,
    }


def _build_entry(entry: dict) -> dict:
    csv_path = _REPO_ROOT / entry["verdict_table_relpath"]
    rows = _read_rows(csv_path)
    by_cell: dict[str, list[dict]] = {}
    for row in rows:
        by_cell.setdefault(row["cell_id"], []).append(row)
    per_cell = {
        cell: _facts_for_cell(cell_rows)
        for cell, cell_rows in sorted(by_cell.items())
    }
    return {
        "round_number": entry["round_number"],
        "verdict_table_relpath": entry["verdict_table_relpath"],
        "repeats": entry["repeats"],
        "comment": entry["comment"],
        "per_cell": per_cell,
    }


def emit() -> Path:
    payload = {
        "schema_version": 1,
        "artifact_name": "label_stability_table",
        "documented_thresholds": {
            "stddev_l2_sector": 0.03,
            "stddev_l1tex_sector": 0.01,
        },
        "entries": [_build_entry(entry) for entry in _ROUND_CLOSE_VERDICTS],
    }
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with _ARTIFACT_PATH.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return _ARTIFACT_PATH


def _main() -> int:
    out = emit()
    relpath = out.relative_to(_REPO_ROOT)
    rounds = ", ".join(str(e["round_number"]) for e in _ROUND_CLOSE_VERDICTS)
    print(
        f"wrote {relpath}: rounds=[{rounds}] "
        f"entries={len(_ROUND_CLOSE_VERDICTS)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
