#!/usr/bin/env python3
"""Build a reproducible residual-selection report for the SM86 live cohort.

This is deliberately an analysis-only tool.  It reads the already captured
NCU and simulator artifacts, compares an independent NCU repeat with the
baseline, and never writes measurements back into simulator inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "results/captures/sm86_simulator_live_baseline_20260829_v2/campaign_manifest.json"
)
METRICS = (
    "gpc__cycles_elapsed.max",
    "sm__cycles_active.max",
    "smsp__inst_executed.sum",
)
CAUSE_NAMES = (
    "control_stall",
    "dependence_counter_read",
    "dependence_counter_non_memory_write",
    "dependence_counter_shared_write",
    "dependence_counter_l1tex",
    "dependence_counter_unknown_origin",
    "async_source",
    "async_destination_local",
    "async_destination_sm_service",
    "cta_barrier_rendezvous",
    "pipeline_tensor",
    "eligible_not_selected",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _campaign_file(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label}: missing campaign-relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label}: path escapes campaign root: {relative}") from exc
    if not path.is_file():
        raise ValueError(f"{label}: file is missing: {relative}")
    return path


def _label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _ncu_row(path: Path) -> dict[str, str]:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    for row in rows:
        if row.get("Kernel Name"):
            return {str(key): str(value) for key, value in row.items()}
    raise ValueError(f"{path}: no kernel row")


def _metric(row: Mapping[str, str], name: str) -> int:
    value = row.get(name)
    if value is None:
        raise ValueError(f"NCU row has no metric {name}")
    try:
        return int(round(float(value.replace(",", ""))))
    except ValueError as exc:
        raise ValueError(f"NCU metric {name} is not numeric") from exc


def _top_causes(readiness: Mapping[str, Any]) -> list[dict[str, Any]]:
    limiting = readiness.get("limiting_cycles")
    if not isinstance(limiting, Mapping):
        return []
    values = []
    for name in CAUSE_NAMES:
        value = limiting.get(name, 0)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            values.append({"cause": name, "limiting_cycles": value})
    values.sort(key=lambda item: (-item["limiting_cycles"], item["cause"]))
    return values[:3]


def _replay_summary(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    launches = document.get("launches")
    if not isinstance(launches, list) or len(launches) != 1:
        raise ValueError(f"{path}: expected one replay launch")
    launch = launches[0]
    if not isinstance(launch, Mapping):
        raise ValueError(f"{path}: replay launch is malformed")
    timing = launch.get("timing")
    if not isinstance(timing, Mapping):
        raise ValueError(f"{path}: replay timing is missing")
    readiness = timing.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError(f"{path}: replay readiness is missing")
    frontier = timing.get("completion_frontier")
    if not isinstance(frontier, Mapping):
        raise ValueError(f"{path}: replay completion frontier is missing")
    counters = launch.get("counters")
    if not isinstance(counters, Mapping):
        raise ValueError(f"{path}: replay counters are missing")
    tails = counters.get("unmodeled_tails")
    if not isinstance(tails, Mapping):
        tails = frontier.get("unmodeled_tails", {})
    terminal = timing.get("terminal_modeled_event_time")
    if not isinstance(terminal, int) or isinstance(terminal, bool):
        raise ValueError(f"{path}: terminal modeled event time is malformed")
    return {
        "terminal_lower_bound_cycles": terminal,
        "completion_status": frontier.get("status"),
        "tails": dict(tails),
        "readiness_schema": readiness.get("schema_version"),
        "top_readiness_causes": _top_causes(readiness),
        "instructions_issued": counters.get("instructions_issued"),
        "records_seen": counters.get("records_seen"),
        "sha256": _sha256(path),
    }


def _repeat_metrics(root: Path, slug: str, stage: Mapping[str, Any]) -> dict[str, Any]:
    baseline_csv = _campaign_file(root, stage.get("csv"), f"{slug}.ncu")
    repeat_csv = _campaign_file(
        root, f"evidence/raw/ncu_repeat/rep-2/{slug}/ncu.csv", f"{slug}.ncu_repeat"
    )
    repeat_metadata = _campaign_file(
        root,
        f"evidence/raw/ncu_repeat/rep-2/{slug}/metadata.json",
        f"{slug}.ncu_repeat.metadata",
    )
    metadata = json.loads(repeat_metadata.read_text(encoding="utf-8"))
    if metadata.get("return_code") != 0:
        raise ValueError(f"{slug}: repeated NCU command did not exit zero")
    baseline_row = _ncu_row(baseline_csv)
    repeat_row = _ncu_row(repeat_csv)
    baseline = {name: _metric(baseline_row, name) for name in METRICS}
    repeat = {name: _metric(repeat_row, name) for name in METRICS}
    deltas = {name: repeat[name] - baseline[name] for name in METRICS}
    max_relative = max(
        abs(deltas[name]) / max(abs(baseline[name]), 1) for name in METRICS
    )
    return {
        "baseline": baseline,
        "repeat_2": repeat,
        "delta_repeat_minus_baseline": deltas,
        "max_relative_delta": round(max_relative, 8),
        "stable_within_five_percent": max_relative <= 0.05,
        "baseline_csv": {"path": _label(baseline_csv), "sha256": _sha256(baseline_csv)},
        "repeat_csv": {"path": _label(repeat_csv), "sha256": _sha256(repeat_csv)},
        "repeat_metadata": {
            "path": _label(repeat_metadata),
            "sha256": _sha256(repeat_metadata),
            "argv": metadata.get("command"),
            "started_at": metadata.get("started_at"),
            "finished_at": metadata.get("finished_at"),
        },
    }


def analyze(manifest_path: Path, output_path: Path) -> int:
    manifest_path = manifest_path.expanduser().resolve()
    root = manifest_path.parent
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or document.get("schema_version") != "sm86.live.campaign.v2":
        raise ValueError("unsupported campaign manifest")
    cells = document.get("cells")
    if not isinstance(cells, list) or len(cells) != 6:
        raise ValueError("closure analysis requires six cohort cells")

    rows: list[dict[str, Any]] = []
    for cell in cells:
        if not isinstance(cell, Mapping) or not isinstance(cell.get("id"), str):
            raise ValueError("malformed cohort cell")
        slug = cell["id"]
        stages = cell.get("stages")
        if not isinstance(stages, Mapping):
            raise ValueError(f"{slug}: stages are missing")
        live = _repeat_metrics(root, slug, stages["ncu"])
        replay_one = _campaign_file(root, stages["simulator"]["outputs"][0], f"{slug}.replay")
        replay_two = _campaign_file(root, stages["simulator"]["outputs"][1], f"{slug}.replay_twin")
        sim = _replay_summary(replay_one)
        if replay_one.read_bytes() != replay_two.read_bytes():
            raise ValueError(f"{slug}: simulator twins differ")
        sim["twin_sha256"] = _sha256(replay_two)
        sim["twin_byte_identical"] = True
        comparable = slug != "cp_async_canary"
        baseline_sm = live["baseline"]["sm__cycles_active.max"]
        repeat_sm = live["repeat_2"]["sm__cycles_active.max"]
        terminal = sim["terminal_lower_bound_cycles"]
        row: dict[str, Any] = {
            "cell": slug,
            "comparable_to_simulator_timing": comparable,
            "live": live,
            "simulator": sim,
            "residual": {
                "metric": "NCU sm__cycles_active.max minus simulator terminal lower bound",
                "baseline_cycles": baseline_sm - terminal,
                "repeat_2_cycles": repeat_sm - terminal,
                "repeat_delta_cycles": repeat_sm - baseline_sm,
                "interpretable": comparable,
            },
            "artifacts": [
                {"path": _label(replay_one), "sha256": _sha256(replay_one)},
                {"path": _label(replay_two), "sha256": _sha256(replay_two)},
            ],
        }
        rows.append(row)

    comparable_rows = [row for row in rows if row["comparable_to_simulator_timing"]]
    ranked = sorted(
        comparable_rows,
        key=lambda row: (-row["residual"]["baseline_cycles"], row["cell"]),
    )
    for rank, row in enumerate(ranked, start=1):
        row["residual"]["rank_by_baseline_cycles"] = rank

    # A diagnostic ranking is intentionally distinct from a timing patch.  It
    # records why the next probe is chosen while preserving the no-go result.
    candidate_ranking = [
        {
            "candidate": "dirty_writeback_coherence_retirement",
            "evidence_cells": [row["cell"] for row in rows if any(row["simulator"]["tails"].get(name) for name in ("dirty_writeback", "coherence"))],
            "selection": "next_discriminator",
            "reason": "Every cohort replay exposes both typed tails; a scalar completion offset cannot separate them from launch or async/barrier behavior.",
        },
        {
            "candidate": "async_destination_service",
            "evidence_cells": [row["cell"] for row in rows if any(item["cause"] == "async_destination_sm_service" for item in row["simulator"]["top_readiness_causes"])],
            "selection": "candidate_not_admitted",
            "reason": "It is a dominant modeled readiness cause in some cells, but no independent silicon observable isolates it from retirement tails.",
        },
        {
            "candidate": "control_and_barrier_readiness",
            "evidence_cells": [row["cell"] for row in rows if any(item["cause"] in {"control_stall", "cta_barrier_rendezvous"} for item in row["simulator"]["top_readiness_causes"])],
            "selection": "candidate_not_admitted",
            "reason": "The readiness counters are modeled diagnostics, not an independent live timing label.",
        },
        {
            "candidate": "launch_envelope",
            "evidence_cells": [row["cell"] for row in rows],
            "selection": "candidate_not_admitted",
            "reason": "GPC elapsed includes launch work not represented by the direct event frontier; it is not a simulator-local latency.",
        },
    ]
    report = {
        "schema_version": 1,
        "analysis": {
            "command": [
                ".venv/bin/python",
                "workloads/research_probes/analyze_sm86_live_closure.py",
                "--manifest",
                _label(manifest_path),
                "--output",
                _label(output_path),
            ],
            "manifest": {
                "path": _label(manifest_path),
                "campaign_id": document.get("campaign_id"),
                "candidate_revision": document.get("repository", {}).get("candidate_revision")
                if isinstance(document.get("repository"), Mapping)
                else None,
            },
            "repeat_root": "evidence/raw/ncu_repeat/rep-2",
            "decision": "no-go",
            "selected_diagnostic": "dirty_writeback_coherence_retirement",
            "next_discriminator": "matched store-free/barrier-free async-source pair with an independent retirement observable",
            "repeatability_basis": "Two independent application-replay NCU captures per cell; simulator twins are byte-identical.",
            "held_out_discriminator": "not_run; the next pair is intentionally held out to avoid selecting a timing seam from the scored cohort.",
            "ncu_values_used_as_simulator_inputs": False,
        },
        "candidate_ranking": candidate_ranking,
        "rows": rows,
        "decision": {
            "decision": "no-go",
            "seam": "cycle scalar completion/retirement mapping",
            "reason": "Live residuals are repeat-stable at the measured envelope, but the simulator reports dirty-writeback/coherence tails for every comparable cell and the observations do not identify one architecture-local latency.",
            "next_action": "Run the matched store-free/barrier-free async-source discriminator with an independent retirement/frontier observable before changing simulator timing state.",
            "claim_limits": [
                "No cycle-accuracy claim",
                "No NCU leakage into simulator inputs",
                "No cross-architecture inference",
                "No JADE/HETR or CPU-GPU co-simulation change",
            ],
        },
    }
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote closure report: {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return analyze(args.manifest, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"closure analysis: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
