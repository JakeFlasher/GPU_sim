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


def _campaign_token_matches(token: object, relative: str, root: Path) -> bool:
    if not isinstance(token, str):
        return False
    candidates = {
        relative,
        "results/captures/sm86_simulator_live_baseline_20260829_v2/" + relative,
        str((root / relative).resolve()),
    }
    return token in candidates


def _repeat_command_contract(
    root: Path,
    cell: Mapping[str, Any],
    stage: Mapping[str, Any],
    repeat_dir: str,
) -> tuple[list[str], list[str]]:
    """Return expected NCU argv and the indices whose paths may be normalized."""

    source = stage.get("argv")
    tool = stage.get("tool_path")
    if not isinstance(source, list) or any(not isinstance(item, str) for item in source):
        raise ValueError(f"{cell.get('id')}: canonical NCU argv is malformed")
    if not isinstance(tool, str) or not tool:
        raise ValueError(f"{cell.get('id')}: canonical NCU tool path is missing")
    try:
        output_index = source.index("--output")
        separator = source.index("--", output_index)
    except ValueError as exc:
        raise ValueError(f"{cell.get('id')}: canonical NCU argv lacks output/target separator") from exc
    if output_index + 1 >= separator:
        raise ValueError(f"{cell.get('id')}: canonical NCU output path is malformed")
    target = source[separator + 1 :]
    if not target or target[0] != cell.get("binary", {}).get("path"):
        raise ValueError(f"{cell.get('id')}: canonical NCU target is not the cell binary")
    def option_value(name: str) -> str | None:
        try:
            index = source.index(name)
        except ValueError:
            return None
        if index + 1 >= separator:
            raise ValueError(f"{cell.get('id')}: canonical NCU option {name} has no value")
        return source[index + 1]
    command = [tool, "--force-overwrite", "--export", repeat_dir + "/ncu-report"]
    for name in ("--launch-skip", "--launch-count", "--replay-mode", "--clock-control", "--metrics"):
        value = option_value(name)
        if value is not None:
            command.extend((name, value))
    command.extend(target)
    return command, target


def _validate_repeat_metadata(
    root: Path,
    cell: Mapping[str, Any],
    stage: Mapping[str, Any],
    repeat_dir: str,
    metadata: Mapping[str, Any],
) -> None:
    slug = str(cell["id"])
    expected_command, target = _repeat_command_contract(root, cell, stage, repeat_dir)
    command = metadata.get("command")
    if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
        raise ValueError(f"{slug}: repeated NCU metadata command is missing")
    if len(command) != len(expected_command):
        raise ValueError(f"{slug}: repeated NCU command length differs from the cell contract")
    for index, (actual, expected) in enumerate(zip(command, expected_command)):
        if index == 3:  # --export value
            if not _campaign_token_matches(actual, expected, root):
                raise ValueError(f"{slug}: repeated NCU export path is misbound")
        elif actual != expected:
            raise ValueError(f"{slug}: repeated NCU command differs at argv[{index}]")
    if metadata.get("return_code") != 0:
        raise ValueError(f"{slug}: repeated NCU command did not exit zero")
    expected_csv = repeat_dir + "/ncu.csv"
    expected_report = repeat_dir + "/ncu-report.ncu-rep"
    for key, expected in (("csv", expected_csv), ("report", expected_report)):
        if not _campaign_token_matches(metadata.get(key), expected, root):
            raise ValueError(f"{slug}: repeated NCU metadata {key} path is misbound")
    if target[0] != cell.get("binary", {}).get("path"):
        raise ValueError(f"{slug}: repeated NCU target binary differs")


def _repeat_metrics(
    root: Path, cell: Mapping[str, Any], stage: Mapping[str, Any]
) -> dict[str, Any]:
    slug = str(cell["id"])
    baseline_csv = _campaign_file(root, stage.get("csv"), f"{slug}.ncu")
    repeat_dir = f"evidence/raw/ncu_repeat/rep-2/{slug}"
    repeat_csv = _campaign_file(
        root, repeat_dir + "/ncu.csv", f"{slug}.ncu_repeat"
    )
    repeat_metadata = _campaign_file(
        root, repeat_dir + "/metadata.json", f"{slug}.ncu_repeat.metadata"
    )
    metadata = json.loads(repeat_metadata.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{slug}: repeated NCU metadata is malformed")
    _validate_repeat_metadata(root, cell, stage, repeat_dir, metadata)
    baseline_row = _ncu_row(baseline_csv)
    repeat_row = _ncu_row(repeat_csv)
    if any(name not in repeat_row for name in METRICS):
        raise ValueError(f"{slug}: repeated NCU CSV does not contain the canonical metric set")
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
            "contract_validated": True,
        },
    }


def _derive_decision(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Derive the closure decision from validated observations.

    A no-go is only eligible when the live repeat and the simulator diagnostic
    lanes are present for every comparable row.  Missing or contradictory
    evidence is an analysis failure, never an implicit no-go.
    """

    comparable = [row for row in rows if row.get("comparable_to_simulator_timing")]
    unstable = [
        row["cell"]
        for row in rows
        if row.get("live", {}).get("stable_within_five_percent") is not True
    ]
    if unstable:
        raise ValueError(
            "insufficient_evidence: live repeat is unstable for " + ", ".join(unstable)
        )
    missing_diagnostics = [
        row["cell"]
        for row in rows
        if not isinstance(row.get("simulator", {}).get("readiness_schema"), int)
        or not row.get("simulator", {}).get("top_readiness_causes")
    ]
    if missing_diagnostics:
        raise ValueError(
            "insufficient_evidence: required readiness diagnostics are absent for "
            + ", ".join(missing_diagnostics)
        )
    tail_rows = [
        row
        for row in comparable
        if row.get("simulator", {}).get("tails", {}).get("dirty_writeback") is True
        and row.get("simulator", {}).get("tails", {}).get("coherence") is True
    ]
    if not comparable or len(tail_rows) != len(comparable):
        raise ValueError(
            "insufficient_evidence: dirty_writeback/coherence retirement tails "
            "are not observed across every comparable row"
        )
    residual_score = sum(
        max(int(row["residual"]["baseline_cycles"]), 0) for row in comparable
    )
    residuals = [row["residual"]["baseline_cycles"] for row in comparable]
    tail_cells = [row["cell"] for row in tail_rows]
    service_cells = [
        row["cell"]
        for row in rows
        if any(
            item.get("cause") == "async_destination_sm_service"
            for item in row.get("simulator", {}).get("top_readiness_causes", [])
        )
    ]
    control_cells = [
        row["cell"]
        for row in rows
        if any(
            item.get("cause") in {"control_stall", "cta_barrier_rendezvous"}
            for item in row.get("simulator", {}).get("top_readiness_causes", [])
        )
    ]
    candidate_ranking = [
        {
            "candidate": "dirty_writeback_coherence_retirement",
            "evidence_cells": tail_cells,
            "selection": "next_discriminator",
            "selection_score": residual_score,
            "reason": (
                f"Typed dirty-writeback/coherence tails occur in {len(tail_cells)}/"
                f"{len(comparable)} comparable rows; the summed repeat-stable residual "
                f"is {residual_score} cycles, but a scalar offset cannot separate "
                "retirement from launch or async/barrier behavior."
            ),
        },
        {
            "candidate": "async_destination_service",
            "evidence_cells": service_cells,
            "selection": "candidate_not_admitted",
            "selection_score": len(service_cells),
            "reason": "A modeled readiness cause is present, but no independent silicon observable isolates it from retirement tails.",
        },
        {
            "candidate": "control_and_barrier_readiness",
            "evidence_cells": control_cells,
            "selection": "candidate_not_admitted",
            "selection_score": len(control_cells),
            "reason": "Readiness counters are modeled diagnostics, not an independent live timing label.",
        },
        {
            "candidate": "launch_envelope",
            "evidence_cells": [row["cell"] for row in rows],
            "selection": "candidate_not_admitted",
            "selection_score": len(rows),
            "reason": "GPC elapsed includes launch work not represented by the direct event frontier; it is not a simulator-local latency.",
        },
    ]
    validation = {
        "rows": len(rows),
        "comparable_rows": len(comparable),
        "repeat_stable_rows": len(rows) - len(unstable),
        "retirement_tail_rows": len(tail_rows),
        "required_diagnostic_rows": len(rows) - len(missing_diagnostics),
        "residual_score_cycles": residual_score,
        "residual_min_cycles": min(residuals),
        "residual_max_cycles": max(residuals),
        "decision_eligible": True,
    }
    decision = {
        "decision": "no-go",
        "seam": "cycle scalar completion/retirement mapping",
        "selected_diagnostic": "dirty_writeback_coherence_retirement",
        "reason": (
            f"Live residuals are repeat-stable across {len(rows)} rows, while the "
            f"simulator reports dirty-writeback/coherence tails in every comparable "
            f"row ({len(tail_rows)}/{len(comparable)}); the observations do not identify "
            "one architecture-local latency."
        ),
        "next_action": "Run the matched store-free/barrier-free async-source discriminator with an independent retirement/frontier observable before changing simulator timing state.",
        "claim_limits": [
            "No cycle-accuracy claim",
            "No NCU leakage into simulator inputs",
            "No cross-architecture inference",
            "No JADE/HETR or CPU-GPU co-simulation change",
        ],
    }
    return candidate_ranking, validation, decision


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
        live = _repeat_metrics(root, cell, stages["ncu"])
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

    candidate_ranking, validation, decision = _derive_decision(rows)
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
            "input_validation": validation,
        },
        "candidate_ranking": candidate_ranking,
        "rows": rows,
        "decision": decision,
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
