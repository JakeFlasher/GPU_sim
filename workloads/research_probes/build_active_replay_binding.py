#!/usr/bin/env python3
"""Build the immutable content binding for the retained replay corpus."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workloads.research_probes import active_replay_binding as binding
from workloads.research_probes import validate_active_replay_pairs as validator


DEFAULT_OUTPUT = REPO_ROOT / "evidence/sm86_active_replay_binding.json"


def _revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def build(output: Path) -> dict[str, object]:
    binary = REPO_ROOT / "sim/target/release/sim-run"
    if not binary.is_file() or binary.stat().st_size == 0:
        raise RuntimeError(f"release simulator is missing: {binary}")
    launch_root = validator.LAUNCH.DEFAULT_OUTPUT.resolve()
    retirement_root = validator.RETIREMENT.DEFAULT_OUTPUT.resolve()
    arrival_root = validator.ARRIVAL.DEFAULT_OUTPUT.resolve()
    admission_root = validator.ADMISSION.DEFAULT_OUTPUT.resolve()
    groups = (
        ("launch", validator.launch_pairs(launch_root, binary)),
        ("retirement", validator.retirement_pairs(retirement_root, binary)),
        ("arrival", validator.arrival_pairs(arrival_root, binary)),
        ("admission", validator.admission_pairs(admission_root, binary)),
        (
            "admission-service-sweep",
            validator.admission_service_sweep_pairs(admission_root, binary),
        ),
        ("write-completion", validator.write_completion_pairs(admission_root, binary)),
        ("red-rmw-completion", validator.red_rmw_completion_pairs(admission_root, binary)),
    )
    dependencies: dict[str, str] = {}
    entries: dict[str, list[tuple[str, str]]] = {}
    expected_directories: set[Path] = set()
    labels: set[str] = set()

    def add_pair(
        group: str,
        label: str,
        first: Path,
        second: Path,
        paths: list[Path],
    ) -> None:
        if label in labels:
            raise RuntimeError(f"duplicate replay binding label: {label}")
        labels.add(label)
        expected_directories.add(first.parent.resolve())
        for path in paths:
            key = binding.canonical_path(path)
            if key not in dependencies:
                dependencies[key] = binding.content_hash(path)
        entries.setdefault(group, []).append(
            (label, binding.pair_digest(label, first, second))
        )

    for group, pairs in groups:
        for label, first, second, paths, _expectation in pairs:
            add_pair(group, label, first, second, paths)

    actual_directories = set().union(
        validator.replay_result_directories(launch_root),
        validator.replay_result_directories(retirement_root),
        validator.replay_result_directories(arrival_root),
        validator.replay_result_directories(admission_root),
    )
    supplemental = sorted(
        path
        for path in actual_directories - expected_directories
        if validator._known_supplemental_replay_directory(path, admission_root)
    )
    for extra in supplemental:
        spec = validator._supplemental_pair_spec(extra, admission_root, binary)
        if spec is None:
            raise RuntimeError(f"cannot resolve supplemental replay: {extra}")
        first, second, paths, _expectation, _trace = spec
        add_pair(
            "supplemental",
            f"supplemental/{extra.parent.name}/{extra.name}",
            first,
            second,
            paths,
        )

    group_records = {
        name: {
            "pair_count": len(values),
            "pair_root": binding.root_digest(values),
        }
        for name, values in sorted(entries.items())
    }
    all_entries = [entry for values in entries.values() for entry in values]
    document: dict[str, object] = {
        "schema_version": 1,
        "kind": "active_replay_content_binding",
        "repository_revision": _revision(),
        "simulator_binary": {
            "path": binding.canonical_path(binary),
            "sha256": binding.sha256(binary),
        },
        "architecture_card": {
            "path": "arch_profiles/cards/sm_86.json",
            "sha256": binding.sha256(REPO_ROOT / "arch_profiles/cards/sm_86.json"),
        },
        "pchase_card": {
            "path": "data/pchase/sm_86_static_card.json",
            "sha256": binding.sha256(REPO_ROOT / "data/pchase/sm_86_static_card.json"),
        },
        "dependencies": dict(sorted(dependencies.items())),
        "groups": group_records,
        "active_pair_count": sum(
            record["pair_count"] for name, record in group_records.items() if name != "supplemental"
        ),
        "supplemental_pair_count": group_records.get("supplemental", {}).get("pair_count", 0),
        "corpus_root": binding.root_digest(all_entries),
        "regeneration_command": [
            ".venv/bin/python",
            "workloads/research_probes/build_active_replay_binding.py",
            "--output",
            "evidence/sm86_active_replay_binding.json",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        document = build(args.output.expanduser().resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"active replay binding: {exc}", file=sys.stderr)
        return 1
    print(
        "wrote active replay binding: "
        f"{args.output} ({document['active_pair_count']} active pairs, "
        f"{document['supplemental_pair_count']} supplemental pairs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
