"""Targets-not-features static guard (successor to the retired NCU quarantine guard).

CLAUDE.md Rule 1 (re-founded 2026-07-02): NCU / hardware-counter values are
prediction *targets and validation references only*. They must never enter
simulator inputs: no label-artifact reads, no NCU invocation, and no
label-pipeline imports inside ``sim/`` — except the exact
``sim/run/src/emit.rs`` observation-contract surface, whose whole job is to name
the metrics being emitted (metric *names* are the output contract and are
allowed everywhere; label *values* are not).

Scope now, growth later: scans ``sim/`` sources (Rust + Python + build files).
The eval harness may read labels (it scores predictions against them); the
label factory obviously touches NCU. Those trees are exempt by construction.

Run:  .venv/bin/python -m pytest eval/guards/test_targets_not_features_guard.py -q
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = REPO_ROOT / "sim"
EMIT_CONTRACT_PARTS = ("run", "src", "emit.rs")

# Token classes forbidden inside sim/ sources. Word-ish boundaries where the
# bare token would otherwise be a common substring.
FORBIDDEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "label artifact path": re.compile(r"data/local_labels|data/annex"),
    "admitted-label artifact": re.compile(r"label_admitted|label_repeatability|label_excluded"),
    "NCU report artifact": re.compile(r"\.ncu-rep|ncu_report|ncu-report"),
    "label-pipeline import": re.compile(
        r"(import_label|parse_ncu_metrics|capture_(count|timing|cache)_labels|"
        r"repeatability_aggregator)"
    ),
    "NCU invocation": re.compile(r"""(?x)
        (subprocess|Command::new|process::Command)[^\n]{0,120}
        ["'](/[\w/.-]*)?ncu["' ]
    """),
}

# The emission layer's contract surface may name metrics/contracts freely; it
# still must not read label artifacts, so only the import/invocation classes
# are relaxed there — artifact-path classes stay enforced.
EMIT_RELAXED_CLASSES = {"label-pipeline import"}
SCAN_SUFFIXES = {".rs", ".py", ".toml", ".cc", ".cpp", ".h", ".hpp"}


def _sim_sources():
    if not SIM_ROOT.is_dir():
        return
    # Cargo build-output dirs, excluded EXPLICITLY (never by bare component
    # name: a future source dir that happens to be called "target" must stay
    # scanned).
    excluded_build_dirs = (
        ("target",),                    # sim/target/
        ("trace", "pbgen", "target"),   # sim/trace/pbgen/target/
    )
    for p in sorted(SIM_ROOT.rglob("*")):
        if not (p.is_file() and p.suffix in SCAN_SUFFIXES):
            continue
        rel_parts = p.relative_to(SIM_ROOT).parts
        if any(rel_parts[: len(pre)] == pre for pre in excluded_build_dirs):
            continue
        yield p


def _is_emit_contract(src: Path) -> bool:
    return src.relative_to(SIM_ROOT).parts == EMIT_CONTRACT_PARTS


def test_sim_sources_never_consume_ncu_labels():
    violations: list[str] = []
    scanned = 0
    for src in _sim_sources():
        scanned += 1
        rel = src.relative_to(REPO_ROOT)
        in_emit = _is_emit_contract(src)
        text = src.read_text(encoding="utf-8", errors="replace")
        for cls, pat in FORBIDDEN_PATTERNS.items():
            if in_emit and cls in EMIT_RELAXED_CLASSES:
                continue
            for m in pat.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                violations.append(f"{rel}:{line}: [{cls}] {m.group(0)[:80]!r}")
    assert not violations, (
        "targets-not-features violations inside sim/ "
        "(NCU labels are targets, never inputs):\n" + "\n".join(violations)
    )
    # sim/ may legitimately not exist before the Phase-0 scaffold lands; the
    # registry assertion below still runs, so the guard is never vacuous.


def test_emit_relaxation_is_exact_to_current_module():
    current = SIM_ROOT.joinpath(*EMIT_CONTRACT_PARTS)
    assert current.is_file()
    assert _is_emit_contract(current)
    for alias in (
        SIM_ROOT / "emit/src/lib.rs",
        SIM_ROOT / "run/emit.rs",
        SIM_ROOT / "run/src/emit/mod.rs",
        SIM_ROOT / "run/tests/emit.rs",
    ):
        assert not _is_emit_contract(alias)


def test_metric_registry_declares_model_io_role_or_deployability():
    """Transition assertion for the Rule-1 enforcement field.

    Registry v2 entries carry ``deployability`` today; the Phase-1B migration
    renames it to ``model_io_role`` (target | input | either). Every metric
    must carry exactly one of the two — a metric with neither is unguardable.
    """
    reg_path = REPO_ROOT / "tools/local_labels/metric_registry.json"
    doc = json.loads(reg_path.read_text(encoding="utf-8"))
    metrics = doc.get("metrics", doc)
    if isinstance(metrics, dict):
        items = metrics.items()
    else:
        items = ((m.get("label_field", str(i)), m) for i, m in enumerate(metrics))
    missing = [
        name for name, spec in items
        if isinstance(spec, dict)
        and "deployability" not in spec
        and "model_io_role" not in spec
    ]
    assert not missing, (
        "metric_registry entries lacking both 'deployability' and "
        f"'model_io_role': {missing}"
    )
