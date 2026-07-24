"""Lock the SOL authoring tools to the wired KB routing contract."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_router_and_linter_share_data_only_contract() -> None:
    contract = _load("kb_routing_contract", ROOT / "scripts/kb/routing_contract.py")
    expected = contract.SUBCORPUS_TO_SKILL

    assert expected["problems"] == "implementing-sol-baselines"
    assert expected["cutlass_constructs"] == "implementing-sol-baselines"
    assert expected["_root"] == "navigating-project-kb"

    router_text = (ROOT / "scripts/kb/route_query.py").read_text(encoding="utf-8")
    linter_text = (ROOT / "tools/sol_kb/lint_cards.py").read_text(encoding="utf-8")
    assert "from routing_contract import SUBCORPUS_TO_SKILL" in router_text
    assert "from routing_contract import (" in linter_text


def test_cutlass_layout_anchor_is_compound_and_narrow() -> None:
    router = _load("kb_route_query", ROOT / "scripts/kb/route_query.py")
    matches = [
        (pattern, subcorpora)
        for pattern, subcorpora, _boost in router.SKILL_ANCHORS
        if "cutlass_constructs" in subcorpora
    ]
    assert matches
    assert any(
        pattern.search("CuTe Layout reference for CUTLASS C++")
        for pattern, _subcorpora in matches
    )
    assert any(
        pattern.search("sm100_blockscaled_layout tcgen05.mma.block_scale")
        for pattern, _subcorpora in matches
    )
    assert not any(
        pattern.search(prompt)
        for pattern, _subcorpora in matches
        for prompt in ("memory layout in Rust", "layout of a manuscript")
    )


def test_generated_bash_gate_matches_python_emitter() -> None:
    emitted = subprocess.check_output(
        ["python3", str(ROOT / "scripts/kb/_relevance.py"), "--emit-bash"]
    )
    checked_in = (ROOT / ".claude/hooks/_gate_regexes.sh").read_bytes()
    assert checked_in == emitted


def test_generated_bash_gate_handles_long_prompt_without_argv_overflow() -> None:
    prompt = ("GPU simulator trace timing " * 20_000).encode()
    result = subprocess.run(
        [
            "bash",
            "-c",
            "source \"$1\"; prompt=$(cat); kb_gate_is_relevant \"$prompt\"",
            "bash",
            str(ROOT / ".claude/hooks/_gate_regexes.sh"),
        ],
        check=False,
        capture_output=True,
        input=prompt,
    )
    assert result.returncode == 0
    assert result.stdout == b""
    assert result.stderr == b""


def test_retired_measurement_skill_is_labeled_as_non_authoritative() -> None:
    router = _load("kb_route_query_formatter", ROOT / "scripts/kb/route_query.py")
    rendered = router._format_reminder(
        [{"path": ".claude/knowledge/measurement/pinned-protocol-v21.md",
          "skill": "validating-nonroot-gpu-measurement"}],
        ["validating-nonroot-gpu-measurement"],
    )
    assert "historical pointer" in rendered
    assert "do not apply its rules" in rendered
    assert "root CLAUDE.md" in rendered


def test_only_unique_sol_authoring_tools_remain() -> None:
    root_files = {
        path.name
        for path in (ROOT / "tools/sol_kb").iterdir()
        if path.is_file() and path.suffix in {".py", ".yaml"}
    }
    test_files = {name for name in root_files if name.startswith("test_")}
    assert test_files == {
        "test_canonical_router_parity.py",
        "test_generate_dossiers.py",
        "test_lint_cards.py",
    }
    assert root_files - test_files == {
        "generate_dossiers.py",
        "generate_patterns.py",
        "lint_cards.py",
    }
