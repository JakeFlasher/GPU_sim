"""Fail-closed ArchProfile schema/Rust admission parity."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "arch_profiles" / "schema" / "arch_profile.schema.json"
CARDS_DIR = REPO_ROOT / "arch_profiles" / "cards"
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
VALIDATOR = Draft7Validator(SCHEMA)


def _card(name: str) -> dict:
    return json.loads((CARDS_DIR / name).read_text(encoding="utf-8"))


def _errors(card: dict) -> list:
    return sorted(VALIDATOR.iter_errors(card), key=lambda error: list(error.path))


ISSUE_RESOURCE_KEYS = (
    "smsp",
    "occupancy",
    "dep_barriers",
    "barrier_latencies",
    "dual_issue",
)


def _synthetic_issue_resources() -> dict:
    """Non-authoritative sentinels used only to exercise the closed schema."""
    return {
        "smsp": {"count": 3, "issue_ports": 5},
        "occupancy": {"max_resident_warps_per_sm": 7, "max_blocks": 11},
        "dep_barriers": {"count": 13},
        "barrier_latencies": {"read": 17, "non_memory_write": 19},
        "dual_issue": {"supported": True},
    }


def test_schema_is_well_formed_and_current_cards_validate() -> None:
    Draft7Validator.check_schema(SCHEMA)
    for path in sorted(CARDS_DIR.glob("*.json")):
        card = json.loads(path.read_text(encoding="utf-8"))
        assert not _errors(card), path


def test_synthetic_issue_resource_blocks_are_strict_and_optional() -> None:
    card = _card("sm_86.json")
    assert all(key not in card for key in ISSUE_RESOURCE_KEYS)
    card.update(_synthetic_issue_resources())
    assert not _errors(card)

    for key in ISSUE_RESOURCE_KEYS:
        gapped = copy.deepcopy(card)
        gapped[key] = {
            "status": "research_gap",
            "reason": f"synthetic {key} gap",
            "challenge": "fixture-only",
        }
        assert not _errors(gapped), key

    nested_gap = copy.deepcopy(card)
    nested_gap["dual_issue"]["supported"] = {
        "status": "research_gap",
        "reason": "synthetic pairing gap",
    }
    assert not _errors(nested_gap)


def test_issue_resource_numeric_fields_are_positive_and_bounded() -> None:
    locations = {
        "smsp": (("count", "issue_ports"), (1 << 32) - 1),
        "occupancy": (
            ("max_resident_warps_per_sm", "max_blocks"),
            (1 << 32) - 1,
        ),
        "dep_barriers": (("count",), (1 << 32) - 1),
        "barrier_latencies": (("read", "non_memory_write"), (1 << 53) - 1),
    }
    for block, (fields, maximum) in locations.items():
        for field in fields:
            card = _card("sm_86.json")
            card.update(_synthetic_issue_resources())
            card[block][field] = 1.0
            assert not _errors(card), (block, field, "integral JSON number")

            card[block][field] = maximum
            assert not _errors(card), (block, field, "maximum")

            card[block][field] = 0
            assert _errors(card), (block, field, 0)

            card[block][field] = maximum + 1
            assert _errors(card), (block, field, "overflow")


@pytest.mark.parametrize(
    "block,value",
    [
        ("smsp", None),
        ("smsp", {"count": 3}),
        ("smsp", {"count": 3, "issue_ports": 5, "unknown": 1}),
        (
            "smsp",
            {
                "status": "research_gap",
                "reason": "collision",
                "count": 3,
                "issue_ports": 5,
            },
        ),
        ("smsp", {"status": "unknown", "reason": "bad status"}),
        ("dual_issue", {"supported": "unknown"}),
        ("dual_issue", {"supported": {"status": "research_gap"}}),
        (
            "dual_issue",
            {
                "supported": {
                    "status": "research_gap",
                    "reason": "pending",
                    "unknown": 1,
                }
            },
        ),
    ],
)
def test_issue_resource_arms_reject_malformed_or_contaminated_shapes(
    block: str, value: object
) -> None:
    card = _card("sm_86.json")
    card.update(_synthetic_issue_resources())
    card[block] = value
    assert _errors(card)


@pytest.mark.parametrize(
    "family",
    [
        [],
        ["sm_120", "sm_120"],
        ["sm_120", "sm_86"],
        ["sm_120", "sm_120z"],
    ],
)
def test_build_arch_family_rejects_empty_duplicate_cross_arch_and_unknown(
    family: list[str],
) -> None:
    card = _card("sm_120-laptop.json")
    card["build_arch_family"] = family
    assert _errors(card)


@pytest.mark.parametrize(
    "entry",
    [
        {"status": "research_gap", "reason": "pending", "value": 1.0},
        {"status": "research_gap", "reason": "pending", "dispersion": 1.0},
        {"status": "research_gap", "reason": "pending", "unit": "cycles"},
        {"status": "research_gap", "reason": "pending", "unknown_payload": 1},
        {"status": "research_gap"},
        {"value": 1.0, "reason": "gap-only"},
        {"value": 1.0, "challenge": "gap-only"},
        {"value": 1.0, "unknown_payload": 1},
    ],
)
def test_measured_scalar_arms_are_exclusive_and_closed(entry: dict) -> None:
    card = _card("sm_86.json")
    card["latencies_cycles"]["l1_hit"] = entry
    assert _errors(card)


def test_valid_scalar_metadata_remains_admitted() -> None:
    card = _card("sm_86.json")
    card["latencies_cycles"]["l1_hit"] = {
        "value": 1.0,
        "dispersion": 0.1,
        "unit": "cycles",
    }
    card["latencies_cycles"]["mshr_outstanding_misses"] = {
        "status": "research_gap",
        "reason": "pending",
        "challenge": "retained",
    }
    assert not _errors(card)


@pytest.mark.parametrize(
    "name,declaration",
    [
        ("sm_86.json", "unknown_typed_unavailable"),
        ("sm_86.json", "bypass"),
        ("sm_120-laptop.json", "ca"),
        ("sm_120-laptop.json", "bypass"),
    ],
)
def test_bare_ldg_declaration_must_match_architecture_capability(
    name: str, declaration: str
) -> None:
    assert "bypass" in SCHEMA["properties"]["bare_ldg_e_default"]["enum"]
    card = copy.deepcopy(_card(name))
    card["bare_ldg_e_default"] = declaration
    assert _errors(card)
