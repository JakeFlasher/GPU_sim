from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from validate_sm86_predictive_config import validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "docs" / "sm86_predictive_closure_config_20261001.json"


def load_config() -> dict:
    return json.loads(CONFIG.read_text())


def test_predictive_config_is_valid() -> None:
    assert validate_config(load_config())["schema"] == "sm86_predictive_closure_config_v1"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("input_contract", "ncu_labels"), "feature"),
        (("route_quotient", "physical_route_labels"), "observed"),
        (("cache_quotient", "hardware_dram_completion"), "observed"),
        (("validation", "score_split"), "random"),
    ],
)
def test_predictive_config_rejects_contract_regressions(path: tuple[str, str], value: str) -> None:
    raw = copy.deepcopy(load_config())
    raw[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        validate_config(raw)


def test_predictive_config_rejects_scope_expansion() -> None:
    raw = copy.deepcopy(load_config())
    raw["scope_policy"] = "recover_vendor_internals"
    with pytest.raises(ValueError):
        validate_config(raw)
