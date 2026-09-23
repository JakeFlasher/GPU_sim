import json

import pytest

pytest.importorskip("jsonschema")

from tools.cache_predictor.l1_static_card import (
    L1StaticCardEntryInvalid,
    _TABLE_PATH,
    load_l1_static_card_table,
)


@pytest.mark.parametrize("payload", ["{", "[]"])
def test_static_card_parse_failures_stay_typed(tmp_path, payload):
    path = tmp_path / "static-card.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(L1StaticCardEntryInvalid, match="predictor_l1_capacity_invalid"):
        load_l1_static_card_table(path)


def test_static_card_duplicate_architecture_is_typed(tmp_path):
    document = json.loads(_TABLE_PATH.read_text(encoding="utf-8"))
    document["entries"].append(dict(document["entries"][0]))
    path = tmp_path / "duplicate-static-card.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(L1StaticCardEntryInvalid, match="duplicate architecture"):
        load_l1_static_card_table(path)


def test_static_card_schema_parse_failure_is_typed(tmp_path, monkeypatch):
    broken_schema = tmp_path / "broken-schema.json"
    broken_schema.write_text("{", encoding="utf-8")
    monkeypatch.setattr(
        "tools.cache_predictor.l1_static_card._TABLE_SCHEMA_PATH",
        broken_schema,
    )
    with pytest.raises(L1StaticCardEntryInvalid, match="schema is not valid JSON"):
        load_l1_static_card_table(_TABLE_PATH)
