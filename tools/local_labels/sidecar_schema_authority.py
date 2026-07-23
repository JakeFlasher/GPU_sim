"""Pinned authority for the active v5 local-label sidecar schema."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    REPO_ROOT / "workloads/simbench_bw/schemas/local_label_sidecar.schema.json"
)
SCHEMA_SHA256 = "d8c06d477050d5522b967fa5e6fc5bdb9e3b6637989ea6fe60bc1c235d741564"
SCHEMA_ID = "simbench_bw/schemas/local_label_sidecar.schema.json"
SCHEMA_VERSION = 5


def parse_authoritative_schema_bytes(raw: bytes) -> dict[str, Any]:
    if hashlib.sha256(raw).hexdigest() != SCHEMA_SHA256:
        raise ValueError("sidecar schema bytes do not match the pinned authority")
    try:
        schema = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"sidecar schema is not valid JSON: {exc}") from exc
    if (
        not isinstance(schema, dict)
        or schema.get("$id") != SCHEMA_ID
        or (schema.get("properties") or {}).get("schema_version", {}).get(
            "const"
        ) != SCHEMA_VERSION
    ):
        raise ValueError("sidecar schema authority is not the v5 contract")
    return schema


def load_authoritative_schema() -> tuple[bytes, dict[str, Any]]:
    try:
        raw = SCHEMA_PATH.read_bytes()
    except OSError as exc:
        raise ValueError(f"sidecar schema unavailable: {exc}") from exc
    return raw, parse_authoritative_schema_bytes(raw)
