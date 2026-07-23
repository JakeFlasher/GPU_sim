"""Digest-keyed sealed-holdout touch ledger + retune-after-holdout guard.

The sealed-holdout discipline says the holdout is scored ONCE per
fitted/scored head, at final evaluation, and producer/model configuration
must not be re-tuned afterwards against the same seal. Today that
discipline is structural (holdout cells never enter a fit); this module
makes it AUDITABLE and fail-closed:

  - every holdout scoring event ("touch") is recorded in a committed
    ledger keyed on the sealed split's ``cohort_manifest_digest`` plus a
    ``scope`` naming what was scored (e.g. ``t1_count_replacement``);
  - a SECOND touch for the same (digest, scope) raises the typed
    ``HoldoutTouchRepeatError`` — one-touch is enforced, not just
    documented;
  - a configuration change after a touch is rejected: scoring machinery
    calls ``assert_retune_legal`` with its current ``config_digest``
    BEFORE consuming holdout labels; if the same (digest, scope) was
    already touched under a DIFFERENT config digest, the typed
    ``HoldoutRetuneAfterTouchError`` is raised — re-tuning against an
    already-touched seal requires a fresh re-seal/split regeneration
    (a new cohort_manifest_digest), never a silent re-run.

The ledger is consumed by the count-label governance trail now and by the
T1 lockstep validator at its one-touch final evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = (
    REPO_ROOT / "data" / "local_labels" / "holdout_touch_ledger_sm_86.json"
)

LEDGER_SCHEMA_VERSION = 1


class HoldoutTouchRepeatError(ValueError):
    """A second holdout scoring touch for the same sealed digest + scope.
    The sealed holdout is scored exactly once per seal; re-scoring
    requires a fresh re-seal (new cohort_manifest_digest)."""

    typed_reason = "holdout_touched_more_than_once"


class HoldoutRetuneAfterTouchError(ValueError):
    """A producer/model configuration change after a holdout touch on the
    same seal. The holdout verdict is bound to the configuration that
    scored it; a changed configuration must re-seal before touching the
    holdout again."""

    typed_reason = "producer_retune_after_holdout_touch"


def empty_ledger() -> dict[str, Any]:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "artifact_kind": "holdout_touch_ledger",
        "arch": "sm_86",
        "touches": [],
    }


def load_ledger(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_LEDGER
    if not p.is_file():
        return empty_ledger()
    return json.loads(p.read_text(encoding="utf-8"))


def _touches_for(
    ledger: dict[str, Any], *, sealed_digest: str, scope: str,
) -> list[dict[str, Any]]:
    return [
        t for t in ledger.get("touches", [])
        if t.get("sealed_digest") == sealed_digest
        and t.get("scope") == scope
    ]


def assert_touch_legal(
    ledger: dict[str, Any], *, sealed_digest: str, scope: str,
) -> None:
    """One-touch enforcement: raise if (sealed_digest, scope) was already
    scored."""
    prior = _touches_for(ledger, sealed_digest=sealed_digest, scope=scope)
    if prior:
        raise HoldoutTouchRepeatError(
            f"{HoldoutTouchRepeatError.typed_reason}: sealed digest "
            f"{sealed_digest[:12]}.../{scope} was already scored at "
            f"{prior[0].get('source_commit', 'unknown')!r}; the sealed "
            "holdout is scored exactly once per seal."
        )


def assert_retune_legal(
    ledger: dict[str, Any],
    *,
    sealed_digest: str,
    scope: str,
    config_digest: str,
) -> None:
    """Retune-after-holdout guard: once (sealed_digest, scope) has been
    touched, ANY differing configuration digest is rejected until a fresh
    re-seal lands."""
    for touch in _touches_for(
        ledger, sealed_digest=sealed_digest, scope=scope,
    ):
        if touch.get("config_digest") != config_digest:
            raise HoldoutRetuneAfterTouchError(
                f"{HoldoutRetuneAfterTouchError.typed_reason}: sealed "
                f"digest {sealed_digest[:12]}.../{scope} was scored under "
                f"config {str(touch.get('config_digest'))[:12]}...; the "
                f"current config {config_digest[:12]}... differs. "
                "Re-tuning after a holdout touch requires a fresh re-seal "
                "(new cohort_manifest_digest)."
            )


def record_touch(
    path: str | Path | None = None,
    *,
    sealed_digest: str,
    scope: str,
    scored_artifact: str,
    config_digest: str,
    source_commit: str,
) -> dict[str, Any]:
    """Record a holdout scoring touch (enforcing one-touch first) and
    persist the ledger. Returns the updated ledger dict."""
    p = Path(path) if path is not None else DEFAULT_LEDGER
    ledger = load_ledger(p)
    assert_touch_legal(ledger, sealed_digest=sealed_digest, scope=scope)
    ledger.setdefault("touches", []).append({
        "sealed_digest": sealed_digest,
        "scope": scope,
        "scored_artifact": scored_artifact,
        "config_digest": config_digest,
        "source_commit": source_commit,
    })
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ledger
