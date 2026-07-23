"""Consumer surface for the committed observation contract.

The lockstep validator (and any other producer-count consumer) MUST
route every metric through :func:`assert_metric_scoreable` before
scoring. Closed-world, fail-closed:

- a metric with NO contract entry is rejected
  (``contract_entry_missing``) — scoring an unpinned semantic is the
  exact silent-divergence failure the contract exists to prevent;
- ``contract_mismatch`` / ``contract_underdetermined`` entries are
  rejected with their own typed reasons (the fixtures could not pin a
  unique semantic);
- ``no_expressible_fixture`` entries are rejected for scoring (the
  validator records them as typed-skipped, never scored vacuously);
- a pinned entry returns its semantic plus any ``semantic_scope``
  restriction the caller must enforce (e.g. sector arithmetic is
  pinned for 128B-aligned warp footprints ONLY).
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_CONTRACT_PATH = (
    REPO_ROOT / "data" / "local_labels" / "observation_contract_sm_86.json"
)


class ContractEntryMissingError(RuntimeError):
    typed_reason = "contract_entry_missing"


class ContractMismatchError(RuntimeError):
    typed_reason = "contract_mismatch"


class ContractUnderdeterminedError(RuntimeError):
    typed_reason = "contract_underdetermined"


class ContractNoExpressibleFixtureError(RuntimeError):
    typed_reason = "no_expressible_fixture"


_STATUS_ERRORS = {
    "contract_mismatch": ContractMismatchError,
    "contract_underdetermined": ContractUnderdeterminedError,
    "no_expressible_fixture": ContractNoExpressibleFixtureError,
}


def load_contract(path: Path | None = None) -> dict:
    contract_path = path or DEFAULT_CONTRACT_PATH
    return json.loads(contract_path.read_text(encoding="utf-8"))


def assert_metric_scoreable(
    metric: str, contract: dict | None = None,
) -> dict:
    """Return the pinned entry for ``metric`` or raise the typed
    rejection. The returned entry carries ``semantic`` and (when
    present) ``semantic_scope`` — the caller MUST honor the scope
    restriction or reject the cell."""
    doc = contract if contract is not None else load_contract()
    entry = doc.get("metrics", {}).get(metric)
    if entry is None:
        raise ContractEntryMissingError(
            f"no observation-contract entry for {metric!r}: scoring an "
            "unpinned counting semantic is rejected (fail-closed)"
        )
    status = entry.get("status")
    if status == "pinned":
        return entry
    err = _STATUS_ERRORS.get(status)
    if err is None:
        raise ContractEntryMissingError(
            f"unknown contract status {status!r} for {metric!r}"
        )
    raise err(
        f"{metric!r} is typed {status!r} in the observation contract; "
        "it must be recorded as typed-unscoreable, never scored"
    )


def scoreable_metrics(contract: dict | None = None) -> tuple[str, ...]:
    doc = contract if contract is not None else load_contract()
    return tuple(sorted(
        m for m, e in doc.get("metrics", {}).items()
        if e.get("status") == "pinned"
    ))
