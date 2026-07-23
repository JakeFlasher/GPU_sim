"""Canonical build-variant identity normalizer (stdlib-only, dependency-free).

This is the SINGLE source of truth for normalizing a record's optional
register-cap build-variant identity (the ``register_cap`` int + ``variant_id``
string pair). It lives in its own module — importing NOTHING beyond the standard
library — so it can be shared by BOTH heavy consumers (``repeatability_aggregator``,
``check_admission_floor`` — which also import jsonschema) AND the lightweight
``sidecar_emitter`` that ``label_driver.sh`` invokes under a bare system
``python3`` that may not have jsonschema installed. Keeping one implementation
avoids the divergence failure class a re-inlined second copy would reintroduce.
"""

from __future__ import annotations

VARIANT_PREFIX = "register_cap_"


def variant_identity(record: dict) -> tuple[str, int | None, str | None]:
    """Normalize a record's OPTIONAL build-variant identity to a single,
    consistent ``(variant_id, register_cap, error)`` triple.

    The two optional wire fields are ``register_cap`` (int) and ``variant_id``
    (str, ``register_cap_<cap>``). Routing every consumer + producer through this
    one helper removes any divergence between how the two fields are preferred:

    - both absent        -> base cell: ``("", None, None)``.
    - only register_cap  -> derive: ``("register_cap_<cap>", cap, None)``.
    - only variant_id    -> require the CANONICAL form ``register_cap_<int>``
      (ASCII-decimal, no leading zeros) and derive register_cap; a noncanonical
      alias (e.g. ``register_cap_064``) returns a non-None ``error``.
    - both present       -> require ``variant_id == register_cap_<register_cap>``
      EXACTLY (canonical string equality, not just numeric-suffix equality);
      a mismatch or noncanonical alias returns a non-None ``error``.

    The canonical-form requirement is what stops one physical cap from being
    admitted under two distinct variant ids (``register_cap_64`` vs.
    ``register_cap_064``). When ``error`` is not None the values are returned
    as-observed (for the diagnostic) but the caller MUST fail closed so a
    contradictory / noncanonical pair never reaches admitted output.
    """
    raw_variant_id = record.get("variant_id")
    raw_register_cap = record.get("register_cap")
    # PRESENCE-vs-MALFORMED type gate (fail closed; do NOT silently drop):
    # None means genuinely-absent; a present value of the wrong type is a
    # malformed wire value that must ERROR, not be read as absent. Silently
    # dropping a present-but-wrong-type field would trust the OTHER field alone
    # and mask a real disagreement -- e.g. a register_cap given as the JSON
    # float 32.0 (which jsonschema draft 2020-12 ACCEPTS as type:integer, so it
    # reaches here from a schema-valid artifact) would be dropped, the
    # variant_id trusted alone, and a cell physically built at cap 24 could be
    # admitted as the register_cap_64 variant. bool is excluded from the int
    # domain because `isinstance(True, int)` is True yet a bool cap is
    # meaningless.
    if raw_variant_id is not None and not isinstance(raw_variant_id, str):
        return (
            "", None,
            f"variant_id {raw_variant_id!r} is not a string "
            f"(got {type(raw_variant_id).__name__})",
        )
    if raw_register_cap is not None and (
        isinstance(raw_register_cap, bool)
        or not isinstance(raw_register_cap, int)
    ):
        return (
            raw_variant_id if isinstance(raw_variant_id, str) else "",
            None,
            f"register_cap {raw_register_cap!r} is not an integer "
            f"(got {type(raw_register_cap).__name__})",
        )
    has_variant_id = isinstance(raw_variant_id, str) and raw_variant_id != ""
    has_register_cap = (
        isinstance(raw_register_cap, int) and not isinstance(raw_register_cap, bool)
    )
    if not has_variant_id and not has_register_cap:
        return ("", None, None)

    # Non-negative value domain (shared by every consumer + producer). A
    # negative register_cap would otherwise derive a schema-invalid
    # variant_id (e.g. "register_cap_-1": register_cap has minimum 0 and the
    # variant_id pattern is ^register_cap_[0-9]+$). The variant_id-only and
    # both-present paths already reject negatives via the ASCII-decimal
    # canonical check below; this guards the register-cap-only path so the
    # producer fails closed before writing a sidecar its own importer rejects.
    if has_register_cap and raw_register_cap < 0:
        return (
            raw_variant_id if has_variant_id else "",
            raw_register_cap,
            f"register_cap {raw_register_cap} is negative (must be >= 0)",
        )

    parsed_cap: int | None = None
    if has_variant_id:
        suffix = raw_variant_id[len(VARIANT_PREFIX):]
        # Require an ASCII-decimal suffix (mirrors the schema's [0-9]); a
        # non-ASCII "digit" would pass str.isdigit() but break int().
        if not (
            raw_variant_id.startswith(VARIANT_PREFIX)
            and suffix.isascii()
            and suffix.isdigit()
        ):
            return (
                raw_variant_id,
                raw_register_cap if has_register_cap else None,
                f"variant_id {raw_variant_id!r} is not of the form "
                f"register_cap_<int>",
            )
        parsed_cap = int(suffix)
        # Canonical-form check: the variant_id must equal the EXACT canonical
        # decimal string, so a noncanonical alias (leading zeros, e.g.
        # register_cap_064) can never be admitted as a build variant distinct
        # from register_cap_64. Comparing only the parsed integer would let the
        # alias through and split/duplicate variant-aware cell accounting.
        canonical = f"{VARIANT_PREFIX}{parsed_cap}"
        if raw_variant_id != canonical:
            return (
                raw_variant_id,
                raw_register_cap if has_register_cap else parsed_cap,
                f"variant_id {raw_variant_id!r} is not canonical "
                f"(expected {canonical!r})",
            )

    if has_variant_id and has_register_cap:
        if parsed_cap != raw_register_cap:
            return (
                raw_variant_id,
                raw_register_cap,
                f"variant_id {raw_variant_id!r} disagrees with "
                f"register_cap {raw_register_cap}",
            )
        return (raw_variant_id, raw_register_cap, None)
    if has_register_cap:  # register_cap only -> derive variant_id
        return (f"{VARIANT_PREFIX}{raw_register_cap}", raw_register_cap, None)
    return (raw_variant_id, parsed_cap, None)  # variant_id only (canonical)
