"""Cold/warm launch protocol provenance for the predictor.

Calibration and prediction must agree on the launch protocol
used to gather per-cell evidence. NCU's standard invocation
`--launch-skip N --launch-count 1` skips N warmup launches and
then measures exactly one launch; the resulting per-cell
ledger entry carries:

* `launch_skip_count` (int >= 1) — the value of `N`. A value
  of `0` means NCU is profiling the first launch, before any
  warmup launch has established a measured cache state; the
  derive helper rejects `0` with a typed reason so a
  measurement always follows at least one warmup launch.
* `warmup_protocol` (string enum) — names the protocol family
  the cell was calibrated under:
    - `ncu_launch_skip_n_count_1`: the standard NCU
      invocation form.
    - `warm_cold_pair`: Zhang USENIX'24 paired-launch
      protocol for the predictor-hostile cohort
      (persisting-window, prefetch-hint, warm-cold-pair
      cells). Calibration uses this; prediction must
      explicitly opt in.
    - `unknown`: typed-unavailable sentinel.
* `cache_initial_state` (string enum) — derived from
  `launch_skip_count` plus the protocol family. Reflects
  the empirical cache state at the start of the measured
  launch under each protocol's invocation form:
    - `cold` under the standard NCU
      `--launch-skip N --launch-count 1` invocation. NCU's
      default `--replay-mode kernel` re-runs the profiled
      kernel multiple times (once per metric subset) and
      flushes the L1 cache between metric replays. The
      "measured" launch the harness observes is therefore
      a cold-start replay even though it follows N nominal
      warmup launches in trace order.
    - `cold` under the `warm_cold_pair` protocol: the
      measured launch follows an explicit cold-cache reset
      between the warm calibration and the cold measurement.
    - `unknown` when the protocol is unknown.

  Calibration-side detail: `--replay-mode application`
  would preserve cache state across metric replays but
  changes the NCU-label measurement contract; the project
  pins kernel replay, so the canonical NCU protocol's
  `cache_initial_state` is `cold`.

A predictor that scores a cell whose calibration ledger and
prediction request disagree on any of the three fields must
fail closed with `typed_unavailable: warmup_protocol_mismatch`
rather than mis-calibrating. A prediction record missing
`cache_initial_state` provenance entirely must fail the
schema validator with
`warmup_protocol_cache_initial_state_missing`.

These checks gate the predictor body's cell scoring. They live
in their own module so the schema-guard and protocol-match
discipline can be tested in isolation before the predictor
body integrates them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Protocol family identifiers.
WARMUP_PROTOCOL_NCU_LAUNCH_SKIP_N_COUNT_1: str = (
    "ncu_launch_skip_n_count_1"
)
WARMUP_PROTOCOL_WARM_COLD_PAIR: str = "warm_cold_pair"
WARMUP_PROTOCOL_UNKNOWN: str = "unknown"

ALL_WARMUP_PROTOCOLS: frozenset[str] = frozenset({
    WARMUP_PROTOCOL_NCU_LAUNCH_SKIP_N_COUNT_1,
    WARMUP_PROTOCOL_WARM_COLD_PAIR,
    WARMUP_PROTOCOL_UNKNOWN,
})


# Cache-initial-state identifiers.
CACHE_INITIAL_STATE_COLD: str = "cold"
CACHE_INITIAL_STATE_WARM: str = "warm"
CACHE_INITIAL_STATE_UNKNOWN: str = "unknown"

ALL_CACHE_INITIAL_STATES: frozenset[str] = frozenset({
    CACHE_INITIAL_STATE_COLD,
    CACHE_INITIAL_STATE_WARM,
    CACHE_INITIAL_STATE_UNKNOWN,
})


# Typed-reason vocabulary. Downstream typed-unavailable bundle
# composition keys decisions off these strings.
WARMUP_PROTOCOL_TYPED_REASONS: frozenset[str] = frozenset({
    "warmup_protocol_mismatch",
    "warmup_protocol_launch_skip_zero",
    "warmup_protocol_launch_skip_negative",
    "warmup_protocol_unknown",
    "warmup_protocol_cache_initial_state_missing",
})


class WarmupProtocolError(Exception):
    """Raised by the protocol helpers when a record / pair of
    records cannot be honestly admitted. Carries the
    typed-reason string for downstream typed-unavailable
    bundle composition.
    """

    def __init__(self, typed_reason: str, message: str) -> None:
        super().__init__(f"{typed_reason}: {message}")
        self.typed_reason = typed_reason
        self.message = message


@dataclass(frozen=True)
class WarmupProtocolRecord:
    """Immutable per-cell record carrying the three
    cold/warm-state provenance fields: `launch_skip_count`,
    `warmup_protocol`, and `cache_initial_state`.
    """

    launch_skip_count: int
    warmup_protocol: str
    cache_initial_state: str


def derive_warmup_protocol(
    launch_skip_count: int,
    *,
    protocol: str = WARMUP_PROTOCOL_NCU_LAUNCH_SKIP_N_COUNT_1,
) -> WarmupProtocolRecord:
    """Build a normalized `WarmupProtocolRecord` from the
    NCU-style invocation parameters.

    `launch_skip_count` MUST be `>= 1` under the recognized
    protocols. `0` raises `warmup_protocol_launch_skip_zero`:
    a launch-skip of zero measures the first launch, which
    has not yet been preceded by any warmup launch, so the
    measured cache state is undefined. Negative values raise
    `warmup_protocol_launch_skip_negative`.

    `protocol == WARMUP_PROTOCOL_UNKNOWN` returns a record
    with `cache_initial_state=unknown`, signaling the cell
    should be routed to typed-unavailable downstream.

    Any other unrecognized `protocol` string raises
    `warmup_protocol_unknown`.
    """
    if launch_skip_count < 0:
        raise WarmupProtocolError(
            "warmup_protocol_launch_skip_negative",
            f"launch_skip_count must be >= 1; got "
            f"{launch_skip_count}",
        )
    if launch_skip_count == 0:
        raise WarmupProtocolError(
            "warmup_protocol_launch_skip_zero",
            "launch_skip_count=0 profiles the warmup launch "
            "itself, not the measured launch. Use "
            "launch_skip_count >= 1 so the measurement target "
            "follows at least one warmup launch.",
        )
    if protocol == WARMUP_PROTOCOL_UNKNOWN:
        return WarmupProtocolRecord(
            launch_skip_count=launch_skip_count,
            warmup_protocol=WARMUP_PROTOCOL_UNKNOWN,
            cache_initial_state=CACHE_INITIAL_STATE_UNKNOWN,
        )
    if protocol == WARMUP_PROTOCOL_NCU_LAUNCH_SKIP_N_COUNT_1:
        # Standard NCU `--launch-skip N --launch-count 1`
        # invocation under the default `--replay-mode kernel`:
        # NCU re-runs the profiled kernel multiple times for
        # different metric subsets and flushes L1 between
        # metric replays. The "measured" launch the harness
        # observes is therefore a cold-start replay even
        # though it follows N nominal warmup launches in
        # trace order. Predictor cold-mode prediction matches
        # this calibration-protocol truth; the earlier
        # `cache_initial_state=warm` derivation was the
        # source of friendly-L1 + L2 over-saturation
        # (predicted=1.0 vs measured=0.0 on cold-pattern
        # cells).
        return WarmupProtocolRecord(
            launch_skip_count=launch_skip_count,
            warmup_protocol=(
                WARMUP_PROTOCOL_NCU_LAUNCH_SKIP_N_COUNT_1
            ),
            cache_initial_state=CACHE_INITIAL_STATE_COLD,
        )
    if protocol == WARMUP_PROTOCOL_WARM_COLD_PAIR:
        # Paired-launch protocol: the measured launch follows
        # an explicit cold-cache reset (e.g., context flush)
        # between the warm calibration and the cold measurement.
        return WarmupProtocolRecord(
            launch_skip_count=launch_skip_count,
            warmup_protocol=WARMUP_PROTOCOL_WARM_COLD_PAIR,
            cache_initial_state=CACHE_INITIAL_STATE_COLD,
        )
    raise WarmupProtocolError(
        "warmup_protocol_unknown",
        f"unrecognized warmup_protocol={protocol!r}; expected "
        f"one of {sorted(ALL_WARMUP_PROTOCOLS)}",
    )


def validate_protocol_match(
    calibration: WarmupProtocolRecord,
    prediction: WarmupProtocolRecord,
) -> None:
    """Raise `WarmupProtocolError(warmup_protocol_mismatch)`
    when the calibration ledger entry and the prediction
    record disagree on any of the three provenance fields.

    Treats `warmup_protocol == WARMUP_PROTOCOL_UNKNOWN` on
    either side as an unconditional mismatch — an unknown
    protocol cannot be honestly compared.

    Does not raise on a clean match.
    """
    if (
        calibration.warmup_protocol == WARMUP_PROTOCOL_UNKNOWN
        or prediction.warmup_protocol == WARMUP_PROTOCOL_UNKNOWN
    ):
        raise WarmupProtocolError(
            "warmup_protocol_mismatch",
            f"warmup_protocol unknown on at least one side "
            f"(calibration={calibration.warmup_protocol!r}, "
            f"prediction={prediction.warmup_protocol!r}); "
            f"cannot honestly match",
        )
    if (
        calibration.launch_skip_count
        != prediction.launch_skip_count
    ):
        raise WarmupProtocolError(
            "warmup_protocol_mismatch",
            f"launch_skip_count disagrees "
            f"(calibration={calibration.launch_skip_count}, "
            f"prediction={prediction.launch_skip_count})",
        )
    if (
        calibration.warmup_protocol
        != prediction.warmup_protocol
    ):
        raise WarmupProtocolError(
            "warmup_protocol_mismatch",
            f"warmup_protocol disagrees "
            f"(calibration={calibration.warmup_protocol!r}, "
            f"prediction={prediction.warmup_protocol!r})",
        )
    if (
        calibration.cache_initial_state
        != prediction.cache_initial_state
    ):
        raise WarmupProtocolError(
            "warmup_protocol_mismatch",
            f"cache_initial_state disagrees "
            f"(calibration={calibration.cache_initial_state!r}, "
            f"prediction={prediction.cache_initial_state!r})",
        )


def validate_cache_initial_state_provenance(
    record_dict: dict[str, Any],
) -> None:
    """Schema-validator guard. A prediction record MUST carry
    a `cache_initial_state` field with a recognized value; a
    record dict that omits the field, sets it to `None`, or
    uses an unrecognized string raises
    `warmup_protocol_cache_initial_state_missing`.

    This enforces the cold/warm-state-provenance schema
    requirement that a prediction emitted without
    `cache_initial_state` provenance fails the schema
    validator.
    """
    state = record_dict.get("cache_initial_state")
    if state is None:
        raise WarmupProtocolError(
            "warmup_protocol_cache_initial_state_missing",
            "prediction record is missing required field "
            "'cache_initial_state'",
        )
    if state not in ALL_CACHE_INITIAL_STATES:
        raise WarmupProtocolError(
            "warmup_protocol_cache_initial_state_missing",
            f"prediction record has unrecognized "
            f"cache_initial_state={state!r}; expected one of "
            f"{sorted(ALL_CACHE_INITIAL_STATES)}",
        )


__all__ = [
    "WarmupProtocolError",
    "WarmupProtocolRecord",
    "WARMUP_PROTOCOL_NCU_LAUNCH_SKIP_N_COUNT_1",
    "WARMUP_PROTOCOL_WARM_COLD_PAIR",
    "WARMUP_PROTOCOL_UNKNOWN",
    "ALL_WARMUP_PROTOCOLS",
    "CACHE_INITIAL_STATE_COLD",
    "CACHE_INITIAL_STATE_WARM",
    "CACHE_INITIAL_STATE_UNKNOWN",
    "ALL_CACHE_INITIAL_STATES",
    "WARMUP_PROTOCOL_TYPED_REASONS",
    "derive_warmup_protocol",
    "validate_protocol_match",
    "validate_cache_initial_state_provenance",
]
