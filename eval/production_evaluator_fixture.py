"""Pure, explicitly unclaimable fixture algebra for evaluator tests.

This module cannot emit an accuracy-report admission envelope or a label-opening
receipt. The source-pinned standalone entrypoint is preflight-only while current
M1 production authorities remain unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.contracts.validation import canonical_launch_key_token
from eval.production_labels import (
    ProductionAuthorityPin,
    load_production_matched_labels,
)
from eval.score import _score_lane_neutral_algebra, load_result_snapshot
from tools.local_labels.immutable_artifact import ImmutableArtifactError

_FIXTURE_MODE = "FIXTURE_ONLY_NON_AUTHORITATIVE_ALGEBRA"


@dataclass(frozen=True, slots=True)
class FixtureLaunchFamily:
    canonical_launch_token: str
    kernel_family: str

    def __post_init__(self) -> None:
        if (
            type(self.canonical_launch_token) is not str
            or not self.canonical_launch_token
            or type(self.kernel_family) is not str
            or not self.kernel_family
        ):
            raise ImmutableArtifactError("fixture launch-family projection is invalid")


@dataclass(frozen=True, slots=True)
class FixtureAlgebraRequest:
    result_path: Path
    labels_path: Path
    label_authority_path: Path
    label_authority_pin: ProductionAuthorityPin
    executable_sha256: str
    binary_architecture: str
    gpu_sku: str
    launches: tuple[FixtureLaunchFamily, ...]

    def __post_init__(self) -> None:
        for name in ("result_path", "labels_path", "label_authority_path"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                object.__setattr__(self, name, Path(value))
        if not isinstance(self.label_authority_pin, ProductionAuthorityPin):
            raise ImmutableArtifactError("fixture label authority pin is invalid")
        if (
            type(self.executable_sha256) is not str
            or len(self.executable_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.executable_sha256
            )
            or type(self.binary_architecture) is not str
            or not self.binary_architecture
            or type(self.gpu_sku) is not str
            or not self.gpu_sku
            or not self.launches
        ):
            raise ImmutableArtifactError("fixture algebra authority is incomplete")
        tokens = tuple(item.canonical_launch_token for item in self.launches)
        if len(tokens) != len(set(tokens)):
            raise ImmutableArtifactError(
                "fixture launch-family projection is duplicated"
            )


def evaluate_non_authoritative_fixture_algebra(
    request: FixtureAlgebraRequest,
) -> dict[str, Any]:
    """Return unclaimable rows and coverage from authenticated fixture labels."""

    if not isinstance(request, FixtureAlgebraRequest):
        raise ImmutableArtifactError("explicit fixture algebra request is required")
    result = load_result_snapshot(request.result_path)
    result_document = result.validated_document()
    if result_document["build_identity"]["binary_sha256"] != (
        request.executable_sha256
    ):
        raise ImmutableArtifactError("fixture result executable identity mismatch")
    card = result_document["arch_card_identity"]
    if result_document["configuration_identity"]["arch_card_sha256"] != card[
        "card_sha256"
    ]:
        raise ImmutableArtifactError("fixture result internal identity mismatch")
    if (
        card["sm_arch"] != request.binary_architecture
        or card["sku"] != request.gpu_sku
    ):
        raise ImmutableArtifactError("fixture result architecture/SKU mismatch")
    result_tokens = tuple(
        canonical_launch_key_token(launch["launch_key"])
        for launch in result_document["launches"]
    )
    diagnostic_tokens = tuple(
        canonical_launch_key_token(launch["launch_key"])
        for launch in result_document["diagnostics"]["per_launch"]
    )
    if diagnostic_tokens != result_tokens:
        raise ImmutableArtifactError("fixture result diagnostic identity mismatch")
    expected_tokens = tuple(item.canonical_launch_token for item in request.launches)
    if result_tokens != expected_tokens:
        raise ImmutableArtifactError(
            "fixture result launch inventory differs from authority"
        )
    labels = load_production_matched_labels(
        request.labels_path,
        request.label_authority_path,
        authority_pin=request.label_authority_pin,
    )

    if (
        labels.binary_architecture != request.binary_architecture
        or labels.gpu_sku != request.gpu_sku
    ):
        raise ImmutableArtifactError("fixture label architecture/SKU mismatch")
    observed = tuple(
        FixtureLaunchFamily(
            launch.identity.canonical_token,
            launch.kernel_family,
        )
        for launch in labels.launches
    )
    if observed != request.launches:
        raise ImmutableArtifactError(
            "fixture label launch-family projection differs from authority"
        )
    algebra = _score_lane_neutral_algebra(result, labels)

    return {
        "contract_mode": _FIXTURE_MODE,
        "claimability": {
            "status": "unclaimable",
            "reason": "fixture_only_non_authoritative_algebra",
        },
        "comparison_failures": algebra["comparison_failures"],
        "coverage": algebra["coverage"],
        "per_metric_family": algebra["per_metric_family"],
    }


__all__ = (
    "FixtureAlgebraRequest",
    "FixtureLaunchFamily",
    "evaluate_non_authoritative_fixture_algebra",
)
