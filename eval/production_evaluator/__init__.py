"""Offline-only boundary for the future sealed-output evaluator.

This package deliberately exposes no scoring implementation.  Scientific and
operational authorities required by a production evaluator remain unresolved.
Ordinary package import is prohibited; the guard uses a retained source snapshot.
"""
from __future__ import annotations

raise RuntimeError(
    "ordinary package import is forbidden; use the descriptor-snapshot boundary"
)

__all__ = ("EvaluatorUnavailable", "evaluate")


class EvaluatorUnavailable(RuntimeError):
    """Raised while the production evaluator boundary remains blocked."""


def evaluate() -> None:
    """Refuse evaluation until the separately governed authorities exist."""
    raise EvaluatorUnavailable(
        "production evaluator is blocked offline; required authorities are "
        "typed-unresolved"
    )
