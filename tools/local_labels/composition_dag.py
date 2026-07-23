"""Composition DAG resolver over the registry's derived metrics.

A derived registry entry (``metric_class`` in ``derived_ratio`` /
``derived_pct_peak``) carries a ``composition`` block:

    {"formula": <str>,
     "dependencies": [<registry metric keys>],
     "device_constants": [<device-constant names>]}

The formula is an arithmetic expression whose bare identifiers are exactly the
dependency metric keys and the device-constant names. A dependency key may be a
PRIMITIVE (any non-derived registry metric, supplied at resolve time) or
ANOTHER derived metric (composition-of-composition); the latter forms an edge in
the dependency DAG so a derived metric is always evaluated after the derived
metrics it consumes.

This module builds that DAG, gives a topological evaluation order (raising on a
cycle), and evaluates each derived metric with a RESTRICTED ``ast`` evaluator
(no ``eval``/``exec``): only an arithmetic-expression node allowlist is honored,
device-constant names are resolved through the fail-closed
``DeviceConstantTable``, and an unavailable dependency cascades the derived value
to a typed-unavailable result with the offending dependency named.

Public surface:

- ``CompositionDAGResolver(registry, device_constant_table)``
  - ``.topological_order(arch) -> list[str]``
  - ``.resolve(arch, primitive_values) -> dict[key -> result dict]``
  - ``.refuse_label_leakage(feature_keys) -> None``
- ``CompositionError`` raised with a typed reason on a structural failure.

A ``resolve`` result for a derived key is
``{"value": float|None, "status": <str>, "reason": <str>}``; ``status`` is
``"resolved"`` when the formula produced a value, otherwise
``"typed_unavailable"`` with a ``reason`` that names the cause.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable

from device_constants import DeviceConstantError, DeviceConstantTable
from metric_registry import MetricEntry, MetricRegistry

# Metric classes that carry a composition block and participate in the DAG.
DERIVED_METRIC_CLASSES: tuple[str, ...] = ("derived_ratio", "derived_pct_peak")

# Only explicitly available upstream dependencies can supply a value to a
# composition. Every other status, including sm_120_deferred_no_live_evidence,
# fails closed as a typed-unavailable dependency.
AVAILABLE_ARCH_STATUSES: frozenset[str] = frozenset({"available"})

# The only callables an allowlisted formula may invoke. These are pure,
# side-effect-free numeric reductions; NEVER expose builtins, attribute access,
# subscripting, comprehensions, or any name resolution outside this map.
_ALLOWED_CALLS: dict[str, Callable[..., float]] = {
    "min": min,
    "max": max,
    "abs": abs,
}

# Allowlisted ast node types for a formula. Anything outside this set is a
# rejected formula (composition_formula_unsafe). Note we deliberately exclude
# Attribute, Subscript, comprehensions, lambdas, BoolOp, Compare, IfExp, etc.
_ALLOWED_NODES: tuple[type, ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Name,
    ast.Load,
    ast.Call,
    # Arithmetic operators.
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.UAdd,
    ast.USub,
)
# Numeric literals are ast.Constant on every supported Python (3.8+). The
# legacy ast.Num alias is deprecated (removed in 3.14) and is intentionally
# not referenced, so merely importing this module emits no DeprecationWarning.
_CONSTANT_NODES: tuple[type, ...] = (ast.Constant,)


class CompositionError(ValueError):
    """Raised on a structural composition failure with a typed reason.

    ``reason`` is one of:
      - ``composition_dag_cyclic``
      - ``composition_unknown_symbol``
      - ``device_constant_missing``
      - ``composition_formula_unsafe``
      - ``composition_formula_syntax``
      - ``composition_not_derived``
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        message = reason if not detail else f"{reason}: {detail}"
        super().__init__(message)


@dataclass(frozen=True)
class _DerivedNode:
    """One derived-metric node in the composition DAG."""

    key: str
    entry: MetricEntry
    formula: str
    dependencies: tuple[str, ...]
    device_constants: tuple[str, ...]

    @property
    def asserts_composition(self) -> bool:
        """True when the entry asserts a predictor-side composition from
        registered primitives. A derived entry that names NO dependencies and
        NO device constants carries a sentinel ``formula`` (e.g.
        ``label_only_no_predictor_composition`` /
        ``speed_of_light_requires_downstream_calibration``) recording that no
        composition from registered primitives is asserted; such a node is not
        evaluable here and resolves typed-unavailable rather than being parsed
        as an arithmetic formula."""
        return bool(self.dependencies) or bool(self.device_constants)


def _is_derived(entry: MetricEntry) -> bool:
    return entry.metric_class in DERIVED_METRIC_CLASSES


def _check_node_allowlist(tree: ast.AST) -> None:
    """Walk the parsed formula and reject any node outside the allowlist."""
    for node in ast.walk(tree):
        if isinstance(node, _CONSTANT_NODES):
            # A constant must be numeric (int/float); reject str/bytes/None/bool.
            # Use ast.Constant.value (3.8+); avoid the deprecated .n alias.
            value = node.value if hasattr(node, "value") else node.n
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CompositionError(
                    "composition_formula_unsafe",
                    f"non-numeric constant {value!r} in formula",
                )
            continue
        if isinstance(node, _ALLOWED_NODES):
            if isinstance(node, ast.Call):
                # Only a bare-name call to an allowlisted reduction, no kwargs,
                # no starred args.
                if not isinstance(node.func, ast.Name):
                    raise CompositionError(
                        "composition_formula_unsafe",
                        "only direct calls to allowlisted functions are permitted",
                    )
                if node.func.id not in _ALLOWED_CALLS:
                    raise CompositionError(
                        "composition_formula_unsafe",
                        f"call to non-allowlisted function {node.func.id!r}",
                    )
                if node.keywords:
                    raise CompositionError(
                        "composition_formula_unsafe",
                        "keyword arguments are not permitted in formula calls",
                    )
                for arg in node.args:
                    if isinstance(arg, ast.Starred):
                        raise CompositionError(
                            "composition_formula_unsafe",
                            "starred arguments are not permitted in formula calls",
                        )
            continue
        raise CompositionError(
            "composition_formula_unsafe",
            f"disallowed expression node {type(node).__name__}",
        )


def _parse_formula(formula: str) -> ast.Expression:
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise CompositionError(
            "composition_formula_syntax", f"{formula!r}: {exc}"
        ) from exc
    _check_node_allowlist(tree)
    return tree


@dataclass(frozen=True)
class CompositionDAGResolver:
    """Builds and evaluates the DAG over the registry's derived metrics."""

    registry: MetricRegistry
    device_constant_table: DeviceConstantTable

    # ----- DAG construction ------------------------------------------------

    def _entry_by_key(self) -> dict[str, MetricEntry]:
        return {e.key: e for e in self.registry.entries}

    def _derived_nodes(self) -> dict[str, _DerivedNode]:
        by_key = self._entry_by_key()
        nodes: dict[str, _DerivedNode] = {}
        for entry in self.registry.entries:
            if not _is_derived(entry):
                continue
            comp = entry.composition or {}
            deps = tuple(comp.get("dependencies", []) or ())
            consts = tuple(comp.get("device_constants", []) or ())
            formula = comp.get("formula", "")
            nodes[entry.key] = _DerivedNode(
                key=entry.key,
                entry=entry,
                formula=str(formula),
                dependencies=deps,
                device_constants=consts,
            )
        # Sanity: every derived dependency that is itself derived must be a
        # known node (a derived dep referencing a missing/unknown key is an
        # unknown symbol; that surfaces during evaluation, but a derived->derived
        # edge to a non-existent key would otherwise silently drop, so we only
        # build edges to keys that exist in the registry).
        del by_key
        return nodes

    def _derived_edges(
        self, nodes: dict[str, _DerivedNode]
    ) -> dict[str, set[str]]:
        """Edge map node -> set of derived dependency nodes it consumes.

        Only dependencies that are themselves derived metrics (present in
        ``nodes``) create ordering edges; primitive dependencies are leaves.
        """
        edges: dict[str, set[str]] = {k: set() for k in nodes}
        for key, node in nodes.items():
            for dep in node.dependencies:
                if dep in nodes:
                    edges[key].add(dep)
        return edges

    def _validate_dependency_keys(self, nodes: dict[str, _DerivedNode]) -> None:
        """Every declared dependency must be a registered metric key."""
        by_key = self._entry_by_key()
        for node in nodes.values():
            for dep in node.dependencies:
                if dep not in by_key:
                    raise CompositionError(
                        "composition_unknown_symbol",
                        f"{node.key} dependency {dep!r} is not a registry key",
                    )

    @staticmethod
    def _arch_status(per_arch: dict[str, Any], arch: str) -> str | None:
        status = per_arch.get(arch)
        if isinstance(status, dict):
            raw = status.get("status")
            return str(raw) if raw is not None else None
        return str(status) if status is not None else None

    def topological_order(self, arch: str) -> list[str]:
        """Return derived-metric keys ordered so every metric follows the
        derived metrics it depends on. Raises ``CompositionError`` with reason
        ``composition_dag_cyclic`` on a dependency cycle.

        ``arch`` is accepted for interface symmetry with ``resolve``; the
        derived-metric topology is architecture-independent (the device-constant
        table only affects evaluable VALUES, not the dependency structure).
        """
        del arch  # topology does not depend on arch; kept for a uniform API.
        nodes = self._derived_nodes()
        self._validate_dependency_keys(nodes)
        edges = self._derived_edges(nodes)
        # Kahn's algorithm over edges (dep -> dependent) reversed: we have
        # edges[node] = its derived deps. Process a node only after all its
        # derived deps are emitted.
        emitted: list[str] = []
        emitted_set: set[str] = set()
        remaining = dict(edges)
        # Deterministic order: sort keys so the output is stable.
        progress = True
        while remaining and progress:
            progress = False
            for key in sorted(remaining):
                if remaining[key] <= emitted_set:
                    emitted.append(key)
                    emitted_set.add(key)
                    del remaining[key]
                    progress = True
        if remaining:
            cyclic = ", ".join(sorted(remaining))
            raise CompositionError(
                "composition_dag_cyclic",
                f"derived metrics form a dependency cycle: {cyclic}",
            )
        return emitted

    # ----- Evaluation ------------------------------------------------------

    def _dep_unavailable_reason(
        self,
        arch: str,
        node: _DerivedNode,
        primitive_values: dict[str, float],
        resolved: dict[str, dict[str, Any]],
    ) -> str | None:
        """If any dependency of ``node`` cannot supply a value on ``arch``,
        return a ``dependency_unavailable:<dep>`` reason for the first such
        dependency; otherwise ``None``."""
        by_key = self._entry_by_key()
        for dep in node.dependencies:
            dep_entry = by_key.get(dep)
            if dep_entry is None:
                # Unknown dependency key: not a value problem, an unknown symbol.
                # Surfaced at evaluation time via the symbol allowlist.
                continue
            # Per-arch availability gate on the dependency.
            per_arch = dep_entry.per_arch_availability or {}
            status = self._arch_status(per_arch, arch)
            if status not in AVAILABLE_ARCH_STATUSES:
                return f"dependency_unavailable:{dep}"
            if _is_derived(dep_entry):
                # A derived dependency must have resolved to a value already.
                dep_result = resolved.get(dep)
                if dep_result is None or dep_result.get("value") is None:
                    return f"dependency_unavailable:{dep}"
            else:
                # A primitive dependency must be supplied (and non-None).
                if dep not in primitive_values or primitive_values[dep] is None:
                    return f"dependency_unavailable:{dep}"
        return None

    def _build_environment(
        self,
        arch: str,
        node: _DerivedNode,
        primitive_values: dict[str, float],
        resolved: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        """Resolve every symbol the formula may reference into a numeric value:
        dependency metric keys (from primitives or previously-resolved derived)
        and device-constant names (from the fail-closed table)."""
        by_key = self._entry_by_key()
        env: dict[str, float] = {}
        for dep in node.dependencies:
            dep_entry = by_key.get(dep)
            if dep_entry is not None and _is_derived(dep_entry):
                env[dep] = float(resolved[dep]["value"])
            else:
                env[dep] = float(primitive_values[dep])
        for const_name in node.device_constants:
            try:
                env[const_name] = self.device_constant_table.get(arch, const_name)
            except DeviceConstantError as exc:
                raise CompositionError(
                    "device_constant_missing", f"{const_name} (arch {arch}): {exc}"
                ) from exc
        return env

    def _eval_node(self, ast_node: ast.AST, env: dict[str, float]) -> float:
        """Evaluate an allowlisted ast node against ``env`` (no eval/exec)."""
        if isinstance(ast_node, ast.Expression):
            return self._eval_node(ast_node.body, env)
        if isinstance(ast_node, _CONSTANT_NODES):
            # ast.Constant.value on modern Python (3.8+); avoid the deprecated
            # .n alias so no DeprecationWarning fires on the hot eval path.
            raw = ast_node.value if hasattr(ast_node, "value") else ast_node.n
            return float(raw)
        if isinstance(ast_node, ast.Name):
            try:
                return float(env[ast_node.id])
            except KeyError as exc:
                raise CompositionError(
                    "composition_unknown_symbol", ast_node.id
                ) from exc
        if isinstance(ast_node, ast.UnaryOp):
            operand = self._eval_node(ast_node.operand, env)
            if isinstance(ast_node.op, ast.UAdd):
                return +operand
            if isinstance(ast_node.op, ast.USub):
                return -operand
            raise CompositionError(
                "composition_formula_unsafe",
                f"unary op {type(ast_node.op).__name__}",
            )
        if isinstance(ast_node, ast.BinOp):
            left = self._eval_node(ast_node.left, env)
            right = self._eval_node(ast_node.right, env)
            op = ast_node.op
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                return left / right
            if isinstance(op, ast.Pow):
                return left ** right
            if isinstance(op, ast.Mod):
                return left % right
            raise CompositionError(
                "composition_formula_unsafe",
                f"binary op {type(op).__name__}",
            )
        if isinstance(ast_node, ast.Call):
            func = ast_node.func
            if not isinstance(func, ast.Name) or func.id not in _ALLOWED_CALLS:
                raise CompositionError(
                    "composition_formula_unsafe", "non-allowlisted call"
                )
            args = [self._eval_node(a, env) for a in ast_node.args]
            return float(_ALLOWED_CALLS[func.id](*args))
        raise CompositionError(
            "composition_formula_unsafe",
            f"disallowed node {type(ast_node).__name__}",
        )

    def _validate_symbols(self, node: _DerivedNode, tree: ast.Expression) -> None:
        """Every Name in the formula must be a declared dependency key or a
        declared device-constant name; otherwise composition_unknown_symbol."""
        known = set(node.dependencies) | set(node.device_constants)
        for sub in ast.walk(tree):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                # Function names (min/max/abs) are not data symbols.
                continue
            if isinstance(sub, ast.Name):
                if sub.id in _ALLOWED_CALLS:
                    continue
                if sub.id not in known:
                    raise CompositionError("composition_unknown_symbol", sub.id)

    def resolve(
        self, arch: str, primitive_values: dict[str, float]
    ) -> dict[str, dict[str, Any]]:
        """Evaluate every derived metric for ``arch`` in topological order.

        ``primitive_values`` maps PRIMITIVE (non-derived) registry metric keys to
        their numeric values. For each derived metric the result is
        ``{"value": float|None, "status": str, "reason": str}``:

          - a metric whose every dependency is available and whose formula
            evaluates returns ``{"value": <float>, "status": "resolved",
            "reason": ""}``;
          - a metric with an unavailable dependency (per-arch status other than
            available, or a primitive missing
            from ``primitive_values``, or an upstream derived that itself did not
            resolve) cascades to ``{"value": None, "status":
            "typed_unavailable", "reason": "dependency_unavailable:<dep>"}``.

        Structural formula failures (unknown symbol, missing device constant,
        unsafe/cyclic formula) raise ``CompositionError`` — they are author-time
        registry defects, not per-run unavailability.
        """
        order = self.topological_order(arch)
        nodes = self._derived_nodes()
        resolved: dict[str, dict[str, Any]] = {}
        for key in order:
            node = nodes[key]
            if not node.asserts_composition:
                # A derived entry that asserts no composition from registered
                # primitives (no dependencies, no device constants) is not
                # evaluable here; its sentinel formula is a marker, not an
                # arithmetic expression. Resolve typed-unavailable, fail-closed,
                # rather than parsing the sentinel.
                resolved[key] = {
                    "value": None,
                    "status": "typed_unavailable",
                    "reason": "no_predictor_composition",
                }
                continue
            tree = _parse_formula(node.formula)
            # Symbol-allowlist check is a structural property of the formula and
            # is enforced even when a dependency is unavailable, so an authoring
            # typo never hides behind a cascade.
            self._validate_symbols(node, tree)
            unavailable = self._dep_unavailable_reason(
                arch, node, primitive_values, resolved
            )
            if unavailable is not None:
                resolved[key] = {
                    "value": None,
                    "status": "typed_unavailable",
                    "reason": unavailable,
                }
                continue
            env = self._build_environment(arch, node, primitive_values, resolved)
            value = self._eval_node(tree, env)
            resolved[key] = {
                "value": float(value),
                "status": "resolved",
                "reason": "",
            }
        return resolved

    # ----- Guards ----------------------------------------------------------

    def refuse_label_leakage(self, feature_keys: set[str]) -> None:
        """Delegate to the registry's no-NCU-label-leakage guard. Fail closed if
        any feature key resolves to a ``label_only`` registry entry."""
        self.registry.assert_no_label_leakage(feature_keys)
