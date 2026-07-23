# Metric Registry &amp; Deployability Leakage

> **Scope note.** This page preserves a historical `registry_v2`/deployability
> snapshot; its 41-entry census and predictor-era deployment prose below are
> non-operative. The current v5 label authority is
> [`local_label_sidecar.schema.json`](../workloads/simbench_bw/schemas/local_label_sidecar.schema.json),
> [`local_label_admitted.schema.json`](../workloads/simbench_bw/schemas/local_label_admitted.schema.json),
> and [`tools/local_labels/metric_registry.json`](../tools/local_labels/metric_registry.json)
> under root `CLAUDE.md`; the current registry contains 61 metric entries. NCU
> values remain validation targets, never simulator inputs. Use
> [`docs/README.md`](README.md) and [`STATUS.md`](STATUS.md) for current precedence
> and publication scope.

The current JSON registry is the **single source of truth for metric identity and
deployability metadata**; it binds every
ground-truth target metric the local-label pipeline can carry to its canonical
Nsight Compute identifier, its unit, its carrier, its measurement geometry, and
— critically — its **deployability**: whether it may ever become a predictor
*feature*, or is forever a *label only*.

This page documents the canonical `registry_v2` ontology, the two carriers
(`flat_field` vs `unit_tagged_metric`), the `deployability` axis and the leakage
guard built on it, the composition DAG that resolves derived metrics, and the
reconciliation of the two registries that the four-tree merge inherited
(the "Q1" seam).

**Current registry source:**
[`tools/local_labels/metric_registry.json`](../tools/local_labels/metric_registry.json)
(loaded + validated by
[`metric_registry.py`](../tools/local_labels/metric_registry.py)).

Sibling docs: [active architecture record](ARCHITECTURE.md) ·
[../OPEN_RESEARCH_CHALLENGES.md](../OPEN_RESEARCH_CHALLENGES.md) (historical
challenge context) · [../MERGE_PROVENANCE.md](../MERGE_PROVENANCE.md) (historical
merge provenance).

---

## 1. Why a registry at all

The deployed predictor runs in **non-root, NCU-free** environments (cloud
containers with `ERR_NVGPUCTRPERM`, no privileged CUPTI, no Nsight Compute). The
labels used to *train* and *calibrate* it, however, come from exactly those
privileged counters. A registry lets us state, per metric and in one committed
place, "this counter is a legitimate ground-truth **label**, but it is
physically unavailable at deploy time, so it may never be wired into a feature
vector." Without that pin, a metric such as achieved occupancy could silently
leak from the label side into the feature side and produce a predictor that
cannot run where it is meant to run.

Honesty posture (carried throughout this suite): **sm_86 cache prediction is
validated**; every entry's **sm_120 (Blackwell) status is deferred / refuted**
(`sm_120_deferred_no_live_evidence`, all 41 entries); wall-clock time is a
**label + refutation** target only, never a deployable feature. See
[../OPEN_RESEARCH_CHALLENGES.md](../OPEN_RESEARCH_CHALLENGES.md).

---

## 2. The `registry_v2` ontology

`metric_registry.json:2` declares `registry_version: 2` and
`label_schema_version: 5`. Version 2 promotes a full metric-ontology layer from
optional to **required on every entry** (enforced in `_parse`,
`metric_registry.py:298-424`). Current census: **41 entries**.

Each entry carries:

| Field | Vocabulary / shape | Source |
|---|---|---|
| `metric_class` | `static`, `replay_count`, `cache_stream`, `time_clock`, `derived_ratio`, `derived_pct_peak`, `typed_unavailable` | `metric_registry.py:79-87` |
| `risk_tier` | `T0` (launch arithmetic) → `T3` (DVFS/scheduler-bound timing) | `metric_registry.py:88` |
| `ncu_observation_contract` | `{counting_unit, rollup, numerator, denominator, op_classes, unsupported_op_behavior}` | `metric_registry.py:98-105` |
| `per_arch_availability` | `{sm_86, sm_120}` — **both required** | `metric_registry.py:113` |
| `composition` | `{formula, dependencies, device_constants}` — required on derived entries | `metric_registry.py:106-111` |

Distribution at HEAD: `metric_class` = 15 `replay_count` / 7 `time_clock` / 5
`cache_stream` / 5 `derived_ratio` / 4 `static` / 3 `derived_pct_peak` / 2
`typed_unavailable`; `risk_tier` = 15 `T1` / 12 `T3` / 10 `T2` / 4 `T0`;
`per_arch_availability.sm_86` = 38 `available` / 2 `excluded` / 1
`requires_runtime_carveout`; **`sm_120` = `sm_120_deferred_no_live_evidence`
for all 41**.

The `ncu_observation_contract` is what makes the registry self-describing rather
than a bare name list — e.g. the mandatory L2 hit rate pins
`numerator: lts__t_sectors_lookup_hit`, `denominator: lts__t_sectors`
(`metric_registry.json:53-66`), so the *geometry* of the ratio is committed, not
just the rollup token.

---

## 3. Carriers: `flat_field` vs `unit_tagged_metric`

Two physical carriers coexist for backward compatibility
(`metric_registry.py:53`, `CARRIER_KINDS`):

- **`flat_field`** — the historical percentage carrier. Only the hit-rate
  fraction family (mandatory L2, L1TEX) uses it; it travels the
  `percent_to_fraction` transform and lands as a flat scalar in the sidecar.
  **2 entries** at HEAD.
- **`unit_tagged_metric`** — the schema-v5 carrier for every non-percentage
  family (`count` / `ns` / `cycles`), tagged with its unit so a count can never
  silently ride the percentage `/100` path. **39 entries** at HEAD.

A coherence assertion (`metric_registry.py:336-340`) forces
`unit == "pct"` iff `transform == "percent_to_fraction"`, and
`validate_unit_tag()` (`metric_registry.py:461-489`) rejects a `unit_tagged`
record whose claimed unit disagrees with the registry
(`unit_tagged_metric_unit_mismatch`).

---

## 4. Deployability &amp; the leakage guard

The `deployability` field (`metric_registry.py:52`, `DEPLOYABILITY_KINDS`) is
the load-bearing policy axis:

- **`feature_eligible`** — statically derivable, available *without* privileged
  counters, so admissible as a predictor input. Only **4 entries**: the four
  theoretical/launch-limit occupancy metrics
  (`occupancy_limit_{warps,blocks,registers,shared_mem}_theoretical`,
  `metric_class: static`, `risk_tier: T0`).
- **`label_only`** — counter-derived ground truth. Captured for training /
  calibration / refutation, but **never** wired into a deployable feature
  surface. **37 entries**, including every timing, occupancy, IPC, replay-count
  and cache-stream metric.

The leakage guard exists in **two mutually reinforcing places**, both keyed on
this one field:

1. `MetricRegistry.assert_no_label_leakage()` — `metric_registry.py:257-275`.
2. Standalone
   [`no_deployability_leakage_guard.py`](../tools/local_labels/no_deployability_leakage_guard.py)
   — loads the same registry (`load_registry`) and filters
   `if e.deployability != "label_only"` (line 39).

Both resolve **every alias** of a `label_only` metric — its registry `key`,
`sidecar_field`, `admitted_field`, and canonical `ncu_metric` — into a forbidden
set (`label_only_aliases`, `metric_registry.py:244-255`), then **fail closed**
if any deployable feature name hits it. The guard's docstring names the two
canonical never-a-feature examples explicitly
(`no_deployability_leakage_guard.py:5-8`):

- achieved occupancy `sm__warps_active.avg.pct_of_peak_sustained_active`
  (registry key `occupancy_achieved_warps_pct`, `label_only`);
- IPC `sm__inst_executed.avg.per_cycle_elapsed` (key `ipc_inst_per_cycle`,
  `label_only`).

Wall-clock `gpu__time_duration.sum` (key `exec_duration_ns`, `time_clock`,
`T3`, `label_only`) is in the same quarantine. **Occupancy, IPC, and gpu\_\_time
can never become features** — that is the invariant the guard exists to prove.

### Eligibility flow

```
                          ┌────────────────────────────────────┐
                          │  metric entry  (metric_registry.json)│
                          │  key · ncu_metric · unit · carrier   │
                          └───────────────────┬──────────────────┘
                                              │  deployability
                        ┌─────────────────────┴─────────────────────┐
                        │                                            │
             deployability == "label_only"            deployability == "feature_eligible"
             (37 entries · counter-derived)            (4 entries · static / T0 launch arithmetic)
                        │                                            │
       ┌────────────────┴───────────────┐                           │
       ▼                                ▼                            ▼
  captured as a LABEL           aliases collected into        MAY enter the deployable
  (train / calibrate /          the forbidden set:            feature vector — resolves
   refute; sm_86 only)          {key, sidecar_field,          without privileged counters
       │                         admitted_field, ncu_metric}   (theoretical occupancy limits)
       │                                │                            │
       │                    ┌───────────┴────────────┐               │
       │                    ▼                        ▼               ▼
       │        assert_no_label_leakage()   no_deployability_    assert_no_deployability_
       │        (in-registry guard)         leakage_guard.py     leakage()  →  PASSES
       │                    │               (standalone guard)        (allowed)
       │                    └───────────┬────────────┘
       ▼                                ▼
   ✗  label_only  ─ ─ ─ ─ ─ ─ ─ ─►  FAIL CLOSED
      NEVER a feature              DeployabilityLeakageError
      (occupancy / IPC /          "label-only metric leaked into
       gpu__time)                  deployable feature surface"

   Legend:  ↛  label_only  can NOT become a feature      →  feature_eligible  CAN
```

---

## 5. The composition DAG

Derived entries (`metric_class` ∈ `{derived_ratio, derived_pct_peak}`) carry a
`composition` block whose `formula` references **dependency metric keys** and
**device-constant names** as bare identifiers.
[`composition_dag.py`](../tools/local_labels/composition_dag.py) builds a DAG
over those entries and evaluates each in topological order.

Design points that matter:

- **Restricted evaluator, no `eval`/`exec`.** Only an arithmetic ast-node
  allowlist is honored (`_ALLOWED_NODES`, `composition_dag.py:67-83`); the only
  callables are `min`/`max`/`abs` (`_ALLOWED_CALLS`, lines 58-62). Attribute
  access, subscripting, comprehensions, and boolean/compare nodes are rejected
  (`composition_formula_unsafe`).
- **Fail-closed device constants.** Constant names (`sm_count`,
  `sector_size_bytes`, `peak_dram_bytes_per_s`, …) resolve through a
  `DeviceConstantTable`; a missing constant raises `device_constant_missing`.
- **Typed-unavailable cascade.** A dependency whose per-arch status is anything
  other than `available` — including every `sm_120_deferred_no_live_evidence`
  entry — cascades the derived value to
  `{"value": None, "status": "typed_unavailable", "reason": "dependency_unavailable:<dep>"}`
  (`AVAILABLE_ARCH_STATUSES = {"available"}`, `composition_dag.py:53`). **This is
  the structural reason no sm_120 derived metric can produce a value: its
  primitives are all deferred.**
- **The guard is reused, not reimplemented.**
  `CompositionDAGResolver.refuse_label_leakage()` delegates straight to
  `registry.assert_no_label_leakage()` (`composition_dag.py:500-503`).

Worked examples from HEAD:

```
ipc_inst_per_cycle           = smsp_inst_executed_sum / (sm_count * exec_cycles_gpc_max)
                               deps: smsp_inst_executed_sum, exec_cycles_gpc_max   const: sm_count
dram_bytes_per_second        = (traffic_dram_read_sectors + traffic_dram_write_sectors)
                               * sector_size_bytes / (exec_duration_ns / 1e9)
throughput_*_pct_peak        = min(dram_bytes_per_second / peak_dram_bytes_per_s, 1.0)
```

Note the second-order edge: the three `throughput_*_pct_peak` metrics depend on
the derived `dram_bytes_per_second`, which is why a topological order is
required rather than a flat pass.

Entries that carry a sentinel formula (e.g.
`label_only_no_predictor_composition` on the mandatory L2 hit rate,
`metric_registry.json` composition note) declare **no** dependencies/constants;
the resolver detects this (`asserts_composition`, `composition_dag.py:119-129`)
and returns `typed_unavailable` with reason `no_predictor_composition` rather
than trying to parse the sentinel as arithmetic.

---

## 6. Two-registry reconciliation (the Q1 seam)

The merge inherited a **second** registry from Tree B (the sm_120 Blackwell
tree):
[`ncu_metric_registry.py`](../tools/local_labels/ncu_metric_registry.py) +
[`typed_ncu_label_path.py`](../tools/local_labels/typed_ncu_label_path.py). At
first glance the two look like "two incompatible type systems." They are not —
they are **different-scope** registries, and the reconciliation is a scope tag,
not a merge conflict.

| | Canonical `metric_registry.json` (registry_v2) | Tree B `ncu_metric_registry.py` |
|---|---|---|
| Version | `registry_version: 2` | `cache_memory_registry_v1` (`ncu_metric_registry.py:36`) |
| Entries | 41 (JSON, full ontology) | 22 `MetricSpec`s, 15 base NCU tokens (`ALLOWLIST_TOKENS`) |
| Type axis | `unit` + `ncu_observation_contract{numerator,denominator}` | `kind` ∈ `{fraction,count,ratio}` (`ncu_metric_registry.py:39-42`) |
| Scope | **all** ground-truth labels | **cache/memory only** (`ncu_metric_registry.py:19-25`) |
| Timing/occupancy/IPC | carried as `label_only` labels | **forbidden** (`_FORBIDDEN_SUBSTRINGS`, lines 60-73) |
| Extra fields | metric_class, risk_tier, per_arch, composition | `comparison_formula`, `prediction_source`, kind-routed admission |

Two facts settle it (grounded in the Q1 seam research report):

1. **Token containment is total and one-directional.** All 15 of Tree B's base
   NCU tokens are present in the canonical registry; **B adds zero tokens the
   canonical registry lacks.** B's extra keys are derived ratios/totals computed
   *from* those shared tokens. The canonical registry is a **strict superset**.
2. **Both trees already agree on the deploy policy.** Tree B *bans*
   `gpu__time`/`occupancy`/`inst_executed`/`__ipc`/`warps_active` from its
   allowlist (`ncu_metric_registry.py:60-73`); the canonical registry keeps them
   but marks them `label_only` and quarantines them with the leakage guard. The
   two mechanisms enforce the **same** invariant — those tokens are never
   deployable features.

### Resolution

- **Canonical = the codex JSON `registry_v2`.** It is the single source of
  truth: strict superset of B's tokens, already arch-aware
  (`per_arch_availability`), already carrying numerator/denominator via
  `ncu_observation_contract`, plus the composition DAG and 27 in-tree consumers.
- **Tree B's registry becomes a scope-tagged cache/memory projection.** Its
  kind-routed admission (`typed_ncu_label_path.admit_fraction/count/ratio`,
  fail-closing on wrong-kind push) and its `comparison_formula` /
  `prediction_source` error-geometry fields are the only genuinely additive
  capabilities; they re-home onto the canonical registry as a `scope=cache_memory`
  filter (ideally regenerated *from* the JSON so B's consumers keep their
  `import ncu_metric_registry` surface).
- **The gpu\_\_time / occupancy / IPC "scope conflict" dissolves.** Those tokens
  stay as `label_only` **labels** in the canonical registry (the validated sm_86
  timing lane needs them as calibration targets), while B's registry-wide *ban*
  demotes to a per-consumer `scope=cache_memory` **filter**. Nothing changes
  policy: they remain guard-protected from ever becoming features.

```
   gpu__time / occupancy / IPC token
             │
     ┌───────┴──────────────────────────────────────┐
     │ canonical registry_v2                         │ cache/memory scope filter (Tree B origin)
     │  → representable as a LABEL (label_only)       │  → NOT admissible (out of cache/mem scope)
     │  → leakage guard: never a FEATURE              │  → _FORBIDDEN_SUBSTRINGS reject
     └───────────────────────────────────────────────┘
        same invariant, two vantage points  ⇒  no contradiction
```

For the full option analysis and the residual user decision (single source of
truth vs peer registries, and whether NCU `gpu__time` stays representable at
all), see [../MERGE_PROVENANCE.md](../MERGE_PROVENANCE.md) and
[../OPEN_RESEARCH_CHALLENGES.md](../OPEN_RESEARCH_CHALLENGES.md).

---

## 7. Invariants (what tests hold)

- `label_schema_version` moves in lockstep with the sidecar/admitted schema
  (`metric_registry.py:289-294`); a drift fails fast.
- No `label_only` metric — by any of its four aliases — may appear in a
  deployable feature surface (`assert_no_label_leakage` /
  `assert_no_deployability_leakage`).
- Only `pct` units take the `percent_to_fraction` transform; all other units
  are `unit_tagged` and `identity_nonneg`.
- Every derived entry's formula parses under the restricted ast allowlist and
  references only declared dependency keys + device constants.
- Every entry declares both `sm_86` and `sm_120` availability; `sm_120` is
  uniformly deferred (no live Blackwell evidence).
