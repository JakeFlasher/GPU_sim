# ArchProfile — loader for architecture-local profile-card facts

> **Scope note.** The active architecture documentation is
> [`docs/ARCHITECTURE.md`](ARCHITECTURE.md), [`docs/STATUS.md`](STATUS.md),
> [`docs/COMPONENTS.md`](COMPONENTS.md), and the `arch_profiles/` cards/schema.
> This page also preserves useful historical merge rationale, but it creates no
> workflow gate.

`arch_profiles/` is the profile-card boundary for architecture-local facts. Other
architecture inputs, including the latency evidence under `data/pchase/`, remain
separate inputs and are not folded into these cards. The historical four-tree
merge replaced scattered per-device Python dicts with a single set of JSON
**cards** read through one loader
([`arch_profiles/loader.py`](../arch_profiles/loader.py)).

The design is grounded in the seam research question **Q9** ("Blackwell
ArchProfile key + canonical token vocabulary"). This page documents the schema,
the loader API, the 3-tier token model, the lockstep contract, and how to add a
card.

---

## 1. The problem: arch facts were scattered across per-device dicts

Before the merge, an architecture's hardware and policy facts lived as literal
constants sprinkled across production modules. Q9's audit found the same fact
keyed differently in different files — a genuine correctness hazard:

| Live constant | Home (pre-merge) | Keyed by | Hazard |
|---|---|---|---|
| `_SM_COUNT_BY_ARCH` | `workloads/simbench_bw/cache_predictor.py`, `canonical_features.py` | **`sm_120`** | codex carried guessed `50`; Tree B measured `26` |
| `_STATIC_CARD_L2_BYTES` | `tools/hrd_profiler/bundle_from_trace.py` | **`sm_120-laptop`** | exact-key `.get(arch, 0)` returns `0` silently on the *other* spelling |
| `_GRIDDIM_PENALTY_ALPHA` | `cache_predictor.py` | `sm_86` scalar | Blackwell value never existed |
| `_BARE_LDG_E_DEFAULTS_TO_CA` | `tools/cache_predictor/l1_cache_op_classifier.py` | frozenset membership | `sm_86 ∈`, `sm_120 ∉` — the L1→L2 miss filter hinges on it |
| L1 / L2 geometry | `l1_static_card_table.json`, `run_suite.py` cc-branches | mixed | sm_120 L1 row simply absent |
| NVML anchor | `data/local_labels/nvml_reference_operating_point_sm_86.json` | file-per-arch | GA102 bands wrong for a 5060 |

The two SM-count vs. L2-bytes maps keyed the *same* Blackwell part under *two
different spellings* (`sm_120` vs `sm_120-laptop`). ArchProfile lifts every one
of these onto a card field read through the loader, so a fact is defined once and
the spelling hazard is fixed by construction.

---

## 2. The card schema

Each card is one JSON document under
[`arch_profiles/cards/`](../arch_profiles/cards/), validated by
[`schema/arch_profile.schema.json`](../arch_profiles/schema/arch_profile.schema.json)
(draft-07). Required fields: `arch_id`, `sku`, `compute_capability`,
`build_arch_family`, `sm_count`, `bare_ldg_e_default`, `cache_layers`,
`evidence_status`.

Key fields (from [`cards/sm_86.json`](../arch_profiles/cards/sm_86.json) and
[`cards/sm_120-laptop.json`](../arch_profiles/cards/sm_120-laptop.json)):

- **Identity**: `arch_id` (canonical bare `sm_XX`, enum `{sm_86, sm_120}`),
  `sku` (`rtx_3090_ti` / `rtx_5060_laptop`), `aliases`, `device_name_substrings`,
  `compute_capability`, `die`, `microarchitecture`, `role`.
- **Build**: `build_arch_family` (fatbin slices, e.g. `["sm_120","sm_120a","sm_120f"]`)
  and `gencode` — a *compile-time* vocabulary, deliberately separate from the
  deploy `arch_id`.
- **Hardware/policy cards**: `sm_count`, `griddim_penalty_alpha`,
  `bare_ldg_e_default` (`ca` | `bypass` | `unknown_typed_unavailable`),
  `cache_layers`, `l2_geometry`, `l1_geometry`, `nvml_reference_operating_point`,
  `duration_crosswalk`.
- **Honesty**: `evidence_status`, `predictor_state` (per-observable prose), and
  the `research_gap` marker (§7) for any field local silicon never produced.

The schema encodes the honesty posture directly: `sm_count`, geometry, and NVML
fields are `anyOf: [<value>, researchGap]` — a card may legitimately carry a
typed-absent gap instead of a number
(`schema/arch_profile.schema.json:32-45`).

---

## 3. The loader API

`from arch_profiles import ...` re-exports the whole surface
([`arch_profiles/__init__.py`](../arch_profiles/__init__.py)):

| Function | Purpose |
|---|---|
| `load_profile(token, sku=None) -> ArchProfile` | Resolve any spelling → card. `loader.py:171` |
| `resolve_arch_id(token) -> str` | Normalise 8 spellings / CC tuple / device name → bare `sm_XX`. `loader.py:191` |
| `resolve_sku(token, strict=True) -> str \| None` | Pin a device-tier SKU. `loader.py:241` |
| `build_arch_family_for(cc) -> list[str]` | Fatbin build family for a CC (may include server-only slices). `loader.py:265` |
| `is_research_gap(value) -> bool` | True if a field is the typed-absent marker. `loader.py:70` |
| `list_profiles() -> list[str]` | Installed `arch_id`s. `loader.py:143` |

`ArchProfile` is a frozen dataclass (`loader.py:74`) with typed accessors over
the raw card: `.sm_count`, `.l2_geometry`, `.l1_geometry`, `.bare_ldg_e_default`,
`.cache_layers`, `.nvml_reference_operating_point`, `.build_arch_family`,
`.compute_capability`, `.evidence_status`.

**Fail-closed by construction.** A field that is a research gap is returned as a
`ResearchGap` — a `dict` subclass whose `__bool__` is `False` (`loader.py:57-67`):

```python
prof = load_profile("sm_120-laptop")
if prof.l1_geometry:                     # ResearchGap → falsy → branch skipped
    walk_l1(prof.l1_geometry)            # never runs on Blackwell
else:
    fail_closed("sm_120 L1 geometry is a research gap")  # correct path
```

A caller that forgets `is_research_gap()` and simply truth-tests the field still
does the safe thing: it must **never** substitute another arch's value. Rejected
server/workstation cards (RTX PRO 6000, B200, H100…) raise `ArchProfileError`
rather than silently resolving to laptop numbers
(`loader.py:216-221`, `aliases.json:28`).

---

## 4. The 3-tier token model

Q9's central finding: the "8 coexisting spellings" are **not** chaos — they are
three deliberate surface roles. ArchProfile formalises them. Any inbound token
funnels down to exactly one canonical `arch_id`:

```
          INBOUND TOKEN (any of 8 spellings, a CC tuple, or a device name)
                                     │
   ┌─────────────────────────────────┼─────────────────────────────────┐
   │  "sm_120"  "sm_120-laptop"  "sm_120a"  (12,0)   "…RTX 5060 Laptop…" │
   │  "sm_86"   "sm_120f"        "3090 ti"  (8,6)    "…RTX PRO 6000…"    │
   └─────────────────────────────────┼─────────────────────────────────┘
                                     ▼
                          resolve_arch_id(token)          loader.py:191
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                       ▼
      (1) rejected?            (2) exact token?       (3) device substring
      "pro 6000","b200",       token_to_arch_id       / CC-tuple lookup
      "gb202"… ─► RAISE        {sm_120-laptop→sm_120} ("5060"→sm_120)
              │                      │                       │
              └──────────────────────┼──────────────────────┘
                                     ▼
                        CANONICAL arch_id ∈ { sm_86 , sm_120 }
                                     │
                     resolve_sku(token)   loader.py:241
                                     ▼
              device-tier SKU ∈ { rtx_3090_ti , rtx_5060_laptop }
                                     │
                                     ▼
                _card_path_for(arch_id, sku)  ──►  ArchProfile
```

The three tiers, each with a distinct role
([`aliases.json`](../arch_profiles/aliases.json),
[`build_arch_families.json`](../arch_profiles/build_arch_families.json)):

1. **Canonical `arch_id` — bare `sm_XX`** (label / comparison / cohort surface).
   `provenance_audit._canonicalize_request_arch` already declared the bare form
   canonical; keeping it stable holds the frozen `label_admitted.json` arch enum
   `{sm_86, sm_120}` and the 30-cell sm_120 cohort byte-stable.
2. **Device-tier `sku`** (`rtx_5060_laptop`) — disambiguates hardware facts that
   differ between two cards sharing a compute capability. A bare `sm_120` cannot
   tell a **GB206 laptop (32 MiB L2, 26 SM)** from a **GB202 workstation (far
   larger L2, ~188 SM)** — so those numbers live on the SKU card, not the
   arch_id.
3. **`build_arch_family`** — a *compile-time* list (`sm_120a`, `sm_120f`,
   `sm_90a`, `sm_100a`) that a fatbin may legitimately carry but which are
   **never** valid deploy `arch_id`s (`build_arch_families.json:20-22`).

**Why `sm_120-laptop` exists.** It is the device/capability *alias* of `sm_120`.
It is the key `_STATIC_CARD_L2_BYTES` used for the 32 MiB card, and it is what
the loader normalises back to `sm_120` (`aliases.json:7`). It is *not* a deploy
arch_id and *not* an nvcc build token — the frozen label schema's negative test
rejects it in an `arch` field. The alias tier lets the resolver accept it while
the canonical tier keeps it out of the frozen artifacts. A future cloud GB202 /
PRO 6000 is therefore a **new SKU card under the same `arch_id`**, not a new
token or a code fork.

---

## 5. Card/source consistency

The focused tests in
[`tests/test_arch_profile_lockstep.py`](../arch_profiles/tests/test_arch_profile_lockstep.py)
check that architecture cards and their live consumers agree. These are
ordinary regression tests: update both sides deliberately when new measurements
change a parameter.

```
   LIVE PRODUCTION SOURCE                 ArchProfile CARD          verdict
   ─────────────────────                  ────────────────          ───────
   cache_predictor.py
     _SM_COUNT_BY_ARCH["sm_86"]=84  ═════ sm_86.sm_count      = 84   ✓ byte-equal
     _GRIDDIM_PENALTY_ALPHA  =0.110 ═════ .griddim_penalty…   =0.110 ✓ byte-equal
   bundle_from_trace.py
     _STATIC_CARD_L2_BYTES["sm_86"] ═════ sm_86.l2_geometry
        = 6*1024*1024                       .capacity_bytes   =6 MiB ✓ byte-equal
   l1_cache_op_classifier.py
     "sm_86" ∈ _BARE_LDG_E_…_CA     ═════ .bare_ldg_e_default ="ca"  ✓ consistent
   ─────────────────────────────────────────────────────────────────────────
   cache_predictor.py                                    DELIBERATE CORRECTION
     _SM_COUNT_BY_ARCH["sm_120"]=26 ═════ sm_120.sm_count = 26  ✓ consistent
```

- The sm_86 tests compare card values with live consumers.
- The measured sm_120 SM count is 26 in both the card and the current source.
- Resolver tests cover supported spellings and compute-capability tuples,
  rejection of unknown device/SKU pairs, and build/deploy token separation.

---

## 6. Adding a new arch / SKU card

1. **Write the card** `cards/<arch>-<sku>.json`. Set `arch_id` to the canonical
   bare token (add it to the schema `arch_id` enum only if it is a genuine new
   deploy arch). Populate every hardware fact you have *measured*; use a
   `research_gap` object (§7) for anything you have not.
2. **Register the aliases** in [`aliases.json`](../arch_profiles/aliases.json):
   `token_to_arch_id`, `device_substring_to_arch_id`,
   `compute_capability_to_arch_id`, `sku_tokens`, `device_substring_to_sku`. If
   the card is a non-local server/workstation part that must not masquerade as a
   local deploy arch, add its name substring to `rejected_device_substrings`.
3. **Register the build family** in
   [`build_arch_families.json`](../arch_profiles/build_arch_families.json) if the
   compute capability is new (keep server-only slices out of `deploy_arch_ids`).
4. **Add lockstep tests.** Any live constant the card mirrors must get a
   byte-identity assertion; any deliberate correction must get an explicit
   `card != source` assertion like the sm_120 SM-count test.
5. **Disambiguation.** If two cards share an `arch_id`, `_card_path_for`
   (`loader.py:148`) raises unless the caller passes `sku=` — that is intended;
   a shared-CC SKU must always be pinned.

Run the guard: `pytest arch_profiles/tests -q`.

---

## 7. Research-gap markers

A field local silicon never produced is stored as
`{"status": "research_gap", "reason": ..., "challenge": ...}`, **not** a silent
default. On the sm_120-laptop card that covers `l1_geometry`,
`nvml_reference_operating_point`, and `duration_crosswalk`
(`cards/sm_120-laptop.json:29-44`). Reading such a field returns a falsy
`ResearchGap`, forcing callers to fail closed.

This is the honesty posture, enforced by data:

- **sm_86 cache prediction is validated** — the L2 sector-hit predictor passes
  3/3 hard gates (card `predictor_state.l2_sector_hit_rate`).
- **sm_120 (Blackwell) predictors are REFUTED** — L2 hit prediction, L1, and
  wall-clock all fail their gates on the 5060; the card records the refutation
  verbatim (`predictor_state`, `evidence_status:
  "Verified-live-sm_120 (LABELS live; every predictor REFUTED)"`). These are
  framed as open research challenges, not solved features.
- **Wall-clock is labels + refutation only** on both arches (sm_86
  `predictor_state.wall_clock`, sm_120 SMAPE 1.766 does not beat a shape/size
  baseline).

Each `research_gap` carries a reason and a stable descriptive identifier. Treat
it as a concrete experiment to design, not as an agent-workflow blocker.

---

## See also

- [ARCHITECTURE.md](ARCHITECTURE.md) — how profile cards feed the active Rust simulator.
- [STATUS.md](STATUS.md) — current implementation and research priorities.
- [`arch_profiles/loader.py`](../arch_profiles/loader.py) · [`aliases.json`](../arch_profiles/aliases.json) · [`build_arch_families.json`](../arch_profiles/build_arch_families.json) · [`schema/arch_profile.schema.json`](../arch_profiles/schema/arch_profile.schema.json) · [`tests/test_arch_profile_lockstep.py`](../arch_profiles/tests/test_arch_profile_lockstep.py)
