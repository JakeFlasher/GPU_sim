# LDGSTS + ATOMS Operand-Semantics Adjudication (sm_86 tracer contract flip)

> task5 (analyze). The design prerequisite for the task4 tracer
> allocation-resolution-gate widening: it decides which memory-operand address
> the NVBit wire carries and which records must be stamped with a global
> `allocation_id`, so the ASYNC_COPY+ATOM flip never mis-stamps a shared
> destination/atomic. Grounded in tracer-code + NVBit-header + SASS operand
> inspection (plan §247). Produced via independent analysis + an analyze-lane
> expert consult (the two agreed on the core; the consult added the
> `getMemorySpace()` discriminator, the per-address wire field, and the
> operand-index-not-hardcoded mitigation, all folded in below). The known-address
> canary (§4) is the task4/task6 empirical confirmation.

## Verdict up front
The transition is sound and implementable **iff the discriminator is the NVBit
per-instruction memory space (`getMemorySpace()`), NOT "does the address resolve
to a global allocation."** The single genuinely-unresolved fact is the MREF
operand *index* ordering for LDGSTS on sm_86, which the canary settles; a reversed
ordering is a one-line producer fix, not a contract redesign.

## Evidence base
- Classifier: `memory_op_class.h:110` (`LDGSTS → kAsyncCopy`), `:121-124`
  (`ATOM*`/`RED*` → `kAtomic`, prefix-matched — does NOT distinguish shared vs
  global); test `tests/memory_op_class_test.cc:54-65` confirms `ATOMS`(shared),
  `ATOMG`(global), `ATOM`(generic), `RED*` all collapse to `kAtomic`.
- MREF capture: `tracer_tool.cu:1320-1329` (per-operand `MEMORY_REF` inject loop),
  `:1399-1400` (`nvbit_add_call_arg_mref_addr64(instr, mem_oper_idx)`,
  `assert(num_mref <= 2)`), `:1409` (index decrements → wire is reverse-operand
  order); `inject_funcs.cu:70-79` (per-lane 32-value address vector).
- NVBit API: `nvbit.h:607-608` ("Typically memory instructions have only 1 MREF
  so in general id = 0" — LDGSTS is the atypical `id ∈ {0,1}`); `Instr::getMemorySpace()`
  `nvbit.h:78` returning the enum `instr_types.h:51-64` (`GLOBAL`, `SHARED`,
  `GENERIC`, `GLOBAL_TO_SHARED`); no per-operand space accessor (the
  `src_memspace`/`dst_memspace` pair `nvbit.h:166-167` is `TMATransferInfo`, i.e.
  TMA-only, NOT LDGSTS); `mref` struct `instr_types.h:161-172` exposes no space.
- Host-side drop: `tracer_tool.cu:718-727` (`is_excluded_op_class`, incl. `kAtomic`
  `:722` + `kAsyncCopy` `:724`); `:3210-3248` routes excluded/unspecified to the
  `unsupported_memory_path` sidecar.
- Alloc-resolution gate: `tracer_tool.cu:3275` (`if (cls == kGlobal && mask.any())`);
  `:3280-3313` per-lane `lookup_allocation_id` (registry "tracks GLOBAL device
  address spaces only", `:3269-3273`); `:3293-3302` any-zero → `unmapped_memory_range`;
  `:3303-3312` cross-allocation span → `cross_allocation_lane_span`; `:3313` stamp.
  Ready-made producer hook: the currently write-only `opcodes_id_ldgsts`
  `std::set<int>` (`:636`, populated `:1281-1282`).
- Validator: `import_trace_contract.py:1007-1011` (`global_class` keyed on
  `MEMORY_OP_CLASS_GLOBAL` only → today FAIL-OPENS on ATOMIC(6)/ASYNC_COPY(8)),
  `:1021-1023` (predicated-off admit), `:1024-1049` (`allocation_id_missing`),
  `:1050-1106` (sidecar/known/range checks), `:1113-1135` (class-present checks).
- Wire: `instruction.proto:29` (`MEMORY_OP_CLASS_ATOMIC=6`, `ASYNC_COPY=8`
  committed values), `:56` (`memory_op_class` is INSTRUCTION-level); `address.proto`
  (`allocation_id = 7` → next free additive tag 8).
- SASS: `.claude/knowledge/sass/sass-instruction-reference.md:69-70`;
  `references/NVIDIA_Manuals/cuda_binary_utility/04_instruction-set-reference.md` §4.2.

## 1. LDGSTS operand semantics
`LDGSTS` (`cp.async` load-global-store-shared) is a **two-memory-operand**
instruction; sm_86 SASS: `LDGSTS.E.BYPASS.<sz> [R_dst_shared], [R_src_global.64]`.
Both bracketed operands are MREFs, matching `assert(num_mref <= 2)`. The tracer's
operand loop treats every MREF uniformly and emits one `mref_addr64` per operand,
so **the wire carries BOTH addresses as two `address` records on the same
instruction** — the global source VA AND the shared destination offset; neither is
synthesized.

- Global source → 64-bit device VA → resolves in the global-only registry to a
  non-zero id.
- Shared destination → per-CTA shared byte offset → returns `0` from the
  global-only registry **by construction**.

**Adjudication.** Allocation resolution applies to the **global-source operand
only**. The shared destination is recorded with `resolved_allocation_id == 0` as a
**legitimate** value and MUST be routed around the any-zero → `unmapped_memory_range`
drop. If the current `cls == kGlobal` per-lane resolution ran over the shared-dest
record, every lane would return 0 and the record would be spuriously dropped — the
exact failure the plan forbids. The gate must key on **per-operand memory space**,
not on the returned value.

## 2. ATOMS (shared) vs ATOMG (global) discrimination
The classifier collapses `ATOMS`/`REDS`(shared), `ATOMG`/`REDG`/`RED`(global), and
`ATOM.E`(generic) all to `kAtomic`, so **opcode class alone gives the gate no
shared/global signal.**

**Why "resolves to a global allocation" is insufficient (decisive):** it cannot
distinguish a correct `ATOMS` to shared offset (→ 0, should be EXEMPT) from a
buggy/out-of-bounds `ATOMG` to a freed VA (→ 0, should be REJECTED). The registry
return is an *outcome*, not an *identity*.

**Discriminator = the NVBit per-instruction memory space** (`getMemorySpace()`,
`nvbit.h:78` → `instr_types.h:51-64`), captured at instrument time, with the opcode
suffix as a cheap cross-check (fail closed on disagreement):

| Opcode | `getMemorySpace()` | Decision |
|---|---|---|
| `ATOMS`, `REDS` | `SHARED` | **exempt** (allocation_id legitimately 0) |
| `ATOMG`, `REDG`, `RED` | `GLOBAL` | **require** allocation_id |
| `ATOM.E` (generic) | `GENERIC` | **fail-closed → require** (§4 residual) |

An atomic is a single-operand instruction, so for atomics this fully resolves the
question — no operand-ordering uncertainty applies.

## 3. task4 implementation contract

**(a) Which records get allocation_id stamping.** Replace the gate guard
`if (cls == kGlobal && mask.any())` (`:3275`) with
`if (requires_global_identity(space_of_this_operand) && mask.any())`, the set being
`cls == kGlobal` OR (`cls == kAtomic` ∧ space ∈ {GLOBAL, GENERIC}) OR
(`cls == kAsyncCopy` ∧ this operand is the global-source MREF). The existing
three-way outcome (stamp / `unmapped_memory_range` / `cross_allocation_lane_span`)
is retained unchanged for those.

**(b) Shared records carry a legitimate 0 — distinguished STRUCTURALLY, not by
value.** Shared atomics + LDGSTS-shared-dest **skip resolution entirely** and emit
with `resolved_allocation_id == 0`. A global record that would yield 0 is converted
to a typed drop *before any wire record is written*, so a shared record's 0 is
"never asked" while a global 0 "never reaches the wire." The value 0 is never
ambiguous downstream because the record's space disambiguates it.

**(c) The per-address wire field (REQUIRED).** Because `memory_op_class` is
INSTRUCTION-level (`instruction.proto:56`) and LDGSTS's two operands share one
instruction, the class cannot carry two spaces — so the validator (which never sees
NVBit state) cannot enforce per-operand from class alone. Add **one additive field**
`optional MemoryOpClass address_space = 8` to the `address` message (proto3-additive,
capability-gated, NO `schema_version` bump), stamped GLOBAL on the global
source/global atomic and SHARED on the shared dest/shared atomic.

**(d) Validator (`import_trace_contract.py`).** Rekey the identity gate
(`:1007-1011`) on the per-address space with a legacy fallback:
`global_identity = addr.HasField("address_space") ? addr.address_space == GLOBAL :
(inst.memory_op_class == MEMORY_OP_CLASS_GLOBAL)`. Then:
- global atomic / LDGSTS-global-source, active, id absent/zero → **`allocation_id_missing`
  REJECT** (`:1032-1049`) — **negative test (a)**;
- global with a present id → sidecar/known/range checks fire unchanged (`:1055-1106`);
- shared atomic / LDGSTS-shared-dest (`address_space == SHARED`) → identity clauses
  do NOT fire; admit with id absent — **negative test (b)**; a producer regression
  mislabeling a shared address GLOBAL is rejected in both directions
  (`allocation_id_missing` or `allocation_id_unknown`/range);
- predicated-off (`and_mask == 0`) → admit, all classes (`:1021-1023`).

**(e) Minimal change surface (gate-level, NOT classifier re-routing).**
1. drop `kAtomic`+`kAsyncCopy` from `is_excluded_op_class` (`:722,:724`); the other
   five stay excluded.
2. capture `instr->getMemorySpace()` at instrument time into an `opcode_id`-keyed
   map (reuse the write-only `opcodes_id_ldgsts` hook, `:636`/`:1281-1282`); for
   LDGSTS also record which MREF index is the global side (from operand order, §4).
3. rekey the resolution guard (`:3275`) to `requires_global_identity(space)`.
4. add `optional MemoryOpClass address_space = 8` to `address.proto`.
5. rekey the validator (`:1007-1011`) with the legacy fallback.

**Rejected alternative:** re-routing `ATOMS→kShared`/`ATOMG→kGlobal` in the
classifier — it strands LDGSTS (one instruction, two spaces cannot fit the
instruction-level class) and silently repurposes committed `MEMORY_OP_CLASS_ATOMIC=6`,
risking any consumer keying on class==ATOMIC. Keep `memory_op_class` faithful
(ATOMIC/ASYNC_COPY, direction in `memory_op_access_kind`); carry enforcement space
per-address.

## 4. Residual risk + the known-address canary (task4/task6 validation)
1. **LDGSTS MREF operand-index ordering on sm_86 — RESOLVED by an in-tree
   runtime-tested precedent.** `tools/nvbit_tracer/metric_extractor/metric_extractor.cu:1346-1379`
   already records the LDGSTS global source with **MREF id = 1** ("LDGSTS carries
   two memory references in SASS order — `[shared_dst]` first, `[global_src.64]`
   second … the address arg uses MREF id=1 (the global source)"), gated on
   `getMemorySpace() == GLOBAL_TO_SHARED`, and the runtime test
   `calib_cpasync_global_to_shared` "validates this by comparing captured sectors
   against the kernel's d_in address; a future SASS form that swaps operand ordering
   breaks the test loudly." So the producer stamps **id 0 = shared dest, id 1 =
   global source** with an existing regression guard; the known-address canary
   (§below) becomes a confirming regression check, not a discovery step. Upgrades
   the operand-index confidence from MEDIUM to HIGH.
2. **Generic atomics (`ATOM.E`, GENERIC)** whose runtime address lands in shared →
   fail-closed drop (safe, typed) — a flagged coverage gap for generic-atomic-into-
   shared kernels, measured separately.

**Canary** (one warp, host-KNOWN global buffer `G` with recorded `[base,base+size)`,
`__shared__ int S[32]`, launched with `flag==0`): (1) ordinary global load `G[t]`;
(2) global atomic `atomicAdd(&G[t],1)` → ATOMG; (3) shared atomic `atomicAdd(&S[t],v)`
→ ATOMS; (4) `__pipeline_memcpy_async(&S[t], &G[(t+1)&31], 4)` → LDGSTS; (5)
predicated-off `if (flag) G[t]=v`. Trace and assert: (1)(2) global-atomic +
LDGSTS-**source** addresses inside `G`'s range with non-zero `allocation_id`; (3)
shared atomic + LDGSTS-**dest** addresses return 0 and are admitted WITHOUT being
dropped; (4) the LDGSTS emits **two** address records — which wire position holds
the global one confirms/refutes the operand-index ordering; (5) predicated-off
admits with no id. REFUTED if: LDGSTS emits one record (NVBit doesn't surface both
MREFs on sm_86); a shared record resolves non-zero (offsets alias `G`'s VA →
exemption must gate on space not range); or the global-atomic address falls outside
`G` (extraction bug). Any of these blocks the transition until fixed.

## Implementation contract summary

| Record type | Discriminator | Gate action (`tracer_tool.cu` ~3275) | Validator (`import_trace_contract.py`) | Fail-closed |
|---|---|---|---|---|
| plain global `LDG`/`STG` | `cls==kGlobal` | resolve; stamp id | GLOBAL → require id + range | 0→`unmapped_memory_range`; multi→`cross_allocation_lane_span` |
| global atomic `ATOMG`/`REDG`/`RED` | `kAtomic` ∧ space GLOBAL | resolve; stamp id; `address_space=GLOBAL` | GLOBAL → require id + range | absent/0 id → `allocation_id_missing` (**neg. a**) |
| generic atomic `ATOM.E` | `kAtomic` ∧ space GENERIC | fail-closed → resolve as global | GLOBAL → require id + range | shared-window → `unmapped_memory_range` (**residual**) |
| shared atomic `ATOMS`/`REDS` | `kAtomic` ∧ space SHARED | **skip resolution**; id=0 legit; `address_space=SHARED` | SHARED → identity clauses do NOT fire; admit | mislabeled GLOBAL → `allocation_id_unknown`/range (**neg. b**) |
| LDGSTS global source | `kAsyncCopy`, global-side MREF | resolve; stamp id; `address_space=GLOBAL` | GLOBAL → require id + range | absent/0 → `allocation_id_missing` |
| LDGSTS shared dest | `kAsyncCopy`, shared-side MREF | **skip resolution**; id=0 legit; `address_space=SHARED` | SHARED → identity clauses do NOT fire; admit | canary-confirmed exempt |
| predicated-off (any) | `active_mask & predicate_mask == 0` | no identity required | `and_mask==0` → admit | n/a |

## Confidence
- LDGSTS is global→shared with two MREFs; allocation resolution belongs to the
  global source only — **HIGH** (SASS/PTX + the two-MREF path + the global-only
  registry).
- `getMemorySpace()` space discriminator for shared vs global atomics + the
  per-address wire field + validator mechanics — **HIGH** (NVBit headers + the
  cited code).
- The exact NVBit LDGSTS operand-*index* attribution (id 1 = global source) —
  **HIGH**, resolved by the in-tree runtime-tested precedent
  `metric_extractor.cu:1346-1379` + `calib_cpasync_global_to_shared`; the canary is
  a confirming regression check, and a reversed ordering is a one-line producer fix
  that "breaks the test loudly."
