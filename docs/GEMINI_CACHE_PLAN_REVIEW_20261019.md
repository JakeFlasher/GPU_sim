I will inspect `docs/PROJECT_MEMORY.md`, `docs/STATUS.md`, `docs/CONTRACT.md`, and `CACHE_LOCALITY_EXECUTION_PLAN.md` first, followed by the relevant simulator source and Python tooling seams, to produce a read-only feasibility and execution analysis.
# Cache & Locality Execution Plan — Read-Only Feasibility Audit & Execution Plan

This review audits [`CACHE_LOCALITY_EXECUTION_PLAN.md`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/CACHE_LOCALITY_EXECUTION_PLAN.md) against [`AGENTS.md`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/AGENTS.md), [`docs/PROJECT_MEMORY.md`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/docs/PROJECT_MEMORY.md), [`docs/STATUS.md`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/docs/STATUS.md), [`docs/CONTRACT.md`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/docs/CONTRACT.md), [`docs/MRC_NCU_VALIDITY_AUDIT.md`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/docs/MRC_NCU_VALIDITY_AUDIT.md), and the live Rust/Python codebase. No files or code have been modified.

---

## 1. Milestone Implementation Status Matrix

| Milestone | Scope | Current Codebase Status | Primary Code Seams |
| :--- | :--- | :--- | :--- |
| **Pre-M1 Baseline** | SM86 `L1WriteLookupObserver`, SM120 `async_global_l1_hit_exclusion`, typed `L2ClientScope` shadow, Tier-1 `sector_mrc_observer_v1`, base `reuse_interval_observer_v1` | **Implemented** | [`sim/mem/src/l1.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L1307-L1420), [`sim/sm/src/metrics.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/metrics.rs#L163-L768), [`sim/sm/src/system.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs#L4267-L4313), [`sim/direct_run/src/emit.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/emit.rs#L935-L1024) |
| **M1** | Split L1 `pending_merges` into `already_requested_merges` vs. `unrequested_pending_merges`; include `already_requested_merges` in SM120 lookup hits | **Missing** (`pending_merges` is still monolithic; SM120 `effective_lookup_hits_for_arch` still returns `resident_hits` only) | [`L1OriginCounters`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L434-L504), [`L1Counters`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L507-L574), [`L1MutationBudget`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L2304-L2325), [`L1Cache::access_core`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L3110-L3223), [`emit.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/emit.rs#L935-L1012), [`lib.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/lib.rs#L2208-L2251) |
| **M2A** | Enable per-SM `L1WriteLookupObserver` for SM120 via card + `GlobalUntagged` store support + report wiring | **Missing** (`ArchId::l1_write_lookup_window` returns `research_gap` for SM120; `stateful_write_lookup_observer_enabled` is hardcoded to `Sm86` and `"STG"`) | [`ArchId::l1_write_lookup_window`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/arch/src/lib.rs#L123-L137), [`sm_120-rtx-5090.json`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/arch_profiles/cards/sm_120-rtx-5090.json#L251-L260), [`stateful_write_lookup_observer_enabled`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L2841-L2851), [`l1_write_hit_terms`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/emit.rs#L1198-L1216), [`cache_lookup_prediction_reports`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/lib.rs#L2250-L2277) |
| **M2B** | Unify `GlobalUntagged` store observer routing by removing `state.admitted_warps <= 8` gate and reading `observer_operation` in `system.rs` | **Missing** (`admitted_warps <= 8` gate active at `issue.rs:14001`; `system.rs` ignores `req.observer_operation` at lines `4230, 4315, 4380`) | [`enqueue_mem_requests_for_class`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L13994-L14003), [`MultiSmSystem::step_cycle`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs#L4228-L4385) |
| **M2C** | Full-sector write-allocate / sector install on store and L2 read-fetch suppression | **Missing — High Risk** (Requires trace-mask derivation and alters timing/IPC if wired into live `L1Cache` instead of a shadow observer) | [`L1Cache`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L1682-L1958), [`access_shared`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L14807-L14908), [`emit_trace_mem_requests`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L9222-L9344) |
| **M3** | Pre-packet L2 write tag-hit snapshot (`line_present_before_packet`) and SM86 L2 write lookup hit reporting | **Missing** (`line_present` mutates per sector at `l2.rs:2220`; furthermore, store instructions emit 1-sector `MemRequest`s in `issue.rs:9292`) | [`L2Cache::access_mask_event_coalesced_transactional`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l2.rs#L2100-L2232), [`MultiSmSystem::step_cycle`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs#L4225-L4401), [`metrics.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/metrics.rs#L3823-L3843), [`lib.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/lib.rs#L2300-L2340) |
| **M4 (Tier 1)** | Class-A `sector_mrc_observer_v1` (Fenwick/Mattson) + upgraded `reuse_interval_observer_v1` (Xiang right-censored tail, 80-bin exact-sum histogram, coupled line/sector clock) | **`sector_mrc_observer_v1` Implemented; `reuse_interval_observer_v1` upgrade Missing** | [`LocalityMrcObservation`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/metrics.rs#L216-L768), [`ReuseIntervalObservation`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/metrics.rs#L163-L339), [`ReuseIntervalState`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/metrics.rs#L1755-L1906), [`validate_reuse_mrc.py`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/tools/validate_reuse_mrc.py#L62-L137) |
| **M4 (Tiers 2 & 3)** | Class-B `sm_l1_locality_observer_v1` and `l2_system_locality_observer_v1` conditional scenario envelopes | **Missing** | [`sim/sm/src/metrics.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/metrics.rs), [`sim/sm/src/system.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs), [`sim/direct_run/src/emit.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/emit.rs) |
| **M5** | Full 12-worker `.jsonl.zst` replay on SM120 (339 traces) & SM86 (79 traces), validator & metric accuracy reports | **Tooling exists, CLI flags in plan are outdated** | [`tools/sm120_corpus_replay.py`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/tools/sm120_corpus_replay.py), [`tools/validate_reuse_mrc.py`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/tools/validate_reuse_mrc.py), [`tools/report_locality_mrc.py`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/tools/report_locality_mrc.py), [`tools/sm120_metric_report.py`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/tools/sm120_metric_report.py) |

---

## 2. Deep-Dive Seam Audit & Critical Algorithmic Traps by Milestone

### Milestone M1: SM120 In-Flight Sector Merge Decomposition

#### Exact Seams to Modify
1. **[`L1OriginCounters`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L434-L504) & [`L1Counters`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L507-L574):**
   - Add `pub already_requested_merges: u64` and `pub unrequested_pending_merges: u64` to both structs.
   - Update [`L1OriginCounters::effective_lookup_hits_for_arch`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L483-L488) and [`L1Counters::effective_lookup_hits_for_arch`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L569-L574):
     - `ArchId::Sm120 => self.resident_hits.saturating_add(self.already_requested_merges)`
     - `_ => self.hits`
   - Maintain the exact conservation invariant: `pending_merges == already_requested_merges + unrequested_pending_merges` and `hits == resident_hits + pending_merges`.
2. **Trap #1 — [`L1MutationBudget`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L2304-L2378) Fixed-Size Array:**
   - `CACHE_LOCALITY_EXECUTION_PLAN.md` misses [`L1CounterField`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L2304-L2325) and `L1MutationBudget`, which uses a fixed array `per_field_increments: [u64; 17]` indexed by `field as usize` and checked in `L1Cache::bump_counter` / `bump_origin_counter`. If new `L1CounterField` variants (`AlreadyRequestedMerges`, `UnrequestedPendingMerges`) are added, the array size at [`l1.rs:2308`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L2308) and `record_counter_increment` must be updated accordingly, or it will panic with an out-of-bounds index at runtime.
3. **Trap #2 — Intra-Instruction vs. Cross-Instruction Line Merge State in [`L1Cache::access_core`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L3110-L3223):**
   - Look closely at how `L1Cache::access_core` interacts with [`access_shared`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L14807-L14908):
     - When a cold 128B (4-sector) warp load executes, `access_shared` calls `access_with_origin` sequentially for each of the 4 sectors in `sectors[off..off+len]`.
     - Sector 0 hits the cold miss path (`MissLineAlloc`, [`l1.rs:3314-3324`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L3314-L3324)), which inserts `line_fills.insert(line, fill_until)` and sets `*self.sectors.entry(line).or_default() |= 1 << sector_0`.
     - Sectors 1, 2, and 3 of the **same warp instruction** then enter `access_core` and match `if let Some(&fill_until) = self.line_fills.get(&line)` ([`l1.rs:3148`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L3148)). Because `self.sectors` only has bit 0 set, `(self.sectors.get(&line).copied().unwrap_or(0) & (1 << sector)) != 0` is `false` for sectors 1..3! Thus sectors 1..3 are properly classified as `unrequested_pending_merges` (0 lookup hits on a cold 128B warp load), while [`l1.rs:3169`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L3169) sets `1 << sector` in `self.sectors` because [`access_shared`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L14866) folds same-cycle sectors (`st.line_folds`) into the initial L2 line request without calling [`reserve_pending_line_sector`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L2211-L2229).
     - If a **second** warp later accesses any of sectors 0..3 while `line_fills` is still in flight, those sector bits are now present in `self.sectors` (or in `self.sector_fills` if requested on an intermediate cycle via `reserve_pending_line_sector`), so `sector_already_known` evaluates to `true` and increments `already_requested_merges`.
   - **One subtle bug to prevent in `access_core`:** In `L1Const` mode ([`issue.rs:14756-14805`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L14756-L14805)), `reserve_pending_line_sector` is not called on `LineMerge` at cycle $t_1 > t_0$, whereas in `L1Shared` (`l1-l2` full replay) it is. In `L1Shared`, when a later warp at cycle $t_1 > t_0$ requests a *new* sector $s$ on a line with an active `line_fill`, `access_core` sets `1 << s` in `self.sectors` at [`l1.rs:3169`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L3169), and then `access_shared` immediately calls `reserve_pending_line_sector` ([`issue.rs:14875`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L14875)), which clears `1 << s` from `self.sectors` and inserts `(line, s, generation)` into `self.sector_fills`. Therefore, any subsequent warp at $t_2 \ge t_1$ accessing sector $s$ will hit `self.sector_fills.get(&(line, sector, generation))` at [`l1.rs:3120`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L3120) (`SectorMerge` -> `already_requested_merges`). This means checking `let sector_already_known = (self.sectors.get(&line).copied().unwrap_or(0) & sbit) != 0 || self.sector_fills.contains_key(&(line, sector, generation));` before mutating `self.sectors` at [`l1.rs:3169`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L3169) is **exact** for both same-cycle folded sectors and cross-cycle secondary sector reservations.
4. **Trap #3 — [`global_resident_hit_adjustment`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/emit.rs#L935-L1012) and [`lib.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/lib.rs#L2208-L2251):**
   - Look at [`emit.rs:943-993`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/emit.rs#L943-L993) (`global_resident_hit_adjustment` and `global_read_resident_hit_adjustment`) and [`emit.rs:1001-1012`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/emit.rs#L1001-L1012) (`async_global_l1_hit_exclusion`). Currently:
     - `async_global_l1_hit_exclusion` returns `counters.by_origin.async_Copy.resident_hits` when `arch.excludes_async_global_from_l1_hit_rates()` is true (SM120). Once `effective_lookup_hits_for_arch(Sm120)` includes `already_requested_merges`, `async_global_l1_hit_exclusion` must return `counters.by_origin.async_copy.effective_lookup_hits_for_arch(arch)` (or `resident_hits + already_requested_merges`) so `LDGSTS` in-flight merges are also excluded instead of leaking into global hit rates.
     - Both [`emit.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/emit.rs#L1217-L1290) and [`lib.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/lib.rs#L2208-L2251) call `effective_lookup_hits_for_arch(arch)` and `async_global_l1_hit_exclusion(arch, counters)`, so updating those two functions updates both native metrics and `cache_lookup_predictions`.

---

### Milestone M2A: SM120 Per-SM Write Lookup Observer Activation

#### Exact Seams to Modify
1. **[`ArchId::l1_write_lookup_window`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/arch/src/lib.rs#L123-L137) & [`sm_120-rtx-5090.json`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/arch_profiles/cards/sm_120-rtx-5090.json#L251-L260):**
   - **Trap #4 — Strict JSON Card Schema (`deny_unknown_fields`):** [`sim/arch/src/card.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/arch/src/card.rs#L172-L195) parses architecture cards using `strict_json_from_slice` with `#[serde(deny_unknown_fields)]` on `CacheHierarchySection` (`l1_d_bytes_per_sm`, `l1_geometry`, `l2_bytes`, `l2_Slice_count`, `memory_bus_bits`). Adding an undeclared JSON key in `sm_120-rtx-5090.json` will cause all SM120 runs to fail closed on startup!
   - Either add an optional typed field to `CacheHierarchySection` in [`sim/arch/src/card.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/arch/src/card.rs#L172-L195) AND update [`ArchId::l1_write_lookup_window`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/arch/src/lib.rs#L123-L137) to return `Capability::with_source(L1WriteLookupObserverConfig { window_lines: 512, max_Sectors_per_line: 4 }, CapabilityStatus::ScopedSimplification, ...)` for `ArchId::Sm120` (128 KB / 256B = 512 lines, or 1024 128B lines—note that `L1WriteLookupObserverConfig` is already wired through `ArchId::l1_write_lookup_window()` in [`issue.rs:2610, 6872, 9225`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L6872)).
2. **[`stateful_write_lookup_observer_enabled`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L2841-L2851):**
   - Currently checks `if schedule.arch != ArchId::Sm86 { return false; }` and `head == "STG"`.
   - Replace `schedule.arch != ArchId::Sm86` with `!schedule.arch.l1_write_lookup_window().status.is_available()` (or `matches!(schedule.arch, ArchId::Sm86 | ArchId::Sm120)`).
   - Replace `head == "STG"` with `matches!(head, "STG" | "ST" | "STL")` so `GlobalUntagged` store instructions (`ST.E.*`) that enter [`record_store_lookup_observer`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L9329-L9334) are not silently dropped by the mnemonic filter.
   - Also check the two call sites in [`issue.rs:6870`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L6870) and [`issue.rs:9222`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L9222) which explicitly check `if schedule.arch == ArchId::Sm86`: replace those with `if schedule.arch.l1_write_lookup_window().value.is_some()` so SM120 `LDG`/`LDS` read observations also populate the SM120 write-lookup observer! (**Critical catch:** `CACHE_LOCALITY_EXECUTION_PLAN.md` Section 2.1 only mentioned `stateful_write_lookup_observer_enabled` at line 2841, missing lines `6870` and `9222` where read accesses populate the observer window!).
3. **Trap #5 — Cross-File Disagreement Between [`emit.rs:1198-1216`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/emit.rs#L1198-L1216) and [`lib.rs:2264`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/lib.rs#L2264):**
   - In [`emit.rs:1198-1216`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/emit.rs#L1198-L1216), `l1_write_hit_terms(arch, counters, trace)` computes `(write_hits, write_hit_misses)` for `l1tex__t_sector_hit_rate.pct`.
   - However, in [`lib.rs:2264`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/lib.rs#L2264) (`cache_lookup_prediction_reports` for `CacheHitRateL1TexSector`, which emits `l1tex__t_sectors_lookup_hit.sum` and `l1tex__t_sectors_lookup_miss.sum`), the code currently writes:
     ```rust
     let write_hits = counters.store_hits;
     ```
     ignoring `l1_write_hit_terms`!
   - Both [`emit.rs:1198-1216`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/emit.rs#L1198-L1216) and [`lib.rs:2264`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/lib.rs#L2264) must use `crate::emit::l1_write_hit_terms(arch, counters, trace)` (with `matches!(arch, ArchId::Sm86 | ArchId::Sm120)`), or else `l1tex__t_sector_hit_rate.pct` and `l1tex__t_sectors_lookup_hit.sum` will violate internal metric conservation.

---

### Milestone M2B: `GlobalUntagged` Store Observer Routing Unification

#### Exact Seams to Modify
1. **[`enqueue_mem_requests_for_class`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L13994-L14003):**
   - Remove the `&& state.admitted_warps <= 8` condition at [`issue.rs:14001`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L14001) so all `GlobalUntagged` store instructions (`ST.*`) unconditionally tag `req.observer_operation = Some(MemoryOperation::Store)`.
   - Also route `GlobalUntagged` stores (`AccessKind::GlobalUntagged` when `inst.mem_store`) in [`emit_trace_mem_requests`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L9328-L9334) through `record_store_lookup_observer` so per-SM `L1WriteLookupObserver` sees untagged stores on both SM86 and SM120.
2. **[`MultiSmSystem::step_cycle`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs#L4228-L4385):**
   - Note: `MemRequest` defines `pub observer_operation: Option<MemoryOperation>` (there is no helper method `effective_observer_operation()` yet; add `pub fn effective_observer_operation(&self) -> MemoryOperation { self.observer_operation.unwrap_or(self.operation) }` on `MemRequest` in [`sim/sm/src/issue.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L180-L235)).
   - Use `req.effective_observer_operation()` at [`system.rs:4230`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs#L4230) (`record_l2_sector_for_locality`), [`system.rs:4315`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs#L4315) (`record_l2_client_scope_shadow`), and [`system.rs:4380`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs#L4380) (`l2_lookup_traffic` store vs. read attribution), while keeping `req.operation` for physical L2 timing/DRAM state (`access_mask_event_coalesced_transactional` at [`system.rs:4241`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs#L4241)) so timing behavior is preserved.

---

### Milestone M2C: Store-to-Load Sector Residency & L2 Read-Fetch Suppression (Scientific & Architectural Audit)

#### Critical Concern & Blocker Analysis
`CACHE_LOCALITY_EXECUTION_PLAN.md` Section 2.3 proposes modifying live [`L1Cache`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs) so full-sector stores set `self.sectors |= 1 << sector` and **suppress subsequent L2 `OrdinaryLdg`/`Local` read requests** in [`access_shared`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L14807-L14908). Two concrete issues must be addressed before touching code:
1. **Missing Byte-Mask Representation in `sector_arena`:**
   - In [`emit_trace_mem_requests`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L9222-L9344), `sectors: &[i64]` only contains deduplicated 32B sector IDs (`addr >> 5`), not per-sector 32-bit byte masks.
   - **Trace-derived, card-only solution:** Derive full-sector write coverage deterministically from the trace instruction metadata already on `TraceInst`: an unpredicated full-warp store (`(inst.active_mask & inst.predicate_mask) == 0xffff_ffff`) with per-lane element size $\ge 4$ bytes (e.g., `STG.E`, `STG.E.64`, `STG.E.128`, `STL`, `ST.E`, excluding `.U8`/`.S8`/`.U16`/`.S16`) whose emitted sector count `len` equals `(32 * elem_bytes) / 32` (`len == 4` for 32-bit, `len == 8` for 64-bit, `len == 16` for 128-bit) and whose sectors are contiguous (`sectors[off + len - 1] - sectors[off] == len - 1`) is mathematically guaranteed to overwrite all 32 bytes of every touched sector without needing raw per-lane addresses.
2. **Behavioral Timing Side Effect vs. Observer Separation:**
   - If `L1Cache` installs full-sector stores into live `self.sectors` and suppresses L2 read requests in `access_shared`, a subsequent load transitions from an L1 miss (L2/DRAM latency $\sim 200+$ cycles) to an L1 resident hit ($33$ cycles). That directly alters warp stall release cycles, scheduler issue order, L2 LRU state, and `Active Cycles` / `Elapsed Cycles`.
   - **Recommendation:** Implement M2A + M2B first and evaluate the full 339-trace SM120 / 79-trace SM86 corpus. Only enable live L1/L2 store-to-load sector installation in M2C if gated behind a card capability (`l1_full_sector_store_install`) and verified not to regress `Active Cycles` / `Elapsed Cycles` or `l2_hit_rate.pct` on the full replay corpus; otherwise, record full-sector store-to-load forwarding in the per-SM `L1WriteLookupObserver` / `L2ClientScope` shadow so lookup-hit and L2 read-demand predictions are updated without destabilizing cycle timing.

---

### Milestone M3: L2 Write Lookup Hit & Pre-Packet Tag-Hit Attribution

#### Critical Architectural Trap Found in [`issue.rs:9292`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L9292) & [`system.rs:4237`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs#L4237)
`CACHE_LOCALITY_EXECUTION_PLAN.md` Section 3.1 correctly identifies that [`L2Cache::access_mask_event_coalesced_transactional`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l2.rs#L2103-L2221) sets `line_present = true` at line `2220` inside the `for sector in 0..4` loop. **However, looking at the caller reveals a second, deeper cause of the 3-of-4 cold store hit bug:**
1. **Trap #6 — Store Instructions Emit Separate 1-Sector `MemRequest`s in [`emit_trace_mem_requests`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L9234-L9320):**
   - Look at [`issue.rs:9234-9320`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L9234-L9320):
     ```rust
     for (request_index, &sector_id) in sectors[off..off + usize::from(len)].iter().enumerate() {
         ...
         let mut request = MemRequest::store(t, sector_id, origin);
         ...
         out.push(request);
     }
     ```
   - Each sector of a coalesced 4-sector store is pushed as a **separate** `MemRequest` with a single-sector mask (`1 << (sector_id & 3)`), whereas coalesced loads use `MemRequest::load_line_sectors`!
   - Consequently, in [`MultiSmSystem::step_cycle`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs#L4225-L4401), each of the 4 sectors of a single warp `STG.E` instruction calls `access_mask_event_coalesced_transactional` **4 separate times**! Even if you snapshot `line_present_before_packet = line_present` inside `access_mask_event_coalesced_transactional`, the first sector's call allocates the L2 line in `self.lines`, so calls #2, #3, and #4 for the **same warp store instruction** will still see `line_present_before_packet == true`!
   - **Exact Fix Required for M3:**
     1. In [`L2Cache::access_mask_event_coalesced_transactional`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l2.rs#L2100-L2232), snapshot `let line_present_before_packet = line_present;` before `for sector in 0..4` (`line 2103`) and add `pub pre_packet_write_tag_hit_mask: u8` to [`L2MaskPreview`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l2.rs#L2225-L2231) (set bits where `line_present_before_packet` is true).
     2. **Crucially**, either coalesce same-line sectors of a single store instruction into one line-level `MemRequest` (matching `load_line_sectors`), **or** track a per-instruction/per-cycle store line set in [`MultiSmSystem::step_cycle`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs#L4359-L4377) (keyed by `(req.sm_ix, req.issue_time, req.warp_ix, line_addr)`) where the pre-instruction `line_present_before_packet` state of `line_addr` on the first sector of that store instruction is reused for the remaining sectors of the same warp store instruction on the same 128B line! Without this, cold 4-sector stores will still produce 3/4 = 75% L2 write tag hits.
2. **Trap #7 — Precedence Between [`metrics.rs:3829-3833`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/metrics.rs#L3829-L3833) and [`lib.rs:2317`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/lib.rs#L2317):**
   - In [`tools/sm120_metric_report.py:242-246`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/tools/sm120_metric_report.py#L242-L246), `native_metrics.` (rank 0, emitted by `sim/direct_run/src/metrics.rs`) takes precedence over `cache_lookup_predictions.` (rank 1, emitted by `sim/direct_run/src/lib.rs`).
   - Look at [`sim/direct_run/src/metrics.rs:3829-3833`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/metrics.rs#L3829-L3833):
     ```rust
     let write_hits = if run_cfg.arch == ArchId::Sm86 {
         split.store_lookup_hits.min(split.store_packet_candidate_sectors)
     } else {
         split.store_packet_lookup_hits
     };
     ```
     And look at [`system.rs:4361-4365`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs#L4361-L4365): on `ArchId::Sm86`, `store_lookup_hits` was set to `requested_sectors` (100% hit)!
   - Therefore, in [`system.rs:4359-4377`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs#L4359-L4377), [`metrics.rs:3829-3833`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/metrics.rs#L3829-L3833), and [`lib.rs:2315-2325`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/lib.rs#L2315-L2325), unify both SM86 and SM120 to use the pre-instruction/pre-packet L2 tag-hit count (`store_packet_lookup_hits` populated from `pre_packet_write_tag_hit_mask` with same-instruction line deduplication).

---

### Milestone M4: Reuse-Distance & MRC Architecture (Class-A vs. Class-B Separation)

#### 1. What Is Already Implemented in [`sim/sm/src/metrics.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/metrics.rs)
- **Tier-1 `sector_mrc_observer_v1` ([`LocalityMrcObservation`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/metrics.rs#L216-L768)):**
  - Already implements the exact Class-A Mattson/Fenwick stack-distance oracle over the 28-point capacity grid (`LOCALITY_MRC_CAPACITY_LINES`), including coupled 128B line-tag and 32B sector reuse distance (`coupled_reuse_distance_lines = max(d_line, d_sector)`), cold vs. capacity miss decomposition, and two-phase same-tick batch-query-before-insert interval envelopes (`lower` and `upper`).
  - **Do not rewrite `sector_mrc_observer_v1`**; it is already verified at `0.0%` discrepancy by [`tools/validate_reuse_mrc.py`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/tools/validate_reuse_mrc.py).
- **Tier-1 `reuse_interval_observer_v1` ([`ReuseIntervalObservation`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/metrics.rs#L163-L339)):**
  - Currently uses a 32-bin logarithmic record-tick histogram over an 8-point `REUSE_PROXY_CAPACITY_RECORDS` grid without right-censored tail intervals ($T - \text{last\_seen}$), without bin-sum tracking, and with a 15–40% gap against the exact Fenwick MRC.

#### 2. Exact Seams & Backward-Compatibility Requirements for Upgrading `reuse_interval_observer_v1`
- **Trap #8 — [`tools/validate_reuse_mrc.py`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/tools/validate_reuse_mrc.py#L62-L137) Schema Checks:**
  - `validate_reuse_mrc.py` checks `reuse["schema_version"] == "reuse_interval_observer_v1"`, `reuse["proxy_capacity_records"] == [32, 64, 128, 256, 512, 1024, 2048, 4096]`, and `sum(line_hist) + line_overflow == line_reused` for each stream (`l1_global_local`, `l1_shared`, `l2_sector`).
  - Therefore, when adding the Xiang/AET solver (`aet_miss_rates_by_capacity_lines` on the 28-point `LOCALITY_MRC_CAPACITY_LINES` grid, right-censored tail intervals `tail_interval_hist` / `tail_interval_sum`, exact per-bin interval sums `coupled_interval_sum`, and line-reference clock $V_t$), **preserve the existing `ReuseIntervalStreamSummary` fields** (or extend them additively and update `validate_reuse_mrc.py` in lockstep) so existing schema conservation checks remain 100% green while adding the AET vs. exact Fenwick WAPE check (`< 1.5%` corpus median WAPE).

#### 3. Tier-2 (`sm_l1_locality_observer_v1`) & Tier-3 (`l2_system_locality_observer_v1`) Class-B Conditional Shadows
- Implement as additive, read-only shadow structs serialized on `FullSimMetrics` in [`sim/sm/src/metrics.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/metrics.rs) and [`sim/sm/src/system.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs):
  - **Tier-2 (`sm_l1_locality_observer_v1`):** Per-SM demand stream (`OrdinaryLdg`, `Local`, plus `AsyncCopy` tagged separately) evaluating the card-configured SM L1 capacity point (`128 KB = 1024 lines` on SM120; `128 KB` minus shared carveout on SM86) under two card-only boundary scenarios:
    - $H_{\max}$: In-flight merge credit enabled (`already_requested_merges` + `L1WriteLookupObserver` hits included, `AsyncCopy` excluded on SM120).
    - $H_{\min}$: Resident-only hits (`resident_hits` only, no write-observer or in-flight merge credit).
  - **Tier-3 (`l2_system_locality_observer_v1`):** Cross-SM interleaved L2 demand stream evaluating the card-configured L2 capacity (`96 MB = 786,432 lines` across `16` slices on SM120; `6 MB = 49,152 lines` on SM86) under two card-only boundary scenarios:
    - $H_{\min}$: Uncached/bypass write policy (`OrdinaryLdg + Local` read demand + pre-instruction `line_present_before_packet` store tag hits).
    - $H_{\max}$: Full-sector store residency / write-allocate policy.

---

## 3. Audit of Scientifically Invalid or Unidentifiable Acceptance Gates

Per [`AGENTS.md`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/AGENTS.md) and [`docs/MRC_NCU_VALIDITY_AUDIT.md`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/docs/MRC_NCU_VALIDITY_AUDIT.md), the following acceptance gates in [`CACHE_LOCALITY_EXECUTION_PLAN.md`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/CACHE_LOCALITY_EXECUTION_PLAN.md) require scientific correction:

1. **Unidentifiable Gate: Multi-Capacity MRC vs. Single-Point NCU Profiles**
   - **Why Invalid:** NCU profiles (`ncu_profiles/*.csv` and `data/validation/ncu_Metrics.json`) capture a **single realized hardware operating point per kernel launch**, not a cache-capacity sweep. Furthermore, hardware L1/L2 hit rates conflate finite set associativity, MSHR merge credits, write-lookup policies, and shared-memory carveout with capacity misses. Treating a multi-capacity MRC curve $M(C)$ as directly comparable to NCU across $C$, or tuning MRC parameters to match NCU, is scientifically invalid and violates the [`AGENTS.md`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/AGENTS.md) ban on NCU fitting.
   - **Trace-Derived, Card-Only Replacement Gate:**
     - **Class-A Gate (Exact Trace Oracles — Zero NCU Input):**
       1. `sector_mrc_observer_v1` vs. exact Fenwick/Mattson stack-distance oracle: **`0.0%` error across all 28 capacities** and **zero monotonicity/conservation violations** across all 339 SM120 and 79 SM86 traces (enforced by [`tools/validate_reuse_mrc.py`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/tools/validate_reuse_mrc.py)).
       2. `reuse_interval_observer_v1` (upgraded AET solver with right-censored tail intervals and coupled line/sector clock) vs. `sector_mrc_observer_v1`: **corpus median WAPE $< 1.5\%$** across the 28-point capacity grid `LOCALITY_MRC_CAPACITY_LINES` on non-trivial workloads ($\ge 64$ unique lines), measured purely trace-to-trace.
     - **Class-B Gate (Conditional Hardware Shadows at Card Capacity $C_{\text{card}}$ — Independent NCU Operating-Point Diagnostic):**
       - Evaluate `sm_l1_locality_observer_v1` and `l2_system_locality_observer_v1` strictly at the **architecture-card capacity** $C_{\text{card}}$ (`sm_120-rtx-5090.json` / `sm_86.json`).
       - Report **envelope bracket coverage** (fraction of active-cache workloads where the parameter-free card scenario interval $[H_{\min}(C_{\text{card}}), H_{\max}(C_{\text{card}})]$ brackets or reduces distance to the realized NCU hit rate) as an **explanatory diagnostic**, never as a fitted parameter or scalar selection gate.

2. **Conflated Gate in M1.3 / M2A.4 / M3.3: Hard Single-Metric Thresholds vs. Non-Degradation Frontier**
   - **Why Risky:** Setting arbitrary pass/fail cutoffs (e.g., SM86 L2 write hit median APE $< 20\%$ on all kernels) ignores traces where NCU reports $0$ L2 write sectors or where compulsory cold writes dominate small launches.
   - **Trace-Derived, Card-Only Replacement Gate:**
     - Require **strict Pareto improvement or non-regression** on the mechanism-local family in [`tools/sm120_metric_report.py`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/tools/sm120_metric_report.py) and [`compare_numeric_full.py`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/compare_numeric_full.py):
       - **M1:** Zero degradation on `Active Cycles`, `Elapsed Cycles`, `Warp Instructions`, and L2 metrics; strictly reduced median APE and increased within-10% accuracy on SM120 `l1tex__t_sector_hit_rate.pct` (`87.64%` baseline) and `l1tex__t_sectors_lookup_hit.sum` (`65.07%` baseline).
       - **M2A + M2B:** Zero degradation on read-only traces; strictly reduced median APE on write-active SM120 traces (`store_accesses > 0`) for `l1tex__t_sector_hit_rate.pct`, `l1tex__t_sectors_lookup_hit.sum`, and `lts__t_sectors_srcunit_tex_op_read.sum`.
       - **M3:** Cold coalesced 4-sector store instructions produce `0` L2 write tag hits (`pre_packet_write_tag_hit_mask == 0`), warm repeat stores to resident lines produce `4` L2 write tag hits, and SM86 `lts__t_sectors_op_write_lookup_hit.sum` median APE decreases monotonically from the `100.0%` baseline without degrading SM120 L2 write accuracy (`5.71%` baseline).

---

## 4. Recommended Step-by-Step Execution Order

1. **Step 1 — Implement M1 (SM120 In-Flight Merge Decomposition):**
   - Edit [`sim/mem/src/l1.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs), [`sim/direct_run/src/emit.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/emit.rs), and [`sim/direct_run/src/lib.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/lib.rs).
   - Smoke-test on 1 representative SM120 trace and 1 SM86 trace to verify `pending_merges == already_requested_merges + unrequested_pending_merges` and `L1MutationBudget` bounds.
2. **Step 2 — Implement M2A + M2B (SM120 Per-SM Write Lookup Observer & `GlobalUntagged` Store Unification):**
   - Edit [`sim/arch/src/lib.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/arch/src/lib.rs), [`sim/arch/src/card.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/arch/src/card.rs), [`arch_profiles/cards/sm_120-rtx-5090.json`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/arch_profiles/cards/sm_120-rtx-5090.json), [`sim/sm/src/issue.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs) (including lines `2841`, `6870`, `9222`, `9329`, `14001`), [`sim/sm/src/system.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs), [`sim/direct_run/src/emit.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/emit.rs), and [`sim/direct_run/src/lib.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/lib.rs#L2264).
   - Keep M2C store-to-load forwarding in the shadow observer first so cycle timing is preserved.
3. **Step 3 — Implement M3 (Pre-Packet & Pre-Instruction L2 Write Tag-Hit Snapshot):**
   - Edit [`sim/mem/src/l2.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l2.rs#L2100-L2232), [`sim/sm/src/system.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs#L4359-L4377) (deduplicating same-instruction multi-sector store `MemRequest`s on the same 128B line), [`sim/direct_run/src/metrics.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/metrics.rs#L3829-L3833), and [`sim/direct_run/src/lib.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/lib.rs#L2315-L2325).
4. **Step 4 — Implement M4 (AET `reuse_interval_observer_v1` Upgrade + Tier-2/3 Class-B Conditional Shadows):**
   - Edit [`sim/sm/src/metrics.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/metrics.rs), [`sim/sm/src/system.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs), [`tools/validate_reuse_mrc.py`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/tools/validate_reuse_mrc.py), and [`tools/report_locality_mrc.py`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/tools/report_locality_mrc.py).
5. **Step 5 — Execute M5 Complete-Corpus 12-Worker `.jsonl.zst` Replay & Validation.**

---

## 5. Corrected Verification Commands & Expected Artifacts

> [!WARNING]
> Section 5.1 of [`CACHE_LOCALITY_EXECUTION_PLAN.md`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/CACHE_LOCALITY_EXECUTION_PLAN.md#L334-L393) contains invalid CLI flags (`cargo build` without `--manifest-path sim/Cargo.toml`, `--corpus-root` instead of `--capture-root`, `--output-dir` instead of `--output`, and non-existent flags on `validate_reuse_mrc.py` and `report_locality_mrc.py`). Use the exact verified commands below.

### 1. Build Release Binaries
```bash
cargo build --release --manifest-path sim/Cargo.toml --workspace
```
- **Expected Binary:** `sim/target/release/sim-run`

### 2. Full 339-Trace SM120 Replay (`workers=12`, Compressed `.jsonl.zst`)
```bash
python3 tools/sm120_corpus_replay.py \
  --capture-root /home/jakeshea/5090_branch_1016/complete_capture_corpus \
  --sim-run sim/target/release/sim-run \
  --arch-card arch_profiles/cards/sm_120-rtx-5090.json \
  --calibration arch_profiles/calibrations/sm_120-rtx-5090.Port.v1.json \
  --mode full \
  --cache-mode l1-l2 \
  --workers 12 \
  --timeout-sec 180 \
  --output runs/sm120_m1_m5_full_l1_l2.jsonl.zst
```
- **Expected Artifact:** `runs/sm120_m1_m5_full_l1_l2.jsonl.zst` (`339/339` `status == "ok"` records).

### 3. Full 79-Trace SM86 Validation Replay (`workers=12`, Compressed `.jsonl.zst`)
```bash
python3 tools/sm120_corpus_replay.py \
  --capture-root data/validation/traces \
  --sim-run sim/target/release/sim-run \
  --arch-card arch_profiles/cards/sm_86.json \
  --calibration arch_profiles/calibrations/sm_86.port.v1.json \
  --mode full \
  --cache-mode l1-l2 \
  --workers 12 \
  --timeout-sec 180 \
  --output runs/sm86_m1_m5_full_l1_l2.jsonl.zst
```
- **Expected Artifact:** `runs/sm86_m1_m5_full_l1_l2.jsonl.zst` (`79/79` `status == "ok"` records).

### 4. Class-A Reuse-Interval & Exact Fenwick MRC Invariant Validation (Zero NCU Input)
```bash
python3 tools/validate_reuse_mrc.py runs/sm120_m1_m5_full_l1_l2.jsonl.zst
python3 tools/validate_reuse_mrc.py runs/sm86_m1_m5_full_l1_l2.jsonl.zst
```
- **Expected Output:** `PASS: validated 339 full-replay records` and `PASS: validated 79 full-replay records` with `0` conservation, monotonicity, or cold+capacity identity failures, plus AET vs. Fenwick WAPE summary.

### 5. Trace-Only Locality & MRC Summary Reports (Zero NCU Input)
```bash
python3 tools/report_locality_mrc.py \
  runs/sm120_m1_m5_full_l1_l2.jsonl.zst \
  --output reports/sm120_m1_m5_locality_mrc.md

python3 tools/report_locality_mrc.py \
  runs/sm86_m1_m5_full_l1_l2.jsonl.zst \
  --output reports/sm86_m1_m5_locality_mrc.md
```
- **Expected Artifacts:** `reports/sm120_m1_m5_locality_mrc.md`, `reports/sm120_m1_m5_locality_mrc.json`, `reports/sm86_m1_m5_locality_mrc.md`, `reports/sm86_m1_m5_locality_mrc.json`.

### 6. Independent Post-Replay NCU Accuracy Evaluation (SM120 & SM86)
```bash
python3 tools/sm120_metric_report.py \
  --full runs/sm120_m1_m5_full_l1_l2.jsonl.zst \
  --fast runs/sm120_m1_m5_full_l1_l2.jsonl.zst \
  --profiles /home/jakeshea/5090_branch_1016/complete_capture_corpus/ncu_profiles \
  --output reports/sm120_m1_m5_ncu_report \
  --include-ordered-fallback

python3 compare_numeric_full.py \
  --replay runs/sm86_m1_m5_full_l1_l2.jsonl.zst \
  --ncu data/validation/ncu_metrics.json \
  --output reports/sm86_m1_m5_ncu_comparison.json
```
- **Expected Artifacts:**
  - `reports/sm120_m1_m5_ncu_report/summary.md` & `metric_accuracy.json` (conforming to the consolidated "currently unavailable" presentation rule in [`AGENTS.md`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/AGENTS.md)).
  - `reports/sm86_m1_m5_ncu_comparison.json` verifying SM86 L1/L2 read+write accuracy and zero timing regression.
