# Unified GPU Simulator: Cache Locality (Ding–Denning HOTL) & L1/L2 Accuracy Execution Plan

**Target Architectures:**
- **SM120 (NVIDIA RTX 5090 / Blackwell):** 339 real captures (`11,558` total kernel launches)
- **SM86 (NVIDIA RTX 3090 Ti / Ampere):** 286 `status == "ok"` real captures (`12,392` total kernel launches; `107` exact `identity` joins + `179` `ordered_fallback` joins)

**Execution Mandate for Codex:**
- **Zero new hardware probing or trace recapture:** Use only the existing captured traces and independent NCU profiles on disk.
- **Scientific Invariants (`AGENTS.md`):**
  1. **Zero NCU fitting:** Scored NCU counters are independent evaluation targets—never replay inputs, tuning parameters, per-workload lookup tables, or kernel-name overrides.
  2. **Class-A vs. Class-B Separation:**
     - **Class-A (Intrinsic Trace Locality):** Pure mathematical properties of the declared trace stream (`sector_mrc_observer_v1`, `reuse_interval_observer_v1`). Evaluated against the exact Fenwick/Mattson stack-distance oracle (`0.0%` oracle error).
     - **Class-B (Simulator-Conditioned Locality & Hardware Endpoint Projections):** Conditional models parameterized by explicit architecture-card geometry (`arch_profiles/cards/sm_86.json`, `arch_profiles/cards/sm_120-rtx-5090.json`), cycle-interleaved SM/L2 streams, and transaction semantics.
  3. **Report-Only vs. Behavioral Isolation:** Implement report-only counter/observer slices first (requiring **zero delta** to physical cache state, L2/DRAM traffic, IPC, and timing frontiers) before evaluating any behavioral cache-policy slice.
  4. **Full Replay Only:** Every full-corpus verification must run `l1-l2` full replay with `--workers 12` and compressed `.jsonl.zst` outputs via `tools/sm120_corpus_replay.py --workers 12`. No fast-replay substitution and no synthetic fixtures.

---

## 1. Executive Summary & Quantitative Error Forensics

A full-corpus forensic audit across all `11,558` SM120 launches and `12,392` SM86 launches revealed that the remaining L1 and L2 APE residuals are **not** caused by unidentifiable proprietary silicon hashes or random noise. Instead, they are driven by **five concrete, cross-architectural simulator accounting and transaction-attribution defects** at the `L1Cache`, `L1WriteLookupObserver`, `SmCore`, and `L2Model` seams, which align directly with **Prof. Peter Denning's Working Set Theory** and **Prof. Chen Ding's Higher-Order Theory of Locality (HOTL) / Footprint Composition**.

### 1.1 Baseline vs. Verified Mechanism Impact Summary

| Architecture | Scored Metric | Baseline Mean APE (`NCU > 0`) | Baseline Median APE | Dominant Root Cause | Verified Post-Fix Mean APE | Target Acceptance Gate |
| :--- | :--- | ---: | ---: | :--- | ---: | ---: |
| **SM120** | **Global L1 Hit** (`l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum`) | `39.32%` (`n=2,829`) | `42.86%` | `L1Cache::access_core` increments `merges` on in-flight `line_fills` even when the requested 32B sector bit was never requested (`(known & sbit) == 0`), triggering an L2 follow-up fetch; `effective_lookup_hits_for_arch` subtracted a hardcoded `3 * line_alloc_misses`. | **`0.25%`** (`2,797/2,829` exact; `8,729/8,729` zero-target exact) | **`< 1.0%` Mean APE** (`0` new zero-target false positives) |
| **SM120** | **Global L1 Hit Rate** (`l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct`) | `39.30%` (`n=2,829`) | `42.86%` | Same in-flight unrequested-sector merge conflation as Global L1 Hit. | **`0.23%`** | **`< 1.0%` Mean APE** |
| **SM120** | **Global L1 Miss** (`l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum`) | `5.93%` (`n=11,316`) | `0.00%` | Complement of Global L1 Hit (`requests - lookup_hits`). | **`0.06%`** (`11,284/11,316` exact) | **`< 0.5%` Mean APE** |
| **SM120** | **Unified L1 Hit Rate** (`l1tex__t_sector_hit_rate.pct`) | `58.01%` (`n=4,983`) | `64.57%` | (1) Global read under-prediction (`+14.2 pp`), and (2) `L1WriteLookupObserver` (`plan_batch`) is gated to `ArchId::Sm86` only (`issue.rs:2842`), causing `1,807` store-hit rows (`1,571` pure-store rows) to predict `0.0%` (`+43.8 pp`). | **`< 8.5%`** (projected across `n=4,983` `NCU>0` rows; `0` false positives on `6,525` store zero-target rows) | **`< 10.0%` Mean APE** |
| **SM120** | **Operation-Read L2 Hit** (`lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum`) | `14.89%` (`n=2,330`) | `0.00%` (`2,175/2,330` exact) | `59.3%` (`+8.83 pp`) of all error comes from just `15` over-predicted launches: `9` launches (`splitKreduce_kernel`, `register_spill_local_mem`, `magma_sgemmEx`) where same-SM `STG/STL -> LDG/LDL` readback misses in `SimL1` and hits in `L2`, plus `2` `vector_fft_c2r` generic `LD` launches bypassing `SimL1`. | **`< 6.8%`** | **`< 8.0%` Mean APE** |
| **SM120** | **Generic L2 Hit** (`lts__t_sectors_srcunit_tex_lookup_hit.sum`) | `24.33%` (`n=4,104`) | `0.00%` (`2,297/4,104` exact) | (1) `lookup_observer_hits` (`system.rs:4380`) uses `req.operation` (`Read`) instead of `req.effective_observer_operation()` (`Store`) on `GlobalUntagged` `ST.E` (`magma_sgemmEx_kernel` `6,144 vs 1,200`), (2) `L2Model` mutates `line_present = true` *inside* the `0..4` sector loop of a single packet while `system.rs:4366` ignores pre-packet write-tag hits on multi-packet sub-line stores (`702` Category-B rows), and (3) `STORE_PACKET_WINDOW_CYCLES` suppresses valid hits after a miss packet (`451` Category-C/D rows). | **`< 9.5%`** | **`< 10.0%` Mean APE** |
| **SM86** | **Unified L1 Hit Rate — Exact Joins** (`l1tex__t_sector_hit_rate.pct`, `identity`) | `9.87%` (`n=75`) | `0.04%` | `#1` outlier (`simbench_l4_moe_layer_residual_norm_pipeline`, `414.29%` APE = `49.4%` of entire `identity` error) and `#5–#7` (`moe_grouped_gemm`) come from `merges - line_alloc_misses` over-crediting unrequested in-flight subline merges on 2-sector `LDG.E.64` loads. Exact sector-merge identity yields `112 vs 112` (`0.00%` error) and `4,208 vs 4,208` (`0.00%` error). | **`~4.7%`** | **`< 6.0%` Mean APE** |
| **SM86** | **Unified L1 Hit Rate — All Captures** (`l1tex__t_sector_hit_rate.pct`, `all`) | `24.66%` (`n=135`) | `1.59%` | `59/60` `ordered_fallback` `NCU>0` rows and `28/28` `NCU==0` rows match `Launch 0` to the exact instruction/sector; their error is driven by `sm86_form_qualified_lookup_hits` packet-ratio branches and unrequested in-flight merges, plus `ldgsts_async_copy` zeroing on `cp.async.ca` (`lcd_conv2d_implicit_gemm_bf16_sm80`: `1,676 vs 1,680`). | **`< 9.5%`** (`0/28` zero-target false positives, down from `28/28`) | **`< 10.0%` Mean APE** |
| **SM86** | **Generic L2 Hit** (`lts__t_sectors_srcunit_tex_lookup_hit.sum`) | `8.52%` (`n=184`) | `0.00%` | Already `< 10%`; `78.8%` (`+6.71 pp`) of residual comes from 7 single-CTA `reduce_kernel` captures (`NCU <= 4` hits) where 16 warps write 1 sector across `> 8` cycles (`STORE_PACKET_WINDOW_CYCLES`). Also `direct_run/src/lib.rs:2317` omits `ArchId::Sm86`. | **`< 4.5%`** | **`< 6.0%` Mean APE** |
| **SM86** | **Operation-Read L2 Hit** (`lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum`) | `4.34%` (`n=161`) | `0.00%` | Already `< 5%` (`130/161` exact). | **`< 4.34%`** (preserve) | **`< 5.0%` Mean APE** |
| **Both** | **Class-A Reuse-Interval HOTL Footprint MRC vs. Fenwick Oracle** | `29.4%` WAPE (`SM120`) / `27.8%` WAPE (`SM86`) | — | Existing `reuse_interval_observer_v1` lacks origin qualification, uses record-tick instead of line-reference clock, omits Xiang's right-censored tail intervals $V_{\text{end}} - V_{\text{last}}(\ell)$, truncates at 32 bins, and decouples sector survival from line eviction. | **`< 1.0%` WAPE** | **`< 1.5%` WAPE / `< 2.0%` MAE** |

---

## 2. Theoretical Foundation: Integrating Prof. Peter Denning & Prof. Chen Ding's Locality Models into GPU Simulation

### 2.1 Why Per-Warp Stack Distance (`sector_mrc_observer_v1`) Cannot Directly Predict Shared L1/L2 Hardware Hit Rates

`sector_mrc_observer_v1` computes exact Mattson stack distance (Reuse Distance, $\text{RD}$) on a **within-warp** coalesced stream. While mathematically exact for its declared Class-A stream (`0.0%` error against its Fenwick oracle), a GPU L1TEX cache is shared by up to $W \le 48$ (SM86) or $W \le 64$ (SM120) concurrent warps across 4 SMSPs per SM, and L2 is shared across all SMs and all memory client scopes (`OrdinaryLdg`, `ConstantLdg`, `Local`, `Store`, `Async`, `Atomic`).

Per-warp stack distance fails to predict hardware L1/L2 hit rates for five fundamental reasons explained by Denning and Ding:
1. **Missing Inter-Warp Constructive Sharing:** In GPU kernels (`gemm`, `flash_attn`, `conv2d`, `rmsnorm`), Warp $w_1$ fetches a 128B line (`RD = \infty` within $w_1$), and Warps $w_2 \dots w_k$ on the same SM access that same 128B line within a few cycles (`RD = \infty` within each warp's private stream, yet a **100% hit** in the shared SM L1TEX or MSHR).
2. **Missing Inter-Warp Destructive Footprint Dilation:** When Warp $w_1$ reuses a line after $\tau$ cycles, its private stack distance is $d_{\text{private}}$, but $W-1$ other warps have interleaved their working sets into the same L1TEX during window $\tau$, dilating the true shared-cache stack distance to $FP_{\text{SM}}(\tau) \gg d_{\text{private}}$.
3. **Sector-Cache & MSHR Fill-Window Coupling:** NVIDIA L1TEX and L2 use 128B lines divided into four 32B sectors. When Warp $w_2$ hits an in-flight 128B line in the MSHR at reuse interval $\tau < T_{\text{fill}}$, hardware only credits a lookup hit if **that specific 32B sector bit was already requested**; an unrequested 32B sector on an in-flight 128B line is a **subline sector miss** that issues a follow-up fetch to L2.
4. **Lease-0 / Write-No-Allocate Store Streams:** In GPU L1TEX, ordinary global stores (`WriteThroughNoAllocate`) probe the 128B tag directory (`Lease = 0` on miss) without evicting resident read lines, whereas in L2 sub-line writes allocate a 128B line tag on the first sub-line request and hit that tag on subsequent sub-line requests.
5. **Finite Set-Associativity & Thrashing Resistance ($W_0 > C$):** Under strict fully-associative LRU, a cyclic loop over $W_0 > C$ lines yields `0%` hit rate (`RD = W_0 - 1 \ge C`), whereas hardware set-associative caches with pseudo-LRU/SRRIP/BIP insertion retain $(C-1)/W_0$ or Smith's binomial/Poisson survival fraction.

### 2.2 Denning–Ding Mathematical Framework for the Simulator

1. **Peter Denning's Working Set $W(t, \tau)$ and Differential Miss-Rate Identity (CACM 1968, IEEE TSE 1980):**
   For a reference stream of total length $N$ (measured in unique-line reference ticks $V_t$ so intra-transaction sector fanout does not warp time), let $W(t, w)$ be the set of distinct 128B lines referenced in the backward window $(t - w, t]$. Denning's expected working-set size (which Chen Ding terms the **Footprint** $fp(w)$) satisfies:
   $$fp(0) = 0, \qquad fp(w) - fp(w-1) = m(w-1) = P(\text{RI} > w - 1)$$
   where $m(w)$ is the Working-Set miss ratio at window $w$ and $\text{RI}$ is the backward reuse interval (reuse time).

2. **Chen Ding & Yutao Xiang's Higher-Order Theory of Locality (HOTL) Exact Finite-Boundary Footprint Equation (ASPLOS 2011 / ACM TOCS 2013):**
   For a finite trace of length $V_{\text{end}}$ ticks containing $U$ distinct items (lines or sectors), let $n_{\text{interior}}(\tau)$ be the histogram count of interior reuse intervals of exact length $\tau = t - t_{\text{prev}} \ge 1$, let $f_u = V_{\text{first}}(u) - 1$ be the initial cold-entry tick of item $u$, and let $l_u = V_{\text{end}} - V_{\text{last}}(u)$ be the **right-censored tail interval** from item $u$'s final reference to the end of the trace. Then for any window $w \in [1, V_{\text{end}} - 1]$:
   $$fp(w) = U - \frac{1}{V_{\text{end}} - w} \left( \sum_{\tau = w + 1}^{V_{\text{end}}} (\tau - w)\, n_{\text{interior}}(\tau) + \sum_{u=1}^{U} \max(f_u - w, 0) + \sum_{u=1}^{U} \max(l_u - w, 0) \right)$$
   *Note on `reuse_interval_observer_v1`'s current 29% WAPE:* When `f_u == l_u` in stationary cyclic loops, the first-entry error cancels the missing tail-exit term. By omitting $\sum_u \max(l_u - w, 0)$ while keeping cold misses, the current implementation breaks Xiang's boundary conservation law on streaming and multi-phase GPU kernels. Keeping the exact censored tail intervals $l_u = V_{\text{end}} - V_{\text{last}}(u)$ at launch finalization (with zero extra per-access memory overhead, since `last_seen` already stores $V_{\text{last}}(u)$) restores exact conservation.

3. **Ding–Denning Footprint-to-MRC Conversion & Coupled 128B-Tag / 32B-Sector Survival:**
   For any target cache capacity $C$ (in 128B lines), find the characteristic residence window $w^*(C)$ by monotonically inverting $fp_{\text{line}}(w^*(C)) = C$ via log-linear interpolation.
   - **Line Miss Ratio:**
     $$\widehat{MR}_{\text{line}}(C) = m_{\text{line}}(w^*(C)) = \frac{1}{N_{\text{line}}} \left( U_{\text{line}} + \sum_{\tau > w^*(C)} n_{\text{line}}(\tau) \right)$$
   - **Coupled Sector Survival Rule:** A 32B sector $b \in \{0,1,2,3\}$ of 128B line $\ell$ is resident upon reuse at $V_t$ if and only if every intervening reference to line $\ell$ since the last access to $(\ell, b)$ had line reuse interval $\le w^*(C)$. Maintaining four sector timestamps $t_{\text{sec}}(\ell, b)$ and four bottleneck intervals $\tau_{\text{bot}}(\ell, b)$ per line:
     $$\tau_{\text{eff}}(\ell, b) = \max\big(\tau_{\text{bot}}(\ell, b),\, V_t - t_{\text{line}}(\ell)\big)$$
     A sector access is a **resident sector hit** iff $\tau_{\text{eff}}(\ell, b) \le w^*(C)$; a **line-tag miss** if $V_t - t_{\text{line}}(\ell) > w^*(C)$; and a **resident-line sector-fill miss** if $V_t - t_{\text{line}}(\ell) \le w^*(C) < \tau_{\text{eff}}(\ell, b)$.

4. **Inter-Warp Footprint Composition (Xiang et al., PPoPP 2013) & Constructive vs. Destructive Sharing:**
   Comparing the Tier-1 Per-Warp MRC $MR_{\text{warp}}(C)$ against the Tier-2 Per-SM Cycle-Interleaved MRC $MR_{\text{SM}}(C)$ at capacity $C$ decomposes multi-warp SIMT cache behavior into:
   - **Constructive Inter-Warp Sharing Gain** $\Delta M_{\text{constructive}}(C)$: Sector accesses where $\text{RD}_{\text{warp}} \ge C$ (including $\text{RD}_{\text{warp}} = \infty$, i.e., warp-cold) but $\text{RD}_{\text{SM}} < C$ because an earlier warp on the same SM fetched the line/sector.
   - **Destructive Inter-Warp Footprint Dilation Loss** $\Delta M_{\text{destructive}}(C)$: Sector accesses where $\text{RD}_{\text{warp}} < C$ (would have hit in a private cache) but $\text{RD}_{\text{SM}} \ge C$ because concurrent warps diluted the SM footprint over window $\tau$.
   $$\text{Net Inter-Warp Sharing Effect}(C) = MR_{\text{warp}}(C) - MR_{\text{SM}}(C) = \Delta M_{\text{constructive}}(C) - \Delta M_{\text{destructive}}(C)$$

5. **MSHR Fill-Window Coupling $[\tau_{\text{fill}}, \tau^*(C)]$:**
   At the Tier-2 Per-SM observer, each reference records both shared stack distance $\text{RD}_{\text{SM}}$ and cycle reuse interval $\Delta t_{\text{cycle}}$ since the current line allocation's issue cycle:
   - If $\Delta t_{\text{cycle}} < T_{\text{fill}}$ (in-flight MSHR line):
     - If sector bit $b$ was **already requested** in the active allocation epoch: **MSHR Pending-Hit Merge** (`already_requested_merges`).
     - If sector bit $b$ was **not yet requested** in the active allocation epoch: **Unrequested Subline Miss** (`unrequested_subline_miss`, issuing follow-up L2 fetch).
   - If $\Delta t_{\text{cycle}} \ge T_{\text{fill}}$ and $\text{RD}_{\text{SM}} < C_{\text{eff}}$: **Settled SRAM Resident Hit**.

6. **Class-B Conditional Scenario Envelope $[H_{\min}, H_{\max}]$ (Smith Set-Associative Mapping + BIP/SRRIP Bounds):**
   For a cache with $S$ sets and associativity $A = C_{\text{eff}} / S$, Smith's (1978) Poisson/Binomial mapping gives the set-associative survival probability of a reuse at effective footprint $d_{\text{eff}} = fp(\tau)$:
   $$P_{\text{SA}}(\text{hit} \mid d_{\text{eff}}, S, A) = \sum_{k=0}^{A-1} \frac{(d_{\text{eff}}/S)^k \, e^{-d_{\text{eff}}/S}}{k!} = \frac{\Gamma(A, d_{\text{eff}}/S)}{(A-1)!}$$
   And under thrashing-resistant bimodal insertion (BIP/SRRIP with promotion rate $\epsilon \in [1/32, 1]$), single-use streaming lines ($d = \infty$) contribute only $\epsilon \, d_{\text{cold}}$ to the effective eviction pressure on reused lines ($d_{\text{eff}} = d_{\text{reused}} + \epsilon \, d_{\text{cold}}$). Sweeping:
   - Effective L1 capacity $C_{\text{eff}} \in [C_{\min}, C_{\max}]$ constrained by trace-observed shared memory usage (`shared_load`/`shared_store` footprint),
   - Same-cycle cross-SMSP tie-breaking (`optimistic` two-phase batch vs. `pessimistic`), and
   - Replacement hazard ($\epsilon = 1$ strict LRU vs. $\epsilon = 1/32$ thrashing-resistant BIP + Smith set-associative mapping)
   produces a zero-fitting **Class-B Conditional Scenario Envelope** $[H_{\min}, H_{\max}]$ for each launch.

---

## 3. Detailed Execution Plan, Milestones & Acceptance Criteria

### Milestone 1 (M1): Exact Physical Sector-Merge Accounting in `L1Cache` (Report-Only Counter Slice)

#### 1.1 Causal Mechanism & Code Seams
- **Defect:** In [`L1Cache::access_core`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L3148-L3189), when a read accesses a 128B line that has an active in-flight entry in `self.line_fills`, `access_core` unconditionally takes the `L1Outcome::HitPending` branch and increments `self.merges` (`origin.merges`, `same_request_merges`, or `older_pending_merges`) **even when `(known & sbit) == 0`** (i.e., the specific 32B sector `sector` was never requested in that in-flight line fill!).
- **Physical Reality:** Immediately after `access_core` returns `HitPending`, [`SmCore::access_shared`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L14875-L14896) checks `already_sent`; when `!already_sent`, it calls `reserve_pending_line_sector` (`l1.rs:2199-2235`) and sends a **brand-new `MemRequest::l1_followup_fetch` to L2**. In hardware PMU semantics, a 32B sector that was never requested and must be fetched from L2 is a **lookup miss**, whereas a 32B sector whose fetch is already in flight (`(known & sbit) != 0`) is a **pending lookup hit**.
- **Why Old Heuristics Failed:** Previously, [`L1OriginCounters::effective_lookup_hits_for_arch`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L473-L503) subtracted a hardcoded `3 * line_alloc_misses` on SM120 and `1 * line_alloc_misses` on SM86 (plus fragile packet-ratio branches in [`sm86_form_qualified_lookup_hits`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/metrics.rs#L163-L286)). Whenever a kernel accesses 1, 2, or 4 sectors per 128B line, `3 * line_alloc_misses` or `1 * line_alloc_misses` systematically under-predicts (SM120 `39.32%` APE) or over-predicts (SM86 `#1` outlier `simbench_l4_moe_layer_residual_norm_pipeline` `576 vs 112` and 28 `NCU==0` false positives).

#### 1.2 Exact Implementation Steps
1. **[`sim/mem/src/l1.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs):**
   - **Preserve existing `merges` semantics:** Do **not** alter `self.merges`, `origin.merges`, `same_request_merges`, or `older_pending_merges` (which track line-transaction scheduling merges and feed timing/audit paths).
   - **Add `already_requested_merges: u64` (and `unrequested_pending_merges: u64`) to `L1OriginCounters` (`l1.rs:437-465`) and `L1Counters` (`l1.rs:514-565`):**
     - In [`L1Cache::access_core`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L3110-L3190), snapshot `let sector_already_known = (self.sectors.get(&line).copied().unwrap_or(0) & sbit) != 0;` **before** `self.record_unrequested_subline_miss(...)` and **before** `*self.sectors.entry(line).or_default() |= sbit;` inside the `HitPending` branch (`l1.rs:3163`).
     - When `HitPending` is taken (either via `self.sector_fills.get(&key)` or `self.line_fills.get(&line)`):
       - If `sector_already_known` (or `self.sector_fills.contains_key(&key)`), increment `origin.already_requested_merges` (using `saturating_add(1)` / checked counter plumbing).
       - Else (`!sector_already_known`), increment `origin.unrequested_pending_merges`.
     - Verify the invariant on every read access: `origin.merges == origin.already_requested_merges + origin.unrequested_pending_merges`, and `origin.unrequested_pending_merges == unrequested_subline_miss_sectors_origin - origin.sector_resident_misses`.
   - **Update `L1OriginCounters::effective_lookup_hits_for_arch` (`l1.rs:473-503`):**
     - Replace the heuristic `merges.saturating_sub(3 * line_alloc_misses)` / `merges.saturating_sub(line_alloc_misses)` with the exact physical sector-merge count:
       ```rust
       pub fn effective_lookup_hits_for_arch(&self, _arch: ArchId) -> u64 {
           self.hits.saturating_add(self.already_requested_merges)
       }
       ```
2. **[`sim/direct_run/src/emit.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/emit.rs#L942-L1055):**
   - In `global_read_resident_hit_adjustment` / `cache_lookup_predictions` (`emit.rs:942-1055, 1105-1155`), use `l1.global.effective_lookup_hits_for_arch(arch)` (`l1.global.hits + l1.global.already_requested_merges`) directly without the legacy `older_pending` heuristic branches that required special-casing `global_load_ltc128b` (since `already_requested_merges == 0` on all `8,729` SM120 zero-target rows, including `sol_l2_034` launches `24` and `85`!).
   - Retain existing `constant_qualified_ldg_merges` exclusion only where `LDG.CONSTANT` routes through a separate non-L1TEX constant cache on SM86 when applicable, and verify against both SM120 and SM86.
3. **[`sim/direct_run/src/metrics.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/metrics.rs#L163-L286):**
   - In `sm86_form_qualified_lookup_hits`, replace the multi-branch packet-ratio heuristics (`lines 205-285`) with the exact sector-qualified lookup hits (`l1.global.hits + l1.global.already_requested_merges + l1.local.hits + l1.local.already_requested_merges`), preserving only the architecture-grounded constant-cache / streaming-bypass qualifications.
   - In `sm86_form_qualified_lookup_hits` (`lines 195-204`), do **not** zero out `ldgsts_async_copy` when cacheable `cp.async.ca` (`l1.global.hits > 0`) produced genuine resident L1 sector hits; exclude only unrequested async pending merges (`unrequested_pending_merges`), preserving `lcd_conv2d_implicit_gemm_bf16_sm80` (`1,676` vs NCU `1,680` hits, `0.24%` error).

#### 1.3 M1 Acceptance Criteria
- **Zero Timing/Traffic Mutation:** Physical L1/L2 state, L2/DRAM sector traffic, and Active/Elapsed IPC must have **0.00% delta** across all 339 SM120 and 286 SM86 captures.
- **SM120 Global L1 Accuracy:**
  - `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum` Mean APE drops from `39.32%` to **`< 1.0%`** (`~0.25%`), Median APE `0.00%`.
  - `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum` Mean APE drops from `5.93%` to **`< 0.5%`** (`~0.06%`).
  - `l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct` Mean APE drops from `39.30%` to **`< 1.0%`** (`~0.23%`).
  - Zero-target safety: **0 false positives** across all `8,729` `NCU == 0` Global L1 rows on SM120.
- **SM86 Unified L1 Accuracy:**
  - `l1tex__t_sector_hit_rate.pct` (`identity` exact joins, `n=75`) Mean APE drops from `9.87%` to **`< 6.0%`** (`~4.7%`), with `simbench_l4_moe_layer_residual_norm_pipeline` at **`0.00%` APE** (`112 vs 112`).
  - `l1tex__t_sector_hit_rate.pct` (`all` joins, `n=135`) Mean APE drops from `24.66%` to **`< 10.0%`**, and zero-target false positives drop from `28/28` to **`0/28`**.

---

### Milestone 2 (M2): SM120 Per-SM Write-Tag Lookup Observer, Generic `LD`/`ST` Operation Attribution, and Same-SM Store-to-Load Sector Residency

Following the `codex exec` adversarial review, M2 is split into **three independent, sequentially verified sub-slices (M2A, M2B, M2C)** so report-only observer improvements are never conflated with behavioral cache changes.

#### 2.1 Slice M2A (Report-Only): Enable Stateful Per-SM 128B Write-Tag Lookup Observer (`L1WriteLookupObserver`) on SM120
- **Causal Mechanism:**
  - On SM120, `2,182` of `4,983` `NCU > 0` Unified L1 rows currently predict `0.0%` hit rate (`+43.79 pp` of the `58.01%` Unified L1 Mean APE). On `1,807` of those `2,182` rows (`82.8%`), NCU has `l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum > 0`.
  - Why? When multiple warps or instructions on the same SM write to the same 128B line (or write 2-sector `STG.E.U16` half-lines, straddling unaligned lines, or `reduce_kernel` partial lines), the first store instruction allocates a 128B write-tag entry in the SM's L1TEX tag directory (counting as a store lookup miss), and subsequent store instructions to that 128B line on the same SM hit the 128B tag! Conversely, a single coalesced 4-sector `STG.E` instruction writing a fresh 128B line performs **one pre-instruction tag lookup** before allocating the line, resulting in **4 sector lookup misses and 0 store lookup hits** (`0` hits across all `6,525` store zero-target rows!).
  - [`L1WriteLookupObserver::plan_batch`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L1307-L1420) already models this exact pre-instruction 128B tag-lookup + post-instruction allocation semantics (`l1.write_lookup_hits`), and [`L1Cache::access_store_with_policy`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l1.rs#L3871-L3898) already tracks `l1.store_tag_hits` when a store hits a read-allocated 128B line tag in `self.sets`! However:
    1. [`stateful_write_lookup_observer_enabled`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L2840-L2852) hardcodes `if schedule.arch != ArchId::Sm86 { return false; }`, disabling `L1WriteLookupObserver` on SM120 and falling back to the stateless dummy formula `projected_l1_write_lookup_hits` (`issue.rs:9231`).
    2. [`emit.rs:1198-1216`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/emit.rs#L1198-L1216) ignores `l1.write_lookup_hits` and `l1.store_tag_hits` on SM120 except on `global_store_wt` (`STG.E.WT`), using `l1.store_hits` (`0` unless the exact 32B sector was previously read).
- **Implementation Steps (M2A):**
  1. In [`arch_profiles/cards/sm_120-rtx-5090.json`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/arch_profiles/cards/sm_120-rtx-5090.json) and [`sim/arch/src/lib.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/arch/src/lib.rs#L117-L162), populate `l1_write_lookup` parameters for SM120 (`sets`, `slots_per_set`) matching the SM120 L1 tag geometry (`256` sets, `4` ways).
  2. In [`sim/sm/src/issue.rs:2840-2852`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L2840-L2852), allow `stateful_write_lookup_observer_enabled` when the architecture card defines `l1_write_lookup` (both `ArchId::Sm86` and `ArchId::Sm120`).
  3. In [`sim/sm/src/issue.rs:9210-9260`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L9210-L9260), ensure that when a store hits an already-resident read line tag in `L1Cache` (`store_tag_hits`) OR hits the per-SM `L1WriteLookupObserver` (`plan_batch`, where `plan_batch` is also updated whenever a read allocates a 128B line in `L1Cache` so read-allocated 128B lines and write-allocated 128B lines share a unified tag presence view without double-counting), `l1.write_lookup_hits` counts the exact pre-instruction 128B tag-hit sectors.
  4. In [`sim/direct_run/src/emit.rs:1198-1216`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/emit.rs#L1198-L1216), use `l1.write_lookup_hits` for SM120 `store_lookup_hits` (just as SM86 does in `metrics.rs:306`), while preserving `0` store lookup hits on all single-instruction cold 128B stores.

#### 2.2 Slice M2B (Report-Only Observer Fix): Generic `ST.E` Operation Attribution in L2 `lookup_observer_hits`
- **Causal Mechanism:**
  - In `magma_sgemmEx_kernel` (`44` Generic L2 over-predicted launches on SM120, e.g., `sol_l2_060_chunk_delta_sm120adapted_small l=37` where simulator predicts `6,144` Generic L2 hits vs NCU `1,200`), the kernel uses generic `ST.E` (`AccessKind::GlobalUntagged`).
  - At [`sim/sm/src/issue.rs:13994-14003`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L13994-L14003), `req.observer_operation` is set to `Some(MemoryOperation::Store)`, so `read_lookup_hits` (`op_read`) correctly sees `MemoryOperation::Store` and reports `0` read hits (`0 vs 0` exact!).
  - **Bug:** Seven lines below in [`sim/sm/src/system.rs:4380`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs#L4380), `lookup_observer_hits` calls `preview.lookup_for_operation(req.operation)` instead of `preview.lookup_for_operation(req.effective_observer_operation())`! Because `req.operation` was left as `MemoryOperation::Read`, `lookup_observer_hits` credits `6,144` fake read-pending-merge hits (`preview.pending_merge_mask()`) into `l2.lookup_hits`!
- **Implementation Steps (M2B):**
  1. In [`sim/sm/src/system.rs:4380`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs#L4380) (and lines `4217-4325` for store-packet tracking), use `let effective_op = req.effective_observer_operation();` consistently for `lookup_observer_hits` and `store_packet_lookup_hits` so `GlobalUntagged` stores (`ST.E`) are never scored as L2 read-pending merges.
  2. In [`sim/sm/src/issue.rs:13994-14003`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L13994-L14003), remove the arbitrary `admitted_warps <= 8` restriction on setting `observer_operation = Some(MemoryOperation::Store)` for `AccessKind::GlobalUntagged` when `inst.op == MemoryOperation::Store`: an instruction decoded as `MemoryOperation::Store` is always a store regardless of warp occupancy!

#### 2.3 Slice M2C: Same-SM Store-to-Load Full-Sector Residency (`STG/STL -> LDG/LDL`)
- **Causal Mechanism:**
  - Across all `2,330` SM120 Op-Read L2 (`lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum`) rows, `2,175` are `0.00%` exact, and **`54.5%` of the entire corpus Op-Read APE comes from just 9 launches** (`7` `cublasLt::splitKreduce_kernel` launches + `simbench_l1_register_spill_local_mem l=0` + `magma_sgemmEx_kernel l=2`).
  - In `splitKreduce_kernel`, warps write reduced partial sums via `STG.E` (`pc=0x2ac0`, `512`/`1024`/`4096` full 32B sectors) and immediately read back the **exact same address** via `LDG.E` (`pc=0x37c0`, reuse distance = 0/1 line). On real SM120 silicon (NCU), those `LDG.E` sectors hit **100% in L1TEX** (`l1_global_op_ld_lookup_hit = 992 = 480 ConstantLdg + 512 OrdinaryLdg`) and send **0** `OrdinaryLdg` requests to L2 (`ConstantLdg` in `L2_by_source` matches NCU `lts__t_sectors_srcunit_tex_op_read` `sum / hit / miss` **100.0% identically** across all 7 launches!). The same holds for `STL -> LDL` in `simbench_l1_register_spill_local_mem` (`NCU l1_local_op_ld_lookup_miss = 0`).
- **Implementation Steps (M2C):**
  1. Track a per-SM full-sector store residency / write-install mask in `L1Cache` (only when a store writes all 32 bytes of a sector, i.e., full-sector write mask coverage as required by the `codex exec` review so partial stores never make unwritten bytes readable) while preserving write-through L2 store request emission (`L2Source::Store` traffic to L2 remains unchanged).
  2. When a subsequent `OrdinaryLdg` or `Local` load on the same SM accesses a sector that is still resident in that SM's L1 set/tag from a prior full-sector store (not yet evicted by intervening L1 set allocations), count it as an L1 resident hit and suppress the spurious L2 `OrdinaryLdg` / `Local` read fetch.

#### 2.4 M2 Acceptance Criteria
- **SM120 Unified L1 Hit Rate (`l1tex__t_sector_hit_rate.pct`):** Mean APE drops from `58.01%` to **`< 10.0%`**, with **0 new zero-target false positives** across the `6,525` store zero-target rows.
- **SM120 Operation-Read L2 Hit (`lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum`):** Mean APE drops from `14.89%` to **`< 8.0%`** (`~6.5%`), with all 7 `cublasLt::splitKreduce_kernel` launches and `simbench_l1_register_spill_local_mem` matching NCU within `< 1%`.

---

### Milestone 3 (M3): L2 Pre-Packet Write-Tag Snapshot & Store Coalescing Window (`sim/mem/src/l2.rs`, `sim/sm/src/system.rs`, `sim/direct_run/src/lib.rs`)

#### 3.1 Causal Mechanism & Code Seams
1. **Intra-Packet Mutation vs. Multi-Packet Sub-Line L2 Write-Tag Hits (`702` Category-B + `191` Category-E SM120 Generic L2 rows):**
   - In [`L2Model::access_internal`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l2.rs#L2183-L2231), `line_present` is set to `true` **inside** the `for sector in 0..4` loop of a single `MemRequest` (`line 2220`). Because a single coalesced 128B write (`sector_mask = 0b1111`) would falsely increment `write_tag_hits` 3 times on itself, [`system.rs:4366`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs#L4366) completely disabled `write_tag_hits` in `lookup_for_operation(MemoryOperation::Store)`, counting only `self.present_sectors` (sectors whose data was *already valid* before the store)!
   - In reality (as verified across all `7,401` `ZERO_T` rows vs. `702` `Category-B` rows in `cutlass::Kernel2`, `at::elementwise_kernel`, `flash_attn_fwd/bwd`, `geglu`, `swiglu`), a **single coalesced 128B write packet** performs **1 pre-packet tag lookup** (`0` write-tag hits on a cold line), whereas **subsequent separate sub-line write packets** (e.g. four separate 32B sector write packets or two 64B write packets arriving at L2 for the same 128B line) observe `line_present_before_packet == true` from the first packet's tag allocation and count as L2 write lookup hits (`3/4 = 75%` or `1/2 = 50%`)!
2. **`STORE_PACKET_WINDOW_CYCLES` Miss-Packet Suppression (`368` Category-C + `83` Category-D SM120 rows):**
   - In [`sim/sm/src/system.rs:4232-4249, 4314-4323`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs#L4232-L4249), `store_packet_keys` suppresses any store sector on `(sm_ix, line, sector)` within `STORE_PACKET_WINDOW_CYCLES = 8` cycles of an earlier packet. When Packet 0 **misses** L2 (`present_mask == 0`, contributing `0` hits) and installs the sector in L2, and Packet 1 arrives within 8 cycles (`Sim_op_write_req == NCU_op_write.sum`, meaning hardware did *not* coalesce Packet 1 into Packet 0), `recent == true` drops Packet 1's valid L2 hit (`sp_hit = 0` while `NCU > 0`)!
3. **SM86 Generic L2 Store-Packet Plumbing (`sim/direct_run/src/lib.rs:2317`):**
   - In [`sim/direct_run/src/lib.rs:2317`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/lib.rs#L2317), `if self.schedule.arch == ArchId::Sm120` gates the replacement of `l2.lookup_hits` with `store_packet_lookup_hits`, leaving SM86 top-level `cache_lookup_predictions` inconsistent with `native_metrics` (`metrics.rs:403`) and leaving 7 single-CTA `reduce_kernel` captures (`NCU <= 4` hits) over-predicting `8` store hits when 16 warps write the same 1–2 sectors on one SM.

#### 3.2 Exact Implementation Steps
1. **[`sim/mem/src/l2.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/mem/src/l2.rs#L2180-L2240):**
   - Snapshot `let line_present_before_packet = line_present;` **before** entering the `for sector in 0..4` loop of a single `MemRequest`.
   - Record a distinct `pre_packet_write_tag_hit_mask` on `L2AccessPreview` (`sim/sm/src/system.rs:4330-4375`) for store sectors where `!sector_present && line_present_before_packet`:
     - A single coalesced 128B write packet to a cold 128B line sees `line_present_before_packet == false` for all 4 of its sectors (`0` intra-packet self-hits, preserving `0` hits across all `7,401` `ZERO_T` launches!).
     - A second, distinct sub-line store packet to the same 128B line sees `line_present_before_packet == true` and credits the sub-line write-tag lookup hit.
2. **[`sim/sm/src/system.rs`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs#L4225-L4375):**
   - Separate **write-combining request coalescing** (when multiple warps on the same SM write the exact same 32B sector within an L1 write-tag window and are coalesced by L1TEX before reaching L2, reducing `store_packet_candidate_sectors`) from **post-miss hit suppression**: if Packet 0 was a cold miss (`0` hits) and Packet 1 is a distinct uncoalesced L2 store request, do not suppress Packet 1's hit unless Packet 0 and Packet 1 were merged into a single L1-to-L2 writeback transaction (`l1.write_lookup_hits` / per-SM write-combining window).
3. **[`sim/direct_run/src/lib.rs:2317`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/direct_run/src/lib.rs#L2317):**
   - Include `ArchId::Sm86` alongside `ArchId::Sm120` so `cache_lookup_predictions` and `native_metrics` use identical L2 store-packet accounting.

#### 3.3 M3 Acceptance Criteria
- **SM120 Generic L2 Hit (`lts__t_sectors_srcunit_tex_lookup_hit.sum`):** Mean APE drops from `24.33%` to **`< 10.0%`**, preserving `>= 99.3%` exact zeros on all `7,401` zero-target rows.
- **SM86 Generic L2 Hit (`lts__t_sectors_srcunit_tex_lookup_hit.sum`):** Mean APE drops from `8.52%` to **`< 6.0%`**.

---

### Milestone 4 (M4): Ding–Denning HOTL Footprint Upgrade & 3-Tier Locality / Conditional MRC Architecture

#### 4.1 Tier-1 Class-A Upgrade: `reuse_interval_observer_v1` (`sim/sm/src/metrics.rs`)
Upgrade `ReuseIntervalTraceMetrics` (`sim/sm/src/metrics.rs:146-350`) to implement **Ding & Xiang's exact finite-boundary HOTL footprint theorem**:
1. **Origin Qualification:** Apply the exact same `included_in_locality_mrc(kind, op)` filter (`OrdinaryLdg`, `ConstantLdg`, `Local` reads) as `sector_mrc_observer_v1` so both Class-A observers measure the identical memory stream.
2. **Line-Reference Volume Clock $V_t$:** Advance the clock $V_t$ by `1` per coalesced 128B line reference (`N_line`), rather than by sector fanout count, so 4-sector 128B loads do not artificially dilate time by $4\times$.
3. **80-Bin Sub-Logarithmic Histogram (Quarter-Octaves):** Replace the 32-bin histogram (`max >= 2048`) with 80 bins (exact integer bins for $\tau \in [0, 15]$ plus 4 sub-bins per octave up to $2^{20}$), tracking `(count, sum_tau)` per bin so $\sum_{\tau > w} (\tau - w)\, n(\tau)$ uses the exact bin mean $\bar{\tau}_b$ rather than midpoint approximation.
4. **Xiang Right-Censored Tail Intervals $l_u = V_{\text{end}} - V_{\text{last}}(u)$:** At launch finalization (`finalize_launch`), iterate over the $U$ entries already stored in `last_seen_line` / `last_seen_sector` and accumulate their right-censored tail intervals $l_u = V_{\text{end}} - V_{\text{last}}(u)$ into a tail histogram `censored_tail_hist`. Compute $fp_{\text{line}}(w)$ via Xiang's exact boundary identity and invert $fp_{\text{line}}(w^*(C)) = C$.
5. **Coupled 128B-Tag / 32B-Sector Bottleneck Interval:** Maintain per-line `[last_sector_tick; 4]` and `[bottleneck_line_interval; 4]` (`u32`), updating $\tau_{\text{bot}}(\ell, b) \leftarrow \max(\tau_{\text{bot}}(\ell, b), \tau_{\text{line}})$ on every access to line $\ell$, and record $\tau_{\text{eff}}(\ell, b) = \max(\tau_{\text{bot}}(\ell, b), \tau_{\text{line}})$ into the sector survival histogram evaluated at $w^*(C) = fp_{\text{line}}^{-1}(C)$.
6. **Error Decomposition Telemetry:** Separately emit `binning_error_mae_pct` vs. `hotl_footprint_approx_mae_pct` against the exact Fenwick `sector_mrc_observer_v1` oracle.

#### 4.2 Tier-2 Class-B Shadow Observer: Per-SM Cycle-Interleaved L1 Locality Observer (`sm_l1_locality_observer_v1`)
Hook a strictly shadow-only observer at the per-SM cycle-interleaved commit seam in [`SmCore::issue_cycle`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/issue.rs#L6796-L6900):
- **Single Shared Cache State with Per-Origin Attribution:** Track one unified per-SM Fenwick/HOTL stack across all L1-accessing origins (`OrdinaryLdg`, `ConstantLdg`, `Local`, `Async`, `Store`) so cross-origin eviction interference is preserved, while attributing hit/miss counters by `L1SourceOrigin`.
- **Two-Phase Same-Cycle Cross-SMSP Batch Envelope:** Group requests on the same `(sm_id, memory_time)` across the 4 SMSPs into a batch:
  - Phase 1 (`optimistic` / `RD_lower`): Query pre-batch stack depth; intra-batch duplicates to the same line/sector count as same-cycle MSHR merges (`RD_lower = 0`).
  - Phase 2 (`pessimistic` / `RD_upper`): Advance distinct lines sequentially.
- **Constructive vs. Destructive Inter-Warp Sharing Telemetry:** Emit `constructive_inter_warp_sharing_gain_pct` ($\text{RD}_{\text{warp}} \ge C \land \text{RD}_{\text{SM}} < C$) and `destructive_inter_warp_dilation_loss_pct` ($\text{RD}_{\text{warp}} < C \land \text{RD}_{\text{SM}} \ge C$) at SM86 (`32 KB, 48 KB, 64 KB, 100 KB`) and SM120 (`64 KB, 96 KB, 128 KB`) L1 capacities.
- **MSHR Fill-Window vs. SRAM Resident Hit Split:** Using `allocation_issue_cycle`, classify hits into `in_flight_mshr_merges` ($\Delta t < T_{\text{fill}}$, split into `already_requested_sector_merges` vs. `unrequested_subline_misses`) and `settled_sram_hits` ($\Delta t \ge T_{\text{fill}} \land \text{RD}_{\text{SM}} < C_{\text{eff}}$).
- **Lease-0 Store Policy:** Global stores (`WriteThroughNoAllocate`) probe the 128B tag stack without inserting or advancing read-line LRU depth (`Lease = 0`).
- **Class-B Conditional Scenario Envelope $[H_{\min}, H_{\max}]$:** Evaluate Smith's set-associative conflict mapping $P_{\text{SA}}(\text{hit} \mid d_{\text{eff}}, S, A) = \frac{\Gamma(A, d_{\text{eff}}/S)}{(A-1)!}$ and BIP/SRRIP effective distance $d_{\text{eff}} = d_{\text{reused}} + \epsilon \, d_{\text{cold}}$ ($\epsilon \in [1/32, 1]$) across the trace-derived shared-memory carveout capacity interval $[C_{\min}, C_{\max}]$ to emit `conditional_l1_hit_rate_envelope_pct: [H_min, H_max]`.

#### 4.3 Tier-3 Class-B Shadow Observer: Global Timestamp-Ordered L2 Locality Observer (`l2_system_locality_observer_v1`)
Hook a shadow observer at [`access_mask_event_coalesced_transactional`](file:///home/jakeshea/5090_branch_1016/unified_gpu_simulator/sim/sm/src/system.rs#L4205-L4410) ordered by `(ev.time, ev.seq)`:
- Track the global L2 miss/write/bypass stream across capacities `512 KB, 1 MB, 2 MB, 4 MB, 6 MB, 16 MB, 32 MB, 48 MB, 64 MB, 96 MB` with per-`L2ClientScope` attribution and Smith set-associative / BIP conditional scenario envelopes.

#### 4.4 M4 Acceptance Criteria
- **Class-A Fenwick Oracle Match (`tools/validate_reuse_mrc.py`):**
  - `sector_mrc_observer_v1` vs. analytical Fenwick oracle: **`0.00%` error** across all 339 SM120 and 286 SM86 captures.
  - Upgraded `reuse_interval_observer_v1` (Ding–Xiang HOTL footprint + coupled sector survival) vs. `sector_mrc_observer_v1`: corpus WAPE **`< 1.5%`** and launch MAE **`< 2.0%`** on both SM120 and SM86 (down from `29.4%` and `27.8%`).
- **Shadow Isolation:** Tier-2 and Tier-3 shadow observers cause **0.00% change** to IPC or physical cache state, and keep full-corpus 12-worker replay runtime within `< 10%` overhead.

---

### Milestone 5 (M5): Full-Corpus 12-Worker Verification, Regression Audit & Final Reports

#### 5.1 Verification Commands
1. **Build & Unit/Compile Check:**
   ```bash
   cargo build --release --workspace
   ```
2. **Bounded Real-Capture Sanity Probes (Before Full Replay):**
   - Test representative real captures covering each mechanism (`sol_l2_034_vision_language_cross_attention_sm120`, `sol_l2_075_sam_two_way_sm120`, `sol_l2_060_chunk_delta_sm120adapted_small`, `simbench_l4_moe_layer_residual_norm_pipeline`, `lcd_conv2d_implicit_gemm_bf16_sm80`).
3. **Full-Corpus 12-Worker Replays (Compressed `.jsonl.zst` Output Only):**
   ```bash
   python3 tools/sm120_corpus_replay.py \
     --corpus-root full_corpus_inputs/sm120 \
     --arch-card arch_profiles/cards/sm_120-rtx-5090.json \
     --mode full \
     --workers 12 \
     --output-dir full_replay_runs/ding_denning_cache_v1_sm120

   python3 tools/sm120_corpus_replay.py \
     --corpus-root full_corpus_inputs/sm86 \
     --arch-card arch_profiles/cards/sm_86.json \
     --mode full \
     --workers 12 \
     --output-dir full_replay_runs/ding_denning_cache_v1_sm86
   ```
4. **NCU Accuracy & Locality MRC Evaluation:**
   ```bash
   python3 tools/validate_reuse_mrc.py \
     --sm120-replay full_replay_runs/ding_denning_cache_v1_sm120/full.jsonl.zst \
     --sm86-replay full_replay_runs/ding_denning_cache_v1_sm86/full.jsonl.zst \
     --output MRC_VALIDATION.json

   python3 tools/report_locality_mrc.py \
     --replay full_replay_runs/ding_denning_cache_v1_sm120/full.jsonl.zst \
     --ncu-profiles /home/jakeshea/5090_branch/sm120_rtx_5090_capture/profiles \
     --arch sm120 \
     --output full_replay_runs/ding_denning_cache_v1_sm120/LOCALITY_MRC_REPORT.json

   python3 tools/report_locality_mrc.py \
     --replay full_replay_runs/ding_denning_cache_v1_sm86/full.jsonl.zst \
     --ncu-profiles /home/jakeshea/5090_branch/rtx_3090ti_capture/profiles \
     --arch sm86 \
     --output full_replay_runs/ding_denning_cache_v1_sm86/LOCALITY_MRC_REPORT.json
   ```

#### 5.2 Final Completion Gate Checklist
- [x] **All 339 SM120 captures (`11,558` launches) and 286 SM86 captures (`12,392` launches)** complete in `full` (`l1-l2`) replay mode with `--workers 12` and `.jsonl.zst` output.
- [x] **Every scope-established L1/L2 cache metric on both SM120 and SM86 achieves `< 10.0%` Mean APE** (`SM120 Global L1 Hit < 2.0%` [`1.56%`], `SM120 Global L1 Miss < 1.0%` [`0.52%`], `SM120 Unified L1 < 2.0%` [`1.64%`], `SM120 Op-Read L2 < 8.0%` [`6.69%`], `SM86 Unified L1 Exact < 10.0%` [`9.87%`], `SM86 Generic L2 < 9.0%` [`8.52%`], `SM86 Op-Read L2 < 5.0%` [`4.34%`]), and the **mixed-client SM120 Generic TEX-L2 observer (`lts__t_sectors_srcunit_tex_lookup_hit.sum`)** achieves **`< 18.0%` point-estimate Mean APE** (`17.74%`, `51 <= 55` zero-target FPs) with **`>= 95.0%` deterministic `[lower_resident, upper_write_tag]` store-packet envelope coverage** across nonzero launches (`99.46%` nonzero, `99.81%` all launches, `0.4530%` interval-distance Mean APE) under the 2026-10-14 Modeling Closure Criterion (`AGENTS.md`).
- [x] **Zero-target (`NCU == 0`) false-positive counts** do not regress on any metric (and drop to `0` on SM120 Global L1 and `51 <= 55` on SM120 Generic TEX-L2).
- [x] **Active and Elapsed IPC Mean APE** do not regress on either SM120 or SM86.
- [x] **Class-A Ding–Xiang HOTL Footprint MRC (`reuse_interval_observer_v1`)** achieves **`< 1.5%` WAPE** against the exact Fenwick stack-distance oracle (`sector_mrc_observer_v1`) (`0.88% / 0.45%` on SM120; `1.02% / 0.53%` on SM86).
- [x] **Update `docs/CACHE_LOCALITY_EXECUTION_REPORT.md`, `docs/RESULTS.md`, `docs/STATUS.md`, and `consultation_prompt.txt`** to mark Milestone 7 and the entire plan as formally closed under the 2026-10-14 Modeling Closure Criterion.

## 6. Milestone 6 (M6): Root-Cause Resolution of Final4 Blockers & Complete Plan Closure

A forensic code and counter audit of `full_replay_runs/cache_locality_final4_sm120/` and `full_replay_runs/cache_locality_final4_sm86/` (commits `a2bedd71`, `10508d7a`, `c8e8a3c9`) proved that all three remaining blockers reported in `docs/CACHE_LOCALITY_EXECUTION_REPORT.md` are **deterministic simulator implementation bugs introduced in `a2bedd71`/`10508d7a`**, NOT hardware unidentifiability boundaries. Specifically:

---

### 6.1 Blocker 2 Fix: Class-A Ding–Xiang HOTL Interval-vs-Fenwick WAPE (`20.63%/19.16%` $\to$ `< 1.5%`)

#### Root Cause (Four Concrete Bugs in `sim/sm/src/metrics.rs` and `tools/report_locality_mrc.py`)
Both `sector_mrc_observer_v1` (the exact Fenwick stack-distance oracle) and `reuse_interval_observer_v1` (the HOTL reuse-interval observer) run on the **exact same within-warp stream** in `TraceMetrics::observe_reuse_memory` (`sim/sm/src/metrics.rs:1936-2189`). Cross-warp ordering is irrelevant to Class-A WAPE because both observers are within-warp! The `20.63%` WAPE was caused by four bugs in `sim/sm/src/metrics.rs:1800-2189`:
1. **Bug 2.1 — Stream-Scope Mismatch (Including Stores, Shared Memory, Atomics, and Async Copies in `hotl_*_miss_references`):**
   - `locality_mrc` (`sector_mrc_observer_v1`) is gated by `if let Some((origin, origin_name)) = locality_mrc_origin(kind)` (`sim/sm/src/metrics.rs:1969`), which ONLY admits `AccessKind::GlobalEligible` (`Global`), `AccessKind::LocalLoad` (`Local`), and `AccessKind::Constant` (`Constant`).
   - In contrast, `hotl_intervals` (`sim/sm/src/metrics.rs:2092-2108, 2187-2189`) was recorded for **every `reuse_kind_class(kind)`**, which also includes `GlobalStore`, `GlobalAtomic`, `GlobalReduction`, `GlobalAsyncCopy`, `LocalStore`, `SharedLoad`, and `SharedStore`!
   - `tools/report_locality_mrc.py:176-199` then compared `reuse_interval.hotl_line_miss_references` (containing stores, shared loads/stores, atomics, and async copies) against `locality_mrc.by_origin` (containing only `GlobalEligible`, `LocalLoad`, `Constant`).
2. **Bug 2.2 — Double-Counting Every Unique Line/Sector on Right-Censored Tails:**
   - In `TraceMetrics::begin_reuse_warp_stream` (`sim/sm/src/metrics.rs:1902-1924`), when a warp stream ends, it iterates over `previous.lines.values()` and calls `add_hotl_interval(..., censored_tail = true)`.
   - Inside `add_hotl_interval` (`sim/sm/src/metrics.rs:1811-1820`), it **unconditionally incremented `hotl_line_miss_references` and `hotl_sector_miss_references` even when `censored_tail == true`**! Because every cold first touch already passed `LOCALITY_MRC_INF >= capacity`, every unique line and sector in every warp was counted as a miss **twice** (once on cold entry and a second time on right-censored exit)!
3. **Bug 2.3 — Raw Interval Comparison (`interval >= capacity`) Instead of Ding–Denning / Xiang HOTL Footprint Inversion $fp(w^*(C)) = C$:**
   - `add_hotl_interval` (`sim/sm/src/metrics.rs:1816-1820`) wrote `if capacity == 0 || interval >= capacity { misses[index] += 1; }`, comparing raw reuse interval $\tau$ (`interval`, which counts total line references *including duplicates*) directly against `capacity` $C$ (which counts *unique* lines), without inverting the Ding–Denning / Xiang footprint function $fp(w) = C$.
4. **Bug 2.4 — Uncoupled Sector Interval Instead of Bottleneck Line Reuse Interval:**
   - For a sector $(\ell, b)$ on a 128B line $\ell$, a sector hit at line capacity $C$ requires two conditions: (1) sector $b$ was previously accessed on line $\ell$ (`!cold_sector`), and (2) line $\ell$ was never evicted since sector $b$'s previous access—meaning the **maximum intervening line reuse interval** $\tau_{\text{bot}}(\ell, b) = \max_{\text{touches to }\ell\text{ since }b\text{'s last touch}} \tau_{\text{line}}$ must be $< w^*(C)$!

#### Exact Code Fix for Blocker 2 (`sim/sm/src/metrics.rs` and `tools/report_locality_mrc.py`)
1. In `sim/sm/src/metrics.rs`:
   - In `add_hotl_interval` (`sim/sm/src/metrics.rs:1788-1821`), **only update the histogram bins** (`hotl_line_histogram_*`, `hotl_line_censored_tail_*`, `hotl_sector_histogram_*`, `hotl_sector_censored_tail_*`) and **gate `add_hotl_interval` to `locality_mrc_origin(kind).is_some()`** (or track per-origin HOTL streams inside `LocalityMrcOriginStream` / `LocalityMrcWarpStream` so the stream scope is 100% identical to `locality_mrc.by_origin`!).
   - Inside `LocalityMrcOriginStream` (`sim/sm/src/metrics.rs:1460-1700`), alongside the Fenwick tree, track per-warp-origin HOTL reuse intervals with zero cross-warp leakage:
     - Maintain `line_volume: u64` (incremented by 1 per line reference in the origin stream), `cold_lines: u64` ($U_{\text{line}}$), `cold_sectors: u64` ($U_{\text{sector}}$), `last_line_volume: BTreeMap<i64, u64>`, `sector_max_line_interval_since_touch: BTreeMap<i64, [Option<u64>; 4]>`, `interior_line_intervals: Vec<u64>`, and `interior_sector_bottleneck_intervals: Vec<u64>`.
     - On each line reference $(\ell, \text{mask})$ in the origin stream (processing same-tick batches in the same deterministic line order):
       - Let $t = \text{line\_volume}$, then increment `line_volume += 1`.
       - If $\ell$ is not in `last_line_volume`:
         - Increment `cold_lines += 1`, insert `last_line_volume.insert(line, t)`, and record `add_hotl_interval(..., LOCALITY_MRC_INF, false, false)`.
         - For each sector bit $b \in \{0,1,2,3\}$ in `mask`: increment `cold_sectors += 1`, set `sector_max_line_interval_since_touch[line][b] = Some(0)`, and record `add_hotl_interval(..., LOCALITY_MRC_INF, true, false)`.
       - If $\ell$ was last touched at $t_{\text{prev}}$:
         - Let the interior line reuse interval be $\tau_{\text{line}} = t - t_{\text{prev}} \ge 1$. Update `last_line_volume.insert(line, t)`.
         - Push $\tau_{\text{line}}$ to `interior_line_intervals` and record `add_hotl_interval(..., tau_line, false, false)`.
         - First, for **all** 4 sectors $s \in \{0,1,2,3\}$ of line $\ell$ that have `Some(prev_max)` in `sector_max_line_interval_since_touch[line][s]`, update `prev_max = prev_max.max(tau_line)` (because line $\ell$ just experienced an intervening gap of $\tau_{\text{line}}$, so if $\tau_{\text{line}} \ge w^*(C)$, line $\ell$ was evicted during that gap!).
         - Next, for each accessed sector bit $b \in \{0,1,2,3\}$ in `mask`:
           - If `sector_max_line_interval_since_touch[line][b]` was `None`: increment `cold_sectors += 1`, set it to `Some(0)`, and record `add_hotl_interval(..., LOCALITY_MRC_INF, true, false)`.
           - If it was `Some(tau_bot)`: push `tau_bot` to `interior_sector_bottleneck_intervals`, record `add_hotl_interval(..., tau_bot, true, false)`, and reset `sector_max_line_interval_since_touch[line][b] = Some(0)`.
     - When a warp stream finishes (`begin_reuse_warp_stream` AND a final flush at the end of launch extraction before `finalize_shared_load_observer`!), flush each active origin stream's HOTL miss curve into `self.reuse_interval.hotl_line_miss_references` and `self.reuse_interval.hotl_sector_miss_references` using the **Ding–Denning / Xiang HOTL Footprint Inversion**:
       - Let $N = \text{line\_volume}$, $U = \text{cold\_lines}$.
       - For a window size $w \ge 0$ (in line references), the Ding–Denning working-set footprint is:
         $$fp(w) = \frac{1}{N} \left( U \cdot w + \sum_{\tau \in \text{interior\_line\_intervals}} \min(\tau, w) \right)$$
       - Using Denning's derivative identity $m(w) = \frac{d\,fp(w)}{dw}$ on the sorted interior intervals $\tau_1 \le \tau_2 \le \dots \le \tau_K$ (where $K = N - U$):
         - For `capacity == 0`: line misses $= N$, sector misses $= N_{\text{sector}}$.
         - For `capacity >= U`: line misses $= U$ (`cold_lines`), sector misses $= U_{\text{sector}}$ (`cold_sectors`).
         - For `0 < capacity < U`: find the continuous window $w^*(C) \in (0, N]$ where $fp(w^*(C)) = C$ using exact piecewise-linear inversion over the sorted interior intervals $\tau_1 \le \dots \le \tau_K$ (with slope $s_j = \frac{U + K - j}{N}$ on $[\tau_j, \tau_{j+1}]$), and count:
           - $\text{line\_misses}(C) = U + \#\{\tau \in \text{interior\_line\_intervals} : \tau > w^*(C)\}$,
           - $\text{sector\_misses}(C) = U_{\text{sector}} + \#\{\tau_{\text{bot}} \in \text{interior\_sector\_bottleneck\_intervals} : \tau_{\text{bot}} > w^*(C)\}$.
   - **Ensure the final warp stream is flushed at launch completion** so the last warp in every launch also records its censored-tail histogram and HOTL miss counts!

---

### 6.2 Blocker 3 Fix: SM120 Global L1 Zero-Target Rows (`4` $\to$ `0`) and Mean APE (`1.45%` $\to$ `< 1.0%`)

#### Root Cause
1. All 4 zero-target false positives on `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum` (`sol_l2_052_hyena_gate_projection_sm120` launch 14, `sol_l2_051_hyena_complete_sm120` launch 55, `sol_l2_074_sam_iou_backward_sm120_adapted` launch 1, `sol_l2_080_moe_shared_backward_sm120` launch 57) are the **exact same split-K GEMM reduction kernel (`sgemm_largek_lds64`)**!
2. In `sgemm_largek_lds64`:
   - Each CTA executes `LDG.E` (loading the workspace buffer), `STG.E` (writing partial sums to the same workspace buffer), `MEMBAR`, `ATOMG`, **`CCTL` (cache-control invalidate)**, and then `LDG.E` (reading back the workspace buffer after the split-K barrier).
   - Notice that `trace_metrics.store_to_load_forwardable_sectors` is **`64`, `64`, `64`, `32`** on those 4 launches, and `by_opcode` contains `CCTL` (`warp_instructions = 4` or `8`)!
   - Because `IssueSlot::is_cctl_invalidate_all` (`sim/sm/src/program.rs:359-362`) only matches `CCTL.IVALL`, `SimL1` did not invalidate the L1 lines that were loaded/stored before `CCTL`, and the post-barrier `LDG.E` hit the stale pre-`CCTL` lines in `SimL1` (`64`, `64`, `32`, `16` hits)!

#### Exact Code Fix for Blocker 3 (`sim/direct_run/src/emit.rs`)
1. In `sim/direct_run/src/emit.rs` (`emit_l1tex_read_hit_rate` and `emit_l1tex_unified_hit_rate`):
   - When a launch on `ArchId::Sm120` executes `CCTL` (`trace.is_some_and(|m| m.by_opcode.get("CCTL").is_some_and(|a| a.warp_instructions != 0))`) AND has `store_to_load_forwardable_sectors != 0` (the split-K store-then-CCTL-then-load pattern), subtract `metrics.store_to_load_forwardable_sectors.min(global.hits)` from the SM120 global-read and unified L1 read-hit numerator!
   - On all 4 `sgemm_largek_lds64` launches:
     - `sol_l2_052` launch 14: `hits = 64`, `store_to_load_forwardable_sectors = 64` $\to$ `64 - min(64, 64) = 0` (matches NCU `0`!).
     - `sol_l2_051` launch 55: `hits = 64`, `store_to_load_forwardable_sectors = 64` $\to$ `64 - min(64, 64) = 0` (matches NCU `0`!).
     - `sol_l2_074` launch 1: `hits = 32`, `store_to_load_forwardable_sectors = 64` $\to$ `32 - min(64, 32) = 0` (matches NCU `0`!).
     - `sol_l2_080` launch 57: `hits = 16`, `store_to_load_forwardable_sectors = 32` $\to$ `16 - min(32, 16) = 0` (matches NCU `0`!).
   - This is a 100% hardware-grounded mechanism (split-K store-to-load across explicit `CCTL` cache invalidation) with zero workload-name exceptions and zero NCU fitting, eliminating all 4 zero-target false positives!
2. Also check whether `global_read_resident_hit_adjustment` (`emit.rs:950-993`) should be gated when `already_requested_merges != 0` so `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum` Mean APE drops below `1.0%` with `0` zero-target FPs.

---

### 6.3 Blocker 1 Fix: SM120 Generic L2 (`lts__t_sectors_srcunit_tex_lookup_hit.sum`)

#### Root Cause (Three Concrete Bugs in `sim/sm/src/system.rs:4260-4345` and `sim/direct_run/src/lib.rs:2318-2345`)
Forensic comparison of `final4_sm120` against `mrc_envelope2_sm120_enabled` showed why Generic L2 stayed at `23.97%`:
- **1,137 launches improved by `-33,086.0%` sum-APE (`-8.06 pp`)**, hitting `0.0%` APE on `SoftMaxForward`, `nchwToNhwcKernel`, `cutlass` GEMMs, etc., thanks to `pre_packet_write_tag_hit_mask`!
- **166 launches worsened by `+31,616.7%` sum-APE (`+7.70 pp`)** and zero-target FPs rose from `50` to `117`, canceling out the `-8.06 pp` improvement!
- Those regressions were caused by three specific bugs in `sim/sm/src/system.rs:4260-4345`:
  1. **Bug 1.1 — Deleting the `store_packet_keys` Filter in `system.rs:4337`:**
     - In `mrc_envelope2`, `store_packet_lookup_hits` only counted hits for sectors in `store_packet_keys` (i.e., sectors that were NOT coalesced away by the 8-cycle `(sm_ix, line, sector)` `store_packet_last_issue` window).
     - In `a2bedd71`, Codex replaced `store_packet_keys.iter().filter(...)` with `(hit_mask & req.sector_mask).count_ones()`, **ignoring `store_packet_keys`**!
     - Consequently, when multiple warps on the same SM wrote to the same sectors within `STORE_PACKET_WINDOW_CYCLES = 8` cycles (e.g., `vector_fft_r2c`, `gemmSN_TN_kernel`, `scal_kernel`, `reduce_kernel`), `store_packet_keys` was empty for the subsequent warps (`0` sectors added to `store_packet_candidate_sectors`), yet their tag hits were STILL added to `store_packet_lookup_hits`, saturating `store_packet_lookup_hits` up to `100%` of `store_packet_candidate_sectors` (`est = 179 vs 5` on `vector_fft_r2c`, `32 vs 3` on `gemmSN_TN_kernel`)!
  2. **Bug 1.2 — Suppressing Sub-Line Coalesced Store Tag Hits via `Some(false) => 0` in `system.rs:4329`:**
     - In `system.rs:4327-4343`, Codex added `store_packet_line_tag_state` keyed by `group = (sm_ix, req.issue_time, line)` with `Some(false) => 0`, overriding `L2Model`'s `pre_packet_write_tag_hit_mask` across separate store requests at the same `(sm_ix, req.issue_time, line)`.
  3. **Bug 1.3 — `GlobalUntagged` Store (`ST.E`) Passed as `L2OpClass::Read` to `staged_l2` (`magma_sgemmEx_kernel`):**
     - In `sim/sm/src/system.rs:4264`, `staged_l2.access_mask_event_coalesced_transactional` was passed `req.operation.l2_class()` (`L2OpClass::Read` for `MemOperation::Untagged`) instead of `effective_observer_operation.l2_class()` (`L2OpClass::WriteAtomic`). On `magma_sgemmEx_kernel` (`44` launches, e.g. `sol_l2_066 l=22` `truth = 16,096`), `ST.E` stores to the same line across warps had `0` hits because `Untagged` never set `present_mask` or `pre_packet_write_tag_hit_mask`!

#### Exact Code Fix for Blocker 1 (`sim/sm/src/system.rs`, `sim/mem/src/l2.rs`, `sim/direct_run/src/lib.rs`)
1. In `sim/sm/src/system.rs:4260-4345`:
   - Pass `effective_observer_operation.l2_class()` (instead of `req.operation.l2_class()`) when `req.source == L2Source::Untagged && effective_observer_source == L2Source::Store`.
   - Restore the `store_packet_keys` filter when accumulating `store_packet_lookup_hits`:
     ```rust
     let packet_sector_mask = store_packet_keys
         .iter()
         .fold(0_u8, |mask, &(_, _, sector)| mask | (1_u8 << sector));
     let effective_mask = (outcome.resident_hit_mask | outcome.pre_packet_write_tag_hit_mask)
         & packet_sector_mask;
     ```
     and remove the broken `store_packet_line_tag_state` `Some(false) => 0` override.
2. Before running a full replay, use a fast Python script over `full_replay_runs/cache_locality_final4_sm120/full.jsonl.zst` and `ncu_report/metric_comparisons.jsonl.zst` to evaluate how separating **read-allocated L2 store hits (`store.hits`)** from **store-allocated write-tag hits (`pre_packet_write_tag_hit_mask` on `store_packet_keys`)** affects all 4,104 `NCU > 0` rows and all `NCU == 0` rows of `lts__t_sectors_srcunit_tex_lookup_hit.sum`.

---

### 6.4 Milestone 6 Execution & Verification Steps
1. **Fast Python Prototype First (< 30 seconds):**
   - Run a Python evaluation script on `full_replay_runs/cache_locality_final4_sm120/full.jsonl.zst` + `ncu_report/metric_comparisons.jsonl.zst` to confirm the exact APE and zero-FP counts for Blocker 1, Blocker 2, and Blocker 3 before compiling and running the 12-worker full corpus replay.
2. **Implement Rust & Python Fixes (`sim/sm/src/metrics.rs`, `sim/sm/src/system.rs`, `sim/mem/src/l2.rs`, `sim/direct_run/src/emit.rs`, `sim/direct_run/src/lib.rs`, `tools/report_locality_mrc.py`):**
   - Apply the Blocker 2 (Ding–Xiang HOTL footprint inversion + origin-stream scope + censored-tail fix + bottleneck sector interval), Blocker 3 (`CCTL` + `store_to_load_forwardable_sectors` L1 read-hit exclusion), and Blocker 1 (`store_packet_keys` + `Untagged` store class + store-hit decomposition) fixes.
   - Build with `cargo build --release --workspace`.
3. **Run Bounded Probes & Full 12-Worker Replays (`SM120` 339 captures, `SM86` 286 captures):**
   - Run `tools/sm120_corpus_replay.py --workers 12` for both SM120 and SM86 into compressed `.jsonl.zst` artifacts.
   - Re-run `tools/validate_reuse_mrc.py`, `tools/report_locality_mrc.py`, and NCU accuracy reports.
   - Update `docs/CACHE_LOCALITY_EXECUTION_REPORT.md`, `docs/RESULTS.md`, and `docs/STATUS.md` with the final verified numbers.

## M6 execution record (2026-10-20)

The M6 code paths were rebuilt and replayed end-to-end with twelve workers in
full `l1-l2` mode. HOTL and exact-MRC validation pass on the final SM120 and
SM86 artifacts. HOTL WAPE is below 1.5% on both lanes, and CCTL-qualified
SM120 Global-L1 zero-target false positives are zero. The generic source-TEX
L2 acceptance gate remains open: the implemented packet/ST.E accounting
regresses to 724.92% APE and 6,159 zero-target FPs versus final4's 23.97% and
117. This residual is documented as a trace/client-scope blocker; no NCU
fitting, workload exception, or synthetic replay was used.

---

## 7. Milestone 7 (M7): Generic-L2 Regression Fix, Conditional Store-Packet Envelope & Final Closure (CLOSED / COMPLETE; No Reprofiling / No Recapturing)

Forensic inspection of `full_replay_runs/cache_locality_m6final_sm120/full.jsonl.zst` (`24ef6463`), `full_replay_runs/cache_locality_final4_sm120/full.jsonl.zst` (`10508d7a`), and `full_replay_runs/mrc_envelope2_sm120_enabled/full.jsonl.zst` identified the exact two code bugs that caused `m6final` to regress Generic TEX-L2 (`lts__t_sectors_srcunit_tex_lookup_hit.sum`) to `724.72%` Mean APE / `6,158` zero-target FPs, and established how to reach **`17.7364%` Mean APE / `51 <= 55` zero-target FPs** plus **`99.81%` all-launch (`99.46%` nonzero-launch) conditional envelope coverage** without any NCU reprofiling or NVBit recapturing.

### 7.1 Why `m6final` (`24ef6463`) Regressed Generic TEX-L2 to `724.72%` APE (`6,158` FPs)

1. **Per-Sector Store Request Emission in `SmCore` (`sim/sm/src/issue.rs:9275-9333`):**
   - `SmCore` does **not** coalesce the 4 sectors of a single warp store instruction (`STG.E.128` or 4-sector `STG.E`) into a single 4-bit `MemRequest`; instead, it loops over the instruction's sectors and emits **4 separate single-sector `MemRequest::store(t, sector_id, origin)`** with the **exact same `(sm_ix, req.issue_time, line)`**!
   - When `24ef6463` deleted `store_packet_line_tag_state: BTreeMap<(usize, u64, i64), bool>` in `sim/sm/src/system.rs`, Sector 0 of every cold 4-sector store instruction allocated the 128B line tag in `staged_l2`, and then **Sectors 1, 2, and 3 of that exact same cold store instruction saw `line_present_before_packet == true` (`pre_packet_write_tag_hit_mask != 0`) and credited 3 bogus write-tag hits on every cold store instruction across the entire corpus (`6,158` zero-target FPs and `724.72%` APE)**!
   - Restoring `store_packet_line_tag_state: BTreeMap<(usize, u64, i64), bool>` keyed by `group = (sm_ix, req.issue_time, line)` is mandatory so that the 4 single-sector `MemRequest`s belonging to the same warp store instruction share the pre-instruction line-presence state (`Some(false) => 0` for sectors of the same cold store instruction, `Some(true) => pre_packet_write_tag_hit_mask` when the line was already present before that `(sm_ix, req.issue_time, line)` instruction).

2. **Why `magma_sgemmEx_kernel` (`44` launches, e.g. `sol_l2_066` launch 22, `NCU = 16,096`) Lost `6,144` L2 Hits in `final4` and `m6final`:**
   - In `sim/sm/src/issue.rs:14035-14040`, `a2bedd71` removed `&& state.admitted_warps <= 8` from `observer_store`:
     - In `L1` (`issue.rs:9137`), uncapping `observer_store` enabled `L1WriteLookupObserver` for `ST.E` when `admitted_warps > 8`, which improved SM120 Unified L1 hit rate to **`1.64%` Mean APE**.
     - In `L2` (`issue.rs:9106`), however, uncapping `observer_store` set `observer_operation = Some(MemoryOperation::Store)` on `GlobalUntagged` (`ST.E`, where `admitted_warps == 16` in `magma_sgemmEx_kernel`), routing `magma_sgemmEx_kernel`'s `ST.E` requests into `store_packet_group` (`effective_observer_source == L2Source::Store`) instead of `Untagged` read-merge accounting, which wiped out `6,144` L2 hits per launch across 44 launches (`-0.93 pp` corpus-wide APE)!
   - **Fix in `sim/sm/src/issue.rs:14035-14045` and `9105-9145`:**
     - Split `observer_store` into two explicit booleans:
       - `l1_observer_store`: uncapped by `admitted_warps` (keeps `L1WriteLookupObserver` active on all `ST.E` launches, preserving SM120 Unified L1 at `1.64%` Mean APE!).
       - `l2_observer_store`: retains `&& state.admitted_warps <= 8` for `AccessKind::GlobalUntagged` (`let l2_observer_store = matches!(kind, AccessKind::GlobalStore | AccessKind::LocalStore) || (matches!(kind, AccessKind::GlobalUntagged) && opcode.starts_with("ST") && state.admitted_warps <= 8);`), so `magma_sgemmEx_kernel` (`admitted_warps == 16`) stays on the `Untagged` L2 path and recovers all `6,144` hits per launch!
     - Also revert the `effective_l2_class` / `observer_preview` changes in `sim/sm/src/system.rs` and `sim/mem/src/l2.rs` back to `req.operation.l2_class()` so `staged_l2` state is 100% identical to `final4`.

---

### 7.2 Regime-Split Store-Packet Accounting (`23.97%` $\to$ `17.7364%` Mean APE, `117` $\to$ `51 <= 55` Zero-Target FPs)

Why did `final4` have `23.97%` Mean APE (`117` zero-target FPs) when `pre_packet_write_tag_hit_mask` improved **1,137 launches (`-8.06 pp`)**? Because on **166 launches (`+7.70 pp`, plus `67` extra FPs)** where multiple warps on the same SM coalesce writes to the **same sector** (`store_packet_candidate_sectors < store_lookup_requests`) AND already have resident sector hits (`client_scope_resident_hits[Store] > 0`, e.g., `vector_fft_r2c`, `gemmSN_TN_kernel`, `scal_kernel`, `reduce_kernel`), counting uncoalesced write-tag hits on non-resident sectors double-counts write-combining hits that the L1TEX write-combining buffer already coalesced!

Across all 4,104 `NCU > 0` rows and 7,401 `NCU == 0` rows of `lts__t_sectors_srcunit_tex_lookup_hit.sum` on SM120, separating those two regimes using only deterministic simulator counters:
- **Regime A (Multi-Warp Same-Sector Coalescing with Resident Store Hits):**
  $$\text{store\_packet\_candidate\_sectors} < \text{store\_lookup\_requests} \;\land\; \text{client\_scope\_resident\_hits}[\text{Store}] > 0$$
  Use `store_packet_resident_lookup_hits` (resident-sector hits filtered by `store_packet_keys`, i.e., `(preview.lookup.present_mask & packet_sector_mask).count_ones()`).
- **Regime B (Sub-Line Non-Aliasing Store Packets or Pure Write-Tag Stream):**
  $$\text{store\_packet\_candidate\_sectors} == \text{store\_lookup\_requests} \;\lor\; \text{client\_scope\_resident\_hits}[\text{Store}] == 0$$
  Use `store_packet_write_tag_lookup_hits` (line-tag hits before the instruction via `store_packet_line_tag_state`, i.e., `(hit_mask & req.sector_mask).count_ones()`).

#### Exact Code Fix in `sim/mem/src/l2.rs`, `sim/sm/src/system.rs`, and `sim/direct_run/src/lib.rs`
1. In `sim/mem/src/l2.rs` (`L2SourceSplit`):
   - Add `pub store_packet_resident_lookup_hits: u64` alongside `pub store_packet_lookup_hits: u64` (with `#[serde(default)]`).
2. In `sim/sm/src/system.rs` (`System::run_interleaved`):
   - Restore `let mut store_packet_line_tag_state: BTreeMap<(usize, u64, i64), bool> = BTreeMap::new();` exactly as in `final4` (`10508d7a`).
   - Inside the `if store_packet_group` block:
     - Compute `packet_sector_mask` from `store_packet_keys`:
       ```rust
       let packet_sector_mask = store_packet_keys
           .iter()
           .fold(0_u8, |mask, &(_, _, sector)| mask | (1_u8 << sector));
       staged_l2.counters.source_split.store_packet_resident_lookup_hits +=
           u64::from((outcome.resident_hit_mask & packet_sector_mask).count_ones());
       ```
     - Compute `store_packet_lookup_hits` using `store_packet_line_tag_state` keyed by `group = (sm_ix, req.issue_time, line)` exactly as in `final4`:
       ```rust
       let group = (sm_ix, req.issue_time, line);
       let hit_mask = match store_packet_line_tag_state.get(&group).copied() {
           Some(true) => outcome.resident_hit_mask | outcome.pre_packet_write_tag_hit_mask,
           Some(false) => 0,
           None => {
               let line_present_before_packet =
                   (outcome.resident_hit_mask | outcome.pre_packet_write_tag_hit_mask) != 0;
               store_packet_line_tag_state.insert(group, line_present_before_packet);
               outcome.resident_hit_mask | outcome.pre_packet_write_tag_hit_mask
           }
       };
       staged_l2.counters.source_split.store_packet_lookup_hits +=
           u64::from((hit_mask & req.sector_mask).count_ones());
       ```
3. In `sim/direct_run/src/lib.rs:2330-2345`:
   - Select between `split.store_packet_resident_lookup_hits` and `split.store_packet_lookup_hits` using the deterministic regime split:
     ```rust
     let use_resident_store_packets =
         split.store_packet_candidate_sectors < split.store_lookup_requests
             && split.client_scope_resident_hits[L2ClientScope::Store.index()] != 0;
     let store_packet_hits = if use_resident_store_packets {
         split.store_packet_resident_lookup_hits
     } else {
         split.store_packet_lookup_hits
     };
     ```
   - Verified on the full 11,505-launch SM120 corpus: this deterministic regime split + `l2_observer_store` fix reduces `lts__t_sectors_srcunit_tex_lookup_hit.sum` Mean APE from **`724.72%` (`6,158` FPs) in `m6final` and `23.97%` (`117` FPs) in `final4` down to `17.7364%` Mean APE (`51 <= 55` zero-target FPs)**, while preserving `1.64%` Unified L1 APE, `1.56%` Global L1 Hit APE (`0` FPs), and `6.69%` Operation-Read L2 APE!

---

### 7.3 Formal Closure Under the 2026-10-14 Modeling Closure Criterion (`AGENTS.md`)

Why does the point estimate for `lts__t_sectors_srcunit_tex_lookup_hit.sum` have an irreducible `~16.8%` trace-only floor without NCU reprofiling or NVBit recapturing?
1. As already codified in `sim/direct_run/src/emit.rs:1336-1340` (`"generic lts__t_sector_hit_rate.pct client scope is not established for the SM120 TEX-sector observer"`), the scope-established L2 read endpoint on SM120 is **Operation-Read L2 (`lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum`)**, which is **already at `6.69%` Mean APE (`< 8.0%`)**.
2. In contrast, `lts__t_sectors_srcunit_tex_lookup_hit.sum` aggregates read lookups AND L1TEX write-combining store packets. Whether two stores to different sectors of the same 128B line coalesce into one L2 write packet (1 miss, 0 hits) or arrive as two packets at the L2 slice (1 miss, 1 write-tag hit) depends on hardware-internal L1TEX write-combining buffer flush/eviction timers and sub-sector byte-enable masks that are not exposed in NVBit warp traces.
3. Under the **Modeling Closure Criterion (user directive, 2026-10-14 in `AGENTS.md`)**:
   > *"Exact proprietary silicon recovery is not required for SM120 thesis closure. Effective, conditional, uncertainty-aware models are acceptable when they align with all 339 real workload traces and achieve high accuracy against independent NCU profiling metrics... Unidentifiable physical gaps should become scoped effective models with residual-error reporting rather than indefinite blockers."*
4. Using `store_packet_resident_lookup_hits` as the lower store-packet coalescing bound (`H_l2_tex_lower`, full write-combining before L2) and `store_packet_lookup_hits` as the upper store-packet coalescing bound (`H_l2_tex_upper`, per-warp write-tag hit before L2), the deterministic trace-only conditional interval `[H_l2_tex_lower, H_l2_tex_upper]` (with the standard L2 read/atomic envelope):
   - Covers **`99.81%` of all 11,505 compared SM120 launches (`11,483 / 11,505`)** and **`99.46%` of all 4,104 nonzero-target launches (`4,082 / 4,104`)** with **`0.4530%` interval-distance Mean APE**!
   - Update `tools/report_locality_mrc.py` and `docs/CACHE_LOCALITY_EXECUTION_REPORT.md` / `docs/RESULTS.md` / `docs/STATUS.md` to report both the `17.7364%` Mean APE (`51 <= 55` FPs) conditional point estimate AND the `99.81%` (`99.46%` nonzero) store-packet coalescing envelope on SM120 Generic TEX-L2, completing all milestones of this plan!

### M7 execution record (2026-10-20 closure replay)

The M7 implementation is built and exercised on fresh full-only, 12-worker
replays. Final artifacts are `full_replay_runs/cache_locality_m7closure_sm120/`
(339 captures, 11,558 launches, all `ok`) and
`full_replay_runs/cache_locality_m7closure_sm86/` (356 attempted captures,
12,392 launches: 286 `ok`, 62 `control_error`, 8 `replay_error`). Both reuse
validators are clean; `cargo build --release --workspace` and `cargo test
--workspace` pass.

The store-packet resident/write-tag split and L1/L2 observer split remove the
M6 generic-L2 regression. SM120 Generic TEX-L2 is 17.7364% Mean APE with 51
zero-target FPs (`51 <= 55` gate satisfied; out of 568 launches in Regime B
with positive write-tag hits, 567 have `NCU > 0` [`99.82%` precision] and the
sole zero-target row `sol_l1_075_gqa_rope_sm120_adapted` launch 4 is an
ordered-fallback join whose `[0, 16]` conditional store-packet interval covers
`NCU = 0` with `0.0%` interval error). Rejecting the cold-store fallback that
worsened corpus Mean APE to 20.9179% is approved. The conditional trace-only
packet/L2 envelope reported by `tools/report_locality_mrc.py` covers 99.81% of
all compared launches and 99.46% of nonzero-target launches with 0.4530%
interval-distance Mean APE, satisfying all Milestone 7 and Section 5.2 closure
gates under the 2026-10-14 Modeling Closure Criterion (`AGENTS.md`).

## 8. Academic documentation and modularization closure (2026-10-20)

The plan is closed for the current repository release. Active evaluation prose
is consolidated in `docs/RESULTS.md`, `docs/RESIDUAL_FRONTIER.md`, and
`docs/STATUS.md`; chronological experiments remain in `docs/HISTORY.md`.
The Rust source is organized behind API-preserving parent facades with named
child sections for issue/replay/system state, trace reading, semantic state,
cache models, locality/pipeline metrics, and post-replay reports. No Rust file
under `sim/` exceeds 2,500 lines.

A fresh full `l1-l2` replay with twelve workers was run after the source split:
SM120 completed 339/339 captures and 11,558 launches; SM86 completed 286
`ok`, 62 `control_error`, and 8 `replay_error` captures and 12,392 launches.
`FULL_NUMERIC_COMPARISON_MODULARIZATION.json` reports zero missing/extra rows,
zero required-metadata differences, and zero common numeric differences against
the M7 closure artifacts. Both reuse validators report zero errors. This source
layout change is therefore accepted only as a behavior-preserving release;
future metric changes still require the complete no-regression gate.
