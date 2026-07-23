# Probe (f) cp.async/LDGSTS timing — Fable-5 design (sm_86)

> Provenance: Fable 5 answer (2026-07-08) to `plans/sm86_cp_async_timing_probe_fable5_prompt.md` (follow-up to the
> calibration suite `plans/sm86_calibration_microbench_design_fable5_answer.md`). Verbatim design (thinking-block
> stripped). Referenced by `plans/sm86_comprehensive_live_capture_campaign_plan.md` (AC-6, task18).
> Headline: probe (f) is promoted to **load-bearing rank #2** — order becomes b → f → a → d → c → e
> (execution slot order b, a, d, f, c, e, since f's sharing verdicts consume (b)'s port width and (d)'s MSHR pool).

---

# Probe (f): `LDGSTS` / cp.async timing — latency, capacity, and streaming surface

Naming it into the suite: **(f) async-copy path**, two sub-probes — **F-L** (dependent-chain completion latency) and **F-T** ((b)-style streaming plateau), plus a shared **F-B** burst-issue readout that both consume. Everything is clock64-only at the v2.1 lock (1350 MHz SM / 10501 MHz mem, persistence on); NCU appears only in `cross_checks` as verdict, never as source.

Operational definitions first, because the sim binds to them:

- **Completion latency** L_f(r,m,s): cycles from `LDGSTS` issue to the earliest cycle a `cp.async.wait_group 0`-gated dependent shared load returns correct data, *minus* the shared-load hop itself. Residency r ∈ {L2, DRAM}, modifier m ∈ {.ca, .cg}, size s ∈ {4,8,16} B.
- **Issue cost** c_iss: LSU-slot occupancy per `LDGSTS` warp-instruction.
- **In-flight capacity** C_f: the admission limit (unit: groups or bytes; scope: warp, partition, or SM) beyond which issue backpressures.
- **Streaming ceiling** BW_f(r,m,s): B/SM-cycle at plateau, one SM.

One legality note that constrains every sweep matrix below: PTX only permits `cp.async.cg` at cp-size 16; `.ca` takes 4/8/16. The matrix is therefore `.ca`×{4,8,16} ∪ `.cg`×{16} — five cells, not six. Encode this in `conditioning`, don't let the harness generator silently skip cells.

---

## 1. Sub-probe F-L — dependent-chain completion latency

### Construction

A Sattolo-permutation pointer chase where the *chased value travels through the async path*: each element of the global array holds the index of the next; the copy lands it in shared memory; the dependent shared load produces the next global address. Reuse the suite's chase generator **verbatim** (same 3 MB / 384 MB footprints, same ≥128 B element spacing, same permutation) so TLB and slice-hash conditioning cancels in the subtraction against the demand-chase baselines (243.6 / 422.6).

One correction to the sketch in the brief: **the primary chain needs no `__syncthreads()`**. Per the PTX memory model, after `cp.async.wait_group 0` the async writes are visible *to the executing thread*; a barrier is only required for cross-thread visibility. Since the chase thread consumes its own copy, the bar-free chain is legal and gives an uncontaminated read. The bar-including variant is kept as a *secondary* cell so the sim gets the wait→bar→consume figure it actually needs for CTA-cooperative kernels — and so bar cost is measured as a separable delta rather than baked in.

```cuda
// F-L core, .cg/16B cell shown; one active thread, others parked past a branch.
// Element = uint4, .x holds next index; elements on distinct 128B lines.
__global__ void fL_chase(const uint4* __restrict__ g, long iters,
                         unsigned long long* t_out, unsigned* sink) {
  __shared__ uint4 slot[4];                       // rotate to kill WAR pressure
  if (threadIdx.x) return;
  unsigned idx = 0;
  unsigned s0 = (unsigned)__cvta_generic_to_shared(&slot[0]);
  for (int w = 0; w < WARMUP; ++w) { /* one full permutation pass, untimed */ }
  unsigned long long t0 = clock64();
  #pragma unroll 1
  for (long i = 0; i < iters; ++i) {
    unsigned sa = s0 + ((i & 3) << 4);
    asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n\t"
      "cp.async.commit_group;\n\t"
      "cp.async.wait_group 0;\n\t"
      :: "r"(sa), "l"(g + idx) : "memory");
    idx = ((volatile uint4*)slot)[i & 3].x;       // dependent ld.shared
  }
  unsigned long long t1 = clock64();
  *t_out = t1 - t0; *sink = idx;                  // consume: defeat elision
}
```

Run at K and 2K iterations; slope is the per-hop figure (intercept absorbs clock64 and loop overhead).

### Sweep matrix

| axis | cells |
|---|---|
| residency | L2 (3 MB), DRAM (384 MB) |
| modifier×size | .ca×{4,8,16}, .cg×16 |
| barrier | none (primary), `bar.sync` after wait (secondary) |
| re-copy | fresh lines vs re-copy of just-copied line (L1-allocation test for .ca vs .cg) |

### What clock64 measures, and the algebra

Per-hop T_F(r,m,s) = c_iss + c_cw + L_mem(r,m) + c_fill + T_sh, where c_cw is commit+wait static cost, c_fill the shared-write/scoreboard-release cost, T_sh=23.0 the known shared hop (same chase convention, so the address arithmetic convention matches). Extract:

    L_f(r,m,s) = T_F(r,m,s) − T_sh,      Δ(r) = T_F(r) − T_sh − T_demand(r)

with T_demand ∈ {243.6, 422.6} re-measured in-session as canaries.

**Expected signature:** Δ(L2) ≈ Δ(DRAM) — a residency-invariant fixed overhead (issue+commit+wait+fill minus the demand pipe's writeback/wake cost), plausibly a few tens of cycles. That invariance is itself a two-equation/one-unknown overdetermination. If Δ differs by residency, the async path routes differently through the hierarchy (e.g., .cg's L1 bypass shaving a lookup) — report per-residency L_f and, if unattributable, type it (§6). The re-copy cells give the modifier semantics without counters: a `.ca` re-copy landing near T_sh+Δ+(L1-class latency) ≈ the 33.0-class confirms L1 allocation; `.cg` re-copy staying in the 243.6 class confirms bypass.

### F-B burst-issue readout (shared by both sub-probes)

One warp issues K *independent* `LDGSTS` (distinct lines, distinct smem slots, fully unrolled), then a single `commit_group; wait_group 0`. T(K) = a + c_iss·K below capacity; the intercept ≈ L_f; **a slope break in T(K) is a latency-side capacity readout** — the third, independent, look at C_f. Sweep K ∈ {1…64} × size {4,16}: if c_iss is size-invariant the cost is a dispatch slot; if it scales with s it's port-metered at issue.

---

## 2. Sub-probe F-T — streaming plateau and in-flight knee

### Construction

One block on one SM ((b)'s isolation), W warps, each streaming its slice of the footprint into a shared-memory ring of `STAGES` per-warp stage buffers, `commit_group` per stage, steady-state gated by `wait_group N`. Correctness rule that also polices confound #2: a stage may be reused only when its prior group has drained, which holds iff `STAGES ≥ N+2` (use exactly N+2 so software depth never silently exceeds the wait bound). Consumption: one lane XORs one word per drained stage into a register accumulator (≈0 bandwidth, defeats elision), written out after timing. Warp 0 takes clock64 inside `bar.sync` fences; run ≥10 ms so bar cost is noise.

```cuda
template<int W, int N, int S /*=N+2*/, int SZ>
__global__ void fT_stream(const char* __restrict__ g, size_t chunks,
                          unsigned long long* t_out, unsigned* sink) {
  extern __shared__ char ring[];             // S stages × W × 32 × SZ, 128B-padded/warp
  int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // prologue: fill N stages, commit each
  // steady state, per warp:
  //   for c in [N, chunks):
  //     cp.async.wait_group N;                       // oldest stage now free
  //     consume-XOR one word of stage (c - N) % S;
  //     cp.async [stage (c % S), lane*SZ], [g + slice(w,c) + lane*SZ], SZ;
  //     cp.async.commit_group;
  //   cp.async.wait_group 0;  epilogue-consume
}
```

### Sweep matrix

| axis | cells | purpose |
|---|---|---|
| W | 1,2,4,8,16,24,32 (1 block); 48 via 2×24 (flag conditioning) | issue vs port vs capacity separation |
| N | 0,1,2,4,6,8,12,…, up to smem limit | knee: group-count structure |
| size×modifier | .ca×{4,8,16}, .cg×16 | issue-limited vs port-limited: plateau flat in s ⇒ port; ∝ s ⇒ issue |
| residency | L2 3 MB, DRAM 384 MB (pass-wise reshuffled) | SM↔XBAR port vs DRAM pull |
| hybrid | (W_async, W_demand) grid, §4 | pool/port sharing |

Note the smem geometry: at W=32, s=16 a stage is 16 KB, so ≤5 stages fit in the ~99 KB block carveout — deep-N cells must run at moderate W (e.g., W=8 → 24 stages) and full-W cells at shallow N. This is a designed-in censoring boundary; if the knee lies beyond it, that's the `CENSORED` branch of §6, not a silent extrapolation.

### What clock64 measures & expected signatures

Achieved B/SM-cycle = (W · chunks · 32 · SZ)/Δclock64. The surface should obey

    BW = min( issue: (32·s/c_iss)·min(W, W_iss),  wire: P_port,  Little: C_f / L_f(r) )

Diagnostics: plateau flat in s and equal to (b)'s B_b ⇒ **shared L1↔XBAR port**; knee position in N invariant to W ⇒ per-warp group limit; knee in W·N·bytes ⇒ SM-level byte pool. The DRAM-residency cells answer a question (d) can't: whether cp.async sustains deeper per-SM DRAM pull than the demand MSHR pool allows — a first-order constant for streaming GEMM epilogue-free main loops.

---

## 3. Disentanglement from (b) and (d) — overdetermination map

Three contrasts, so every verdict has ≥2 independent readouts:

**Port sharing (vs b).** (i) Compare BW_f(L2, .cg16) to B_b. (ii) Hybrid kernel: W_a async warps + W_d demand-load streaming warps; if the sum of achieved bandwidths is capped at max(B_b, BW_f) ⇒ one port; if it approaches B_b + BW_f ⇒ separate paths. (iii) Embedded-RTT: a single chase warp of (b)-type inside async saturation — RTT inflation onset at the shared-port saturation point corroborates (ii).

**Pool sharing (vs d).** (i) Compare C_f (bytes) to (d)'s MSHR capacity M_d·32 B — equality is suggestive only. (ii) Hybrid occupancy: infer each side's in-flight bytes via Little (BW_i · L_i); if inflight_a + inflight_d ≈ const ≈ (d)'s pool as W_a grows ⇒ shared; if demand holds M_d while async holds C_f on top ⇒ separate. (iii) Demand-chase RTT under async load: queueing inflation before port saturation ⇒ shared trackers.

**Capacity value C_f itself:** N-knee (F-T), W-knee (F-T), burst slope-break (F-B), and the Little product BW_plateau·L_f — four readouts; accept the constant only if ≥2 agree within ~15%, else type a gap.

---

## 4. Card constants and the sim `LDGSTS` model

| constant | source readouts | sim binding |
|---|---|---|
| `f1 ldgsts_issue_cycles(m,s)` | F-B slope; F-T issue-limited plateau (.ca4: 32·4/BW) | LSU port occupancy per warp-instr |
| `f2 ldgsts_complete_latency{L2,DRAM}(m,s)` + fixed Δ over demand | F-L; Little inversion C_f/BW; bar-variant additivity | completion-event scheduling |
| `f3 ldgsts_inflight_capacity{value,unit,scope,shared_with_mshr}` | N-knee, W-knee, F-B break, BW·L | admission/backpressure at issue |
| `f4 ldgsts_stream_bw{L2,DRAM}` + `shares_l1_port` | F-T plateau; predicted C_f/L_f; contrast with B_b | consistency check; port accounting |
| `f5 wait_bar_delta` | bar vs no-bar F-L cells | `wait_group`+bar unblock cost |

Event-model shape replacing the "bypass load to L2" pricing: at issue, an `LDGSTS` occupies the LSU for `f1`, must win an entry in pool `f3` (the (d) pool if `shared_with_mshr`, else its own — stall the warp on exhaustion), schedules completion at `f2[residency]`, and its bytes debit the (b) port if `shares_l1_port` (at (a)'s sector granularity, keeping the probes composable). `commit_group` snapshots pending copies into a group; `wait_group N` blocks the warp until ≤N groups pending. This is exactly where the undershoot should live: in cutlass-style multistage pipelines the per-stage time is max(mma, copy) plus wait-release, and a sim that prices `LDGSTS` as free-ish never stalls at `wait_group`.

---

## 5. Confound table

| # | confound | defeat | verification signal | residual |
|---|---|---|---|---|
| 1 | smem bank conflicts at the fill sink throttle plateau | 128 B-aligned per-warp stage slots, lane-contiguous 16 B writes (conflict-free wavefronts) | deliberately conflicted canary layout must show a drop; insensitivity ⇒ harness suspect | conflicts internal to the LDGSTS fill port are invisible; note on card |
| 2 | `wait_group` over-serialization fakes the knee | STAGES = N+2 always; sweep N past suspected knee; require plateau non-decreasing and flat beyond knee | knee position reproduced by independent W-sweep and F-B break | knee beyond smem-max ⇒ `CENSORED`, never extrapolated |
| 3 | bar cost contaminates latency | bar-free same-thread chain is primary (PTX visibility rule) | \|T_bar−T_nobar−c_bar\|≤2 cy, with c_bar from an empty-bar loop | bar×wait interaction ⇒ report `f5` as measured pair, type `MIXTURE` if non-additive |
| 4 | compiler elides copy or falls back to LDG+STS | inline `asm volatile` + memory clobber; checksum consumed to global | SASS audit: `LDGSTS` opcodes, static count matches; NCU inst count as out-of-band verdict only | none material |
| 5 | residency drift (L2 set evicted / DRAM set cached) & TLB mismatch vs baselines | suite chase generator verbatim; 3 MB ≪ 6 MB L2, 384 MB ≫ L2; pass-wise reshuffle for F-T DRAM cells | per-cell latency histogram unimodal in the expected class (≤5% foreign-class mass); in-session demand canaries hit 243.6/422.6 | slice-hash pathologies inherited from (e)'s permutation defeat |
| 6 | v2.1 lock drift | NVML poll pre/post each rep: 1350/10501, throttle-reason mask clean | shared-chase canary 23.0±0.5 brackets each session; reject rep otherwise | — |
| 7 | divergence/predication breaks per-thread group counters | convergent warps, no predicated cp.async, lane-uniform control | per-lane checksum equality | — |
| 8 | W=48 needs 2 blocks (occupancy contamination) | primary cells W≤32 single-block; 48-warp cells flagged | plateau continuity across the 32→48 seam | conditioning recorded, not hidden |

---

## 6. Typed-gap fallback and provenance record

Triggers, in decreasing specificity: plateau still rising at the smem-max ring ⇒ `CENSORED` (lower bound C_f ≥ tested in-flight bytes; censoring mechanism = 99 KB carveout, recorded). Knee readouts disagree on unit (N-knee says groups, W·bytes-knee says bytes, F-B ambiguous) ⇒ `UNRESOLVED_STRUCTURE` with both candidate parameterizations and their bounds; sim consumes the *conservative* (more-stalling) one, flagged. Soft knee but consistent unit ⇒ `RANGE` [C_lo, C_hi] from the 10%/90% plateau crossings. Bimodal F-L histograms not attributable to residency ⇒ `MIXTURE` with class means/weights. Plateau matching none of the three predicted binders within 15% ⇒ `SURFACE_GAP` (surface can't be expressed by the card model; do not force a constant).

Record shape (one measured, one gap, same schema as a–e):

```json
{ "probe": "f/F-L", "quantity": "ldgsts_complete_latency_L2(.cg,16B)",
  "decision": "measured",
  "value": {"cycles": "<T_F - 23.0>", "delta_over_demand": "<Δ>", "units": "SM-cycles@v2.1"},
  "conditioning": {"modifier": ".cg", "size": 16, "residency": "L2/3MB", "barrier": "none",
                   "chase": "sattolo/128B-spaced, suite-generator@v2.1"},
  "operating_point": {"sm_mhz": 1350, "mem_mhz": 10501, "sms_active": 1, "driver": "<ver>"},
  "evidence": ["K/2K slope agreement", "Δ(L2)=Δ(DRAM) invariance", "Little inversion C_f/BW within 15%",
               "bar-additivity pass"],
  "confound_verdicts": {"1":"n/a","3":"pass","4":"pass(SASS: LDGSTS.E.BYPASS.128)","5":"pass","6":"pass"},
  "cross_checks": ["NCU lts sector count consistent with (a) granularity — verdict only"],
  "consumer_binding": ["sim.ldgsts.completion_latency.l2"] }

{ "probe": "f/F-T", "quantity": "ldgsts_inflight_capacity",
  "decision": "gap", "gap": {"type": "CENSORED", "bound": ">= <bytes> per SM",
  "mechanism": "smem carveout caps STAGES at W=32"},
  "conditioning": {"W": 32, "N_max_tested": 4, "size": 16, "modifier": ".cg", "residency": "DRAM/384MB"},
  "operating_point": {"sm_mhz": 1350, "mem_mhz": 10501, "sms_active": 1},
  "evidence": ["plateau monotone rising through N=4", "W-knee absent ≤32", "F-B break not reached at K=64"],
  "confound_verdicts": {"2":"censored-by-design","1":"pass","6":"pass"},
  "cross_checks": [], "consumer_binding": ["sim.ldgsts.pool.size (use bound, conservative)"] }
```

---

## 7. Ranking

Load-bearing order becomes **b → f → a → d → c → e**. Rationale: the 2.8× undershoot geomean is carried by the tensor-core cells, all of which move their A/B tiles through `LDGSTS` multistage pipelines; an unpriced `wait_group` critical path and unmodeled in-flight backpressure produce *exactly* a too-fast simulator on exactly those cells, whereas (a)'s granularity and (d)'s demand MSHRs bind those kernels only second-hand once cp.async is priced. Two caveats keep this honest: **execution order ≠ rank** — f's sharing verdicts consume (b)'s B_b and (d)'s M_d, so it *runs* after both (slot 4: b, a, d, f, c, e); and the #2 rank is provisional under the disentanglement doctrine — confirm it with a read-only attribution pass (swap only f's constants into the sim, measure per-cell delta on the GEMM/conv/FA rows; never the reverse fit). If that pass moves those cells by less than the residual error, demote f below d and the unattributed remainder goes back on the wall where it belongs.
