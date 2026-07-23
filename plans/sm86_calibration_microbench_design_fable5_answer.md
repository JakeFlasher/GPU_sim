# Calibration Probe Suite — Fable-5 design (sm_86)

> Provenance: Fable 5 answer (2026-07-08) to `plans/sm86_calibration_microbench_design_fable5_prompt.md`. Verbatim design portion extracted from `results.md` (thinking-block stripped). Referenced by `plans/sm86_comprehensive_live_capture_campaign_plan.md`.

---

# Calibration Probe Suite v2.1-ext: five clock64-only probes for GA102 card constants

**Shared operating-point arithmetic (used everywhere below).** At the lock (SM 1350 MHz, mem 10501 MHz), vendor-peak DRAM = 1008 GB/s ÷ 1.35 GHz = **746.7 B/SM-cycle GPU-aggregate** (per-SM fair share 8.9 B/cy). Your 668 B/cy observation = 89.5% of that. The latency landmarks in hand give a **hit/miss classifier contrast of ~179 cy** (243.6 L2 vs 422.6 DRAM); midpoint 333 cy, but every probe below carries *in-run* positive/negative controls and sets its threshold adaptively (control-hit p95 + 30 cy), because the compressed L2:DRAM contrast is exactly what killed your step detectors before. All timed loads are made compiler/HW-order-safe the same way as your existing `chase` kernel: the loaded value feeds the address of the next access (or is folded into the second `clock64()` via `+ (v & 0)`), and clock64 overhead (~2–6 cy, measured by back-to-back reads at run start) is subtracted and logged.

---

## Probe (a) — L2↔DRAM fetch granularity (32 B / 64 B pair / 128 B line)

**Question restated as a measurable:** after a demand miss on sector `A`, is the partner sector resident in L2 when probed *after A's MSHR has retired*? Report not a binary but a **mixture weight** `p_pair(δ, parity, class)` — this is what reconciles your NCU bracket (37.5% / 62.5% modeled, ~55% measured): a heuristic or class-dependent pair-fill produces a p strictly between the two pure models.

**Kernel / pattern.** Single lane, dependent chain over a 512 MB pool of single-use, randomly-ordered 128 B-aligned lines (each line touched in exactly one trial → cold at L2 by construction; a 64 MB scrub stream runs between init and probe to purge init residue). Per trial: demand `A` with `__ldcg` (`.cg` on sm_8x caches at L2 only — keeps L1 sectoring out of the experiment), spin a **dependent IMAD chain ≥ 1200 cy** (> DRAM RTT, so the partner probe cannot merge into A's in-flight MSHR and read out as a bogus "hit"), then issue the timed probe at `A+δ`:

```cuda
unsigned a = __ldcg(&p->self);                 // demand sector A -> DRAM
unsigned d = a;
#pragma unroll 1
for (int j = 0; j < 300; ++j) d = d*2654435761u + j;   // ~1.2-1.5k cy, reg-only
const unsigned* q = (const unsigned*)((const char*)p + delta + (d & 0)); // data-dep addr
uint64_t t0 = clock64();
unsigned b  = __ldcg(q);
uint64_t t1 = clock64() + (b & 0);             // serialize via data dep
lat[i] = t1 - t0;
p = next_dependent(pool, p, b);                // chain to next trial's A
```

**Sweep axes.** δ ∈ {0 (positive control, must read ≈244), +32, +64, +96, far-line (negative control, ≈423)}; **demanded-sector parity** A@{0,32,64,96} within the line, including δ = −32 — an *aligned-pair* policy fills the partner regardless of which half was demanded, a *directional next-sector prefetch* only fills forward, and the parity×δ matrix separates these two policy shapes cleanly; inter-access gap Δ ∈ {~300, ~1.2k, ~10k cy} (a real L2 fill persists at 10k cy; a read-combine/transient buffer decays); access order regular vs shuffled (a fixed +32 rhythm might *train* a heuristic — if p_pair drops when trial order is randomized, the policy is adaptive and you record that); access class {`ld.cg`, `ld.ca`, `st`, `cp.async`} since fill policy may be class-dependent.

**clock64 measures:** per-trial dependent-load latency samples → a latency histogram per (δ, parity, Δ, class) cell, ~64k trials/cell.

**Expected signature.** δ=+32 histogram is bimodal at ≈244 / ≈423 with mixture weight p_pair; δ=+64/+96 ≈ pure 423 unless full-line fill (then all ≈244). Pure 32 B granule → all probe δ pure DRAM. GDDR6X's 32 B channel atom makes 32 B *physically* available to the controller, so a 64 B answer is a *policy*, not a necessity — expect either a clean 0/1 or a genuinely intermediate p (your 55% NCU reading predicts p_pair ≈ 0.5–0.7 if the 62.5% model corresponds to p=1).

**Confounds & defeats.** DRAM row-buffer hits mimicking L2 hits: an open-row DRAM access saves ~40–55 cy → lands ≈370–380, still >45 cy above the adaptive threshold; verify by the negative control's histogram left tail. MSHR merge: killed by the ≥1200 cy dependent delay (and the Δ sweep proves it). L1 pollution: `.cg` everywhere + the δ=0 control (if δ=0 read ~33 cy, `.cg` leaked into L1 — abort). Compression: pool filled with xorshift noise. TLB/first-touch: full pre-touch pass + scrub before timing.

**Fallback (typed gap).** If the δ=+32 histogram smears rather than splitting bimodally, emit `MIXTURE_UNRESOLVED{granule ∈ [32B,64B], p_pair ∈ [p_lo,p_hi]}` with all histograms attached; the simulator runs both bracket models and reports both. The out-of-band NCU sector-hit cross-check (~55%) goes into the record as a *verdict*, never as the fitted p.

---

## Probe (b) — Per-SM L1↔XBAR port width (B/SM-cycle to L2)

**Kernel / pattern.** One block on one SM (rest of the GPU idle), streaming fully-coalesced `uint4` (`__ldcg`) over a footprint chosen to pin the backing level, with a compile-time unroll of ≥8 independent loads per thread so per-warp MLP is never the binder at low warp counts:

```cuda
uint4 acc = {0,0,0,0};
__syncthreads(); uint64_t t0 = (threadIdx.x==0) ? clock64() : 0; __syncthreads();
for (int it = 0; it < iters; ++it)
  #pragma unroll 8
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    uint4 v = __ldcg(&src[perm(i)]);           // 128B-aligned, coalesced
    acc.x += v.x; acc.y ^= v.y; acc.z += v.z; acc.w ^= v.w;
  }
__syncthreads(); if (threadIdx.x==0) t_out = clock64() - t0;
if (acc.x == 13) *sink = acc;                   // DCE guard, never taken
```

**Config / sweeps.** blockDim 32→1024 (1→32 warps); footprint ∈ {96 KB (L1-leak control), 3 MB (**L2-resident**, RTT 243.6), 384 MB (**DRAM-resident**, RTT 422.6 — safe single-SM, one SM can't approach the 746.7 B/cy aggregate)}; direction variants: read (`ld.cg`), write (`st.cg` full-sector streaming stores), copy — port width may be asymmetric; page-permuted vs linear order (slice-camping check).

**clock64 measures:** block-elapsed cycles around fixed work → achieved **B/SM-cy**; the warp-count rise curve and its plateau.

**Expected signature.** Rise then hard plateau in the {16, 32, 48, 64} B/cy family. If port = 32 B/cy, bandwidth-delay product = 32 × 243.6 ≈ 7.8 KB (L2) / 32 × 422.6 ≈ 13.2 KB (DRAM); with 8×16 B in flight per thread (4 KB/warp), knee at **~2 warps (L2) / ~4 warps (DRAM)** — saturation far below 32 warps is itself confirmation that the plateau is structural.

**The decisive disentanglement — RTT contrast.** Plateau(3 MB) vs plateau(384 MB): a true **port width is RTT-invariant in B/cy**; an outstanding-capacity limit obeys Little's law and scales as M·32 B/RTT, i.e. drops ~42% from L2 to DRAM footprint. Same B/cy at both RTTs ⇒ you measured the port. Scaling ⇒ you measured capacity; emit `CENSORED{port ≥ plateau_L2}` and hand the number to probe (d), which measures capacity properly.

**Confounds & defeats.** L1 leakage under `.cg`: the 96 KB control must NOT show >64 B/cy L1-class bandwidth. Issue-rate limit: at 32 B/cy target, one LDG.128 per SM per 16 cy — orders below the 4-scheduler issue capability; verified because plateau is reached at ≥2 warps, not 1. Slice camping: permuted page order; a dip only in linear order flags hash-stride resonance. Cross-SM noise: idle grid, NVML process check.

**Fallback.** No plateau by 32 warps → `CENSORED{port ≥ max_observed}` + rise curve. Plateau but RTT-scaling → typed handoff to (d) as above. Either way the sim gets a bound, never a default.

---

## Probe (c) — Achievable (not peak) DRAM bandwidth

**Kernel / pattern.** All-84-SM grid-stride streams over ≥1.5 GB (≫ 6 MB L2, random-fill to defeat GA10x bandwidth compression), `uint4` coalesced; three kernels: read (checksum), write (full-sector streaming stores — full 32 B sector writes need no fetch-on-write; measure and record separately since write-allocate behavior differs), copy. Occupancy: 4 blocks/SM × 512 threads (enough MLP that per-SM limits from (b)/(d) don't bind at the per-SM fair share of ~8.9 B/cy — this probe deliberately sits *below* per-SM ceilings so DRAM is the only binder).

**Timing.** Per-block `clock64()` at entry/exit, atomicMin(start)/atomicMax(end) to globals; runtime ≥ 100 ms so cross-SM clock64 skew (sub-µs) is <0.01% error; also record per-SM elapsed dispersion (must be <1%) as a self-check. Aggregate B/cy = bytes / (max_end − min_start).

**Sweeps.** Footprint 256 MB→2 GB (flat = L2-filter and TLB verified quiet); linear vs 2 MB-page-shuffled order (row-locality band: linear = upper, shuffled = lower — record both; the card constant is the linear sustained value, the shuffled value ships as the pessimistic band); read:write mix {1:0, 0:1, 1:1, 2:1} (the sim wants the mix curve, GDDR6X bus-turnaround costs make mixed < pure); **#active-SM sweep 1→84** — this is not optional decoration, it is the cross-check that welds (b) and (c) together (see disentanglement).

**Expected signature.** Read ≈ **660–680 B/cy** (88–91% of 746.7); write similar or a few % lower; 1:1 copy sum slightly below pure read. SM-sweep: linear rise at slope ≈ per-SM port from (b), kink at n* ≈ BW_achieved/port (≈ 21 SMs if port = 32 B/cy), flat after.

**Confounds & defeats.** Compression: random fill (also run an all-zeros labeled variant to *quantify* compression, kept out of the card constant). Thermal/DVFS: NVML 10 ms trace of sm_clock + throttle_reasons; abort trial on any non-idle reason (the lock makes this a verification, not a mitigation). ECC: n/a on GeForce GDDR6X; recorded as a verdict. L2 filtering: footprint-sweep flatness.

**Fallback.** If read/write/copy disagree beyond the row-locality band or NVML shows lock breaks, emit `RANGE{[shuffled, linear]}` with the full variant table. This probe rarely fails; its risk is mislabeling, not noise.

---

## Probe (d) — Per-SM outstanding-miss capacity (MLP limit)

Your single-thread `mshr` probe proved per-*thread* tracking ≥ 32 with no knee — it never stressed the per-SM pool. The fix is to make in-flight-per-SM a controlled product **W × k** and push it past the port's bandwidth-delay product.

**Kernel / pattern.** Per warp, lane 0 only (lanes 1–31 predicated off → **each in-flight load = exactly one 32 B sector**, no coalescing ambiguity), running k interleaved dependent pointer-chases over k private regions:

```cuda
if (lane != 0) return;
const Node* p[KMAX];
for (int j = 0; j < k; ++j) p[j] = heads[warp_id*k + j];
uint64_t t0 = clock64();
for (int h = 0; h < hops; ++h)
  #pragma unroll
  for (int j = 0; j < k; ++j) p[j] = (Node*)__ldcg((uintptr_t*)&p[j]->next);
uint64_t t1 = clock64() + ((uintptr_t)p[0] & 1);
per_warp_cy[warp_id] = t1 - t0;
```

**Config.** One SM active. W warps × k chains, sweep the grid W ∈ {1..48} (48 via two 768-thread blocks forced co-resident by a dynamic-shared-memory carveout so blocks/SM is pinned, not scheduler-lucky) × k ∈ {1,2,4,8,16} → in-flight 1…768. **TLB discipline:** each chain walks a random permutation of 128 B nodes inside its own 2 MB-aligned region — one large page per chain, ≤768 pages total (inside TLB reach), while total working set (up to 1.5 GB) swamps L2 so hops stay cold.

**clock64 measures:** per-warp elapsed / hops = effective hop latency L(W·k); derived per-SM miss throughput X = W·k / L.

**Expected signature.** L flat at ≈422.6 while W·k ≤ M, then linear inflation L = (W·k)·RTT/M; equivalently X plateaus at M/RTT. **Knee position in in-flight units = M_sector.** Critical pre-computed collision: if port = 32 B/cy, the port itself saturates at in-flight ≈ 32·RTT/32 B = **423 sectors (DRAM) / 244 (L2)**. So run the whole (W,k) surface at BOTH footprints: knee fixed in in-flight units across RTTs ⇒ true capacity M; knee moving ∝ RTT ⇒ the port binds first and MSHR capacity exceeds the BDP — in which case the correct card entry is `CENSORED{M ≥ 423 sectors; non-binding, port binds}`, which is *exactly* the verdict your simulator needs to stop letting a fictitious MSHR limit (or its absence) mask the port. Second axis: repeat with lane-0-per-warp replaced by 32-lane same-line coalesced chases vs 32-lane distinct-line chases — if knees differ ~by the sector-per-line factor, entries are line-granular with sector masks (report M_line and M_sector separately; the sim's MSHR model needs to know which it is).

**Confounds & defeats.** Scheduler issue: required issue ≤ X < 2 loads/cy — far under 4 schedulers. DRAM queueing inflating L before the knee: visible as super-linear L growth *without* an X plateau; the joint (L, X) fit separates queueing (both grow) from capacity (X hard-plateaus). TLB: the 2 MB-region design above; verify by re-running with 4 KB-page (cudaMallocHost-mapped) variant — if knees move, TLB was in play. L2 hits contaminating "DRAM" hops: mean-hop sanity check ≥ 400 cy.

**Fallback.** `SURFACE_GAP{(W,k)→L,X attached; binder = port|capacity|queueing verdict}` — the labeled surface is consumable by the sim as a bound either way.

---

## Probe (e) — L2 associativity / slice hash / replacement, clock64-first

Your `conflict` probe failed because it was L1-scoped and stride-based; index hashing defeats *stride* probes, but it does not defeat **eviction-set group testing**, which is hash-agnostic.

**Primitive kernels (host-driven loop).** (1) `timed_reload(P)`: dependent `__ldcg` of P bracketed by clock64 → hit/miss vs adaptive threshold; (2) `touch_chain(S)`: dependently-chained `__ldcg` walk of a candidate list (chained to prevent MSHR-merge ambiguity and enforce access order — order matters for replacement inference).

**Algorithm.** Candidate pool R = 48 MB (8× L2) of random lines. Test: load P → touch subset S → timed reload P. Group-test reduction (Vila-style): split S into A+1 groups, discard any group whose removal still evicts P, recurse → **minimal eviction set; its cardinality = associativity** (expect 16). O(A log|R|) tests, each a kernel launch — minutes, not hours. Repeat for ~10³ probe addresses P.

**Hash recovery, honestly bounded.** Minimal-set members co-map ⇒ each set of 16 addresses gives linear constraints on the (assumed XOR-linear) index+slice function. Hard confound: virtual→physical opacity. cudaMalloc gives 2 MB-frame-contiguous ranges, so bits 0–20 are trustworthy; solve the linear system restricted to bits <21 and emit the hash **as a partial function**: `UNRESOLVED_STRUCTURE{hash bits ≥21 unresolved; low-bit XOR matrix attached}`. Complement with a clock64-only *slice-camping* bandwidth scan (stream at stride 2^s, s = 7…20, measure aggregate B/cy as in (c)): strides that alias to few slices dent bandwidth → the dent pattern independently constrains hash periodicity without any physical-address knowledge.

**Replacement.** With a verified minimal set E (|E| = A): prime P, touch j ∈ {A−4…A+4} members, reload P → survival vs j (LRU: cliff at A; random: smooth (1−1/A)^j decay ≈ 37% at j=A; RRIP-family: scan-resistance — P survives long streams of never-reused lines). Permute touch orders to test order sensitivity. Emit the **survival curve + best-fit policy label + parameters**, not just a label.

**Confounds.** L1: `.cg` throughout. MSHR merge during touches: dependent chaining. Compression changing effective capacity: random data. The compressed L2:DRAM contrast: same adaptive-threshold classifier as (a) with in-run controls.

**Fallback.** If reduction won't converge to a stable cardinality: `RANGE{A ∈ [12,24]}` + convergence traces; hash gap as above (that partial result is still directly consumable by a slice-hash model restricted to the same bits). NCU `lts__t_sectors` per-slice counters may *validate* co-mapping out-of-band — recorded as a verdict, never fitted.

---

## Ranking by load-bearing value (reviewer order: a→b→d→c→e — partially reordered)

1. **(b) per-SM port width.** The 2.8× cycles wall is the headline error and your own attribution says per-SM limits bind before aggregate DRAM. A missing/too-generous per-SM ceiling is the single most probable cause of a *multiplicative* whole-sim speedup, and (b) is the cheapest decisive probe in the suite (hours, hard plateau signature). It also anchors the (c) SM-sweep identity and the interpretation of (d).
2. **(a) fetch granularity.** Closes the named 13.96 pp L2 wall, is the *only admissible source* (contested literature), and feeds the cycles wall too: granule 32 B vs 64 B is up to 2× DRAM bytes per miss in bandwidth-bound phases. The p_pair mixture output dissolves the 37.5/62.5/55 contradiction rather than picking a side. Reviewer had this #1; I demote it only because it cannot by itself move a 2.8× cycles error, while (b) can.
3. **(d) MSHR/MLP capacity.** Co-binder with (b) and its disambiguator — if (b)'s plateau RTT-scales, (d) is promoted to co-#1; run them as a pair regardless. Note its most likely outcome ("port binds first, M ≥ BDP") is still load-bearing: it licenses the sim to make MSHR non-binding with a measured bound.
4. **(c) achievable DRAM.** Replacing 1008 GB/s with ~668 B/cy is a ~10% correction — real but second-order against 2.8×, and your attribution explicitly says the binding limits are upstream. Its SM-sweep, however, is the consistency weld for (b) and should run early even if the headline constant lands late.
5. **(e) associativity/hash/replacement.** Polish on whatever L2-pp residual survives (a); most expensive; documented defaults (16-way) plausibly survive. Runs last.

One flag: your attribution names **async-copy (cp.async/LDGSTS) timing** among first-binders, and no probe here measures it. A sixth probe (dependent cp.async→`cp.async.wait_group`→consume latency, plus a (b)-style cp.async streaming plateau) is cheap to bolt onto (b)'s harness; without it, part of the 2.8× may stay unattributed.

## Disentanglement: real HW limit vs simulator-compensating error

The rule that prevents "plausible-but-useless" constants: **every card constant must be overdetermined by at least two probe-level readouts that agree under the hardware hypothesis and disagree under misattribution — and must never be tuned against the simulator's end-to-end output.** Concretely: (i) *Little's-law triangulation* — throughput plateau × RTT must equal knee position in in-flight bytes ((b)×(d)); (ii) *RTT contrast* — widths are B/cy-invariant across L2/DRAM footprints, capacities are in-flight-invariant, queueing is neither; (iii) *persistence/Δ contrast* in (a) — policy fills persist to 10k cy, combining artifacts don't; (iv) *composition identities with zero free parameters* — (c)'s SM-sweep must kink at n* = BW_c/port_b and single-SM streaming time must be predictable from port width alone within ~10% *before* any constant enters the simulator; (v) if the simulator then still mispredicts, that is a sim-model bug ticket, never a license to refit the constant. Cross-probe contradictions are recorded as typed contradictions in the ledger, not averaged away.

## Consolidated confound table

| Confound | Probes hit | Defeat | Verification signal |
|---|---|---|---|
| DVFS/thermal drift | all | v2.1 lock | NVML 10 ms trace, throttle_reasons == idle, else abort |
| L1 interference / `.cg` leakage | a,b,d,e | `__ldcg` everywhere | 96 KB-footprint control ≈ L2 latency/BW, never L1-class |
| MSHR merge / transient combine | a,e | ≥1200 cy dependent delay; Δ sweep | hit-rate invariant for Δ ∈ [1.2k,10k] cy |
| DRAM row-buffer variance | a,d | classification guard bands; randomized addresses | negative-control histogram right-mode ≥378 cy |
| Compressed L2:DRAM contrast | a,e | in-run positive/negative controls, adaptive threshold | bimodality (dip statistic) per run |
| L2 compression | a,c,e | xorshift data fill | zeros-fill labeled variant quantifies effect |
| ECC | c | n/a (GeForce) | recorded verdict |
| TLB | a,c,d | pre-touch; 2 MB-region-per-chain in (d) | 4 KB-page variant leaves knee unmoved |
| Sector/next-line prefetch training | a | shuffled trial order axis | p_pair(ordered) vs p_pair(shuffled) |
| Slice camping / hash-stride resonance | b,c | page-permuted order | linear-vs-permuted BW delta <2% |
| Cross-SM interference | b,d | single-block grids, exclusive GPU | NVML process list; per-run idle check |
| clock64 skew/overhead | c / all | ≥100 ms runs; overhead calibration | per-SM elapsed dispersion <1% |
| Compiler reorder / DCE | all | data-dependent addresses; checksum sink | SASS audit of the timed loop (one-time) |
| Write-allocate ambiguity | c,b-write | full-sector streaming stores, separate read/write constants | write vs read plateau recorded separately |

## Occupancy sweeps and the NCU carve-out

Multi-warp/occupancy sweeps are *essential* for (b) (1→32 warps, the rise curve is evidence) and (d) (the W×k surface is the measurement); (c) needs high fixed occupancy plus the 1→84 SM sweep; (a) and (e) are strictly single-warp dependent-latency probes (multi-warp would only add MSHR-merge confounds). **No probe requires NCU in the loop.** Two probes accept NCU as out-of-band validation verdicts: (a) against the ~55% sector-hit reading, (e) against per-slice `lts` counters for co-mapping — in both cases the NCU number lands in the provenance record as `cross_check.verdict`, never in the `value` field.

## Record / provenance shape

Gap types: `RANGE` (interval), `CENSORED` (one-sided bound + identified binder), `MIXTURE` (probabilistic parameter, e.g. p_pair), `UNRESOLVED_STRUCTURE` (partial function + resolved domain), `SURFACE_GAP` (labeled evidence surface). Example record:

```json
{
  "probe": "a_l2_dram_granule", "harness": "v2.1-ext", "quantity": "l2_fill_pair_prob",
  "decision": "measured", "value": {"granule_B": 64, "p_pair": 0.58, "ci95": [0.55, 0.61]},
  "conditioning": {"delta": 32, "parity": "both", "class": "ld.cg", "gap_cy": 1200},
  "units": "prob | bytes", 
  "operating_point": {"sm_mhz": 1350, "mem_mhz": 10501, "persistence": true,
                      "driver": "...", "vbios": "...", "gpu": "RTX3090Ti/GA102/sm_86"},
  "evidence": {"histograms": "a_hist.parquet", "controls": {"pos_cy": 244.1, "neg_cy": 421.8},
               "threshold_cy": 301, "bimodality_dip_p": 0.001, "trials_per_cell": 65536},
  "covariates": {"nvml_trace": "a_nvml.csv", "throttle": "none", "clock64_overhead_cy": 4},
  "confound_verdicts": {"mshr_merge": "pass(Δ-invariant)", "row_buffer": "pass",
                        "l1_leak": "pass", "prefetch_training": "p_ordered-p_shuffled=0.01",
                        "compression": "pass(random fill)", "tlb": "pass"},
  "cross_checks": {"ncu_out_of_band": {"metric": "l2 sector hit rate", "value": 0.55,
                   "verdict": "consistent", "fitted": false}},
  "composition_checks": [],
  "consumer_binding": "card.ga102.l2.dram_fetch"
}
```

A `bounded_gap` record swaps `value` for `gap: {type, lo, hi, binder}` and *must* carry the full evidence curve — the simulator brackets, reports both predictions, and the gap stays visible in every downstream error attribution instead of hardening into a silent default.
