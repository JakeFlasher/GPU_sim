//! Sectored L1 cache core — a lockstep port of the cache state machine in
//! `tools/cache_predictor/l1_warp_pool_walker.py` (`_access_once` +
//! `_apply_fills`), the model whose semantics the Phase-1 L1 predictor
//! validated. The Python module's warp-pool SCHEDULER is deliberately NOT
//! ported: it was a latency-proportional stand-in for warp interleaving,
//! and the sim's SM issue model provides the real interleaving (traced
//! placement + control-bits timing). This module is only the cache.
//!
//! Ported semantics (two-layer, 32 B sector / 128 B line):
//! - hit decision is sector-with-line-allocation: a hit needs the LINE
//!   resident in the (fully-associative, LRU) line pool AND the touched
//!   SECTOR already fetched into it;
//! - two distinguishable in-flight fill kinds, both consuming MSHR
//!   capacity: line-ALLOCATION fills keyed by line (any access to the line
//!   meanwhile MERGES by line — counts as a miss, ready at the fill, and
//!   the fill services every sector the merged accesses touched), and
//!   resident-line SECTOR fills keyed by (line, sector) (only the SAME
//!   missing sector merges; the line's other resident sectors still HIT);
//! - fills apply lazily in CHRONOLOGICAL order (line before sector only at
//!   exactly equal timestamps; same-kind ties in issue order): an in-flight
//!   line is not in the LRU until its fill time, and a sector fill whose
//!   line was evicted mid-flight is DROPPED (sector residency dies with
//!   the line);
//! - when every MSHR entry is busy the access STALLS and retries when the
//!   earliest fill (of either kind) frees an entry;
//! - the miss stream splits into `line_alloc_misses` (fresh allocation +
//!   the eviction it forces) and `sector_resident_misses` (partial-line
//!   sector fetch); invariant: `misses == line_alloc + sector_resident +
//!   merges`.
//!
//! v0 EXTENSIONS beyond the ported model (kept in separate counters so the
//! ported semantics stay auditable): global-class STORE sectors are
//! hit-TESTED against the same state without allocating on miss (NCU's
//! unified `l1tex__t_sector_hit_rate.pct` counts store sectors; the L1 is
//! write-through no-allocate for global stores in this v0 hypothesis), and
//! bypass/untagged sector counting for the unified-denominator emission.

use std::collections::{BTreeMap, HashMap};

use sim_arch::ArchId;

pub const SECTOR_BYTES: u32 = 32;
pub const SECTORS_PER_LINE: u32 = 4;
pub const LINE_BYTES: u32 = SECTOR_BYTES * SECTORS_PER_LINE;

/// Configuration (card-resolved by `pchase_card`; latencies are integer
/// cycles — the caller ceils the card's fractional cycle values).
#[derive(Debug, Clone, Copy)]
pub struct L1Config {
    /// Deployment architecture whose geometry/latency hypotheses produced this config.
    pub arch: ArchId,
    pub line_capacity: u64,
    /// `None` = unlimited (ablation only; the card/documented default is
    /// always finite in production).
    pub mshr_count: Option<u32>,
    pub hit_latency: u64,
    pub miss_latency: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum L1ConfigError {
    #[error("L1 cache geometry is unsupported for deployment architecture {arch}")]
    ArchitectureUnsupported { arch: ArchId },
    #[error("line_capacity must be positive")]
    CapacityZero,
    #[error("mshr_count must be positive when finite")]
    MshrZero,
    #[error("hit latency {hit} must be strictly below miss latency {miss}")]
    LatencyOrderInverted { hit: u64, miss: u64 },
}

/// Outcome of one eligible sector access (after any MSHR stall retries).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessOutcome {
    Hit,
    MissLineAlloc,
    MissSectorResident,
    Merge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L1Counters {
    /// Architecture provenance bound by the owning [`L1Cache`]. `None` is
    /// reserved for synthetic/unbound snapshots and is rejected at emission
    /// or aggregation boundaries.
    arch: Option<ArchId>,
    /// Ported counters (modeled, manifest-eligible accesses only).
    pub hits: u64,
    pub misses: u64,
    pub merges: u64,
    pub stall_events: u64,
    pub line_fills: u64,
    pub line_alloc_misses: u64,
    pub sector_resident_misses: u64,
    /// Manifest-tagged ineligible sectors (the ported bypass counter).
    pub bypass_sectors: u64,
    /// Subset of `bypass_sectors` whose cache-op token is ATOM (the
    /// write-class bypass stream). Kept as a subset counter so the
    /// emission layer can fold the READ-class bypass stream (LDGSTS)
    /// into the global-load sector surface without pulling atomics in
    /// with it — NCU books those under op_atom/op_red, not op_ld.
    pub bypass_atom_sectors: u64,
    /// v0 extensions for the unified-denominator emission.
    pub untagged_sectors: u64,
    pub store_sectors: u64,
    pub store_hits: u64,
    /// §6.4 write-class rule at the L1 probe: store sectors whose LINE
    /// tag was resident regardless of the sector's valid bit (a
    /// superset of `store_hits`). Counted alongside — the sector-valid
    /// `store_hits` stays the v0 unified diagnostic until l1tex store
    /// accounting is scored against labels and one rule wins.
    pub store_tag_hits: u64,
}

impl Default for L1Counters {
    fn default() -> Self {
        Self {
            arch: None,
            hits: 0,
            misses: 0,
            merges: 0,
            stall_events: 0,
            line_fills: 0,
            line_alloc_misses: 0,
            sector_resident_misses: 0,
            bypass_sectors: 0,
            bypass_atom_sectors: 0,
            untagged_sectors: 0,
            store_sectors: 0,
            store_hits: 0,
            store_tag_hits: 0,
        }
    }
}

impl L1Counters {
    pub fn for_arch(arch: ArchId) -> Self {
        Self {
            arch: Some(arch),
            ..Self::default()
        }
    }

    pub fn arch(&self) -> Option<ArchId> {
        self.arch
    }

    pub fn add_checked(
        &mut self,
        other: &L1Counters,
    ) -> Result<(), sim_arch::ArchitectureProvenanceError> {
        let expected = self
            .arch
            .ok_or(sim_arch::ArchitectureProvenanceError::Unbound {
                component: "L1 counters",
            })?;
        let got = other
            .arch
            .ok_or(sim_arch::ArchitectureProvenanceError::Unbound {
                component: "L1 counters",
            })?;
        if expected != got {
            return Err(sim_arch::ArchitectureProvenanceError::Mismatch {
                component: "L1 counters",
                expected,
                got,
            });
        }
        self.hits += other.hits;
        self.misses += other.misses;
        self.merges += other.merges;
        self.stall_events += other.stall_events;
        self.line_fills += other.line_fills;
        self.line_alloc_misses += other.line_alloc_misses;
        self.sector_resident_misses += other.sector_resident_misses;
        self.bypass_sectors += other.bypass_sectors;
        self.bypass_atom_sectors += other.bypass_atom_sectors;
        self.untagged_sectors += other.untagged_sectors;
        self.store_sectors += other.store_sectors;
        self.store_hits += other.store_hits;
        self.store_tag_hits += other.store_tag_hits;
        Ok(())
    }
}

enum SingleOutcome {
    Done(AccessOutcome, u64),
    Stall { retry_at: u64 },
}

/// The one shared classification result of [`L1Cache::access_core`];
/// each fill-timing mode maps it to its own public outcome type.
enum CoreAccess {
    Hit,
    /// Merged into an in-flight line allocation; `pending` is the
    /// entry's stored fill time (real in constant mode, the sentinel in
    /// event mode).
    LineMerge {
        pending: u64,
    },
    /// Merged into an in-flight same-sector fetch.
    SectorMerge {
        pending: u64,
    },
    Stalled,
    MissLineAlloc,
    MissSectorResident,
}

/// Which fill-timing discipline an [`L1Cache`] instance runs under —
/// latched on first use so the two families can never mix on one
/// instance (they share the in-flight entry maps).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FillTiming {
    /// Fill times known at issue (`now + miss_latency`), applied lazily.
    Constant,
    /// Fill times decided by the next level; entries carry `u64::MAX`
    /// until [`L1Cache::complete_fill`].
    Event,
}

/// Identifies one in-flight fill for the event-driven mode: the key the
/// next level's RESPONSE must complete. Line-allocation fills are keyed
/// by line; resident-sector fills by (line, sector) — the same two MSHR
/// kinds as the constant-latency mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FillKey {
    Line(i64),
    Sector(i64, u8),
}

/// Outcome of one eligible sector access in EVENT mode: fill times are
/// unknown at issue (the shared L2 decides them), so misses surface as
/// requests and the caller wires completions back via [`L1Cache::complete_fill`].
/// The fill request fetches the ACCESSED 32 B sector; sectors that merge
/// into an outstanding line allocation are delivered to the L2 exactly
/// once per fill BY THE CALLER (the §6.4 slice-2 machinery in sim-sm:
/// same-batch merges fold into the fill request's sector mask, later
/// merges send one follow-up fetch — the pre-slice-2 undercount is
/// FIXED there, not here). The ported L1 still installs the whole
/// merged mask at one fill event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvAccess {
    /// Sector hit: data ready at a known time.
    Hit { ready: u64 },
    /// Fresh miss: send one request to the next level; the response must
    /// call [`L1Cache::complete_fill`] with this key.
    MissRequest { fill: FillKey },
    /// Merged into an ALREADY outstanding fill: no new request; the
    /// merging warp waits on the same key's completion.
    MergedPending { fill: FillKey },
    /// Every MSHR entry is busy: the caller parks the access and retries
    /// after the next completion frees an entry (the event-mode analogue
    /// of the constant mode's retry-at-earliest-fill).
    Stalled,
}

pub struct L1Cache {
    cfg: L1Config,
    /// Fully-associative LRU line pool: monotonic recency stamps.
    by_stamp: BTreeMap<u64, i64>,
    stamp_of: HashMap<i64, u64>,
    next_stamp: u64,
    /// Per-line fetched-sector bitmask (bit = sector index in line, 0..4).
    /// Also accumulates for in-flight line allocations (the fill services
    /// every merged sector).
    sectors: HashMap<i64, u8>,
    /// In-flight fills: value = (fill_time, issue_seq) — the seq keeps
    /// same-timestamp same-kind fills in issue order like the Python
    /// stable sort.
    line_fills: HashMap<i64, (u64, u64)>,
    sector_fills: HashMap<(i64, u8), (u64, u64)>,
    fill_seq: u64,
    /// Lower bound on the earliest in-flight fill time: `apply_fills`
    /// early-outs when `now` has not reached it (the overwhelmingly
    /// common case on the hot path). Maintained as a min on insert and
    /// re-derived after applying.
    min_fill_due: u64,
    /// Latched fill-timing discipline (see [`FillTiming`]).
    fill_timing: Option<FillTiming>,
    pub counters: L1Counters,
}

impl L1Cache {
    pub fn new(cfg: L1Config) -> Result<Self, L1ConfigError> {
        if cfg.arch != ArchId::Sm86 {
            return Err(L1ConfigError::ArchitectureUnsupported { arch: cfg.arch });
        }
        if cfg.line_capacity == 0 {
            return Err(L1ConfigError::CapacityZero);
        }
        if cfg.mshr_count == Some(0) {
            return Err(L1ConfigError::MshrZero);
        }
        if cfg.hit_latency >= cfg.miss_latency {
            return Err(L1ConfigError::LatencyOrderInverted {
                hit: cfg.hit_latency,
                miss: cfg.miss_latency,
            });
        }
        let arch = cfg.arch;
        Ok(Self {
            cfg,
            by_stamp: BTreeMap::new(),
            stamp_of: HashMap::new(),
            next_stamp: 0,
            sectors: HashMap::new(),
            line_fills: HashMap::new(),
            sector_fills: HashMap::new(),
            fill_seq: 0,
            min_fill_due: u64::MAX,
            fill_timing: None,
            counters: L1Counters::for_arch(arch),
        })
    }

    pub fn arch(&self) -> ArchId {
        self.cfg.arch
    }

    fn mshr_in_use(&self) -> usize {
        self.line_fills.len() + self.sector_fills.len()
    }

    fn earliest_fill(&self) -> u64 {
        self.line_fills
            .values()
            .chain(self.sector_fills.values())
            .map(|&(t, _)| t)
            .min()
            .expect("earliest_fill called with fills in flight")
    }

    fn touch_mru(&mut self, line: i64) {
        if let Some(old) = self.stamp_of.remove(&line) {
            self.by_stamp.remove(&old);
        }
        let stamp = self.next_stamp;
        self.next_stamp += 1;
        self.by_stamp.insert(stamp, line);
        self.stamp_of.insert(line, stamp);
    }

    fn evict_beyond_capacity(&mut self) {
        while self.by_stamp.len() as u64 > self.cfg.line_capacity {
            let (&stamp, &line) = self
                .by_stamp
                .first_key_value()
                .expect("non-empty by capacity check");
            self.by_stamp.remove(&stamp);
            self.stamp_of.remove(&line);
            // Partial-line eviction: sector residency dies with the line.
            self.sectors.remove(&line);
        }
    }

    fn line_resident(&self, line: i64) -> bool {
        self.stamp_of.contains_key(&line)
    }

    /// Replay due fills in CHRONOLOGICAL order; each fill's effect is
    /// decided by the cache state AT ITS OWN FILL TIME (line-before-sector
    /// only at exactly equal timestamps; same-kind ties in issue order).
    ///
    /// Integer-time note (deliberate divergence from the float-timed
    /// Python walker): the sim's clock is integer cycles, so fills whose
    /// float times would differ inside one cycle land on the same
    /// integer timestamp and resolve by the (kind, issue-order) tie rule
    /// instead of sub-cycle chronology. Bit-parity with Python holds
    /// whenever both sides run integer latencies (proven by the gate's
    /// differential); fractional-latency sub-cycle ordering is below the
    /// engine's resolution by design.
    fn apply_fills(&mut self, now: u64) {
        if now < self.min_fill_due {
            return;
        }
        let mut due: Vec<(u64, u8, u64, i64, u8)> = Vec::new();
        for (&line, &(t, seq)) in &self.line_fills {
            if t <= now {
                due.push((t, 0, seq, line, 0));
            }
        }
        for (&(line, sector), &(t, seq)) in &self.sector_fills {
            if t <= now {
                due.push((t, 1, seq, line, sector));
            }
        }
        due.sort_unstable();
        for (_, kind, _, line, sector) in due {
            if kind == 0 {
                self.line_fills.remove(&line);
                self.touch_mru(line);
                self.evict_beyond_capacity();
            } else {
                self.sector_fills.remove(&(line, sector));
                if self.line_resident(line) {
                    // The sector arrives into its (still-resident) line
                    // and re-references it at MRU.
                    *self.sectors.entry(line).or_default() |= 1 << sector;
                    self.touch_mru(line);
                }
                // else: the line was evicted before this fill's own
                // timestamp — the fill is DROPPED.
            }
        }
        self.min_fill_due = self
            .line_fills
            .values()
            .chain(self.sector_fills.values())
            .map(|&(t, _)| t)
            .min()
            .unwrap_or(u64::MAX);
    }

    /// One classification/state step shared by BOTH fill-timing modes —
    /// the single copy of the ported ladder (line-merge → sector-present
    /// hit → sector-merge → MSHR stall → fresh miss), with all counter
    /// accounting inside. `fill_time` is the timestamp a fresh miss's
    /// in-flight entry carries: `now + miss_latency` in constant mode,
    /// the `u64::MAX` sentinel in event mode (the lazy applier ignores
    /// it; `complete_fill` installs at the response event).
    fn access_core(&mut self, sector_id: i64, fill_time: u64) -> CoreAccess {
        let line = sector_id >> 2;
        let sector = (sector_id & 3) as u8;
        if let Some(&(pending, _)) = self.line_fills.get(&line) {
            // Access while the line ALLOCATION is in flight: MERGE by line
            // (the fill will install every merged sector).
            self.counters.merges += 1;
            self.counters.misses += 1;
            *self.sectors.entry(line).or_default() |= 1 << sector;
            return CoreAccess::LineMerge { pending };
        }
        let line_resident = self.line_resident(line);
        // Sector-present HIT testing precedes sector-merge handling.
        let sector_present = line_resident
            && self
                .sectors
                .get(&line)
                .is_some_and(|mask| mask & (1 << sector) != 0);
        if sector_present {
            self.counters.hits += 1;
            self.touch_mru(line);
            return CoreAccess::Hit;
        }
        if line_resident && let Some(&(pending, _)) = self.sector_fills.get(&(line, sector)) {
            // The SAME missing sector is already being fetched: MERGE.
            self.counters.merges += 1;
            self.counters.misses += 1;
            return CoreAccess::SectorMerge { pending };
        }
        // Miss path: needs an MSHR entry (both fill kinds consume capacity).
        if let Some(cap) = self.cfg.mshr_count
            && self.mshr_in_use() >= cap as usize
        {
            self.counters.stall_events += 1;
            return CoreAccess::Stalled;
        }
        self.counters.misses += 1;
        let seq = self.fill_seq;
        self.fill_seq += 1;
        self.min_fill_due = self.min_fill_due.min(fill_time);
        if line_resident {
            // Partial-line sector fetch: no allocation, no eviction; the
            // sector becomes resident at its FILL time (lazy).
            self.counters.sector_resident_misses += 1;
            self.sector_fills.insert((line, sector), (fill_time, seq));
            CoreAccess::MissSectorResident
        } else {
            self.counters.line_alloc_misses += 1;
            self.counters.line_fills += 1;
            self.line_fills.insert(line, (fill_time, seq));
            *self.sectors.entry(line).or_default() |= 1 << sector;
            CoreAccess::MissLineAlloc
        }
    }

    /// Latch the instance to one fill-timing mode on first use; mixing
    /// modes is a programming error (constant-mode entries carry real
    /// times the event mode would complete out of order, and event-mode
    /// sentinels would send the constant retry loop to `u64::MAX`).
    fn latch(&mut self, mode: FillTiming) {
        match self.fill_timing {
            None => self.fill_timing = Some(mode),
            Some(latched) => assert!(
                latched == mode,
                "L1Cache fill-timing modes must not mix on one instance \
                 (latched {latched:?}, called {mode:?})"
            ),
        }
    }

    fn access_single(&mut self, sector_id: i64, now: u64) -> SingleOutcome {
        self.apply_fills(now);
        let fill_time = now + self.cfg.miss_latency;
        match self.access_core(sector_id, fill_time) {
            CoreAccess::Hit => SingleOutcome::Done(AccessOutcome::Hit, now + self.cfg.hit_latency),
            CoreAccess::LineMerge { pending } | CoreAccess::SectorMerge { pending } => {
                SingleOutcome::Done(AccessOutcome::Merge, pending)
            }
            CoreAccess::Stalled => SingleOutcome::Stall {
                retry_at: self.earliest_fill(),
            },
            CoreAccess::MissSectorResident => {
                SingleOutcome::Done(AccessOutcome::MissSectorResident, fill_time)
            }
            CoreAccess::MissLineAlloc => {
                SingleOutcome::Done(AccessOutcome::MissLineAlloc, fill_time)
            }
        }
    }

    /// One modeled (manifest-eligible) sector access at time `now`.
    /// MSHR stalls retry internally at the earliest fill time (the ported
    /// walker's retry contract). Returns the outcome and the data-ready
    /// time.
    pub fn access_eligible(&mut self, sector_id: i64, now: u64) -> (AccessOutcome, u64) {
        self.latch(FillTiming::Constant);
        let mut t = now;
        loop {
            match self.access_single(sector_id, t) {
                SingleOutcome::Done(outcome, ready) => return (outcome, ready),
                SingleOutcome::Stall { retry_at } => t = retry_at.max(t + 1),
            }
        }
    }

    /// One modeled sector access in EVENT mode (shared-L2 integration):
    /// the fill time is decided by the next level, so misses register an
    /// UNTIMED in-flight entry (sentinel time `u64::MAX` — the lazy
    /// applier never applies it; [`Self::complete_fill`] installs it at
    /// the response event). Hit/miss/merge accounting matches
    /// [`Self::access_eligible`]; `stall_events` does NOT — the constant
    /// mode's internal retry-at-earliest-fill always succeeds (one stall
    /// per stalled access), while event-mode parking re-counts a stall
    /// on every failed re-attempt after a completion, so `stall_events`
    /// reads "stall ATTEMPTS" in this mode (deliberate: the retry
    /// discipline is the caller's, and no emitted metric consumes the
    /// counter yet).
    ///
    /// The two modes must not be mixed on one cache instance: constant-
    /// mode entries carry real times the event mode would complete out
    /// of order.
    pub fn access_eligible_ev(&mut self, sector_id: i64, now: u64) -> EvAccess {
        self.latch(FillTiming::Event);
        let line = sector_id >> 2;
        let sector = (sector_id & 3) as u8;
        match self.access_core(sector_id, u64::MAX) {
            CoreAccess::Hit => EvAccess::Hit {
                ready: now + self.cfg.hit_latency,
            },
            CoreAccess::LineMerge { .. } => EvAccess::MergedPending {
                fill: FillKey::Line(line),
            },
            CoreAccess::SectorMerge { .. } => EvAccess::MergedPending {
                fill: FillKey::Sector(line, sector),
            },
            CoreAccess::Stalled => EvAccess::Stalled,
            CoreAccess::MissSectorResident => EvAccess::MissRequest {
                fill: FillKey::Sector(line, sector),
            },
            CoreAccess::MissLineAlloc => EvAccess::MissRequest {
                fill: FillKey::Line(line),
            },
        }
    }

    /// Complete an event-mode fill AT the response event's time. A
    /// sector fill whose line was evicted while it was in flight is
    /// DROPPED, like the lazy applier; returns whether the fill
    /// installed.
    ///
    /// Deliberate divergence from the constant path: `apply_fills`
    /// breaks SAME-TIMESTAMP ties line-before-sector (a batch-applier
    /// artifact the Python lockstep pins), while event mode completes
    /// same-cycle fills in response order — under eviction pressure the
    /// two orders can pick different LRU victims. Sub-cycle completion
    /// order is below the model's resolution either way; the event
    /// order is the shared-L2 system's single source of truth.
    pub fn complete_fill(&mut self, fill: FillKey) -> bool {
        self.latch(FillTiming::Event);
        match fill {
            FillKey::Line(line) => {
                self.line_fills
                    .remove(&line)
                    .expect("completion for a line fill that is not in flight");
                self.touch_mru(line);
                self.evict_beyond_capacity();
                true
            }
            FillKey::Sector(line, sector) => {
                self.sector_fills
                    .remove(&(line, sector))
                    .expect("completion for a sector fill that is not in flight");
                if self.line_resident(line) {
                    *self.sectors.entry(line).or_default() |= 1 << sector;
                    self.touch_mru(line);
                    true
                } else {
                    false
                }
            }
        }
    }

    /// v0 extension: hit-TEST a global-class store sector against the
    /// current state without allocating on miss (write-through
    /// no-allocate hypothesis). A store hit re-references the line.
    /// Counts BOTH candidate hit rules — sector-valid (`store_hits`,
    /// the ported-era diagnostic) and the §6.4 write-class line-TAG
    /// rule (`store_tag_hits`) — without changing state behavior (MRU
    /// touch stays on the sector-valid hit only, preserving the L1
    /// state trajectory the scored op_ld exactness was proven on).
    pub fn probe_store(&mut self, sector_id: i64, now: u64) -> bool {
        let line = sector_id >> 2;
        let sector = (sector_id & 3) as u8;
        self.apply_fills(now);
        self.counters.store_sectors += 1;
        let tag_resident = self.line_resident(line);
        if tag_resident {
            self.counters.store_tag_hits += 1;
        }
        let hit = tag_resident
            && self
                .sectors
                .get(&line)
                .is_some_and(|mask| mask & (1 << sector) != 0);
        if hit {
            self.counters.store_hits += 1;
            self.touch_mru(line);
        }
        hit
    }

    /// Manifest-tagged ineligible sectors (the ported bypass counter).
    /// `atom` marks the write-class (ATOM-token) subset — the caller
    /// owns the token, this cache never sees opcodes.
    pub fn count_bypass(&mut self, sectors: u64, atom: bool) {
        self.counters.bypass_sectors += sectors;
        if atom {
            self.counters.bypass_atom_sectors += sectors;
        }
    }

    /// Global-class sectors whose pc the classifier did not tag (v0
    /// unified-denominator extension).
    pub fn count_untagged(&mut self, sectors: u64) {
        self.counters.untagged_sectors += sectors;
    }
}

/// Expand one traced address record's lane vector into UNIQUE global
/// sector ids, in first-seen order (lane-major, ascending sector within a
/// lane) — the `l1_trace_adapter` dedup contract: NCU's sector metric
/// counts unique 32 B sectors per warp INSTRUCTION, and emission order
/// feeds the LRU, so both the SET and the ORDER are ported. `dedup_from`
/// marks where the current instruction's sectors begin in `out`, so a
/// multi-address-record instruction dedups across ALL its records.
pub fn push_unique_sectors(
    lane_vector: &[i64; 32],
    active_lanes: impl Iterator<Item = usize>,
    data_width_bytes: i32,
    out: &mut Vec<i64>,
    dedup_from: usize,
) {
    let width = i64::from(data_width_bytes.max(1));
    for lane in active_lanes {
        let base = lane_vector[lane];
        let first = base >> 5;
        let last = (base + width - 1) >> 5;
        for sector in first..=last {
            if !out[dedup_from..].contains(&sector) {
                out.push(sector);
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn cache(capacity: u64, mshr: Option<u32>) -> L1Cache {
        L1Cache::new(L1Config {
            arch: ArchId::Sm86,
            line_capacity: capacity,
            mshr_count: mshr,
            hit_latency: 34,
            miss_latency: 244,
        })
        .unwrap()
    }

    fn counters_invariant(c: &L1Counters) {
        assert_eq!(
            c.misses,
            c.line_alloc_misses + c.sector_resident_misses + c.merges,
            "miss == line_alloc + sector_resident + merge"
        );
    }

    #[test]
    fn sector_with_line_allocation_hit_decision() {
        let mut l1 = cache(8, None);
        // Sector 0 of line 0: line-alloc miss; fills at t=244.
        let (o, ready) = l1.access_eligible(0, 0);
        assert_eq!(o, AccessOutcome::MissLineAlloc);
        assert_eq!(ready, 244);
        // After the fill: same sector hits, but sector 1 of the SAME
        // (now-resident) line is a sector-resident miss, not a hit.
        let (o, _) = l1.access_eligible(0, 300);
        assert_eq!(o, AccessOutcome::Hit);
        let (o, _) = l1.access_eligible(1, 300);
        assert_eq!(o, AccessOutcome::MissSectorResident);
        counters_invariant(&l1.counters);
    }

    #[test]
    fn line_merge_while_allocation_in_flight() {
        let mut l1 = cache(8, None);
        let (_, fill) = l1.access_eligible(0, 0);
        // Different sector of the same line while the allocation is in
        // flight: merge (counts as a miss), ready at the SAME fill time,
        // and the fill services the merged sector too.
        let (o, ready) = l1.access_eligible(1, 10);
        assert_eq!(o, AccessOutcome::Merge);
        assert_eq!(ready, fill);
        let (o, _) = l1.access_eligible(1, 250);
        assert_eq!(
            o,
            AccessOutcome::Hit,
            "merged sector serviced by the line fill"
        );
        counters_invariant(&l1.counters);
    }

    #[test]
    fn lazy_fill_is_not_a_hit_before_fill_time() {
        let mut l1 = cache(8, None);
        l1.access_eligible(0, 0);
        // Same sector before the fill time: merge, never an LRU hit.
        let (o, _) = l1.access_eligible(0, 100);
        assert_eq!(o, AccessOutcome::Merge);
    }

    #[test]
    fn same_missing_sector_on_resident_line_merges() {
        let mut l1 = cache(8, None);
        l1.access_eligible(0, 0);
        l1.apply_fills(244);
        // Sector 1 fetch in flight on the resident line...
        let (o, fill) = l1.access_eligible(1, 250);
        assert_eq!(o, AccessOutcome::MissSectorResident);
        // ...same sector merges; OTHER resident sector still hits.
        let (o, ready) = l1.access_eligible(1, 260);
        assert_eq!(o, AccessOutcome::Merge);
        assert_eq!(ready, fill);
        let (o, _) = l1.access_eligible(0, 260);
        assert_eq!(
            o,
            AccessOutcome::Hit,
            "sector fetch must not shadow its line"
        );
        counters_invariant(&l1.counters);
    }

    #[test]
    fn eviction_drops_sector_residency_and_inflight_sector_fill() {
        let mut l1 = cache(1, None);
        // Line 0 resident.
        l1.access_eligible(0, 0);
        l1.apply_fills(244);
        // Sector 1 of line 0 in flight (fills at 250+244=494)...
        l1.access_eligible(1, 250);
        // ...line 1 allocation evicts line 0 (capacity 1) at its own fill
        // time 251+244=495 > 494. The sector fill applies FIRST (chrono
        // order) into the still-resident line, then the line fill evicts.
        l1.access_eligible(4, 251);
        l1.apply_fills(1000);
        // Line 0 evicted: sector 0 must MISS as a fresh line allocation.
        let (o, _) = l1.access_eligible(0, 1000);
        assert_eq!(o, AccessOutcome::MissLineAlloc);
        counters_invariant(&l1.counters);
    }

    #[test]
    fn sector_fill_dropped_when_line_evicted_before_its_timestamp() {
        let mut l1 = cache(1, None);
        l1.access_eligible(0, 0); // line 0
        l1.apply_fills(244);
        // Line 1 alloc at t=245 → fills 489, evicting line 0 at 489.
        l1.access_eligible(4, 245);
        // Sector fetch into line 0 at t=246 → would fill at 490, AFTER
        // the eviction: must be dropped.
        let (o, _) = l1.access_eligible(1, 246);
        assert_eq!(o, AccessOutcome::MissSectorResident);
        l1.apply_fills(10_000);
        // Line 0 fully gone; re-access = fresh allocation, and the
        // dropped sector never resurrects.
        let (o, _) = l1.access_eligible(1, 10_000);
        assert_eq!(o, AccessOutcome::MissLineAlloc);
        counters_invariant(&l1.counters);
    }

    #[test]
    fn mshr_full_stalls_and_retries() {
        let mut l1 = cache(8, Some(1));
        let (_, fill) = l1.access_eligible(0, 0); // occupies the only MSHR
        // A different line at t=1 stalls until the fill frees the entry,
        // then misses: ready = fill_time + miss_latency.
        let (o, ready) = l1.access_eligible(8, 1);
        assert_eq!(o, AccessOutcome::MissLineAlloc);
        assert_eq!(ready, fill + 244);
        assert_eq!(l1.counters.stall_events, 1);
        counters_invariant(&l1.counters);
    }

    #[test]
    fn lru_eviction_order() {
        let mut l1 = cache(2, None);
        l1.access_eligible(0, 0); // line 0
        l1.access_eligible(4, 1); // line 1
        l1.apply_fills(300);
        l1.access_eligible(0, 300); // touch line 0 → MRU
        // Line 2 evicts the LRU line 1 (not line 0).
        l1.access_eligible(8, 301);
        l1.apply_fills(1000);
        let (o, _) = l1.access_eligible(0, 1000);
        assert_eq!(o, AccessOutcome::Hit);
        let (o, _) = l1.access_eligible(4, 1001);
        assert_eq!(o, AccessOutcome::MissLineAlloc);
        counters_invariant(&l1.counters);
    }

    #[test]
    fn store_probe_tests_without_allocating() {
        let mut l1 = cache(8, None);
        assert!(!l1.probe_store(0, 0), "cold store misses");
        // The store must NOT have allocated: a load still line-alloc
        // misses.
        let (o, _) = l1.access_eligible(0, 10);
        assert_eq!(o, AccessOutcome::MissLineAlloc);
        l1.apply_fills(300);
        assert!(l1.probe_store(0, 300), "store hits the resident sector");
        // A store to a DIFFERENT sector of the resident line: sector-
        // valid rule misses, §6.4 line-tag rule hits.
        assert!(!l1.probe_store(1, 300), "sector-valid rule still misses");
        assert_eq!(l1.counters.store_sectors, 3);
        assert_eq!(l1.counters.store_hits, 1);
        assert_eq!(
            l1.counters.store_tag_hits, 2,
            "tag rule counts the resident-line probes (both post-fill)"
        );
    }

    #[test]
    fn unique_sector_expansion_dedups_in_first_seen_order() {
        let mut lanes = [i64::MIN; 32];
        // 4 active lanes, 4B apart starting at 0: all in sector 0.
        for (i, l) in lanes.iter_mut().enumerate().take(4) {
            *l = (i * 4) as i64;
        }
        // Lane 4 crosses sectors 3→4 (base 127, width 4).
        lanes[4] = 127;
        let mut out = Vec::new();
        push_unique_sectors(&lanes, 0..5, 4, &mut out, 0);
        assert_eq!(out, vec![0, 3, 4]);
        // A second address record of the SAME instruction dedups against
        // the instruction's whole slice.
        push_unique_sectors(&lanes, 0..5, 4, &mut out, 0);
        assert_eq!(out, vec![0, 3, 4]);
    }

    #[test]
    #[should_panic(expected = "fill-timing modes must not mix")]
    fn mixing_fill_timing_modes_panics() {
        let mut l1 = cache(8, None);
        l1.access_eligible(0, 0); // latches Constant
        l1.access_eligible_ev(4, 1); // Event on the same instance: panic
    }

    #[test]
    fn ev_mode_miss_merge_completion_then_hit() {
        let mut l1 = cache(8, Some(4));
        // Fresh line miss → request keyed by line.
        let a = l1.access_eligible_ev(0, 0);
        assert_eq!(
            a,
            EvAccess::MissRequest {
                fill: FillKey::Line(0)
            }
        );
        // Another sector of the same line merges into the SAME fill.
        let b = l1.access_eligible_ev(1, 5);
        assert_eq!(
            b,
            EvAccess::MergedPending {
                fill: FillKey::Line(0)
            }
        );
        // Response installs line + both merged sectors; hits follow.
        assert!(l1.complete_fill(FillKey::Line(0)));
        assert!(matches!(
            l1.access_eligible_ev(0, 400),
            EvAccess::Hit { ready: 434 }
        ));
        assert!(matches!(
            l1.access_eligible_ev(1, 400),
            EvAccess::Hit { .. }
        ));
        // Sector 2 on the now-resident line: sector-resident request, and
        // the same missing sector merges while in flight.
        assert_eq!(
            l1.access_eligible_ev(2, 401),
            EvAccess::MissRequest {
                fill: FillKey::Sector(0, 2)
            }
        );
        assert_eq!(
            l1.access_eligible_ev(2, 402),
            EvAccess::MergedPending {
                fill: FillKey::Sector(0, 2)
            }
        );
        assert!(l1.complete_fill(FillKey::Sector(0, 2)));
        assert!(matches!(
            l1.access_eligible_ev(2, 900),
            EvAccess::Hit { .. }
        ));
        assert_eq!(l1.counters.hits, 3);
        assert_eq!(l1.counters.misses, 4);
        assert_eq!(l1.counters.merges, 2);
        assert_eq!(l1.counters.line_alloc_misses, 1);
        assert_eq!(l1.counters.sector_resident_misses, 1);
        counters_invariant(&l1.counters);
    }

    #[test]
    fn ev_mode_sector_fill_dropped_when_line_evicted_in_flight() {
        let mut l1 = cache(1, None);
        l1.access_eligible_ev(0, 0); // line 0 alloc
        l1.complete_fill(FillKey::Line(0));
        // Sector fetch in flight on line 0...
        l1.access_eligible_ev(1, 10);
        // ...line 1 allocation completes first and evicts line 0
        // (capacity 1).
        l1.access_eligible_ev(4, 11);
        assert!(l1.complete_fill(FillKey::Line(1)));
        // The sector response then finds its line gone: DROPPED.
        assert!(!l1.complete_fill(FillKey::Sector(0, 1)));
        assert_eq!(
            l1.access_eligible_ev(1, 500),
            EvAccess::MissRequest {
                fill: FillKey::Line(0)
            },
            "dropped sector never resurrects; the line itself was evicted"
        );
        counters_invariant(&l1.counters);
    }

    #[test]
    fn ev_mode_mshr_full_stalls_until_completion_frees_an_entry() {
        let mut l1 = cache(8, Some(1));
        l1.access_eligible_ev(0, 0); // occupies the only MSHR
        assert_eq!(l1.access_eligible_ev(8, 1), EvAccess::Stalled);
        assert_eq!(l1.counters.stall_events, 1);
        l1.complete_fill(FillKey::Line(0));
        assert_eq!(
            l1.access_eligible_ev(8, 2),
            EvAccess::MissRequest {
                fill: FillKey::Line(2)
            }
        );
        counters_invariant(&l1.counters);
    }

    #[test]
    fn counter_aggregation_rejects_mixed_architecture() {
        let mut sm86 = L1Counters::for_arch(ArchId::Sm86);
        let sm120 = L1Counters::for_arch(ArchId::Sm120);
        assert!(matches!(
            sm86.add_checked(&sm120),
            Err(sim_arch::ArchitectureProvenanceError::Mismatch {
                component: "L1 counters",
                expected: ArchId::Sm86,
                got: ArchId::Sm120,
            })
        ));
    }

    #[test]
    fn bypass_atom_subset_counter_tracks_only_the_atom_stream() {
        let mut l1 = cache(8, None);
        l1.count_bypass(3, false);
        l1.count_bypass(2, true);
        assert_eq!(l1.counters.bypass_sectors, 5);
        assert_eq!(l1.counters.bypass_atom_sectors, 2);
    }
}
