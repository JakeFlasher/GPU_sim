//! Cohesive child mechanism extracted from the parent module.

use super::*;

#[derive(Debug, Clone)]
pub struct L1Cache {
    pub(super) cfg: L1Config,
    pub(super) store_policy: L1StorePolicy,
    /// Fully-associative LRU line pool: monotonic recency stamps.
    pub(super) by_stamp: BTreeMap<u64, i64>,
    pub(super) stamp_of: HashMap<i64, u64>,
    pub(super) next_stamp: NextSequence,
    /// Per-line fetched-sector bitmask (bit = sector index in line, 0..4).
    /// Also accumulates for in-flight line allocations (the fill services
    /// every merged sector).
    pub(super) sectors: HashMap<i64, u8>,
    /// Explicit opt-in dirty-sector state. The default replay policy does not
    /// mark stores dirty; the resident-hit experiment uses this map and the
    /// transactional flush API below.
    pub(super) dirty: HashMap<i64, u8>,
    /// Optional known-byte payloads for the strict value-bearing flush lane.
    /// Ordinary replay remains mask-only until a producer supplies these.
    pub(super) dirty_payloads: HashMap<(i64, u8), SectorPayloadV1>,
    /// In-flight fills: value = (due state, issue_seq) — the seq keeps
    /// same-timestamp same-kind timed fills in issue order like the Python
    /// stable sort. Event fills remain explicitly untimed. Sector fills are
    /// keyed by line, sector, and allocation generation so an old response
    /// remains representable while a same-tag reallocation has already
    /// issued a new request for that sector.
    pub(super) line_fills: HashMap<i64, (FillDue, u64, u32)>,
    pub(super) sector_fills: HashMap<(i64, u8, u32), (FillDue, u64)>,
    /// Allocation-generation witnesses for event-mode fills.  A sector fill
    /// can outlive the resident line that spawned it; retaining the line's
    /// allocation sequence lets completion reject an old response after the
    /// same tag has been evicted and reallocated (the L2 model already uses
    /// the analogous generation guard).
    pub(super) line_generations: HashMap<i64, u32>,
    pub(super) fill_seq: NextSequence,
    /// Active logical request boundary for the lookup observer only. This does
    /// not participate in tags, MSHR ownership, replacement, or timing.
    pub(super) active_lookup_batch: Option<L1LookupBatch>,
    pub(super) merge_audit: L1MergeAudit,
    /// Exact earliest timed in-flight fill. `None` means there is no timed
    /// fill (empty or event-driven), while `Some(u64::MAX)` is a real due
    /// cycle. `apply_fills` early-outs when `now` has not reached it.
    pub(super) min_fill_due: Option<u64>,
    /// Latched fill-timing discipline (see [`FillTiming`]).
    pub(super) fill_timing: Option<FillTiming>,
    /// Lazily allocated observer-only write tag window.  Keeping it boxed and
    /// optional leaves cold/read-only replays on the compact hot path.
    pub(super) write_lookup_observer: Option<Box<L1WriteLookupObserver>>,
    /// Optional effective set-local LRU state. The global recency maps remain
    /// the canonical typed boundary; this index is an executable projection
    /// used only by the card-backed modulo scenario.
    pub(super) set_lru: Option<Vec<VecDeque<i64>>>,
    /// Actual set chosen by the pressure-bounded unknown-hash scenario.
    /// Fully associative and fixed-index modes leave this empty.
    pub(super) set_locations: HashMap<i64, usize>,
    pub counters: L1Counters,
}
