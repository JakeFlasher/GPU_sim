# SM86 cross-SM coherence domain — 2026-09-25

The replay now exposes a conservative, trace-only prerequisite for future
coherence modeling.  During address-decoded extraction it groups global-memory
sectors by 128-byte physical line and records the traced SMs and access classes
that touched each line.  The resulting rows are:

* `coherence_decoded_global_lines` — unique decoded global lines;
* `coherence_cross_sm_lines` — lines touched by more than one traced SM;
* `coherence_cross_sm_sectors` — unique sectors on those shared lines;
* `coherence_cross_sm_sector_overlap` — sectors touched by more than one
  traced SM, a tighter contention domain than line-level sharing;
* `coherence_cross_sm_read_write_lines` — shared lines containing both read and
  write classes;
* `coherence_cross_sm_write_write_lines` — shared lines written by multiple
  SMs;
* `coherence_cross_sm_read_write_sectors` and
  `coherence_cross_sm_write_write_sectors` — corresponding access-class
  overlap at 32-byte-sector granularity; and
* `coherence_unknown_sm_or_line_records` — records excluded because the SM or
  sector identity was unavailable.

These are **static sharing-domain observations**, not hardware invalidation,
atomicity, ordering, writeback, or DRAM-traffic counters.  Sector overlap
narrows the line-level contention upper bound; it does not estimate protocol
transaction multiplicity.  The observer does not affect scheduling, cache
state, or completion timing.  Flat replay leaves
the rows typed `unavailable` because it intentionally does not decode sector
addresses.  A bounded one-million-line arena fails closed into the unknown
record count rather than permitting unbounded diagnostic memory growth.

The distinction matters for NCU comparison: a profiler's coherence or write
traffic row includes protocol actions and flush visibility that cannot be
recovered from a static NVBit memory trace alone.  The new domain makes the
missing input explicit and gives a principled next target for protocol work
without leaking NCU measurements into the simulator.
