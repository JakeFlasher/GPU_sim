# Gemini 4 Argon High review: valid GPU locality/MRC metrics

Date: 2026-10-19  
Advisor: `gemini-4-argon-high`, invoked read-only with `agy --dangerously-skip-permissions`  
Scope: SM86/RTX 3090 Ti and SM120/RTX 5090 unified simulator

This file records the advisor's technical review before any implementation
change. The command timed out at its five-minute limit after emitting the
substantive definitions below; the timeout is not evidence that a metric is
valid, so all claims remain subject to the contracts in
`docs/MRC_METRICS_DESIGN.md` and the implementation tests.

## Decisions from the consultation

* NCU supplies realized, single-operating-point counters, not a
  capacity-indexed miss-ratio curve. A universal MRC-vs-NCU APE claim is
  undefined. Keep intrinsic trace metrics (Class A) separate from conditional
  hardware endpoint comparisons (Class B).
* The existing `reuse_interval_observer_v1` is a within-warp record-gap
  survival diagnostic. It must not be relabeled as reuse distance or MRC.
* The canonical trace atom is a non-empty coalesced transaction with an
  origin-qualified 128-byte line key and a four-bit 32-byte sector mask.
  Origin qualification prevents global/local address aliasing.
* Primary Class-A metrics are: line-tag reuse distance in
  `distinct_128B_lines`; sector-weighted line-LRU validity distance in the
  same units; fully-associative line and sector MRCs (ratios); working-set
  footprint/survival; and reuse interval as a separately named diagnostic.
* Same-tick coalesced lines must be queried as a batch before any are inserted
  into the recency structure. Address-order insertion would manufacture
  distance. Tier-1 streams are within-warp trace order; Tier-2 and Tier-3
  streams require an explicitly deterministic SM/cycle or transaction order.
* A fully-associative MRC is a zero-latency LRU locality reference. It does not
  model MSHR/pending-fill merges, sector-fill latency, set hashing, or slice
  placement. Those effects remain in the live cache model or explicit Class-B
  bounds.

## Metric contract (implementation-facing)

For a stream of logical ticks `t`, each tick contains `(origin, line_128B,
sector_mask, op, scope)`:

* `RD_line(t) = infinity` for a first line reference; otherwise it is the
  number of distinct other 128-byte lines between the previous reference to
  that line and `t`. A line tag is resident in a fully associative LRU of `C`
  lines iff `RD_line(t) < C`.
* Sector validity is line-LRU coupled: a line touch refreshes the tag for all
  sectors, while an eviction invalidates every sector. The observer therefore
  tracks, for each line/sector, the largest intervening line distance that
  could have evicted the line, and evaluates the requested sector at `C` using
  that threshold. A sector-first-touch and a post-eviction partial refill are
  distinct counters from line-tag eviction.
* `MRC_line(C) = line_misses(C) / line_references` and
  `MRC_sector(C) = sector_misses(C) / sector_references`, with cold references
  explicit and `MRC(0)=1`. Capacity is measured in 128-byte lines (convertible
  to bytes); sector counts are weighted in 32-byte units.
* A set-associative projection is valid only when the set/hash function and
  way count are declared. Report the conflict premium against the
  fully-associative curve; do not imply that a generic FA curve reproduces
  silicon geometry.
* Reuse interval is the logical tick/cycle gap between two references to the
  same line or sector. Report it as interval/survival, never as a distance.
  Footprint is a window-length-to-working-set curve, not an interval CCDF.

## Validity and scoring gates

Class A is scored against an independent exact Mattson/Fenwick reference on
the same declared stream, not NCU. The reference must pass conservation,
line/sector bounds, monotonic MRC, explicit cold and right-censored handling,
same-tick batch semantics, and deterministic replay. Approximation targets are
reported as WAPE/MAE in percentage points, signed bias, median/P90, and
monotonicity violations; no NCU labels may enter fitting or model selection.

Class B may compare only a scope-compatible endpoint (exact/accepted join,
non-zero denominator, matching client/op scope, and documented coverage). Use
percentage-point error, sector-volume WAPE, exact-zero false-positive counts,
and interval coverage/width. Do not score an entire MRC against one NCU point.

Any observer must be shadow-only until it is proven non-interfering: existing
IPC, timing, L1/L2/DRAM counters, statuses, and full-replay determinism must
remain bit-identical. Hardware MRC claims remain out of scope where cross-warp
arrival order, set/slice/hash/replacement, or TMA completion linkage is not
observable.

## Raw advisor transcript

The complete output available at timeout is retained in the working-session
artifact `/tmp/gemini_mrc_definition_20261019.txt`; it is not a source of
runtime behavior. The repository decisions above are the normalized contract
used for implementation and review.
