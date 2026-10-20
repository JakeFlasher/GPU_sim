# Simulator Terminology and Metric Definitions

This glossary is the human-facing vocabulary for the unified simulator. Machine
field names may retain historical spellings for compatibility.

## Architecture and execution

- **event-driven cycle-level simulation**: advances the modeled GPU from one
  causally relevant issue, response, or synchronization event to the next
  instead of polling every idle cycle.
- **conservative lookahead horizon**: the earliest time at which an unresolved
  remote memory event could affect an SM; this is a standard conservative
  discrete-event synchronization bound.
- **completion timestamp**: a timestamp for a named stage such as issue,
  modeled service, response visibility, or launch drain. Avoid the unqualified
  word “frontier.”
- **compiler-guided greedy-then-youngest (CGGTY)**: the warp-selection policy
  described by Huerta et al.; use the full name on first mention.
- **SMSP/sub-partition**: one of the independent warp-scheduling partitions
  within an SM.
- **effective model**: an explicitly scoped approximation used when retained
  traces do not identify proprietary hardware state.

## Memory and locality

- **sectored cache**: a cache with a line tag and independently valid data
  sectors; this project models 128-byte lines and four 32-byte sectors.
- **resident-sector hit**: the requested sector's data is present before the
  access.
- **line-tag hit**: the parent line tag is present even if the requested sector
  data is absent.
- **pending merge**: an access joins an in-flight fill. An
  **already-requested-sector merge** is distinct from an unrequested subline
  miss.
- **footprint**: average number of distinct data items in windows of a given
  length.
- **lifetime**: the inverse footprint at a specified cache capacity.
- **reuse distance**: number of distinct items referenced between two
  references to the same item.
- **miss-ratio curve (MRC)**: miss ratio as a function of cache capacity.
- **HOTL estimator**: this project's interval-based footprint-to-MRC estimator
  derived from Ding and Xiang's Higher-Order Theory of Locality.
- **Fenwick/Mattson reference**: the exact within-warp stack-distance
  implementation used as the Class-A reference curve.
- **trace-local locality analysis**: measurements over a declared trace order;
  not a hardware-cache measurement.
- **conditional envelope**: a lower and upper prediction interval derived from
  trace-visible mechanisms and explicit assumptions.

## Counter and validation language

- **hardware-counter attribution**: assigning modeled events to the population
  measured by an NCU counter.
- **report-only observer**: an observer that records or attributes events but
  does not mutate cache state, traffic, timing, or completion timestamps.
- **scope limitation**: a mismatch between the counter's hardware client
  population and what retained traces identify.
- **exact identity join**: a simulator launch and NCU row matched by retained
  launch identity.
- **ordered fallback join**: a deterministic order-based association used only
  when identity is unavailable; it is lower confidence and must be reported
  separately.
- **unmatched row**: a launch or NCU row with no defensible association.
- **Mean APE/MAPE**:
  \[
  100\frac{1}{N}\sum_i\frac{|\hat y_i-y_i|}{|y_i|},
  \]
  computed only for nonzero targets.
- **WAPE**:
  \[
  100\frac{\sum_i|\hat y_i-y_i|}{\sum_i|y_i|}.
  \]
  It weights rows by target magnitude and is not interchangeable with Mean
  APE.
- **zero-target false positive**: a nonzero prediction when the independent
  target is zero.

## Evidence boundaries

NCU data is an independent evaluation target. It must never select a model
variant, tune a parameter, populate a lookup table, or create a workload
exception. When a physical mechanism is not identifiable from traces and
architecture cards, the release reports an effective model or an explicit
scope limitation instead of claiming proprietary hardware recovery.
