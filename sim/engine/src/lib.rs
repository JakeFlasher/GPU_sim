//! sim-engine: the discrete-event core (design §6.1).
//!
//! v0 is a binary-heap min-queue over events carrying
//! `(time, seq, sequence_span, target, kind)`, with a strict deterministic total
//! order on `(time, seq)`, where `seq` is the insertion sequence number. Payload data
//! belongs in the caller's typed `kind`; there is no separate FFI payload
//! handle. Two runs over the same inputs therefore produce bit-identical
//! schedules — a prerequisite for the M2 accuracy gates (a simulator whose
//! outputs wobble run-to-run cannot be compared against deterministic-class
//! NCU counts).
//!
//! The planned hybrid timing-wheel front-end and per-SM actor batching are
//! performance work layered behind this same API once profiles justify them
//! (throughput target 3–10 M events/s v0 → 10–30 M optimized).

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZeroU64;

pub mod pipeline;

/// Simulation time in cycles of the modeled clock domain.
pub type Time = u64;

/// Opaque component id (SM index, L2 slice index, DRAM channel, ...).
/// The engine never interprets it; dispatch tables live in the callers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TargetId(pub u32);

/// A scheduled event. `kind` is the caller's tagged enum (no virtual
/// dispatch on the hot path); the engine orders only by `(time, seq)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Event<K> {
    pub time: Time,
    /// Insertion sequence number: unique, monotonically increasing, and the
    /// deterministic tie-breaker for same-cycle events (FIFO within a cycle).
    pub seq: u64,
    /// Logical insertion positions reserved by this physical event, including
    /// `seq`. Preserve the actual reservation independently of its payload.
    pub sequence_span: NonZeroU64,
    pub target: TargetId,
    pub kind: K,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum EngineError {
    #[error(
        "cannot schedule event at t={requested} — the clock already advanced to {now} (time travel would silently corrupt causality)"
    )]
    SchedulePast { requested: Time, now: Time },
    #[error("delay {delay} from now={now} overflows the u64 cycle domain")]
    TimeOverflow { now: Time, delay: Time },
    #[error("event insertion sequence would overflow after assigning {next_seq}")]
    SeqOverflow { next_seq: u64 },
    #[error(
        "event queue yielded t={event_time} before its current clock t={now} — queue causality is corrupt"
    )]
    EventBeforeClock { event_time: Time, now: Time },
    #[error("event queue {counter} overflowed")]
    StatsOverflow { counter: &'static str },
}

/// Heap entry ordered by `(time, seq)` only; `kind` deliberately carries no
/// ordering obligation.
struct HeapEntry<K>(Event<K>);

impl<K> PartialEq for HeapEntry<K> {
    fn eq(&self, other: &Self) -> bool {
        self.0.time == other.0.time && self.0.seq == other.0.seq
    }
}
impl<K> Eq for HeapEntry<K> {}
impl<K> PartialOrd for HeapEntry<K> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<K> Ord for HeapEntry<K> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.0.time, self.0.seq).cmp(&(other.0.time, other.0.seq))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NextSequence {
    Available(u64),
    Exhausted,
}

/// Low-cost queue telemetry for distinguishing virtual clock advancement from
/// actual handler work. `clock_gap_cycles` counts the cycles crossed between
/// consecutive popped events; it is not a claim about host speedup.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QueueStats {
    pub events_popped: u64,
    pub clock_gap_cycles: Time,
    pub last_clock_gap_cycles: Time,
}

/// Deterministic discrete-event queue with pure next-event clock advance.
pub struct EventQueue<K> {
    heap: BinaryHeap<Reverse<HeapEntry<K>>>,
    now: Time,
    next_seq: NextSequence,
    stats: QueueStats,
}

impl<K> Default for EventQueue<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K> EventQueue<K> {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            now: 0,
            next_seq: NextSequence::Available(0),
            stats: QueueStats::default(),
        }
    }

    /// Current simulation time: the timestamp of the last popped event.
    pub fn now(&self) -> Time {
        self.now
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Queue telemetry accumulated by successful pops. Scheduling an event
    /// does not advance the clock and therefore does not affect these values.
    pub fn stats(&self) -> QueueStats {
        self.stats
    }

    /// Next unused insertion position; `None` means the sequence space is
    /// exhausted. This cannot be recovered from pending events after pops.
    pub fn next_sequence(&self) -> Option<u64> {
        match self.next_seq {
            NextSequence::Available(sequence) => Some(sequence),
            NextSequence::Exhausted => None,
        }
    }

    /// Schedule at absolute time `at` (>= now, or a typed error).
    pub fn schedule_at(&mut self, at: Time, target: TargetId, kind: K) -> Result<u64, EngineError> {
        self.schedule_at_spanning(at, target, kind, NonZeroU64::MIN)
    }

    /// Schedule one physical event while reserving `sequence_span` logical
    /// insertion positions. The event owns the first position and later events
    /// begin after the reserved span.
    pub fn schedule_at_spanning(
        &mut self,
        at: Time,
        target: TargetId,
        kind: K,
        sequence_span: NonZeroU64,
    ) -> Result<u64, EngineError> {
        if at < self.now {
            return Err(EngineError::SchedulePast {
                requested: at,
                now: self.now,
            });
        }
        let (seq, next_seq) = match self.next_seq {
            NextSequence::Available(seq) => {
                let last = seq
                    .checked_add(sequence_span.get() - 1)
                    .ok_or(EngineError::SeqOverflow { next_seq: u64::MAX })?;
                (
                    seq,
                    match last.checked_add(1) {
                        Some(next) => NextSequence::Available(next),
                        None => NextSequence::Exhausted,
                    },
                )
            }
            NextSequence::Exhausted => {
                return Err(EngineError::SeqOverflow { next_seq: u64::MAX });
            }
        };
        self.next_seq = next_seq;
        self.heap.push(Reverse(HeapEntry(Event {
            time: at,
            seq,
            sequence_span,
            target,
            kind,
        })));
        Ok(seq)
    }

    /// Schedule `delay` cycles after the current time. Overflowing the
    /// cycle domain is a typed error, never a silent clamp.
    pub fn schedule_in(
        &mut self,
        delay: Time,
        target: TargetId,
        kind: K,
    ) -> Result<u64, EngineError> {
        let at = self
            .now
            .checked_add(delay)
            .ok_or(EngineError::TimeOverflow {
                now: self.now,
                delay,
            })?;
        self.schedule_at(at, target, kind)
    }

    /// Timestamp of the next event without popping it.
    pub fn peek_time(&self) -> Option<Time> {
        self.heap.peek().map(|Reverse(HeapEntry(e))| e.time)
    }

    /// Ordered timestamps and insertion sequences for pending events. Payloads
    /// remain owned by the caller's typed event domain.
    pub fn pending_order(&self) -> Vec<(Time, u64)> {
        let mut order = self
            .heap
            .iter()
            .map(|Reverse(HeapEntry(event))| (event.time, event.seq))
            .collect::<Vec<_>>();
        order.sort_unstable();
        order
    }

    /// Debug-only payload projection for bounded differential traces. The
    /// caller owns the event semantics; this method never participates in
    /// ordering or execution.
    pub fn pending_debug(&self) -> Vec<(Time, u64, String)>
    where
        K: std::fmt::Debug,
    {
        let mut events = self
            .heap
            .iter()
            .map(|Reverse(HeapEntry(event))| (event.time, event.seq, format!("{:?}", event.kind)))
            .collect::<Vec<_>>();
        events.sort_unstable_by_key(|(time, seq, _)| (*time, *seq));
        events
    }

    /// Project pending events into a caller-owned typed representation while
    /// retaining the queue's canonical `(time, sequence)` order. The engine
    /// never interprets the event payload; this is the typed alternative to
    /// `pending_debug` for boundary traces.
    pub fn pending_projected<R>(&self, project: impl FnMut(&Event<K>) -> R) -> Vec<R> {
        // Sort references to the heap entries before invoking the projection.
        // A typed payload can own a large request batch; sorting `(key, R)`
        // would move those payloads O(n log n) times and would invoke the
        // callback in the heap's unspecified iterator order.
        let mut events = self
            .heap
            .iter()
            .map(|Reverse(HeapEntry(event))| event)
            .collect::<Vec<_>>();
        events.sort_unstable_by_key(|event| (event.time, event.seq));
        events.into_iter().map(project).collect()
    }

    /// Pop the next event and advance the clock to its timestamp.
    pub fn pop(&mut self) -> Result<Option<Event<K>>, EngineError> {
        if let Some(Reverse(HeapEntry(event))) = self.heap.peek()
            && event.time < self.now
        {
            return Err(EngineError::EventBeforeClock {
                event_time: event.time,
                now: self.now,
            });
        }
        let Some(Reverse(HeapEntry(head))) = self.heap.peek() else {
            return Ok(None);
        };
        let clock_gap = head.time - self.now;
        let events_popped =
            self.stats
                .events_popped
                .checked_add(1)
                .ok_or(EngineError::StatsOverflow {
                    counter: "events_popped",
                })?;
        let clock_gap_cycles = self.stats.clock_gap_cycles.checked_add(clock_gap).ok_or(
            EngineError::StatsOverflow {
                counter: "clock_gap_cycles",
            },
        )?;
        let Some(Reverse(HeapEntry(event))) = self.heap.pop() else {
            return Ok(None);
        };
        self.now = event.time;
        self.stats = QueueStats {
            events_popped,
            clock_gap_cycles,
            last_clock_gap_cycles: clock_gap,
        };
        Ok(Some(event))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // unwrap in tests is an assertion, not error handling
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Kind {
        IssueBatch,
        MemResponse,
        Drain,
    }

    #[test]
    fn orders_by_time_then_insertion_seq() {
        let mut q = EventQueue::new();
        q.schedule_at(10, TargetId(0), Kind::MemResponse).unwrap();
        q.schedule_at(5, TargetId(1), Kind::IssueBatch).unwrap();
        q.schedule_at(10, TargetId(2), Kind::IssueBatch).unwrap();
        q.schedule_at(5, TargetId(3), Kind::Drain).unwrap();

        let order: Vec<(Time, u32)> = std::iter::from_fn(|| q.pop().unwrap())
            .map(|e| (e.time, e.target.0))
            .collect();
        // t=5 first (FIFO: target 1 before 3), then t=10 (FIFO: 0 before 2).
        assert_eq!(order, vec![(5, 1), (5, 3), (10, 0), (10, 2)]);
    }

    #[test]
    fn reports_virtual_clock_gaps_separately_from_handler_events() {
        let mut q = EventQueue::new();
        q.schedule_at(0, TargetId(0), Kind::IssueBatch).unwrap();
        q.schedule_at(7, TargetId(1), Kind::MemResponse).unwrap();
        q.schedule_at(7, TargetId(2), Kind::Drain).unwrap();

        q.pop().unwrap();
        assert_eq!(
            q.stats(),
            QueueStats {
                events_popped: 1,
                clock_gap_cycles: 0,
                last_clock_gap_cycles: 0,
            }
        );
        q.pop().unwrap();
        assert_eq!(q.stats().clock_gap_cycles, 7);
        assert_eq!(q.stats().last_clock_gap_cycles, 7);
        q.pop().unwrap();
        assert_eq!(q.stats().events_popped, 3);
        assert_eq!(q.stats().clock_gap_cycles, 7);
        assert_eq!(q.stats().last_clock_gap_cycles, 0);
    }

    #[test]
    fn same_cycle_fifo_is_stable_across_many_events() {
        let mut q = EventQueue::new();
        for i in 0..1000u32 {
            q.schedule_at(42, TargetId(i), Kind::IssueBatch).unwrap();
        }
        let targets: Vec<u32> = std::iter::from_fn(|| q.pop().unwrap())
            .map(|e| e.target.0)
            .collect();
        assert_eq!(targets, (0..1000).collect::<Vec<_>>());
    }

    #[test]
    fn clock_advances_monotonically_and_rejects_time_travel() {
        let mut q = EventQueue::new();
        q.schedule_at(100, TargetId(0), Kind::Drain).unwrap();
        assert_eq!(q.pop().unwrap().unwrap().time, 100);
        assert_eq!(q.now(), 100);
        assert_eq!(
            q.schedule_at(99, TargetId(0), Kind::Drain),
            Err(EngineError::SchedulePast {
                requested: 99,
                now: 100
            })
        );
        // Scheduling exactly at `now` is legal (zero-latency follow-up).
        q.schedule_at(100, TargetId(0), Kind::IssueBatch).unwrap();
        assert_eq!(q.pop().unwrap().unwrap().time, 100);
    }

    #[test]
    fn deterministic_against_reference_sort() {
        // Deterministic pseudo-random schedule (LCG — no external RNG dep,
        // and no wall-clock seeding: runs must be reproducible).
        let mut q = EventQueue::new();
        let mut expected: Vec<(Time, u64)> = Vec::new();
        let mut x: u64 = 0x2545_f491_4f6c_dd1d;
        for _ in 0..10_000 {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let t = x >> 40; // ~24-bit times → plenty of same-time collisions
            let seq = q
                .schedule_at(t, TargetId((x & 0xff) as u32), Kind::MemResponse)
                .unwrap();
            expected.push((t, seq));
        }
        expected.sort();
        let popped: Vec<(Time, u64)> = std::iter::from_fn(|| q.pop().unwrap())
            .map(|e| (e.time, e.seq))
            .collect();
        assert_eq!(popped, expected);
    }

    #[test]
    fn projected_events_are_callback_ordered_without_reordering_payloads() {
        let mut q = EventQueue::new();
        q.schedule_at(10, TargetId(2), Kind::Drain).unwrap();
        q.schedule_at(5, TargetId(1), Kind::IssueBatch).unwrap();
        q.schedule_at(10, TargetId(3), Kind::MemResponse).unwrap();
        q.schedule_at(5, TargetId(4), Kind::Drain).unwrap();

        let mut callback_order = Vec::new();
        let projected = q.pending_projected(|event| {
            callback_order.push((event.time, event.seq));
            (event.time, event.target.0)
        });
        assert_eq!(callback_order, vec![(5, 1), (5, 3), (10, 0), (10, 2)]);
        assert_eq!(projected, vec![(5, 1), (5, 4), (10, 2), (10, 3)]);
    }

    #[test]
    fn schedule_in_overflow_is_typed() {
        let mut q = EventQueue::new();
        q.schedule_at(1, TargetId(0), Kind::Drain).unwrap();
        assert_eq!(q.pop().unwrap().unwrap().time, 1);
        assert_eq!(
            q.schedule_in(u64::MAX, TargetId(0), Kind::Drain),
            Err(EngineError::TimeOverflow {
                now: 1,
                delay: u64::MAX,
            })
        );
        assert!(q.is_empty());
    }

    #[test]
    fn maximum_sequence_is_inserted_before_typed_exhaustion() {
        let mut q = EventQueue::new();
        q.next_seq = NextSequence::Available(u64::MAX - 1);
        assert_eq!(
            q.schedule_at(0, TargetId(0), Kind::IssueBatch),
            Ok(u64::MAX - 1)
        );
        assert_eq!(q.schedule_at(0, TargetId(1), Kind::Drain), Ok(u64::MAX));
        assert_eq!(q.len(), 2);
        assert_eq!(q.next_seq, NextSequence::Exhausted);

        assert_eq!(
            q.schedule_at(0, TargetId(2), Kind::MemResponse),
            Err(EngineError::SeqOverflow { next_seq: u64::MAX })
        );
        assert_eq!(q.len(), 2, "exhaustion inserted a partial event");
        assert_eq!(q.next_seq, NextSequence::Exhausted);
        assert_eq!(q.pop().unwrap().unwrap().seq, u64::MAX - 1);
        assert_eq!(q.pop().unwrap().unwrap().seq, u64::MAX);
    }

    #[test]
    fn sequence_spans_preserve_logical_capacity_and_fifo() {
        let mut q = EventQueue::new();
        assert_eq!(
            q.schedule_at_spanning(
                0,
                TargetId(0),
                Kind::IssueBatch,
                NonZeroU64::new(4).unwrap(),
            ),
            Ok(0)
        );
        assert_eq!(q.next_sequence(), Some(4));
        assert_eq!(q.schedule_at(0, TargetId(1), Kind::Drain), Ok(4));
        assert_eq!(q.next_sequence(), Some(5));
        let first = q.pop().unwrap().unwrap();
        assert_eq!(first.sequence_span, NonZeroU64::new(4).unwrap());
        assert_eq!(q.pop().unwrap().unwrap().target, TargetId(1));

        let mut near_limit = EventQueue::new();
        near_limit.next_seq = NextSequence::Available(u64::MAX - 1);
        assert_eq!(
            near_limit.schedule_at_spanning(
                0,
                TargetId(0),
                Kind::IssueBatch,
                NonZeroU64::new(3).unwrap(),
            ),
            Err(EngineError::SeqOverflow { next_seq: u64::MAX })
        );
        assert!(near_limit.is_empty());
        assert_eq!(
            near_limit.schedule_at(0, TargetId(1), Kind::Drain),
            Ok(u64::MAX - 1)
        );
    }

    #[test]
    fn past_event_from_corrupt_queue_is_typed_in_release_builds() {
        let mut q = EventQueue::new();
        q.now = 7;
        q.heap.push(Reverse(HeapEntry(Event {
            time: 6,
            seq: 0,
            sequence_span: NonZeroU64::MIN,
            target: TargetId(0),
            kind: Kind::Drain,
        })));
        let expected = Err(EngineError::EventBeforeClock {
            event_time: 6,
            now: 7,
        });
        assert_eq!(q.pop(), expected);
        assert_eq!(q.len(), 1, "typed failure must retain the corrupt head");
        assert_eq!(q.peek_time(), Some(6));
        assert_eq!(
            q.pop(),
            Err(EngineError::EventBeforeClock {
                event_time: 6,
                now: 7,
            }),
            "repeated pop must fail on the same retained head"
        );
        assert_eq!(q.now(), 7);
    }

    #[test]
    fn interleaved_schedule_and_pop() {
        let mut q = EventQueue::new();
        q.schedule_at(1, TargetId(0), Kind::IssueBatch).unwrap();
        q.schedule_at(3, TargetId(0), Kind::MemResponse).unwrap();
        let e = q.pop().unwrap().unwrap();
        assert_eq!(e.time, 1);
        // React to the event by scheduling a follow-up before the t=3 one.
        q.schedule_in(1, TargetId(1), Kind::Drain).unwrap();
        assert_eq!(
            q.pop().unwrap().unwrap(),
            Event {
                time: 2,
                seq: 2,
                sequence_span: NonZeroU64::MIN,
                target: TargetId(1),
                kind: Kind::Drain
            }
        );
        assert_eq!(q.pop().unwrap().unwrap().time, 3);
        assert!(q.pop().unwrap().is_none());
        assert_eq!(q.now(), 3);
    }
}
