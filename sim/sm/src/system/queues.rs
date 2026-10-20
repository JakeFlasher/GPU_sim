//! Pending-response and due-time queue state; mutates event scheduling state.

use super::*;

/// Per-SM lower bounds on responses to requests whose `L2Access` has
/// been SCHEDULED but not yet PROCESSED (so the true response time is
/// not in `response_due` yet). `SmCore::advance`'s internal cap covers
/// requests emitted during ONE call; this map covers the window BETWEEN
/// calls — without it, any event that resumes the SM inside that window
/// (a stale wake, an unrelated response) hands it a horizon that
/// ignores the in-flight requests, and the SM runs issue cycles past
/// its own future responses (stale L1 state, late barrier clears; found
/// by the gate's event-vs-constant differential, seed 133).
/// A `BTreeMap<bound, count>` rather than a heap because entries leave
/// in L2Access-processing order, which is not bound order.
pub(super) struct PendingBounds(pub(super) Vec<BTreeMap<u64, u32>>);

#[derive(Debug, Clone, Copy)]
pub(super) enum PendingRemoval {
    Decrement { sm: usize, bound: u64, next: u32 },
    Remove { sm: usize, bound: u64 },
}

impl PendingRemoval {
    pub(super) fn commit(self, bounds: &mut PendingBounds) {
        match self {
            Self::Decrement { sm, bound, next } => {
                bounds.0[sm].insert(bound, next);
            }
            Self::Remove { sm, bound } => {
                bounds.0[sm].remove(&bound);
            }
        }
    }
}

impl PendingBounds {
    pub(super) fn insert(&mut self, sm: usize, bound: u64) -> Result<(), SystemError> {
        let next = self.0[sm]
            .get(&bound)
            .copied()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or(SystemError::CounterOverflow {
                counter: "pending response-bound multiplicity",
            })?;
        self.0[sm].insert(bound, next);
        Ok(())
    }

    pub(super) fn prepare_remove(
        &self,
        sm: usize,
        bound: u64,
    ) -> Result<PendingRemoval, SystemError> {
        match self.0[sm].get(&bound).copied() {
            Some(n) if n > 1 => Ok(PendingRemoval::Decrement {
                sm,
                bound,
                next: n - 1,
            }),
            Some(_) => Ok(PendingRemoval::Remove { sm, bound }),
            None => Err(SystemError::PendingBoundMissing { sm, bound }),
        }
    }

    pub(super) fn min(&self, sm: usize) -> Option<u64> {
        self.0[sm].first_key_value().map(|(&t, _)| t)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.0.iter().all(BTreeMap::is_empty)
    }

    pub(super) fn pending_entries(&self) -> usize {
        self.0.iter().map(BTreeMap::len).sum()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DueEntry {
    pub(super) time: u64,
    pub(super) insertion: u64,
    pub(super) response: Option<ResponseBoundaryId>,
    pub(super) l2_allocation_generation: Option<u64>,
}

impl Ord for DueEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.time, self.insertion).cmp(&(other.time, other.insertion))
    }
}

impl PartialOrd for DueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Per-actor pending due-times with the one choreography the driver repeats
/// for responses and wakes. Response entries retain their semantic identity,
/// so a typed snapshot can normalize an exact due cycle and delivery checks
/// cannot accidentally pair two same-cycle responses by timestamp alone.
pub(super) struct DueTimes {
    pub(super) entries: Vec<BinaryHeap<Reverse<DueEntry>>>,
    pub(super) next_insertion: u64,
}

impl DueTimes {
    pub(super) fn new(n: usize) -> Self {
        Self {
            entries: (0..n).map(|_| BinaryHeap::new()).collect(),
            next_insertion: 0,
        }
    }

    pub(super) fn push(&mut self, ix: usize, t: u64) -> Result<(), SystemError> {
        self.push_entry(ix, t, None, None)
    }

    pub(super) fn push_response(
        &mut self,
        ix: usize,
        t: u64,
        response: Option<ResponseBoundaryId>,
        l2_allocation_generation: Option<u64>,
    ) -> Result<(), SystemError> {
        self.push_entry(ix, t, response, l2_allocation_generation)
    }

    pub(super) fn push_entry(
        &mut self,
        ix: usize,
        t: u64,
        response: Option<ResponseBoundaryId>,
        l2_allocation_generation: Option<u64>,
    ) -> Result<(), SystemError> {
        let insertion = self
            .next_insertion
            .checked_add(1)
            .ok_or(SystemError::CounterOverflow {
                counter: "due-time insertion sequence",
            })?;
        self.entries[ix].push(Reverse(DueEntry {
            time: t,
            insertion: self.next_insertion,
            response,
            l2_allocation_generation,
        }));
        self.next_insertion = insertion;
        Ok(())
    }

    pub(super) fn pop_expect(
        &mut self,
        component: &'static str,
        ix: usize,
        t: u64,
    ) -> Result<(), SystemError> {
        let got = self.next(ix);
        if got != Some(t) {
            return Err(SystemError::DueTimeMismatch {
                component,
                sm: ix,
                expected: t,
                got,
            });
        }
        self.entries[ix].pop();
        Ok(())
    }

    pub(super) fn pop_response_expect(
        &mut self,
        component: &'static str,
        ix: usize,
        t: u64,
        response: Option<ResponseBoundaryId>,
    ) -> Result<(), SystemError> {
        let Some(Reverse(entry)) = self.entries[ix].peek().copied() else {
            return Err(SystemError::DueTimeMismatch {
                component,
                sm: ix,
                expected: t,
                got: None,
            });
        };
        if entry.time != t {
            return Err(SystemError::DueTimeMismatch {
                component,
                sm: ix,
                expected: t,
                got: Some(entry.time),
            });
        }
        if entry.response != response {
            return Err(SystemError::ResponseIdentityMismatch {
                component,
                sm: ix,
                expected: response,
                got: entry.response,
            });
        }
        self.entries[ix].pop();
        Ok(())
    }

    pub(super) fn next(&self, ix: usize) -> Option<u64> {
        self.entries[ix].peek().map(|&Reverse(entry)| entry.time)
    }

    #[cfg(feature = "reference")]
    pub(super) fn response_entries(
        &self,
        ix: usize,
    ) -> Vec<(ResponseBoundaryId, u64, Option<u64>)> {
        let mut entries = self.entries[ix]
            .iter()
            .filter_map(|Reverse(entry)| {
                entry.response.map(|response| {
                    (
                        response,
                        entry.time,
                        entry.l2_allocation_generation,
                        entry.insertion,
                    )
                })
            })
            .collect::<Vec<_>>();
        entries.sort_unstable_by_key(|(_, _, _, insertion)| *insertion);
        entries
            .into_iter()
            .map(|(response, time, generation, _)| (response, time, generation))
            .collect()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.entries.iter().all(BinaryHeap::is_empty)
    }

    pub(super) fn pending_entries(&self) -> usize {
        self.entries.iter().map(BinaryHeap::len).sum()
    }
}

/// Backstop against event-loop LIVELOCK — a regression class this driver
/// actually shipped once (same-time SmWake ping-pong, before the horizon
/// contract was narrowed to an SM's own responses). The signature is
/// unmistakable: SmWake events that issue nothing and emit nothing while
/// the clock stays frozen, forever. In any healthy run a parked SM
/// yields such an idle wake at most once per REAL event at that
/// timestamp (a response or L2 access resets the streak, and a
/// future-time wake advances the clock), so consecutive idle wakes are
/// bounded by the SM count; crossing `2 × cores + 8` proves a broken
/// invariant. Fail closed in milliseconds — plain `cargo test` has no
/// timeout, and the pre-fix binaries spun at full CPU for hours.
pub(super) struct SpinGuard {
    pub(super) limit: u64,
    pub(super) last_time: u64,
    pub(super) idle_wakes: u64,
}

impl SpinGuard {
    pub(super) fn new(cores: usize) -> Result<Self, SystemError> {
        let cores =
            u64::try_from(cores).map_err(|_| SystemError::ActorCountOutOfRange { count: cores })?;
        let limit = cores.checked_mul(2).and_then(|n| n.checked_add(8)).ok_or(
            SystemError::CounterOverflow {
                counter: "spin-guard limit",
            },
        )?;
        Ok(Self {
            limit,
            last_time: 0,
            idle_wakes: 0,
        })
    }

    pub(super) fn on_progress_event(&mut self, time: u64) {
        self.last_time = time;
        self.idle_wakes = 0;
    }

    /// Record one processed event. `idle_wake` marks an `SmWake` whose
    /// actor neither issued an instruction nor emitted a request; every
    /// other event kind (and any clock advance) resets the streak.
    pub(super) fn on_event(&mut self, time: u64, idle_wake: bool) -> Result<(), SystemError> {
        if time != self.last_time {
            self.last_time = time;
            self.idle_wakes = 0;
        }
        if !idle_wake {
            self.idle_wakes = 0;
            return Ok(());
        }
        self.idle_wakes = self
            .idle_wakes
            .checked_add(1)
            .ok_or(SystemError::CounterOverflow {
                counter: "consecutive idle wakes",
            })?;
        if self.idle_wakes > self.limit {
            return Err(SystemError::Livelock {
                time,
                idle_wakes: self.idle_wakes,
            });
        }
        Ok(())
    }
}

#[derive(Default)]
pub(super) struct PendingL2Ready(pub(super) HashMap<L2PendingSectorKey, PendingL2ReadyEntry>);

impl PendingL2Ready {
    pub(super) fn merged_ready(
        &self,
        preview: L2EventMaskPreview,
    ) -> Result<Option<u64>, SystemError> {
        let mut latest = None;
        for sector in 0..4 {
            let Some(key) = preview.pending_sector_key(sector) else {
                continue;
            };
            let ready = self
                .0
                .get(&key)
                .map(|entry| entry.ready)
                .ok_or(SystemError::PendingL2ReadyMissing { key })?;
            latest = Some(latest.map_or(ready, |current: u64| current.max(ready)));
        }
        Ok(latest)
    }

    pub(super) fn preflight_register(
        &self,
        preview: L2EventMaskPreview,
    ) -> Result<(), SystemError> {
        for sector in 0..4 {
            let Some(key) = preview.fill_sector_key(sector) else {
                continue;
            };
            if let Some(entry) = self.0.get(&key) {
                return Err(SystemError::PendingL2ReadyAlreadyRegistered {
                    key,
                    ready: entry.ready,
                });
            }
        }
        Ok(())
    }

    #[cfg(not(feature = "reference"))]
    pub(super) fn register(&mut self, fill: L2FillToken, ready: u64) -> Result<(), SystemError> {
        self.register_entry(
            fill,
            PendingL2ReadyEntry {
                ready,
                #[cfg(feature = "reference")]
                producer: None,
            },
        )
    }

    #[cfg(feature = "reference")]
    pub(super) fn register_with_producer(
        &mut self,
        fill: L2FillToken,
        ready: u64,
        producer: Option<L2RequestKey>,
    ) -> Result<(), SystemError> {
        self.register_entry(fill, PendingL2ReadyEntry { ready, producer })
    }

    pub(super) fn register_entry(
        &mut self,
        fill: L2FillToken,
        entry: PendingL2ReadyEntry,
    ) -> Result<(), SystemError> {
        for sector in 0..4 {
            let Some(key) = fill.pending_sector_key(sector) else {
                continue;
            };
            if let Some(existing) = self.0.get(&key) {
                return Err(SystemError::PendingL2ReadyAlreadyRegistered {
                    key,
                    ready: existing.ready,
                });
            }
        }
        for sector in 0..4 {
            if let Some(key) = fill.pending_sector_key(sector) {
                self.0.insert(key, entry.clone());
            }
        }
        Ok(())
    }

    pub(super) fn preflight_remove(&self, fill: L2FillToken, now: u64) -> Result<(), SystemError> {
        for sector in 0..4 {
            let Some(key) = fill.pending_sector_key(sector) else {
                continue;
            };
            let ready = self
                .0
                .get(&key)
                .map(|entry| entry.ready)
                .ok_or(SystemError::PendingL2ReadyRemovalMissing { key })?;
            if ready != now {
                return Err(SystemError::PendingL2ReadyTimeMismatch {
                    key,
                    expected: ready,
                    got: now,
                });
            }
        }
        Ok(())
    }

    pub(super) fn commit_remove(&mut self, fill: L2FillToken) {
        for sector in 0..4 {
            if let Some(key) = fill.pending_sector_key(sector) {
                let removed = self.0.remove(&key);
                debug_assert!(removed.is_some(), "pending L2 removal was preflighted");
            }
        }
    }

    pub(super) fn len(&self) -> usize {
        self.0.len()
    }

    #[cfg(feature = "reference")]
    pub(super) fn typed_entries(&self) -> Vec<(i64, u64, u8, u64, Option<L2RequestKey>)> {
        let mut entries = self
            .0
            .iter()
            .map(|(key, entry)| {
                let identity = key.boundary_identity();
                (
                    identity.line_id,
                    identity.generation,
                    identity.sector,
                    entry.ready,
                    entry.producer.clone(),
                )
            })
            .collect::<Vec<_>>();
        entries.sort_unstable();
        entries
    }
}
