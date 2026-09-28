//! Explicit value-bearing memory-order/read-from admission seam.
//!
//! This is a reference contract for traces that carry synchronization and
//! read-from identities. It is not a reconstruction of NVIDIA's proprietary
//! fence implementation. Missing read-from/value/HB evidence is rejected
//! rather than converted into unconditional invalidation or a guessed value.

use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrderScope {
    Block,
    Device,
    System,
}

impl MemoryOrderScope {
    const fn rank(self) -> u8 {
        match self {
            Self::Block => 0,
            Self::Device => 1,
            Self::System => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrder {
    Relaxed,
    Acquire,
    Release,
    AcquireRelease,
    SequentiallyConsistent,
}

impl MemoryOrder {
    const fn has_acquire(self) -> bool {
        matches!(
            self,
            Self::Acquire | Self::AcquireRelease | Self::SequentiallyConsistent
        )
    }

    const fn has_release(self) -> bool {
        matches!(
            self,
            Self::Release | Self::AcquireRelease | Self::SequentiallyConsistent
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryVisibilityQualifier {
    Cacheable,
    Volatile,
    Proxy,
    Atomic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryWrite {
    pub event_id: u64,
    pub requester: u8,
    pub line_id: i64,
    pub sector: u8,
    pub value_digest: u64,
    pub scope: MemoryOrderScope,
    pub order: MemoryOrder,
    pub sequence: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryRead {
    pub event_id: u64,
    pub requester: u8,
    pub line_id: i64,
    pub sector: u8,
    pub value_digest: Option<u64>,
    pub scope: MemoryOrderScope,
    pub order: MemoryOrder,
    pub sequence: u64,
    pub read_from: Option<u64>,
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum MemoryOrderError {
    #[error("memory event id {0} is duplicated")]
    DuplicateEvent(u64),
    #[error("memory sector {0} is outside 0..=3")]
    SectorOutOfRange(u8),
    #[error("memory sequence regresses for requester {requester}: {requested} < {previous}")]
    SequenceRegression {
        requester: u8,
        requested: u64,
        previous: u64,
    },
    #[error("read {read} names unknown read-from write {write}")]
    ReadFromUnknown { read: u64, write: u64 },
    #[error("read {read} read-from write {write} targets a different location")]
    ReadFromLocationMismatch { read: u64, write: u64 },
    #[error("read {read} value digest disagrees with write {write}")]
    ReadFromValueMismatch { read: u64, write: u64 },
    #[error("read {read} lacks read-from identity for a cross-requester access")]
    ReadFromMissing { read: u64 },
    #[error("read {read} lacks acquire ordering for cross-requester visibility")]
    AcquireMissing { read: u64 },
    #[error("write {write} lacks release ordering for cross-requester visibility")]
    ReleaseMissing { write: u64 },
    #[error("read {read} scope is weaker than write {write} scope")]
    ScopeInsufficient { read: u64, write: u64 },
    #[error(
        "read {read} uses a cacheable visibility qualifier for cross-requester synchronization"
    )]
    VisibilityQualifierMissing { read: u64 },
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MemoryOrderLedgerV1 {
    writes: BTreeMap<u64, MemoryWrite>,
    events: BTreeMap<u64, u64>,
    last_sequence: BTreeMap<u8, u64>,
}

impl MemoryOrderLedgerV1 {
    pub fn observe_write(&mut self, write: MemoryWrite) -> Result<(), MemoryOrderError> {
        self.validate_identity(
            write.event_id,
            write.requester,
            write.sequence,
            write.sector,
        )?;
        self.last_sequence.insert(write.requester, write.sequence);
        self.writes.insert(write.event_id, write);
        Ok(())
    }

    pub fn observe_read(
        &mut self,
        read: MemoryRead,
    ) -> Result<Option<MemoryWrite>, MemoryOrderError> {
        self.validate_identity(read.event_id, read.requester, read.sequence, read.sector)?;
        let Some(write_id) = read.read_from else {
            if self.writes.values().any(|write| {
                write.requester != read.requester
                    && write.line_id == read.line_id
                    && write.sector == read.sector
            }) {
                return Err(MemoryOrderError::ReadFromMissing {
                    read: read.event_id,
                });
            }
            self.last_sequence.insert(read.requester, read.sequence);
            self.events.insert(read.event_id, read.sequence);
            return Ok(None);
        };
        let Some(write) = self.writes.get(&write_id).copied() else {
            return Err(MemoryOrderError::ReadFromUnknown {
                read: read.event_id,
                write: write_id,
            });
        };
        if write.line_id != read.line_id || write.sector != read.sector {
            return Err(MemoryOrderError::ReadFromLocationMismatch {
                read: read.event_id,
                write: write_id,
            });
        }
        if read
            .value_digest
            .is_some_and(|value| value != write.value_digest)
        {
            return Err(MemoryOrderError::ReadFromValueMismatch {
                read: read.event_id,
                write: write_id,
            });
        }
        if write.requester != read.requester {
            if !read.order.has_acquire() {
                return Err(MemoryOrderError::AcquireMissing {
                    read: read.event_id,
                });
            }
            if !write.order.has_release() {
                return Err(MemoryOrderError::ReleaseMissing { write: write_id });
            }
            if read.scope.rank() < write.scope.rank() {
                return Err(MemoryOrderError::ScopeInsufficient {
                    read: read.event_id,
                    write: write_id,
                });
            }
        }
        self.last_sequence.insert(read.requester, read.sequence);
        self.events.insert(read.event_id, read.sequence);
        Ok(Some(write))
    }

    pub fn observe_qualified_read(
        &mut self,
        read: MemoryRead,
        qualifier: MemoryVisibilityQualifier,
    ) -> Result<Option<MemoryWrite>, MemoryOrderError> {
        if let Some(write_id) = read.read_from
            && self
                .writes
                .get(&write_id)
                .is_some_and(|write| write.requester != read.requester)
            && matches!(qualifier, MemoryVisibilityQualifier::Cacheable)
        {
            return Err(MemoryOrderError::VisibilityQualifierMissing {
                read: read.event_id,
            });
        }
        self.observe_read(read)
    }

    fn validate_identity(
        &self,
        event_id: u64,
        requester: u8,
        sequence: u64,
        sector: u8,
    ) -> Result<(), MemoryOrderError> {
        if sector >= 4 {
            return Err(MemoryOrderError::SectorOutOfRange(sector));
        }
        if self.events.contains_key(&event_id) || self.writes.contains_key(&event_id) {
            return Err(MemoryOrderError::DuplicateEvent(event_id));
        }
        if let Some(previous) = self.last_sequence.get(&requester).copied()
            && sequence < previous
        {
            return Err(MemoryOrderError::SequenceRegression {
                requester,
                requested: sequence,
                previous,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(order: MemoryOrder) -> MemoryWrite {
        MemoryWrite {
            event_id: 1,
            requester: 0,
            line_id: 9,
            sector: 0,
            value_digest: 77,
            scope: MemoryOrderScope::Device,
            order,
            sequence: 1,
        }
    }

    fn read(order: MemoryOrder, read_from: Option<u64>) -> MemoryRead {
        MemoryRead {
            event_id: 2,
            requester: 1,
            line_id: 9,
            sector: 0,
            value_digest: Some(77),
            scope: MemoryOrderScope::Device,
            order,
            sequence: 2,
            read_from,
        }
    }

    #[test]
    fn release_acquire_read_from_is_admitted() {
        let mut ledger = MemoryOrderLedgerV1::default();
        ledger.observe_write(write(MemoryOrder::Release)).unwrap();
        assert_eq!(
            ledger
                .observe_read(read(MemoryOrder::Acquire, Some(1)))
                .unwrap()
                .unwrap()
                .value_digest,
            77
        );
    }

    #[test]
    fn cross_requester_read_without_read_from_refuses() {
        let mut ledger = MemoryOrderLedgerV1::default();
        ledger.observe_write(write(MemoryOrder::Release)).unwrap();
        assert_eq!(
            ledger.observe_read(read(MemoryOrder::Acquire, None)),
            Err(MemoryOrderError::ReadFromMissing { read: 2 })
        );
    }

    #[test]
    fn weak_ordering_refuses_cross_requester_visibility() {
        let mut ledger = MemoryOrderLedgerV1::default();
        ledger.observe_write(write(MemoryOrder::Relaxed)).unwrap();
        assert_eq!(
            ledger.observe_read(read(MemoryOrder::Acquire, Some(1))),
            Err(MemoryOrderError::ReleaseMissing { write: 1 })
        );
    }

    #[test]
    fn value_and_location_mismatch_refuse() {
        let mut ledger = MemoryOrderLedgerV1::default();
        ledger.observe_write(write(MemoryOrder::Release)).unwrap();
        let mut bad = read(MemoryOrder::Acquire, Some(1));
        bad.value_digest = Some(88);
        assert_eq!(
            ledger.observe_read(bad),
            Err(MemoryOrderError::ReadFromValueMismatch { read: 2, write: 1 })
        );
    }

    #[test]
    fn cross_requester_cacheable_read_requires_visibility_qualifier() {
        let mut ledger = MemoryOrderLedgerV1::default();
        ledger.observe_write(write(MemoryOrder::Release)).unwrap();
        assert_eq!(
            ledger.observe_qualified_read(
                read(MemoryOrder::Acquire, Some(1)),
                MemoryVisibilityQualifier::Cacheable,
            ),
            Err(MemoryOrderError::VisibilityQualifierMissing { read: 2 })
        );
        ledger
            .observe_qualified_read(
                read(MemoryOrder::Acquire, Some(1)),
                MemoryVisibilityQualifier::Volatile,
            )
            .unwrap();
    }
}
