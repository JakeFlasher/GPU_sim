//! Explicit byte-bearing sector payloads for the next resident-dirty-flush
//! slice.
//!
//! The native trace contract currently provides dirty masks but not store
//! bytes, load return values, or initialized device-memory contents.  This
//! module therefore does not guess those values: it accepts a bounded,
//! content-bearing sidecar record and refuses unknown baselines, identity
//! mismatches, overlapping generations, and out-of-range writes.  It is a
//! state/value primitive; the ordinary replay path remains mask-only until a
//! producer can satisfy this contract.

use serde::Deserialize;
use std::io::Read;
use std::path::Path;

pub const SECTOR_BYTES: usize = 32;
const FULL_KNOWN_MASK: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectorPayloadV1 {
    allocation_id: u64,
    allocation_generation: u64,
    line_id: i64,
    sector: u8,
    sequence: u64,
    bytes: [u8; SECTOR_BYTES],
    known_mask: u32,
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum PayloadError {
    #[error("sector index {sector} is outside 0..=3")]
    SectorOutOfRange { sector: u8 },
    #[error("payload byte mask {mask:#010x} is empty")]
    KnownMaskEmpty { mask: u32 },
    #[error("payload byte range offset {offset} + length {length} exceeds 32-byte sector")]
    ByteRange { offset: u8, length: usize },
    #[error("payload sequence {requested} is not newer than {previous}")]
    SequenceRegression { requested: u64, previous: u64 },
    #[error("payload identity mismatch")]
    IdentityMismatch,
    #[error("payload bytes for mask {mask:#010x} are unknown")]
    BytesUnknown { mask: u32 },
    #[error("payload allocation identity is zero")]
    AllocationIdZero,
    #[error("payload commit is out of order: expected {expected}, got {got}")]
    CommitOrder {
        expected: &'static str,
        got: &'static str,
    },
    #[error("payload identity does not match the ledger baseline")]
    LedgerIdentityMismatch,
    #[error("payload differs from the preceding cache-level commit")]
    LedgerPayloadMismatch,
    #[error("payload sidecar JSON is invalid: {detail}")]
    SidecarJson { detail: String },
    #[error("payload sidecar schema/status is invalid")]
    SidecarSchema,
    #[error("payload sidecar exceeds the 64 MiB input limit")]
    SidecarTooLarge,
    #[error("payload sidecar record {index} has invalid hex bytes")]
    SidecarBytes { index: usize },
    #[error("payload sidecar record {index} has inconsistent byte mask")]
    SidecarMask { index: usize },
    #[error("payload sidecar record {index} has invalid generation")]
    SidecarGeneration { index: usize },
    #[error("payload sidecar contains no payload records")]
    SidecarPayloadsEmpty,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct NativePayloadSidecarRaw {
    schema: String,
    status: String,
    payloads: Vec<NativePayloadRecordRaw>,
    #[serde(default)]
    baseline: Option<NativePayloadBaselineRaw>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct NativePayloadBaselineRaw {
    #[serde(default)]
    input_address_u64: Option<u64>,
    #[serde(default)]
    output_address_u64: Option<u64>,
    #[serde(default)]
    input_words_u32: Option<Vec<u32>>,
    #[serde(default)]
    output_initial_word_u32: Option<u32>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct NativePayloadRecordRaw {
    allocation_id: u64,
    generation: u64,
    line_id: i64,
    sector: u8,
    byte_offset: u8,
    bytes_hex: String,
    known_mask: u32,
    sequence: u64,
    #[serde(default)]
    address_u64: Option<u64>,
    #[serde(default)]
    launch_id: Option<u64>,
    #[serde(default)]
    function_name: Option<String>,
    #[serde(default)]
    function_unique_id: Option<i64>,
    #[serde(default)]
    pc: Option<u32>,
    #[serde(default)]
    guard: Option<u32>,
    #[serde(default)]
    occurrence: Option<u32>,
    #[serde(default)]
    cta: Option<[u32; 3]>,
    #[serde(default)]
    warp: Option<u32>,
}

/// Load the strict v1/v2 sidecar emitted by the native value differential
/// probe. V2 retains richer producer metadata, but this loader still returns
/// only the sector payload primitive: normal replay does not consume it unless
/// a caller explicitly wires a complete value-bearing cache transaction.
pub fn load_native_payload_sidecar(path: &Path) -> Result<Vec<SectorPayloadV1>, PayloadError> {
    const MAX_BYTES: usize = 64 * 1024 * 1024;
    let file = std::fs::File::open(path).map_err(|error| PayloadError::SidecarJson {
        detail: format!("cannot read {}: {error}", path.display()),
    })?;
    let mut data = Vec::new();
    file.take((MAX_BYTES + 1) as u64)
        .read_to_end(&mut data)
        .map_err(|error| PayloadError::SidecarJson {
            detail: format!("cannot read {}: {error}", path.display()),
        })?;
    if data.len() > MAX_BYTES {
        return Err(PayloadError::SidecarTooLarge);
    }
    let raw: NativePayloadSidecarRaw =
        serde_json::from_slice(&data).map_err(|error| PayloadError::SidecarJson {
            detail: error.to_string(),
        })?;
    if raw.schema != "native_value_payload_sidecar_v1"
        && raw.schema != "native_value_payload_sidecar_v2"
        || raw.status != "ok"
    {
        return Err(PayloadError::SidecarSchema);
    }
    if raw.payloads.is_empty() {
        return Err(PayloadError::SidecarPayloadsEmpty);
    }
    let mut payloads = Vec::with_capacity(raw.payloads.len());
    for (index, record) in raw.payloads.into_iter().enumerate() {
        if record.generation == 0 || record.byte_offset >= SECTOR_BYTES as u8 {
            return Err(PayloadError::SidecarGeneration { index });
        }
        if record.bytes_hex.len() % 2 != 0 {
            return Err(PayloadError::SidecarBytes { index });
        }
        let mut bytes = Vec::with_capacity(record.bytes_hex.len() / 2);
        for pair in record.bytes_hex.as_bytes().chunks_exact(2) {
            let high = (pair[0] as char).to_digit(16);
            let low = (pair[1] as char).to_digit(16);
            let (Some(high), Some(low)) = (high, low) else {
                return Err(PayloadError::SidecarBytes { index });
            };
            bytes.push(((high << 4) | low) as u8);
        }
        if bytes.len() != 4
            || usize::from(record.byte_offset)
                .checked_add(bytes.len())
                .is_none_or(|end| end > SECTOR_BYTES)
        {
            return Err(PayloadError::SidecarBytes { index });
        }
        let expected_mask = (((1_u64 << bytes.len()) - 1) as u32) << record.byte_offset;
        if record.known_mask != expected_mask {
            return Err(PayloadError::SidecarMask { index });
        }
        let mut sector_bytes = [0_u8; SECTOR_BYTES];
        sector_bytes
            [usize::from(record.byte_offset)..usize::from(record.byte_offset) + bytes.len()]
            .copy_from_slice(&bytes);
        payloads.push(SectorPayloadV1::new_with_generation(
            record.allocation_id,
            record.generation,
            record.line_id,
            record.sector,
            record.sequence,
            sector_bytes,
            record.known_mask,
        )?);
    }
    Ok(payloads)
}

/// Merge ordered sidecar stores into complete sectors. The helper is
/// deliberately separate from replay: a sector is admitted to the payload
/// commit ledger only when every byte is known and all records share allocation
/// generation/line/sector identity.
pub fn coalesce_payloads(
    mut payloads: Vec<SectorPayloadV1>,
) -> Result<Vec<SectorPayloadV1>, PayloadError> {
    payloads.sort_unstable_by_key(|payload| {
        (
            payload.allocation_id(),
            payload.allocation_generation(),
            payload.line_id(),
            payload.sector(),
            payload.sequence(),
        )
    });
    let mut merged: Vec<SectorPayloadV1> = Vec::new();
    for payload in payloads {
        if let Some(current) = merged.last_mut()
            && current.allocation_id() == payload.allocation_id()
            && current.allocation_generation() == payload.allocation_generation()
            && current.line_id() == payload.line_id()
            && current.sector() == payload.sector()
        {
            *current = current.merge(payload)?;
        } else {
            merged.push(payload);
        }
    }
    for payload in &merged {
        if !payload.is_fully_known() {
            return Err(PayloadError::BytesUnknown {
                mask: !payload.known_mask(),
            });
        }
    }
    Ok(merged)
}

impl SectorPayloadV1 {
    pub fn new(
        allocation_id: u64,
        line_id: i64,
        sector: u8,
        sequence: u64,
        bytes: [u8; SECTOR_BYTES],
        known_mask: u32,
    ) -> Result<Self, PayloadError> {
        if allocation_id == 0 {
            return Err(PayloadError::AllocationIdZero);
        }
        if sector >= 4 {
            return Err(PayloadError::SectorOutOfRange { sector });
        }
        if known_mask == 0 {
            return Err(PayloadError::KnownMaskEmpty { mask: known_mask });
        }
        Ok(Self {
            allocation_id,
            allocation_generation: 0,
            line_id,
            sector,
            sequence,
            bytes,
            known_mask,
        })
    }

    pub fn new_with_generation(
        allocation_id: u64,
        allocation_generation: u64,
        line_id: i64,
        sector: u8,
        sequence: u64,
        bytes: [u8; SECTOR_BYTES],
        known_mask: u32,
    ) -> Result<Self, PayloadError> {
        let mut payload = Self::new(allocation_id, line_id, sector, sequence, bytes, known_mask)?;
        payload.allocation_generation = allocation_generation;
        Ok(payload)
    }

    pub const fn allocation_id(self) -> u64 {
        self.allocation_id
    }

    pub const fn allocation_generation(self) -> u64 {
        self.allocation_generation
    }

    pub const fn line_id(self) -> i64 {
        self.line_id
    }

    pub const fn sector(self) -> u8 {
        self.sector
    }

    pub const fn sequence(self) -> u64 {
        self.sequence
    }

    pub const fn known_mask(self) -> u32 {
        self.known_mask
    }

    pub const fn bytes(self) -> [u8; SECTOR_BYTES] {
        self.bytes
    }

    pub const fn is_fully_known(self) -> bool {
        self.known_mask == FULL_KNOWN_MASK
    }

    pub fn require_known(self, mask: u32) -> Result<(), PayloadError> {
        if mask & !self.known_mask != 0 {
            return Err(PayloadError::BytesUnknown {
                mask: mask & !self.known_mask,
            });
        }
        Ok(())
    }

    /// Apply one byte-range store and return a new payload.  The old payload
    /// is untouched if validation fails, so a lower-level admission failure
    /// cannot silently mutate resident dirty state.
    pub fn with_store(self, offset: u8, data: &[u8], sequence: u64) -> Result<Self, PayloadError> {
        if sequence <= self.sequence {
            return Err(PayloadError::SequenceRegression {
                requested: sequence,
                previous: self.sequence,
            });
        }
        let end = usize::from(offset)
            .checked_add(data.len())
            .ok_or(PayloadError::ByteRange {
                offset,
                length: data.len(),
            })?;
        if data.is_empty() || end > SECTOR_BYTES {
            return Err(PayloadError::ByteRange {
                offset,
                length: data.len(),
            });
        }
        let mut next = self;
        next.sequence = sequence;
        next.bytes[usize::from(offset)..end].copy_from_slice(data);
        let mask = (((1_u64 << data.len()) - 1) as u32) << offset;
        next.known_mask |= mask;
        Ok(next)
    }

    /// Merge a newer payload from the same allocation.  Unknown bytes in the
    /// newer record remain unknown and cannot erase bytes already known in the
    /// older record.
    pub fn merge(self, newer: Self) -> Result<Self, PayloadError> {
        if self.allocation_id != newer.allocation_id
            || self.allocation_generation != newer.allocation_generation
            || self.line_id != newer.line_id
            || self.sector != newer.sector
            || newer.sequence <= self.sequence
        {
            return Err(
                if self.allocation_id != newer.allocation_id
                    || self.allocation_generation != newer.allocation_generation
                    || self.line_id != newer.line_id
                    || self.sector != newer.sector
                {
                    PayloadError::IdentityMismatch
                } else {
                    PayloadError::SequenceRegression {
                        requested: newer.sequence,
                        previous: self.sequence,
                    }
                },
            );
        }
        let mut next = self;
        for byte in 0..SECTOR_BYTES {
            let bit = 1_u32 << byte;
            if newer.known_mask & bit != 0 {
                next.bytes[byte] = newer.bytes[byte];
            }
        }
        next.known_mask |= newer.known_mask;
        next.sequence = newer.sequence;
        Ok(next)
    }

    /// Stable non-cryptographic digest used to prove that a test-side
    /// payload was conserved through a transactional merge.  It is not an
    /// evidence/identity hash; file identity still uses the repository's
    /// content digests.
    pub fn digest64(self) -> u64 {
        let mut hash = 0xcbf29ce484222325_u64;
        for byte in self.allocation_id.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
        for byte in self.allocation_generation.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
        for byte in self.line_id.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
        hash = (hash ^ u64::from(self.sector)).wrapping_mul(0x100000001b3);
        hash = (hash ^ self.sequence).wrapping_mul(0x100000001b3);
        for byte in self.known_mask.to_le_bytes().into_iter().chain(self.bytes) {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
        hash
    }
}

/// A one-sector transactional L1→L2→DRAM conservation ledger.  It is an
/// explicit payload mechanism primitive, not a claim that native replay has
/// captured these values or that the proprietary cache protocol uses this
/// exact transition timing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadCommitLedgerV1 {
    initial: SectorPayloadV1,
    l1: Option<SectorPayloadV1>,
    l2: Option<SectorPayloadV1>,
    dram: Option<SectorPayloadV1>,
}

impl PayloadCommitLedgerV1 {
    pub fn new(initial: SectorPayloadV1) -> Result<Self, PayloadError> {
        if !initial.is_fully_known() {
            return Err(PayloadError::BytesUnknown {
                mask: !initial.known_mask(),
            });
        }
        Ok(Self {
            initial,
            l1: None,
            l2: None,
            dram: None,
        })
    }

    pub const fn initial(&self) -> SectorPayloadV1 {
        self.initial
    }

    pub const fn l1(&self) -> Option<SectorPayloadV1> {
        self.l1
    }

    pub const fn l2(&self) -> Option<SectorPayloadV1> {
        self.l2
    }

    pub const fn dram(&self) -> Option<SectorPayloadV1> {
        self.dram
    }

    fn check_identity(&self, payload: SectorPayloadV1) -> Result<(), PayloadError> {
        if payload.allocation_id() != self.initial.allocation_id()
            || payload.allocation_generation() != self.initial.allocation_generation()
            || payload.line_id() != self.initial.line_id()
            || payload.sector() != self.initial.sector()
        {
            return Err(PayloadError::LedgerIdentityMismatch);
        }
        payload.require_known(u32::MAX)
    }

    pub fn commit_l1(&mut self, payload: SectorPayloadV1) -> Result<(), PayloadError> {
        self.check_identity(payload)?;
        if self.l1.is_some() {
            return Err(PayloadError::CommitOrder {
                expected: "one L1 commit",
                got: "duplicate L1 commit",
            });
        }
        if payload.sequence() <= self.initial.sequence() {
            return Err(PayloadError::SequenceRegression {
                requested: payload.sequence(),
                previous: self.initial.sequence(),
            });
        }
        self.l1 = Some(payload);
        Ok(())
    }

    pub fn commit_l2(&mut self, payload: SectorPayloadV1) -> Result<(), PayloadError> {
        self.check_identity(payload)?;
        let Some(l1) = self.l1 else {
            return Err(PayloadError::CommitOrder {
                expected: "L1 commit",
                got: "L2 commit",
            });
        };
        if self.l2.is_some() {
            return Err(PayloadError::CommitOrder {
                expected: "one L2 commit",
                got: "duplicate L2 commit",
            });
        }
        if payload != l1 {
            return Err(PayloadError::LedgerPayloadMismatch);
        }
        self.l2 = Some(payload);
        Ok(())
    }

    pub fn commit_dram(&mut self, payload: SectorPayloadV1) -> Result<(), PayloadError> {
        self.check_identity(payload)?;
        let Some(l2) = self.l2 else {
            return Err(PayloadError::CommitOrder {
                expected: "L2 commit",
                got: "DRAM commit",
            });
        };
        if self.dram.is_some() {
            return Err(PayloadError::CommitOrder {
                expected: "one DRAM commit",
                got: "duplicate DRAM commit",
            });
        }
        if payload != l2 {
            return Err(PayloadError::LedgerPayloadMismatch);
        }
        self.dram = Some(payload);
        Ok(())
    }

    pub fn final_digest(&self) -> Option<u64> {
        self.dram.map(SectorPayloadV1::digest64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn initial() -> SectorPayloadV1 {
        SectorPayloadV1::new(7, 4, 1, 1, [0; SECTOR_BYTES], FULL_KNOWN_MASK).unwrap()
    }

    #[test]
    fn full_sector_store_preserves_known_bytes_and_identity() {
        let next = initial().with_store(4, &[1, 2, 3, 4], 2).unwrap();
        assert_eq!(next.bytes()[4..8], [1, 2, 3, 4]);
        assert!(next.is_fully_known());
        assert_eq!(next.allocation_id(), 7);
        assert_eq!(next.line_id(), 4);
        assert_eq!(next.sector(), 1);
    }

    #[test]
    fn unknown_baseline_refuses_uncovered_bytes() {
        let payload = SectorPayloadV1::new(7, 4, 0, 1, [0; SECTOR_BYTES], 0b11).unwrap();
        assert_eq!(
            payload.require_known(0b111),
            Err(PayloadError::BytesUnknown { mask: 0b100 })
        );
        assert!(payload.with_store(2, &[9], 2).is_ok());
    }

    #[test]
    fn merge_is_transactional_and_rejects_identity_or_order_errors() {
        let base = initial();
        let newer = base.with_store(0, &[9], 2).unwrap();
        let merged = base.merge(newer).unwrap();
        assert_eq!(merged.bytes()[0], 9);
        assert_eq!(base.bytes()[0], 0);
        assert_eq!(
            base.merge(base),
            Err(PayloadError::SequenceRegression {
                requested: 1,
                previous: 1
            })
        );
        let other = SectorPayloadV1::new(8, 4, 1, 3, [0; SECTOR_BYTES], FULL_KNOWN_MASK).unwrap();
        assert_eq!(merged.merge(other), Err(PayloadError::IdentityMismatch));
    }

    #[test]
    fn digest_changes_with_payload_and_sequence() {
        let a = initial();
        let b = a.with_store(0, &[1], 2).unwrap();
        assert_ne!(a.digest64(), b.digest64());
    }

    #[test]
    fn commit_ledger_requires_l1_l2_dram_order_and_conserves_digest() {
        let initial = initial();
        let l1 = initial.with_store(4, &[9, 8, 7, 6], 2).unwrap();
        let mut ledger = PayloadCommitLedgerV1::new(initial).unwrap();
        assert_eq!(
            ledger.commit_l2(l1),
            Err(PayloadError::CommitOrder {
                expected: "L1 commit",
                got: "L2 commit",
            })
        );
        ledger.commit_l1(l1).unwrap();
        ledger.commit_l2(l1).unwrap();
        ledger.commit_dram(l1).unwrap();
        assert_eq!(ledger.final_digest(), Some(l1.digest64()));
    }

    #[test]
    fn native_payload_sidecar_loader_admits_generation_and_mask() {
        let path = std::env::temp_dir().join(format!(
            "sim_mem_payload_sidecar_{}_{}.json",
            std::process::id(),
            1_u64
        ));
        let document = serde_json::json!({
            "schema": "native_value_payload_sidecar_v1",
            "status": "ok",
            "payloads": [{
                "allocation_id": 2,
                "generation": 1,
                "line_id": 3,
                "sector": 0,
                "byte_offset": 0,
                "bytes_hex": "01020304",
                "known_mask": 15,
                "sequence": 1
            }]
        });
        std::fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();
        let payloads = load_native_payload_sidecar(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0].allocation_generation(), 1);
        assert_eq!(payloads[0].bytes()[..4], [1, 2, 3, 4]);
        assert_eq!(payloads[0].known_mask(), 15);
    }

    #[test]
    fn native_payload_sidecar_loader_rejects_mask_mismatch() {
        let path = std::env::temp_dir().join(format!(
            "sim_mem_payload_sidecar_{}_{}.json",
            std::process::id(),
            2_u64
        ));
        let document = serde_json::json!({
            "schema": "native_value_payload_sidecar_v1",
            "status": "ok",
            "payloads": [{
                "allocation_id": 2,
                "generation": 1,
                "line_id": 3,
                "sector": 0,
                "byte_offset": 4,
                "bytes_hex": "01020304",
                "known_mask": 15,
                "sequence": 1
            }]
        });
        std::fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();
        assert!(matches!(
            load_native_payload_sidecar(&path),
            Err(PayloadError::SidecarMask { index: 0 })
        ));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn native_payload_sidecar_admits_retained_sm86_probe_fixture() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../workloads/research_probes/sm86_value_differential/native_payload_sidecar_delta_2pow31_seed17.json",
        );
        let payloads = load_native_payload_sidecar(&path).unwrap();
        assert_eq!(payloads.len(), 32);
        assert!(payloads.iter().all(|payload| {
            payload.allocation_id() == 2
                && payload.allocation_generation() == 1
                && payload.known_mask() != 0
        }));
    }

    #[test]
    fn retained_sm86_sidecar_coalesces_into_complete_sectors_and_commits() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../workloads/research_probes/sm86_value_differential/native_payload_sidecar_delta_2pow31_seed17.json",
        );
        let payloads = load_native_payload_sidecar(&path).unwrap();
        let sectors = coalesce_payloads(payloads).unwrap();
        assert_eq!(sectors.len(), 4);
        assert!(sectors.iter().all(|payload| payload.is_fully_known()));
        let initial = SectorPayloadV1::new_with_generation(
            2,
            1,
            sectors[0].line_id(),
            sectors[0].sector(),
            0,
            [0; SECTOR_BYTES],
            FULL_KNOWN_MASK,
        )
        .unwrap();
        let mut ledger = PayloadCommitLedgerV1::new(initial).unwrap();
        ledger.commit_l1(sectors[0]).unwrap();
        ledger.commit_l2(sectors[0]).unwrap();
        ledger.commit_dram(sectors[0]).unwrap();
        assert_eq!(ledger.final_digest(), Some(sectors[0].digest64()));
    }
}
