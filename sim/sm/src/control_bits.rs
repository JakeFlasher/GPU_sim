//! SASS control-bits ingestion: the per-instruction static scheduling info
//! (stall count, yield, dependence barriers, reuse) that the SM model reads
//! INSTEAD of simulating scoreboards. The reviewed Ampere path is the active
//! implementation; other architectures require their own provenance-bound
//! layout before this decoder can be enabled.
//!
//! Input: `*.control_bits.json` sidecars produced by
//! `tools/sm120_replay_bench.py`. This loader fail-closes
//! on: layout drift vs the producer's `parser_contract`, undecoded records,
//! summary/record count mismatches, and — as a transcription guard — any
//! record whose JSON control fields disagree with re-decoding its own
//! `high_word` under the sm_86 layout (bit 41, 23 bits; turingas lineage,
//! web-verified 2026-07-06).

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use serde::Deserialize;
use serde::de::{DeserializeSeed, Error as _, MapAccess, SeqAccess, Visitor};
use sim_arch::{ArchError, ArchId, CapabilityError};

/// The exact `parser_contract.control_layout` this decoder implements. If a
/// sidecar declares anything else, the producer moved and this consumer must
/// not guess.
const EXPECTED_LAYOUT: [(&str, &str); 8] = [
    ("source", "high_word >> 41 (23 bits)"),
    ("stall", "bits 0:3"),
    ("yield_flag", "bit 4"),
    ("write_barrier", "bits 5:7 (7 = none)"),
    ("read_barrier", "bits 8:10 (7 = none)"),
    ("wait_mask", "bits 11:16"),
    ("reuse", "bits 17:20"),
    ("reserved_must_be_zero", "bits 21:22"),
];
const MAX_CONTROL_SIDECAR_BYTES: u64 = 64 * 1024 * 1024;

pub const BARRIER_NONE: Option<DepBarrier> = None;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ControlDomainError {
    #[error("stall value {stall} exceeds the 4-bit control domain")]
    StallOutOfRange { stall: u8 },
    #[error("{field} value {value} is neither dependence barrier 0..5 nor wire sentinel 7")]
    BarrierOutOfRange { field: &'static str, value: u8 },
    #[error("wait mask {wait_mask:#010b} exceeds the six dependence barriers")]
    WaitMaskOutOfRange { wait_mask: u8 },
    #[error("reuse mask {reuse:#010b} exceeds the 4-bit control domain")]
    ReuseOutOfRange { reuse: u8 },
    #[error("yield flag {value} is not a binary wire bit")]
    YieldFlagOutOfRange { value: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepBarrier(u8);

impl DepBarrier {
    pub fn new(value: u8) -> Result<Self, ControlDomainError> {
        if value < 6 {
            Ok(Self(value))
        } else {
            Err(ControlDomainError::BarrierOutOfRange {
                field: "dependence barrier",
                value,
            })
        }
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn wire(self) -> u8 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaitMask(u8);

impl WaitMask {
    pub const NONE: Self = Self(0);

    pub fn new(wait_mask: u8) -> Result<Self, ControlDomainError> {
        if wait_mask & !0x3f == 0 {
            Ok(Self(wait_mask))
        } else {
            Err(ControlDomainError::WaitMaskOutOfRange { wait_mask })
        }
    }

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub const fn contains(self, barrier: DepBarrier) -> bool {
        self.0 & (1 << barrier.0) != 0
    }

    pub const fn contains_index(self, index: usize) -> bool {
        index < 6 && self.0 & (1 << index) != 0
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ControlBitsError {
    #[error("could not read sidecar {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("control sidecar {path} is {observed} bytes, exceeding implementation limit {limit}")]
    ResourceLimit {
        path: String,
        observed: u64,
        limit: u64,
    },
    #[error("could not parse sidecar {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("sidecar is for arch {got}, expected {expected}")]
    ArchMismatch { got: String, expected: String },
    #[error("sidecar architecture {arch:?} is not a canonical deployment token: {source}")]
    ArchitectureParse {
        arch: String,
        #[source]
        source: ArchError,
    },
    #[error("architecture capability unavailable for {arch}: {source}")]
    UnsupportedArchitecture {
        arch: String,
        #[source]
        source: CapabilityError,
    },
    #[error("control_layout drift: key '{key}' is {got:?}, this decoder implements {expected:?}")]
    LayoutDrift {
        key: String,
        got: Option<String>,
        expected: String,
    },
    #[error(
        "function {function} pc={pc}: high_word {high_word:#x} re-decodes to {redecoded:?} but sidecar records {recorded:?} (transcription drift)"
    )]
    RedecodeMismatch {
        function: String,
        pc: u64,
        high_word: u64,
        recorded: ControlBits,
        redecoded: ControlBits,
    },
    #[error("function {function} pc={pc}: control domain is invalid: {source}")]
    InvalidControlDomain {
        function: String,
        pc: u64,
        #[source]
        source: ControlDomainError,
    },
    #[error("function {function} pc={pc}: reserved control bits 21:22 nonzero in {high_word:#x}")]
    ReservedNonZero {
        function: String,
        pc: u64,
        high_word: u64,
    },
    #[error("function {function} pc={pc}: malformed hex word {word:?}")]
    MalformedWord {
        function: String,
        pc: u64,
        word: String,
    },
    #[error("function {function} pc={pc}: duplicate pc in sidecar")]
    DuplicatePc { function: String, pc: u64 },
    #[error(
        "function {function} pc={pc}: status is 'decoded' but the control block is missing (structurally corrupt sidecar)"
    )]
    MissingControlBlock { function: String, pc: u64 },
    #[error(
        "function {function} pc={pc}: status is 'decoded' but high_word is missing (structurally corrupt sidecar)"
    )]
    MissingHighWord { function: String, pc: u64 },
    #[error(
        "function {function} pc={pc}: unrecognized control-record status {status:?}; expected 'decoded' or 'unknown_encoding' (structurally corrupt sidecar)"
    )]
    InvalidRecordStatus {
        function: String,
        pc: u64,
        status: String,
    },
    #[error(
        "sidecar reports {count} unsupported control encoding(s) (e.g. {function} pc={pc}); the simulator has no admitted decode for these records — type-exclude this binary or extend the extractor (this does not assert that hardware lacks the instruction)"
    )]
    UnsupportedEncoding {
        count: u64,
        function: String,
        pc: u64,
    },
    #[error("summary drift: total_decoded={summary} but {records} decoded record(s) were parsed")]
    SummaryDrift { summary: u64, records: u64 },
}

/// Decoded control word for one SASS instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlBits {
    /// Static stall cycles the scheduler must insert after issue (0-15).
    pub stall: u8,
    /// Yield hint bit as encoded (semantic interpretation is the SM model's
    /// job; this layer records the wire value).
    pub yield_flag: bool,
    /// Write-dependence barrier this instruction sets (`None` = wire 7).
    pub write_barrier: Option<DepBarrier>,
    /// Read-dependence barrier this instruction sets (`None` = wire 7).
    pub read_barrier: Option<DepBarrier>,
    /// Set of dependence barriers this instruction waits on.
    pub wait_mask: WaitMask,
    /// Register-reuse cache flags (4 bits).
    pub reuse: u8,
}

#[derive(Debug, thiserror::Error)]
enum ControlDecodeError {
    #[error("reserved control bits are nonzero")]
    ReservedNonZero,
    #[error(transparent)]
    Domain(#[from] ControlDomainError),
}

impl ControlBits {
    pub const fn stall(self) -> u8 {
        self.stall
    }

    pub const fn yield_flag(self) -> bool {
        self.yield_flag
    }

    pub const fn write_barrier(self) -> Option<DepBarrier> {
        self.write_barrier
    }

    pub const fn read_barrier(self) -> Option<DepBarrier> {
        self.read_barrier
    }

    pub const fn wait_mask(self) -> WaitMask {
        self.wait_mask
    }

    pub const fn reuse(self) -> u8 {
        self.reuse
    }

    pub fn new(
        stall: u8,
        yield_flag: bool,
        write_barrier: u8,
        read_barrier: u8,
        wait_mask: u8,
        reuse: u8,
    ) -> Result<Self, ControlDomainError> {
        if stall > 0xf {
            return Err(ControlDomainError::StallOutOfRange { stall });
        }
        if reuse > 0xf {
            return Err(ControlDomainError::ReuseOutOfRange { reuse });
        }
        Ok(Self {
            stall,
            yield_flag,
            write_barrier: Self::barrier_from_wire("write barrier", write_barrier)?,
            read_barrier: Self::barrier_from_wire("read barrier", read_barrier)?,
            wait_mask: WaitMask::new(wait_mask)?,
            reuse,
        })
    }

    fn barrier_from_wire(
        field: &'static str,
        value: u8,
    ) -> Result<Option<DepBarrier>, ControlDomainError> {
        match value {
            0..=5 => Ok(Some(DepBarrier(value))),
            7 => Ok(None),
            _ => Err(ControlDomainError::BarrierOutOfRange { field, value }),
        }
    }

    fn yield_from_wire(value: u8) -> Result<bool, ControlDomainError> {
        match value {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(ControlDomainError::YieldFlagOutOfRange { value }),
        }
    }

    pub fn validate(self) -> Result<(), ControlDomainError> {
        if self.stall > 0xf {
            return Err(ControlDomainError::StallOutOfRange { stall: self.stall });
        }
        if self.reuse > 0xf {
            return Err(ControlDomainError::ReuseOutOfRange { reuse: self.reuse });
        }
        Ok(())
    }

    /// Decode from the 128-bit instruction's high word: control =
    /// `(high_word >> 41) & 0x7fffff` (23 bits).
    fn decode(high_word: u64) -> Result<Self, ControlDecodeError> {
        let control = (high_word >> 41) & 0x7f_ffff;
        if (control >> 21) & 0b11 != 0 {
            return Err(ControlDecodeError::ReservedNonZero);
        }
        Self::new(
            (control & 0xf) as u8,
            (control >> 4) & 1 == 1,
            ((control >> 5) & 0x7) as u8,
            ((control >> 8) & 0x7) as u8,
            ((control >> 11) & 0x3f) as u8,
            ((control >> 17) & 0xf) as u8,
        )
        .map_err(ControlDecodeError::Domain)
    }
}

#[derive(Debug, Deserialize)]
struct RawControl {
    stall: u8,
    yield_flag: u8,
    write_barrier: u8,
    read_barrier: u8,
    wait_mask: u8,
    reuse: u8,
}

#[derive(Debug, Deserialize)]
struct RawRecord {
    pc: u64,
    #[serde(default)]
    predicate: Option<String>,
    opcode_text: String,
    #[serde(default)]
    high_word: Option<String>,
    status: String,
    #[serde(default)]
    control: Option<RawControl>,
}

#[derive(Debug, Deserialize)]
struct RawParserContract {
    control_layout: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct RawSummary {
    total_decoded: u64,
    total_unknown_encoding: u64,
}

#[derive(Debug, Deserialize)]
struct RawSidecar {
    arch: String,
    functions: BTreeMap<String, Vec<RawRecord>>,
    parser_contract: RawParserContract,
    binary: String,
    summary: RawSummary,
}

struct StrictJsonSeed;

struct StrictJsonVisitor;

impl<'de> DeserializeSeed<'de> for StrictJsonSeed {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_string<E>(self, _value: String) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(())
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        StrictJsonSeed.deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while sequence.next_element_seed(StrictJsonSeed)?.is_some() {}
        Ok(())
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut keys = BTreeMap::new();
        while let Some(key) = object.next_key::<String>()? {
            if keys.insert(key.clone(), ()).is_some() {
                return Err(A::Error::custom(format!(
                    "duplicate JSON object key {key:?}"
                )));
            }
            object.next_value_seed(StrictJsonSeed)?;
        }
        Ok(())
    }
}

fn reject_duplicate_json_keys(raw: &[u8]) -> Result<(), serde_json::Error> {
    let mut deserializer = serde_json::Deserializer::from_slice(raw);
    StrictJsonSeed.deserialize(&mut deserializer)?;
    deserializer.end()
}

fn read_bounded_control_sidecar(path: &Path, limit: u64) -> Result<Vec<u8>, ControlBitsError> {
    let file = std::fs::File::open(path).map_err(|source| ControlBitsError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let mut raw = Vec::new();
    file.take(limit.saturating_add(1))
        .read_to_end(&mut raw)
        .map_err(|source| ControlBitsError::Io {
            path: path.display().to_string(),
            source,
        })?;
    let observed = u64::try_from(raw.len()).unwrap_or(u64::MAX);
    if observed > limit {
        return Err(ControlBitsError::ResourceLimit {
            path: path.display().to_string(),
            observed,
            limit,
        });
    }
    Ok(raw)
}

/// One instruction's static scheduling record, keyed by pc in
/// [`FunctionControl::by_pc`].
#[derive(Debug, Clone)]
pub struct InstructionControl {
    pub control: ControlBits,
    pub opcode_text: String,
    pub predicate: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FunctionControl {
    pub by_pc: BTreeMap<u64, InstructionControl>,
}

/// All decoded functions of one binary's control-bits sidecar.
#[derive(Debug, Clone)]
pub struct ControlSidecar {
    pub arch: String,
    pub binary: String,
    pub functions: BTreeMap<String, FunctionControl>,
}

impl ControlSidecar {
    pub fn load(path: &Path, expected_arch: &str) -> Result<Self, ControlBitsError> {
        let expected = ArchId::parse_deploy(expected_arch).map_err(|source| {
            ControlBitsError::ArchitectureParse {
                arch: expected_arch.to_owned(),
                source,
            }
        })?;
        Self::load_for_arch(path, expected)
    }

    /// Resolve the stored wire token through the canonical deployment
    /// vocabulary; callers must not infer capabilities from string prefixes.
    pub fn arch_id(&self) -> Result<ArchId, ArchError> {
        ArchId::parse_deploy(&self.arch)
    }

    /// Load and validate a sidecar for a canonical deployment architecture.
    pub fn load_for_arch(path: &Path, expected_arch: ArchId) -> Result<Self, ControlBitsError> {
        let raw = read_bounded_control_sidecar(path, MAX_CONTROL_SIDECAR_BYTES)?;
        Self::parse_for_arch(path, &raw, expected_arch)
    }

    /// Parse already-snapshotted sidecar bytes under one deployment capability.
    pub fn parse_for_arch(
        path: &Path,
        raw: &[u8],
        expected_arch: ArchId,
    ) -> Result<Self, ControlBitsError> {
        Self::parse_for_arch_with_alpha(path, raw, expected_arch, false)
    }

    /// Parse a control sidecar for the explicit SM120 alpha timing lane.  The
    /// exporter supplies the same high-word bit layout used by the mature
    /// decoder, but the production SM120 control-layout capability remains a
    /// research gap.  Alpha therefore admits the wire layout only under an
    /// explicit caller flag and keeps the deployment identity as `sm_120`.
    pub fn parse_for_arch_with_alpha(
        path: &Path,
        raw: &[u8],
        expected_arch: ArchId,
        alpha_layout: bool,
    ) -> Result<Self, ControlBitsError> {
        reject_duplicate_json_keys(raw).map_err(|source| ControlBitsError::Parse {
            path: path.display().to_string(),
            source,
        })?;
        let sidecar: RawSidecar =
            serde_json::from_slice(raw).map_err(|source| ControlBitsError::Parse {
                path: path.display().to_string(),
                source,
            })?;
        Self::from_raw(sidecar, expected_arch.token(), alpha_layout)
    }

    fn from_raw(
        raw: RawSidecar,
        expected_arch: &str,
        alpha_layout: bool,
    ) -> Result<Self, ControlBitsError> {
        let expected = ArchId::parse_deploy(expected_arch).map_err(|source| {
            ControlBitsError::ArchitectureParse {
                arch: expected_arch.to_owned(),
                source,
            }
        })?;
        let actual = ArchId::parse_deploy(&raw.arch).map_err(|source| {
            ControlBitsError::ArchitectureParse {
                arch: raw.arch.clone(),
                source,
            }
        })?;
        if actual != expected {
            return Err(ControlBitsError::ArchMismatch {
                got: raw.arch,
                expected: expected.token().to_owned(),
            });
        }
        if !alpha_layout {
            actual
                .control_layout()
                .require("control_layout")
                .map_err(|source| ControlBitsError::UnsupportedArchitecture {
                    arch: actual.token().to_owned(),
                    source,
                })?;
        } else if actual != ArchId::Sm120 {
            return Err(ControlBitsError::ArchMismatch {
                got: actual.token().to_owned(),
                expected: ArchId::Sm120.token().to_owned(),
            });
        }
        for (key, expected) in EXPECTED_LAYOUT {
            match raw.parser_contract.control_layout.get(key) {
                Some(got) if got == expected => {}
                got => {
                    return Err(ControlBitsError::LayoutDrift {
                        key: key.to_string(),
                        got: got.cloned(),
                        expected: expected.to_string(),
                    });
                }
            }
        }

        // The producer writes unknown_encoding records into the sidecar (then
        // exits 3) rather than dropping them; a sidecar containing ANY is
        // inadmissible for the SM model, which needs complete static
        // scheduling info. Count them and reject typed — never per-record
        // silently, never via a serde explosion.
        let mut invalid_status: Option<(String, u64, String)> = None;
        let mut unknown: Option<(String, u64)> = None;
        let mut unknown_count: u64 = 0;
        for (fname, records) in &raw.functions {
            for rec in records {
                match rec.status.as_str() {
                    "decoded" => {}
                    "unknown_encoding" => {
                        unknown_count += 1;
                        unknown.get_or_insert_with(|| (fname.clone(), rec.pc));
                    }
                    status => {
                        invalid_status
                            .get_or_insert_with(|| (fname.clone(), rec.pc, status.to_owned()));
                    }
                }
            }
        }
        if let Some((function, pc, status)) = invalid_status {
            return Err(ControlBitsError::InvalidRecordStatus {
                function,
                pc,
                status,
            });
        }
        if let Some((function, pc)) = unknown {
            return Err(ControlBitsError::UnsupportedEncoding {
                count: unknown_count,
                function,
                pc,
            });
        }

        let mut functions = BTreeMap::new();
        let mut decoded_records: u64 = 0;
        for (fname, records) in raw.functions {
            let mut by_pc = BTreeMap::new();
            for rec in records {
                let high_word_str =
                    rec.high_word
                        .as_ref()
                        .ok_or_else(|| ControlBitsError::MissingHighWord {
                            function: fname.clone(),
                            pc: rec.pc,
                        })?;
                let high_word = parse_hex_word(high_word_str).ok_or_else(|| {
                    ControlBitsError::MalformedWord {
                        function: fname.clone(),
                        pc: rec.pc,
                        word: high_word_str.clone(),
                    }
                })?;
                let redecoded = if alpha_layout {
                    // SM120 alpha consumes the sidecar's explicit control
                    // fields as an opaque timing hypothesis.  The raw word is
                    // retained for provenance, but its Ampere bit layout is
                    // not asserted or re-decoded.
                    None
                } else {
                    Some(match ControlBits::decode(high_word) {
                        Ok(control) => control,
                        Err(ControlDecodeError::ReservedNonZero) => {
                            return Err(ControlBitsError::ReservedNonZero {
                                function: fname.clone(),
                                pc: rec.pc,
                                high_word,
                            });
                        }
                        Err(ControlDecodeError::Domain(source)) => {
                            return Err(ControlBitsError::InvalidControlDomain {
                                function: fname.clone(),
                                pc: rec.pc,
                                source,
                            });
                        }
                    })
                };
                let recorded_raw =
                    rec.control
                        .as_ref()
                        .ok_or_else(|| ControlBitsError::MissingControlBlock {
                            function: fname.clone(),
                            pc: rec.pc,
                        })?;
                // The SM120 alpha sidecar currently stores the raw SASS wire
                // bit, whereas the model field is semantic (`true` means
                // yield).  Public SASS control notation uses the zero wire
                // value for `Y`; invert only for the explicitly alpha
                // transport path.  The native SM86 decoder retains its
                // established mapping.
                let yield_flag = if alpha_layout {
                    match recorded_raw.yield_flag {
                        0 => true,
                        1 => false,
                        value => {
                            return Err(ControlBitsError::InvalidControlDomain {
                                function: fname.clone(),
                                pc: rec.pc,
                                source: ControlDomainError::YieldFlagOutOfRange { value },
                            });
                        }
                    }
                } else {
                    ControlBits::yield_from_wire(recorded_raw.yield_flag).map_err(|source| {
                        ControlBitsError::InvalidControlDomain {
                            function: fname.clone(),
                            pc: rec.pc,
                            source,
                        }
                    })?
                };
                let recorded = ControlBits::new(
                    recorded_raw.stall,
                    yield_flag,
                    recorded_raw.write_barrier,
                    recorded_raw.read_barrier,
                    recorded_raw.wait_mask,
                    recorded_raw.reuse,
                )
                .map_err(|source| ControlBitsError::InvalidControlDomain {
                    function: fname.clone(),
                    pc: rec.pc,
                    source,
                })?;
                if let Some(redecoded) = redecoded
                    && recorded != redecoded
                {
                    return Err(ControlBitsError::RedecodeMismatch {
                        function: fname.clone(),
                        pc: rec.pc,
                        high_word,
                        recorded,
                        redecoded,
                    });
                }
                let inserted = by_pc.insert(
                    rec.pc,
                    InstructionControl {
                        control: recorded,
                        opcode_text: rec.opcode_text,
                        predicate: rec.predicate,
                    },
                );
                if inserted.is_some() {
                    return Err(ControlBitsError::DuplicatePc {
                        function: fname.clone(),
                        pc: rec.pc,
                    });
                }
                decoded_records += 1;
            }
            functions.insert(fname, FunctionControl { by_pc });
        }

        if raw.summary.total_unknown_encoding != 0 {
            // Records all claimed status 'decoded' (checked above) yet the
            // summary counts unknown encodings: transcription drift.
            return Err(ControlBitsError::SummaryDrift {
                summary: raw.summary.total_unknown_encoding,
                records: 0,
            });
        }
        if raw.summary.total_decoded != decoded_records {
            return Err(ControlBitsError::SummaryDrift {
                summary: raw.summary.total_decoded,
                records: decoded_records,
            });
        }

        Ok(Self {
            arch: expected_arch.to_string(),
            binary: raw.binary,
            functions,
        })
    }
}

fn parse_hex_word(s: &str) -> Option<u64> {
    u64::from_str_radix(s.strip_prefix("0x")?, 16).ok()
}
