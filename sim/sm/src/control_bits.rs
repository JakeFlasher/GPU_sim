//! SASS control-bits ingestion: the per-instruction static scheduling info
//! (stall count, yield, dependence barriers, reuse) that the SM model reads
//! INSTEAD of simulating scoreboards. The reviewed Ampere path is the active
//! implementation; other architectures require their own provenance-bound
//! layout before this decoder can be enabled.
//!
//! Input: `*.control_bits.json` sidecars produced by
//! `tools/sass_control/emit_control_bits_sidecar.py`. This loader fail-closes
//! on: layout drift vs the producer's `parser_contract`, undecoded records,
//! summary/record count mismatches, and — as a transcription guard — any
//! record whose JSON control fields disagree with re-decoding its own
//! `high_word` under the sm_86 layout (bit 41, 23 bits; turingas lineage,
//! web-verified 2026-07-06).

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;
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

pub const BARRIER_NONE: u8 = 7;

#[derive(Debug, thiserror::Error)]
pub enum ControlBitsError {
    #[error("could not read sidecar {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
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
        "sidecar reports {count} unknown_encoding instruction(s) (e.g. {function} pc={pc}); the \
         SM model needs complete static scheduling info — type-exclude this binary or extend the \
         extractor"
    )]
    UnknownEncodingsPresent {
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
    /// Write-dependence barrier index this instruction SETS (7 = none).
    pub write_barrier: u8,
    /// Read-dependence barrier index this instruction SETS (7 = none).
    pub read_barrier: u8,
    /// Bitmask of barrier indices this instruction WAITS on (6 bits).
    pub wait_mask: u8,
    /// Register-reuse cache flags (4 bits).
    pub reuse: u8,
}

impl ControlBits {
    /// Decode from the 128-bit instruction's high word: control =
    /// `(high_word >> 41) & 0x7fffff` (23 bits). Returns `None` when the
    /// reserved bits 21:22 are nonzero (the producer types such records
    /// `unknown_encoding`; a decoded record must never carry them).
    fn decode(high_word: u64) -> Option<Self> {
        let control = (high_word >> 41) & 0x7f_ffff;
        if (control >> 21) & 0b11 != 0 {
            return None;
        }
        let write_barrier = ((control >> 5) & 0x7) as u8;
        let read_barrier = ((control >> 8) & 0x7) as u8;
        // Barrier index 6 is wire-invalid on this layout: the six
        // dependence counters are SB0..SB5 and 7 encodes "none". A 6
        // can only come from corruption or decode skew — fail closed
        // like the reserved bits rather than silently treating it as
        // "none" downstream.
        if write_barrier == 6 || read_barrier == 6 {
            return None;
        }
        Some(Self {
            stall: (control & 0xf) as u8,
            yield_flag: (control >> 4) & 1 == 1,
            write_barrier,
            read_barrier,
            wait_mask: ((control >> 11) & 0x3f) as u8,
            reuse: ((control >> 17) & 0xf) as u8,
        })
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
        let raw = std::fs::read_to_string(path).map_err(|source| ControlBitsError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let sidecar: RawSidecar =
            serde_json::from_str(&raw).map_err(|source| ControlBitsError::Parse {
                path: path.display().to_string(),
                source,
            })?;
        Self::from_raw(sidecar, expected_arch.token())
    }

    fn from_raw(raw: RawSidecar, expected_arch: &str) -> Result<Self, ControlBitsError> {
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
        actual
            .control_layout()
            .require("control_layout")
            .map_err(|source| ControlBitsError::UnsupportedArchitecture {
                arch: actual.token().to_owned(),
                source,
            })?;
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
        let mut unknown: Option<(String, u64)> = None;
        let mut unknown_count: u64 = 0;
        for (fname, records) in &raw.functions {
            for rec in records {
                if rec.status != "decoded" {
                    unknown_count += 1;
                    unknown.get_or_insert_with(|| (fname.clone(), rec.pc));
                }
            }
        }
        if let Some((function, pc)) = unknown {
            return Err(ControlBitsError::UnknownEncodingsPresent {
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
                let redecoded = ControlBits::decode(high_word).ok_or_else(|| {
                    ControlBitsError::ReservedNonZero {
                        function: fname.clone(),
                        pc: rec.pc,
                        high_word,
                    }
                })?;
                let recorded_raw =
                    rec.control
                        .as_ref()
                        .ok_or_else(|| ControlBitsError::MissingControlBlock {
                            function: fname.clone(),
                            pc: rec.pc,
                        })?;
                let recorded = ControlBits {
                    stall: recorded_raw.stall,
                    yield_flag: recorded_raw.yield_flag == 1,
                    write_barrier: recorded_raw.write_barrier,
                    read_barrier: recorded_raw.read_barrier,
                    wait_mask: recorded_raw.wait_mask,
                    reuse: recorded_raw.reuse,
                };
                if recorded != redecoded {
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

#[cfg(test)]
#[allow(clippy::unwrap_used)] // unwrap in tests is an assertion, not error handling
mod tests {
    use super::*;

    /// Real record from gemm_hmma_bf16_sm80 (pc=0):
    /// high_word 0x000fe400078e00ff → control 0x7f2 →
    /// stall=2, yield=1, wb=7, rb=7, wait=0, reuse=0.
    #[test]
    fn decode_real_gemm_word() {
        let cb = ControlBits::decode(0x000f_e400_078e_00ff).expect("reserved bits zero");
        assert_eq!(
            cb,
            ControlBits {
                stall: 2,
                yield_flag: true,
                write_barrier: 7,
                read_barrier: 7,
                wait_mask: 0,
                reuse: 0,
            }
        );
    }

    /// Real record: S2UR sets write barrier 1, stall 1 (0x000e620000002600).
    #[test]
    fn decode_real_s2ur_word() {
        let cb = ControlBits::decode(0x000e_6200_0000_2600).expect("reserved bits zero");
        assert_eq!(cb.stall, 1);
        assert!(cb.yield_flag);
        assert_eq!(cb.write_barrier, 1);
        assert_eq!(cb.read_barrier, BARRIER_NONE);
    }

    fn mini_sidecar_json() -> String {
        r#"{
          "arch": "sm_86",
          "functions": {
            "k": [
              {"pc": 0, "predicate": null,
               "opcode_text": "IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28]",
               "low_word": "0x00000a00ff017624",
               "high_word": "0x000fe400078e00ff",
               "control": {"stall": 2, "yield_flag": 1, "write_barrier": 7,
                            "read_barrier": 7, "wait_mask": 0, "reuse": 0},
               "status": "decoded"}
            ]
          },
          "skipped_arch_sections": [],
          "parser_contract": {
            "supported_toolkit_versions": ["V13.2"],
            "supported_arches": ["sm_86"],
            "control_layout": {
              "source": "high_word >> 41 (23 bits)",
              "stall": "bits 0:3",
              "yield_flag": "bit 4",
              "write_barrier": "bits 5:7 (7 = none)",
              "read_barrier": "bits 8:10 (7 = none)",
              "wait_mask": "bits 11:16",
              "reuse": "bits 17:20",
              "reserved_must_be_zero": "bits 21:22"
            }
          },
          "toolkit_version": "V13.2",
          "binary": "test_bin",
          "summary": {"functions": {"k": {"instructions": 1, "decoded": 1,
                       "unknown_encoding": 0, "stall_cycles_static_sum": 2}},
                       "total_decoded": 1, "total_unknown_encoding": 0}
        }"#
        .to_string()
    }

    #[test]
    fn mini_sidecar_loads() {
        let raw: RawSidecar = serde_json::from_str(&mini_sidecar_json()).unwrap();
        let sc = ControlSidecar::from_raw(raw, "sm_86").expect("valid sidecar");
        let f = &sc.functions["k"];
        assert_eq!(f.by_pc.len(), 1);
        assert_eq!(f.by_pc[&0].control.stall, 2);
        assert!(f.by_pc[&0].opcode_text.starts_with("IMAD.MOV"));
    }

    #[test]
    fn arch_mismatch_rejected() {
        let raw: RawSidecar = serde_json::from_str(&mini_sidecar_json()).unwrap();
        assert!(matches!(
            ControlSidecar::from_raw(raw, "sm_120"),
            Err(ControlBitsError::ArchMismatch { .. })
        ));
    }

    #[test]
    fn layout_drift_rejected() {
        let json = mini_sidecar_json().replace("bits 0:3", "bits 0:4");
        let raw: RawSidecar = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            ControlSidecar::from_raw(raw, "sm_86"),
            Err(ControlBitsError::LayoutDrift { .. })
        ));
    }

    #[test]
    fn transcription_drift_rejected() {
        // Corrupt the recorded stall so it disagrees with the high_word.
        let json = mini_sidecar_json().replace(r#""stall": 2"#, r#""stall": 3"#);
        let raw: RawSidecar = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            ControlSidecar::from_raw(raw, "sm_86"),
            Err(ControlBitsError::RedecodeMismatch { .. })
        ));
    }

    #[test]
    fn unknown_encoding_record_rejected_typed() {
        let json = mini_sidecar_json()
            .replace(r#""status": "decoded""#, r#""status": "unknown_encoding""#);
        let raw: RawSidecar = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            ControlSidecar::from_raw(raw, "sm_86"),
            Err(ControlBitsError::UnknownEncodingsPresent { count: 1, .. })
        ));
    }

    /// The producer's _flush_pending path emits unknown_encoding records
    /// WITHOUT a high_word key; the sidecar must still PARSE (typed
    /// rejection, not a serde explosion).
    #[test]
    fn unknown_encoding_without_high_word_parses_then_rejects_typed() {
        let json = mini_sidecar_json()
            .replace(r#""status": "decoded""#, r#""status": "unknown_encoding""#)
            .replace(r#""high_word": "0x000fe400078e00ff","#, "");
        let raw: RawSidecar = serde_json::from_str(&json).expect("must parse without high_word");
        assert!(matches!(
            ControlSidecar::from_raw(raw, "sm_86"),
            Err(ControlBitsError::UnknownEncodingsPresent { .. })
        ));
    }

    /// status 'decoded' but no control block = structural corruption with
    /// its own variant, never smuggled through another one.
    #[test]
    fn decoded_without_control_block_typed() {
        let json = mini_sidecar_json().replace(
            r#""control": {"stall": 2, "yield_flag": 1, "write_barrier": 7,
                            "read_barrier": 7, "wait_mask": 0, "reuse": 0},"#,
            "",
        );
        let raw: RawSidecar = serde_json::from_str(&json).expect("must parse without control");
        assert!(matches!(
            ControlSidecar::from_raw(raw, "sm_86"),
            Err(ControlBitsError::MissingControlBlock { pc: 0, .. })
        ));
    }

    #[test]
    fn summary_drift_rejected() {
        let json = mini_sidecar_json().replace(r#""total_decoded": 1"#, r#""total_decoded": 2"#);
        let raw: RawSidecar = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            ControlSidecar::from_raw(raw, "sm_86"),
            Err(ControlBitsError::SummaryDrift { .. })
        ));
    }
}
