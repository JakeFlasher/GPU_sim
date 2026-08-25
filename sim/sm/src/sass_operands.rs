//! Static fixed-latency SASS operand projection.
//!
//! The trusted control sidecar already carries both the complete `nvdisasm`
//! instruction text and the four encoded reuse bits.  This module projects a
//! deliberately small, exact scalar-opcode vocabulary into register operands,
//! source order, register spans, and textual reuse annotations.  The textual
//! reuse mask must equal the independently decoded control mask.  Unsupported
//! forms remain a typed [`FixedLatencyOperandStatus::Gap`]; they never acquire
//! guessed banks or timing.

use sim_arch::ArchId;

use crate::control_bits::ControlBits;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedLatencyFamily {
    ScalarAlu,
    Fma,
    Half,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedLatencyOpcode {
    Iadd3,
    Iadd3X,
    Lop3Lut,
    Imad,
    ImadIadd,
    Ffma,
    Fadd,
    Fmul,
    Hadd2,
}

impl FixedLatencyOpcode {
    pub const fn family(self) -> FixedLatencyFamily {
        match self {
            Self::Iadd3 | Self::Iadd3X | Self::Lop3Lut => FixedLatencyFamily::ScalarAlu,
            Self::Imad | Self::ImadIadd | Self::Ffma | Self::Fadd | Self::Fmul => {
                FixedLatencyFamily::Fma
            }
            Self::Hadd2 => FixedLatencyFamily::Half,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SassRegisterFile {
    Regular,
    Uniform,
    Predicate,
    UniformPredicate,
}

/// One SASS register name. `index == None` represents the file's constant
/// register (`RZ`, `URZ`, `PT`, or `UPT`), which consumes no register-file
/// port. `span_registers` retains `.64`/`.128` consecutive-register width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SassRegister {
    file: SassRegisterFile,
    index: Option<u16>,
    span_registers: u8,
}

impl SassRegister {
    pub const fn file(self) -> SassRegisterFile {
        self.file
    }

    pub const fn index(self) -> Option<u16> {
        self.index
    }

    pub const fn span_registers(self) -> u8 {
        self.span_registers
    }

    pub const fn is_constant(self) -> bool {
        self.index.is_none()
    }

    /// Ampere regular-register bank for the first register in this operand.
    /// Constant registers and non-regular files have no regular bank.
    pub const fn sm86_regular_bank(self) -> Option<u8> {
        match (self.file, self.index) {
            (SassRegisterFile::Regular, Some(index)) => Some((index & 1) as u8),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SassRegisterOperand {
    register: SassRegister,
    reused: bool,
    arithmetic_negated: bool,
    bitwise_not: bool,
    logical_not: bool,
    absolute: bool,
}

impl SassRegisterOperand {
    pub const fn register(self) -> SassRegister {
        self.register
    }

    pub const fn reused(self) -> bool {
        self.reused
    }

    pub const fn arithmetic_negated(self) -> bool {
        self.arithmetic_negated
    }

    pub const fn bitwise_not(self) -> bool {
        self.bitwise_not
    }

    pub const fn logical_not(self) -> bool {
        self.logical_not
    }

    pub const fn absolute(self) -> bool {
        self.absolute
    }

    fn has_destination_modifier(self) -> bool {
        self.reused
            || self.arithmetic_negated
            || self.bitwise_not
            || self.logical_not
            || self.absolute
            || self.register.span_registers != 1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SassSourceOperand {
    Register(SassRegisterOperand),
    Immediate,
    ConstantBank,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedLatencyOperandProjection {
    opcode: FixedLatencyOpcode,
    destinations: Box<[SassRegisterOperand]>,
    sources: Box<[SassSourceOperand]>,
    predicate_sources: Box<[SassRegisterOperand]>,
    reuse_mask: u8,
}

impl FixedLatencyOperandProjection {
    pub const fn opcode(&self) -> FixedLatencyOpcode {
        self.opcode
    }

    pub const fn family(&self) -> FixedLatencyFamily {
        self.opcode.family()
    }

    pub fn destinations(&self) -> &[SassRegisterOperand] {
        &self.destinations
    }

    pub fn sources(&self) -> &[SassSourceOperand] {
        &self.sources
    }

    pub fn predicate_sources(&self) -> &[SassRegisterOperand] {
        &self.predicate_sources
    }

    pub const fn reuse_mask(&self) -> u8 {
        self.reuse_mask
    }

    /// Raw regular-register read demand before any dynamic reuse-cache hit.
    pub fn sm86_regular_source_bank_reads(&self) -> [u8; 2] {
        let mut reads = [0u8; 2];
        for source in &self.sources {
            let SassSourceOperand::Register(operand) = source else {
                continue;
            };
            let register = operand.register;
            let Some(index) = register.index else {
                continue;
            };
            if register.file != SassRegisterFile::Regular {
                continue;
            }
            for offset in 0..u16::from(register.span_registers) {
                let bank = usize::from((index.wrapping_add(offset) & 1) as u8);
                reads[bank] = reads[bank].saturating_add(1);
            }
        }
        reads
    }

    /// One regular-bank identity per encoded data-source position. Non-regular,
    /// constant-register, immediate, and constant-bank sources are `None`.
    /// Multi-register operands return `None` for the whole projection because
    /// their several reads cannot be represented by one stage slot.
    pub fn sm86_regular_source_bank_slots(&self) -> Option<Vec<Option<u8>>> {
        let mut result = Vec::with_capacity(self.sources.len());
        for source in &self.sources {
            let bank = match source {
                SassSourceOperand::Register(operand) if operand.register.span_registers != 1 => {
                    return None;
                }
                SassSourceOperand::Register(operand) => operand.register.sm86_regular_bank(),
                SassSourceOperand::Immediate | SassSourceOperand::ConstantBank => None,
            };
            result.push(bank);
        }
        Some(result)
    }

    pub fn sm86_regular_destination_bank_writes(&self) -> [u8; 2] {
        let mut writes = [0u8; 2];
        for operand in &self.destinations {
            let register = operand.register;
            let Some(index) = register.index else {
                continue;
            };
            if register.file != SassRegisterFile::Regular {
                continue;
            }
            for offset in 0..u16::from(register.span_registers) {
                let bank = usize::from((index.wrapping_add(offset) & 1) as u8);
                writes[bank] = writes[bank].saturating_add(1);
            }
        }
        writes
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedLatencyOperandGap {
    opcode: FixedLatencyOpcode,
    reason: FixedLatencyOperandGapReason,
}

impl FixedLatencyOperandGap {
    pub const fn opcode(&self) -> FixedLatencyOpcode {
        self.opcode
    }

    pub const fn reason(&self) -> &FixedLatencyOperandGapReason {
        &self.reason
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FixedLatencyOperandGapReason {
    ArchitectureUnsupported { arch: ArchId },
    MalformedOperandList,
    UnexpectedOperandCount { expected: &'static str, got: usize },
    UnsupportedDestination { operand: String },
    UnsupportedPredicate { operand: String },
    UnsupportedSource { slot: usize, operand: String },
    ReuseMaskMismatch { encoded: u8, textual: u8 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FixedLatencyOperandStatus {
    NotApplicable,
    Projected(FixedLatencyOperandProjection),
    Gap(FixedLatencyOperandGap),
}

impl FixedLatencyOperandStatus {
    pub const fn projected(&self) -> Option<&FixedLatencyOperandProjection> {
        match self {
            Self::Projected(projection) => Some(projection),
            Self::NotApplicable | Self::Gap(_) => None,
        }
    }

    pub const fn gap(&self) -> Option<&FixedLatencyOperandGap> {
        match self {
            Self::Gap(gap) => Some(gap),
            Self::NotApplicable | Self::Projected(_) => None,
        }
    }
}

fn classify_opcode(opcode: &str) -> Option<FixedLatencyOpcode> {
    match opcode {
        "IADD3" => Some(FixedLatencyOpcode::Iadd3),
        "IADD3.X" => Some(FixedLatencyOpcode::Iadd3X),
        "LOP3.LUT" => Some(FixedLatencyOpcode::Lop3Lut),
        // Deliberately exact: the live pipe census distinguishes bare IMAD and
        // the separately exercised IMAD.IADD form from MOV/WIDE/X modifiers.
        "IMAD" => Some(FixedLatencyOpcode::Imad),
        "IMAD.IADD" => Some(FixedLatencyOpcode::ImadIadd),
        value if value == "FFMA" || value.starts_with("FFMA.") => Some(FixedLatencyOpcode::Ffma),
        value if value == "FADD" || value.starts_with("FADD.") => Some(FixedLatencyOpcode::Fadd),
        value if value == "FMUL" || value.starts_with("FMUL.") => Some(FixedLatencyOpcode::Fmul),
        "HADD2" => Some(FixedLatencyOpcode::Hadd2),
        _ => None,
    }
}

fn split_operands(text: &str) -> Option<Vec<&str>> {
    let mut result = Vec::new();
    let mut start = 0usize;
    let mut depth = 0u32;
    for (index, character) in text.char_indices() {
        match character {
            '[' | '(' | '{' => depth = depth.checked_add(1)?,
            ']' | ')' | '}' => depth = depth.checked_sub(1)?,
            ',' if depth == 0 => {
                let operand = text[start..index].trim();
                if operand.is_empty() {
                    return None;
                }
                result.push(operand);
                start = index + character.len_utf8();
            }
            _ => {}
        }
    }
    if depth != 0 {
        return None;
    }
    let last = text[start..].trim();
    if last.is_empty() {
        return None;
    }
    result.push(last);
    Some(result)
}

fn parse_index(text: &str) -> Option<u16> {
    (!text.is_empty()).then(|| text.parse().ok()).flatten()
}

fn parse_register(text: &str) -> Result<Option<SassRegisterOperand>, ()> {
    let mut value = text.trim();
    let mut arithmetic_negated = false;
    let mut bitwise_not = false;
    let mut logical_not = false;
    loop {
        if let Some(rest) = value.strip_prefix('-') {
            arithmetic_negated = true;
            value = rest;
        } else if let Some(rest) = value.strip_prefix('~') {
            bitwise_not = true;
            value = rest;
        } else if let Some(rest) = value.strip_prefix('!') {
            logical_not = true;
            value = rest;
        } else {
            break;
        }
    }
    let absolute = value.starts_with('|') && value.ends_with('|') && value.len() >= 2;
    if absolute {
        value = &value[1..value.len() - 1];
    } else if value.starts_with('|') || value.ends_with('|') {
        return Err(());
    }

    let mut components = value.split('.');
    let base = components.next().ok_or(())?;
    let (file, index) = if base == "RZ" {
        (SassRegisterFile::Regular, None)
    } else if let Some(index) = base.strip_prefix('R').and_then(parse_index) {
        (SassRegisterFile::Regular, Some(index))
    } else if base == "URZ" {
        (SassRegisterFile::Uniform, None)
    } else if let Some(index) = base.strip_prefix("UR").and_then(parse_index) {
        (SassRegisterFile::Uniform, Some(index))
    } else if base == "PT" {
        (SassRegisterFile::Predicate, None)
    } else if let Some(index) = base.strip_prefix('P').and_then(parse_index) {
        (SassRegisterFile::Predicate, Some(index))
    } else if base == "UPT" {
        (SassRegisterFile::UniformPredicate, None)
    } else if let Some(index) = base.strip_prefix("UP").and_then(parse_index) {
        (SassRegisterFile::UniformPredicate, Some(index))
    } else {
        // A decimal immediate also contains dots. Decide whether the base is a
        // register before interpreting dot-separated register modifiers.
        return Ok(None);
    };
    let mut reused = false;
    let mut span_registers = 1u8;
    for modifier in components {
        match modifier {
            "reuse" if !reused => reused = true,
            "64" if span_registers == 1 => span_registers = 2,
            "128" if span_registers == 1 => span_registers = 4,
            // Sub-register selectors do not span an additional register.
            "H0" | "H1" | "B0" | "B1" | "B2" | "B3" => {}
            _ => return Err(()),
        }
    }

    Ok(Some(SassRegisterOperand {
        register: SassRegister {
            file,
            index,
            span_registers,
        },
        reused,
        arithmetic_negated,
        bitwise_not,
        logical_not,
        absolute,
    }))
}

fn parse_destination(text: &str, predicate: bool) -> Result<SassRegisterOperand, String> {
    let operand = parse_register(text)
        .map_err(|()| text.to_owned())?
        .ok_or_else(|| text.to_owned())?;
    let expected_file = if predicate {
        SassRegisterFile::Predicate
    } else {
        SassRegisterFile::Regular
    };
    if operand.register.file != expected_file || operand.has_destination_modifier() {
        return Err(text.to_owned());
    }
    Ok(operand)
}

fn looks_like_immediate(text: &str) -> bool {
    let value = text
        .strip_prefix('-')
        .or_else(|| text.strip_prefix('+'))
        .unwrap_or(text);
    value
        .as_bytes()
        .first()
        .is_some_and(|byte| byte.is_ascii_digit() || *byte == b'.')
}

fn parse_source(
    text: &str,
    slot: usize,
) -> Result<SassSourceOperand, FixedLatencyOperandGapReason> {
    let nonregister_value = text
        .strip_prefix('-')
        .or_else(|| text.strip_prefix('~'))
        .unwrap_or(text);
    match parse_register(text) {
        Ok(Some(register)) => Ok(SassSourceOperand::Register(register)),
        Err(()) => Err(FixedLatencyOperandGapReason::UnsupportedSource {
            slot,
            operand: text.to_owned(),
        }),
        Ok(None) if nonregister_value.starts_with("c[") && nonregister_value.ends_with(']') => {
            Ok(SassSourceOperand::ConstantBank)
        }
        Ok(None) if looks_like_immediate(text) => Ok(SassSourceOperand::Immediate),
        Ok(None) => Err(FixedLatencyOperandGapReason::UnsupportedSource {
            slot,
            operand: text.to_owned(),
        }),
    }
}

fn parse_predicate_source(text: &str) -> Result<SassRegisterOperand, String> {
    let operand = parse_register(text)
        .map_err(|()| text.to_owned())?
        .ok_or_else(|| text.to_owned())?;
    if !matches!(
        operand.register.file,
        SassRegisterFile::Predicate | SassRegisterFile::UniformPredicate
    ) || operand.reused
        || operand.arithmetic_negated
        || operand.bitwise_not
        || operand.absolute
        || operand.register.span_registers != 1
    {
        return Err(text.to_owned());
    }
    Ok(operand)
}

fn project(
    opcode: FixedLatencyOpcode,
    operands: &[&str],
    control: ControlBits,
) -> Result<FixedLatencyOperandProjection, FixedLatencyOperandGapReason> {
    let mut destinations = Vec::new();
    let mut predicate_sources = Vec::new();
    let source_texts: &[&str];

    match opcode {
        FixedLatencyOpcode::Iadd3 => match operands.len() {
            4 => {
                destinations.push(parse_destination(operands[0], false).map_err(|operand| {
                    FixedLatencyOperandGapReason::UnsupportedDestination { operand }
                })?);
                source_texts = &operands[1..4];
            }
            5 => {
                destinations.push(parse_destination(operands[0], false).map_err(|operand| {
                    FixedLatencyOperandGapReason::UnsupportedDestination { operand }
                })?);
                destinations.push(parse_destination(operands[1], true).map_err(|operand| {
                    FixedLatencyOperandGapReason::UnsupportedDestination { operand }
                })?);
                source_texts = &operands[2..5];
            }
            6 => {
                destinations.push(parse_destination(operands[0], false).map_err(|operand| {
                    FixedLatencyOperandGapReason::UnsupportedDestination { operand }
                })?);
                destinations.push(parse_destination(operands[1], true).map_err(|operand| {
                    FixedLatencyOperandGapReason::UnsupportedDestination { operand }
                })?);
                predicate_sources.push(parse_predicate_source(operands[2]).map_err(|operand| {
                    FixedLatencyOperandGapReason::UnsupportedPredicate { operand }
                })?);
                source_texts = &operands[3..6];
            }
            got => {
                return Err(FixedLatencyOperandGapReason::UnexpectedOperandCount {
                    expected: "IADD3 has Rdst, optional Pdst/Psrc, and three data sources",
                    got,
                });
            }
        },
        FixedLatencyOpcode::Iadd3X => {
            if operands.len() != 6 {
                return Err(FixedLatencyOperandGapReason::UnexpectedOperandCount {
                    expected: "IADD3.X has Rdst, three data sources, and two predicate sources",
                    got: operands.len(),
                });
            }
            destinations.push(parse_destination(operands[0], false).map_err(|operand| {
                FixedLatencyOperandGapReason::UnsupportedDestination { operand }
            })?);
            source_texts = &operands[1..4];
            for text in &operands[4..6] {
                predicate_sources.push(parse_predicate_source(text).map_err(|operand| {
                    FixedLatencyOperandGapReason::UnsupportedPredicate { operand }
                })?);
            }
        }
        FixedLatencyOpcode::Lop3Lut => match operands.len() {
            6 => {
                destinations.push(parse_destination(operands[0], false).map_err(|operand| {
                    FixedLatencyOperandGapReason::UnsupportedDestination { operand }
                })?);
                source_texts = &operands[1..4];
                predicate_sources.push(parse_predicate_source(operands[5]).map_err(|operand| {
                    FixedLatencyOperandGapReason::UnsupportedPredicate { operand }
                })?);
            }
            7 => {
                destinations.push(parse_destination(operands[0], true).map_err(|operand| {
                    FixedLatencyOperandGapReason::UnsupportedDestination { operand }
                })?);
                destinations.push(parse_destination(operands[1], false).map_err(|operand| {
                    FixedLatencyOperandGapReason::UnsupportedDestination { operand }
                })?);
                source_texts = &operands[2..5];
                predicate_sources.push(parse_predicate_source(operands[6]).map_err(|operand| {
                    FixedLatencyOperandGapReason::UnsupportedPredicate { operand }
                })?);
            }
            got => {
                return Err(FixedLatencyOperandGapReason::UnexpectedOperandCount {
                    expected: "LOP3.LUT has optional Pdst, Rdst, three data sources, LUT, predicate",
                    got,
                });
            }
        },
        FixedLatencyOpcode::Imad | FixedLatencyOpcode::ImadIadd | FixedLatencyOpcode::Ffma => {
            if operands.len() != 4 {
                return Err(FixedLatencyOperandGapReason::UnexpectedOperandCount {
                    expected: "four operands",
                    got: operands.len(),
                });
            }
            destinations.push(parse_destination(operands[0], false).map_err(|operand| {
                FixedLatencyOperandGapReason::UnsupportedDestination { operand }
            })?);
            source_texts = &operands[1..4];
        }
        FixedLatencyOpcode::Fadd | FixedLatencyOpcode::Fmul | FixedLatencyOpcode::Hadd2 => {
            if operands.len() != 3 {
                return Err(FixedLatencyOperandGapReason::UnexpectedOperandCount {
                    expected: "three operands",
                    got: operands.len(),
                });
            }
            destinations.push(parse_destination(operands[0], false).map_err(|operand| {
                FixedLatencyOperandGapReason::UnsupportedDestination { operand }
            })?);
            source_texts = &operands[1..3];
        }
    }

    let mut sources = Vec::with_capacity(source_texts.len());
    let mut textual_reuse = 0u8;
    for (slot, text) in source_texts.iter().enumerate() {
        let source = parse_source(text, slot)?;
        if matches!(source, SassSourceOperand::Register(operand) if operand.reused) {
            textual_reuse |= 1u8 << slot;
        }
        sources.push(source);
    }
    if textual_reuse != control.reuse {
        return Err(FixedLatencyOperandGapReason::ReuseMaskMismatch {
            encoded: control.reuse,
            textual: textual_reuse,
        });
    }

    Ok(FixedLatencyOperandProjection {
        opcode,
        destinations: destinations.into_boxed_slice(),
        sources: sources.into_boxed_slice(),
        predicate_sources: predicate_sources.into_boxed_slice(),
        reuse_mask: textual_reuse,
    })
}

/// Project one trusted sidecar instruction. Only the exact fixed-latency
/// scalar vocabulary is considered; everything else is `NotApplicable`.
pub fn project_fixed_latency_operands(
    opcode_text: &str,
    control: ControlBits,
    arch: ArchId,
) -> FixedLatencyOperandStatus {
    let Some((opcode_text_token, operands_text)) =
        opcode_text.trim().split_once(char::is_whitespace)
    else {
        return FixedLatencyOperandStatus::NotApplicable;
    };
    let Some(opcode) = classify_opcode(opcode_text_token) else {
        return FixedLatencyOperandStatus::NotApplicable;
    };
    if arch != ArchId::Sm86 {
        return FixedLatencyOperandStatus::Gap(FixedLatencyOperandGap {
            opcode,
            reason: FixedLatencyOperandGapReason::ArchitectureUnsupported { arch },
        });
    }
    let Some(operands) = split_operands(operands_text) else {
        return FixedLatencyOperandStatus::Gap(FixedLatencyOperandGap {
            opcode,
            reason: FixedLatencyOperandGapReason::MalformedOperandList,
        });
    };
    match project(opcode, &operands, control) {
        Ok(projection) => FixedLatencyOperandStatus::Projected(projection),
        Err(reason) => FixedLatencyOperandStatus::Gap(FixedLatencyOperandGap { opcode, reason }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn control(reuse: u8) -> ControlBits {
        ControlBits::new(1, false, 7, 7, 0, reuse).expect("valid control")
    }

    fn projection(text: &str, reuse: u8) -> FixedLatencyOperandProjection {
        match project_fixed_latency_operands(text, control(reuse), ArchId::Sm86) {
            FixedLatencyOperandStatus::Projected(projection) => projection,
            other => panic!("expected projection, got {other:?}"),
        }
    }

    #[test]
    fn exact_bare_imad_boundary_excludes_imad_modifiers() {
        assert!(matches!(
            project_fixed_latency_operands("IMAD R7, R7, R8, 0x3c6ef35f", control(0), ArchId::Sm86),
            FixedLatencyOperandStatus::Projected(_)
        ));
        for text in [
            "IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28]",
            "IMAD.WIDE R2, R4, R7, c[0x0][0x160]",
            "IMAD.X R5, R3, 0x1, ~R5, P0",
        ] {
            assert_eq!(
                project_fixed_latency_operands(text, control(0), ArchId::Sm86),
                FixedLatencyOperandStatus::NotApplicable
            );
        }
    }

    #[test]
    fn exact_imad_iadd_form_projects_the_bypass_address_producer() {
        let projected = projection("IMAD.IADD R5, R3, 0x1, R9", 0);
        assert_eq!(projected.opcode(), FixedLatencyOpcode::ImadIadd);
        assert_eq!(projected.family(), FixedLatencyFamily::Fma);
        assert_eq!(projected.sources().len(), 3);
        assert_eq!(projected.sm86_regular_source_bank_reads(), [0, 2]);
        assert_eq!(projected.sm86_regular_destination_bank_writes(), [0, 1]);
    }

    #[test]
    fn iadd3_predicate_roles_do_not_shift_reuse_slots() {
        let projected = projection("IADD3 R12, P3, P4, R12, c[0x0][0x160], R8.reuse", 0b100);
        assert_eq!(projected.opcode(), FixedLatencyOpcode::Iadd3);
        assert_eq!(projected.destinations().len(), 2);
        assert_eq!(projected.predicate_sources().len(), 1);
        assert_eq!(projected.sources().len(), 3);
        assert_eq!(projected.reuse_mask(), 0b100);
        assert_eq!(projected.sm86_regular_source_bank_reads(), [2, 0]);
    }

    #[test]
    fn iadd3_x_retains_two_predicate_sources() {
        let projected = projection("IADD3.X R21, RZ, R21, RZ, P2, !PT", 0);
        assert_eq!(projected.opcode(), FixedLatencyOpcode::Iadd3X);
        assert_eq!(projected.predicate_sources().len(), 2);
        assert!(projected.predicate_sources()[1].logical_not());
        assert_eq!(projected.sm86_regular_source_bank_reads(), [0, 1]);
    }

    #[test]
    fn lop3_predicate_destination_form_is_projected() {
        let projected = projection("LOP3.LUT P0, R2, R2, 0x3, RZ, 0xc0, !PT", 0);
        assert_eq!(projected.destinations().len(), 2);
        assert_eq!(
            projected.destinations()[0].register().file(),
            SassRegisterFile::Predicate
        );
        assert_eq!(
            projected.destinations()[1].register().sm86_regular_bank(),
            Some(0)
        );
        assert_eq!(projected.sm86_regular_source_bank_reads(), [1, 0]);
    }

    #[test]
    fn ffma_reuse_and_bank_demand_are_independent() {
        let projected = projection("FFMA.RM R16, R7.reuse, R6.reuse, R7.reuse", 0b111);
        assert_eq!(projected.family(), FixedLatencyFamily::Fma);
        assert_eq!(projected.reuse_mask(), 0b111);
        assert_eq!(projected.sm86_regular_source_bank_reads(), [1, 2]);
        assert_eq!(projected.sm86_regular_destination_bank_writes(), [1, 0]);
    }

    #[test]
    fn exact_hadd2_form_projects_two_regular_sources() {
        let projected = projection("HADD2 R24, R24, R7.reuse", 0b10);
        assert_eq!(projected.opcode(), FixedLatencyOpcode::Hadd2);
        assert_eq!(projected.family(), FixedLatencyFamily::Half);
        assert_eq!(projected.sources().len(), 2);
        assert_eq!(projected.reuse_mask(), 0b10);
        assert_eq!(projected.sm86_regular_source_bank_reads(), [1, 1]);
        assert_eq!(projected.sm86_regular_destination_bank_writes(), [1, 0]);
    }

    #[test]
    fn hadd2_32i_remains_outside_the_admitted_form() {
        assert_eq!(
            project_fixed_latency_operands(
                "HADD2_32I R0, R1, 0x3c003c00",
                control(0),
                ArchId::Sm86
            ),
            FixedLatencyOperandStatus::NotApplicable
        );
    }

    #[test]
    fn source_modifiers_preserve_register_identity() {
        let projected = projection("FADD.FTZ R0, -|R3|, ~R4", 0);
        let SassSourceOperand::Register(first) = projected.sources()[0] else {
            panic!("first source is not a register")
        };
        let SassSourceOperand::Register(second) = projected.sources()[1] else {
            panic!("second source is not a register")
        };
        assert!(first.arithmetic_negated());
        assert!(first.absolute());
        assert!(second.bitwise_not());
        assert_eq!(projected.sm86_regular_source_bank_reads(), [1, 1]);
    }

    #[test]
    fn register_span_projects_consecutive_banks() {
        let register = parse_register("R3.64.reuse")
            .expect("syntax")
            .expect("register");
        assert_eq!(register.register().span_registers(), 2);
        assert_eq!(register.register().sm86_regular_bank(), Some(1));
        assert!(register.reused());
    }

    #[test]
    fn encoded_and_textual_reuse_must_match() {
        let status = project_fixed_latency_operands(
            "IMAD R26, R27, c[0x0][0x180], R29.reuse",
            control(0b001),
            ArchId::Sm86,
        );
        assert!(matches!(
            status,
            FixedLatencyOperandStatus::Gap(FixedLatencyOperandGap {
                reason: FixedLatencyOperandGapReason::ReuseMaskMismatch {
                    encoded: 0b001,
                    textual: 0b100,
                },
                ..
            })
        ));
    }

    #[test]
    fn malformed_target_is_typed_but_unrelated_opcode_is_not_applicable() {
        assert!(matches!(
            project_fixed_latency_operands("FFMA R0, R1", control(0), ArchId::Sm86),
            FixedLatencyOperandStatus::Gap(FixedLatencyOperandGap {
                reason: FixedLatencyOperandGapReason::UnexpectedOperandCount { got: 2, .. },
                ..
            })
        ));
        assert_eq!(
            project_fixed_latency_operands("LDG.E R0, [R2.64]", control(0), ArchId::Sm86),
            FixedLatencyOperandStatus::NotApplicable
        );
    }

    #[test]
    fn target_projection_is_architecture_scoped() {
        assert!(matches!(
            project_fixed_latency_operands("IMAD R0, R1, R2, R3", control(0), ArchId::Sm120),
            FixedLatencyOperandStatus::Gap(FixedLatencyOperandGap {
                reason: FixedLatencyOperandGapReason::ArchitectureUnsupported {
                    arch: ArchId::Sm120
                },
                ..
            })
        ));
    }
}
