//! Static SASS operand projection for register readiness.
//!
//! The trusted control sidecar already carries both the complete `nvdisasm`
//! instruction text and the four encoded reuse bits.  This module projects a
//! deliberately small, exact scalar-opcode vocabulary into register operands,
//! source order, register spans, and textual reuse annotations.  The textual
//! reuse mask must equal the decoded control mask on SM86. SM120 projects
//! explicit register roles without importing an SM86 bank/reuse layout. Unsupported
//! forms remain a typed [`FixedLatencyOperandStatus::Gap`]; they never acquire
//! guessed banks or timing.

use sim_arch::ArchId;

use crate::control_bits::ControlBits;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedLatencyFamily {
    ScalarAlu,
    Fma,
    Half,
    Generic,
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
    Generic,
}

impl FixedLatencyOpcode {
    pub const fn family(self) -> FixedLatencyFamily {
        match self {
            Self::Iadd3 | Self::Iadd3X | Self::Lop3Lut => FixedLatencyFamily::ScalarAlu,
            Self::Imad | Self::ImadIadd | Self::Ffma | Self::Fadd | Self::Fmul => {
                FixedLatencyFamily::Fma
            }
            Self::Hadd2 => FixedLatencyFamily::Half,
            Self::Generic => FixedLatencyFamily::Generic,
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
    /// Constituent 32-bit register cells, preserving partial-register writes.
    pub(crate) fn components(self) -> impl Iterator<Item = Self> {
        (0..self.span_registers).filter_map(move |offset| {
            Some(Self {
                file: self.file,
                index: Some(self.index?.checked_add(u16::from(offset))?),
                span_registers: 1,
            })
        })
    }
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

fn classify_opcode(opcode: &str, arch: ArchId) -> Option<FixedLatencyOpcode> {
    if arch == ArchId::Sm120 && (generic_opcode(opcode) || opcode.starts_with("IADD3") || opcode.starts_with("LOP3")) {
        return Some(FixedLatencyOpcode::Generic);
    }
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

/// Blackwell scalar/register families for which the static text exposes a
/// conventional destination followed by register/immediate/predicate
/// operands. This is a role projection only: it does not assert a physical
/// issue port, scoreboard implementation, or latency coefficient.
fn generic_opcode(opcode: &str) -> bool {
    let head = opcode.split('.').next().unwrap_or(opcode);
    matches!(
        head,
        "IADD"
            | "UIADD"
            | "UIADD3"
            | "IMAD"
            | "UIMAD"
            | "LEA"
            | "ULEA"
            | "SHF"
            | "USHF"
            | "SHFL"
            | "VOTE"
            | "P2R"
            | "R2P"
            | "R2UR"
            | "FSEL"
            | "FMNMX"
            | "USEL"
            | "SEL"
            | "IABS"
            | "POPC"
            | "FLO"
            | "FCHK"
            | "MUFU"
            | "PRMT"
            | "MOV"
            | "UMOV"
            | "FADD"
            | "FMUL"
            | "FFMA"
            | "DADD"
            | "DMUL"
            | "DFMA"
            | "HADD2"
            | "HFMA2"
            | "HMUL2"
            | "F2F"
            | "F2I"
            | "I2F"
            | "I2I"
            | "F2FP"
            | "F2IP"
            | "I2FP"
            | "IMNMX"
            | "VIMNMX"
            | "UVIMNMX"
            | "BFE"
            | "BFI"
            | "SHL"
            | "SHR"
            | "USHL"
            | "USHR"
            | "ISET"
            | "ISETP"
            | "FSET"
            | "FSETP"
            | "UISETP"
            | "UFSETP"
            | "PLOP3"
            | "UPLOP3"
            | "ULOP3"
            | "LDMATRIX"
            | "LDSM"
            | "LD" | "LDG" | "LDL" | "LDS" | "LDC" | "LDCU" | "ULDC"
            | "STG" | "STL" | "STS"
            | "ULOP"
            | "LOP"
    )
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
            "H0" | "H1" | "B0" | "B1" | "B2" | "B3"
            | "H0_H0" | "H1_H1" | "B0_B0" | "B1_B1" | "B2_B2" | "B3_B3" => {}
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
    let valid_file = if predicate {
        operand.register.file == SassRegisterFile::Predicate
            && matches!(operand.register.index, Some(0..=6))
    } else {
        operand.register.file == SassRegisterFile::Regular
    };
    if !valid_file || operand.has_destination_modifier() {
        return Err(text.to_owned());
    }
    Ok(operand)
}

fn parse_generic_destination(text: &str) -> Result<SassRegisterOperand, String> {
    let operand = parse_register(text)
        .map_err(|()| text.to_owned())?
        .ok_or_else(|| text.to_owned())?;
    if operand.has_destination_modifier() {
        return Err(text.to_owned());
    }
    match operand.register.file {
        SassRegisterFile::Regular
        | SassRegisterFile::Uniform
        | SassRegisterFile::Predicate
        | SassRegisterFile::UniformPredicate => Ok(operand),
    }
}

fn parse_generic_source(
    text: &str,
    slot: usize,
) -> Result<(Option<SassSourceOperand>, Option<SassRegisterOperand>), FixedLatencyOperandGapReason> {
    match parse_register(text) {
        Ok(Some(operand)) => match operand.register.file {
            SassRegisterFile::Regular | SassRegisterFile::Uniform
                if !operand.logical_not => Ok((Some(SassSourceOperand::Register(operand)), None)),
            SassRegisterFile::Predicate | SassRegisterFile::UniformPredicate => {
                Ok((None, Some(operand)))
            }
            _ => Err(FixedLatencyOperandGapReason::UnsupportedSource {
                slot,
                operand: text.to_owned(),
            }),
        },
        Ok(None) if looks_like_immediate(text) => Ok((Some(SassSourceOperand::Immediate), None)),
        Ok(None) if text.trim_start().starts_with("c[") => {
            Ok((Some(SassSourceOperand::ConstantBank), None))
        }
        Ok(None) => Err(FixedLatencyOperandGapReason::UnsupportedSource {
            slot,
            operand: text.to_owned(),
        }),
        Err(()) => Err(FixedLatencyOperandGapReason::UnsupportedSource {
            slot,
            operand: text.to_owned(),
        }),
    }
}

fn project_generic(
    opcode_text: &str,
    operands: &[&str],
    _control: ControlBits,
) -> Result<FixedLatencyOperandProjection, FixedLatencyOperandGapReason> {
    let head = opcode_text.split('.').next().unwrap_or(opcode_text);
    let modifiers: Vec<_> = opcode_text.split('.').skip(1).collect();
    let has = |modifier: &str| modifiers.contains(&modifier);
    let is_store = matches!(head, "STG" | "STL" | "STS");
    let is_memory = is_store || matches!(head, "LD" | "LDG" | "LDL" | "LDS" | "LDSM" | "LDC" | "LDCU" | "ULDC");
    // SHFL/VOTE transfer values between lanes; a same-lane ready ledger is
    // insufficient. PR pack/unpack and uniform-from-lane selection also need
    // dedicated semantics. Do not fabricate destinations from position zero.
    if matches!(head, "SHFL" | "VOTE" | "P2R" | "R2P" | "R2UR" | "LDMATRIX") {
        return Err(FixedLatencyOperandGapReason::UnsupportedSource {
            slot: 0, operand: format!("{head}: cross-lane or aggregate-register semantics unavailable"),
        });
    }
    if operands.len() < 2 || (matches!(head, "SHF" | "USHF") && (has("U64") || has("S64"))) {
        return Err(FixedLatencyOperandGapReason::UnexpectedOperandCount {
            expected: "supported explicit scalar or memory register form",
            got: operands.len(),
        });
    }
    let mut destination_count = if is_store { 0 } else { 1 };
    if matches!(head, "ISETP" | "UISETP" | "FSETP" | "UFSETP" | "PLOP3" | "UPLOP3") {
        destination_count = 2;
    } else if matches!(head, "IADD" | "IADD3" | "UIADD" | "UIADD3" | "LEA" | "ULEA" | "IMAD" | "UIMAD") {
        // Carry outputs precede the data sources. Trailing predicates are
        // inputs. PT/UPT outputs are discarded constants, not missing roles.
        while destination_count < operands.len()
            && parse_register(operands[destination_count]).ok().flatten().is_some_and(|operand| {
                matches!(operand.register.file, SassRegisterFile::Predicate | SassRegisterFile::UniformPredicate)
            })
        {
            destination_count += 1;
        }
    } else if matches!(head, "LOP3" | "ULOP3")
        && parse_register(operands[0]).ok().flatten().is_some_and(|operand| {
            matches!(operand.register.file, SassRegisterFile::Predicate | SassRegisterFile::UniformPredicate)
        })
    {
        destination_count = 2;
    }
    if destination_count >= operands.len() {
        return Err(FixedLatencyOperandGapReason::UnexpectedOperandCount {
            expected: "at least one source after destinations", got: operands.len(),
        });
    }
    let conversion = matches!(head, "F2F" | "F2I" | "I2F" | "I2I");
    let types: Vec<_> = modifiers.iter().copied()
        .filter(|token| matches!(*token, "F64" | "F32" | "F16" | "BF16" | "U64" | "S64" | "U32" | "S32" | "U16" | "S16" | "U8" | "S8"))
        .collect();
    // Conversion suffixes omit default types. A lone S64 on I2F is
    // the source width, not a two-register FP32 destination; a lone F64
    // on F2I is likewise a source type.
    let destination_pair = conversion && if matches!(head, "I2F" | "F2F") {
        types.iter().find(|t| t.starts_with('F')).is_some_and(|t| *t == "F64")
    } else {
        types.iter().find(|t| t.starts_with('S') || t.starts_with('U')).is_some_and(|t| t.ends_with("64"))
    };
    let source_pair = conversion && if head == "I2F" {
        types.iter().find(|t| t.starts_with('S') || t.starts_with('U')).is_some_and(|t| t.ends_with("64"))
    } else if head == "F2I" {
        types.iter().find(|t| t.starts_with('F')).is_some_and(|t| *t == "F64")
    } else {
        types.get(1).is_some_and(|t| t.ends_with("64"))
    };
    let pair_alu = matches!(head, "DADD" | "DMUL" | "DFMA") || (has("64") && matches!(head, "IADD" | "IADD3" | "UIADD" | "UIADD3" | "MOV" | "UMOV"))
        || (head == "IMNMX" && (has("S64") || has("U64")));
    let width = if head == "LDSM" {
        // Observed SM120 matrix-load destination register vector, not the
        // .16 element width. Other forms remain an explicit projection gap.
        if has("M88") && has("4") { 4 } else if has("M88") && has("2") { 2 }
        else if has("M88") && has("1") { 1 } else {
            return Err(FixedLatencyOperandGapReason::UnsupportedDestination {
                operand: opcode_text.to_owned(),
            });
        }
    } else if has("128") { 4 } else if has("64") { 2 } else { 1 };
    let wide_imad = matches!(head, "IMAD" | "UIMAD") && has("WIDE");
    let mut destinations = Vec::new();
    for text in &operands[..destination_count] {
        let mut operand = parse_generic_destination(text).map_err(|operand| {
            FixedLatencyOperandGapReason::UnsupportedDestination { operand }
        })?;
        if matches!(operand.register.file, SassRegisterFile::Regular | SassRegisterFile::Uniform) {
            operand.register.span_registers = if is_memory { width } else if pair_alu || wide_imad
                || destination_pair { 2 } else { 1 };
        }
        destinations.push(operand);
    }
    let mut sources = Vec::new();
    let mut predicate_sources = Vec::new();
    let mut textual_reuse = 0u8;
    for (slot, text) in operands[destination_count..].iter().enumerate() {
        if text.contains('[') {
            // Constant-bank and memory descriptors can contain register
            // offsets (including UR descriptor pointers). Preserve these RAWs
            // without interpreting the address or cache policy.
            for token in text.split(|c: char| !c.is_ascii_alphanumeric() && c != '.') {
                if token.is_empty() { continue; }
                if let Ok(Some(operand)) = parse_register(token) {
                    sources.push(SassSourceOperand::Register(operand));
                }
            }
            continue;
        }
        let (source, predicate) = parse_generic_source(text, slot)?;
        if let Some(mut source) = source {
            if let SassSourceOperand::Register(ref mut operand) = source {
                let pair = pair_alu || (wide_imad && slot == 2)
                    || source_pair
                    || (matches!(head, "ISETP" | "UISETP" | "IMNMX") && (has("S64") || has("U64")));
                if is_store { operand.register.span_registers = width; }
                else if pair { operand.register.span_registers = 2; }
                if operand.reused && slot < 8 { textual_reuse |= 1 << slot; }
            }
            sources.push(source);
        }
        if let Some(predicate) = predicate { predicate_sources.push(predicate); }
    }
    // PACK_*_MERGE_C retains destination components, hence a real RAW on C.
    if opcode_text.contains("MERGE") {
        sources.extend(destinations.iter().copied().map(SassSourceOperand::Register));
    }
    // These are logical register roles, not register-bank/reuse-cache timing.
    // A hypothesized SM120 high-word reuse layout must not discard an explicit
    // SASS RAW/WAW edge. The strict SM86 projection below is unchanged.
    Ok(FixedLatencyOperandProjection {
        opcode: FixedLatencyOpcode::Generic,
        destinations: destinations.into_boxed_slice(),
        sources: sources.into_boxed_slice(),
        predicate_sources: predicate_sources.into_boxed_slice(),
        reuse_mask: textual_reuse,
    })
}

pub(crate) fn project_guard_predicate(text: Option<&str>) -> Option<SassRegister> {
    let operand = parse_register(text?.trim_start_matches('@')).ok()??;
    matches!(operand.register.file, SassRegisterFile::Predicate | SassRegisterFile::UniformPredicate)
        .then_some(operand.register)
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
        Ok(Some(register))
            if matches!(
                register.register.file,
                SassRegisterFile::Regular | SassRegisterFile::Uniform
            ) && !register.logical_not =>
        {
            Ok(SassSourceOperand::Register(register))
        }
        Ok(Some(_)) => Err(FixedLatencyOperandGapReason::UnsupportedSource {
            slot,
            operand: text.to_owned(),
        }),
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
    let mut predicate_text = text.trim();
    if let Some(rest) = predicate_text.strip_prefix('!') {
        predicate_text = rest;
    }
    if predicate_text.starts_with('!') || predicate_text.contains('.') {
        return Err(text.to_owned());
    }
    let operand = parse_register(text)
        .map_err(|()| text.to_owned())?
        .ok_or_else(|| text.to_owned())?;
    if operand.register.file != SassRegisterFile::Predicate
        || !matches!(operand.register.index, None | Some(0..=6))
        || operand.reused
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
        FixedLatencyOpcode::Generic => unreachable!("SM120 roles are projected separately"),
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
    let Some(opcode) = classify_opcode(opcode_text_token, arch) else {
        return FixedLatencyOperandStatus::NotApplicable;
    };
    if !matches!(arch, ArchId::Sm86 | ArchId::Sm120) {
        return FixedLatencyOperandStatus::Gap(FixedLatencyOperandGap {
            opcode,
            reason: FixedLatencyOperandGapReason::ArchitectureUnsupported { arch },
        });
    }
    let Some(operands) = split_operands(operands_text.trim().trim_end_matches(';').trim()) else {
        return FixedLatencyOperandStatus::Gap(FixedLatencyOperandGap {
            opcode,
            reason: FixedLatencyOperandGapReason::MalformedOperandList,
        });
    };
    let result = if opcode == FixedLatencyOpcode::Generic {
        project_generic(opcode_text_token, &operands, control)
    } else {
        project(opcode, &operands, control)
    };
    match result {
        Ok(projection) => FixedLatencyOperandStatus::Projected(projection),
        Err(reason) => FixedLatencyOperandStatus::Gap(FixedLatencyOperandGap { opcode, reason }),
    }
}

