//! Bounded protobuf wire preflight for repeated decoded structures.
//!
//! Raw artifact byte limits do not bound decoded heap amplification: a repeated
//! empty submessage occupies two wire bytes but still constructs a Rust value.
//! These implementation-owned ceilings are checked before Prost sees the bytes;
//! map-key sets are bounded by the corresponding repeated-entry ceilings.

use std::collections::BTreeSet;

const MAX_GROUP_NESTING: u32 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WirePreflightError {
    Malformed(&'static str),
    ResourceLimit {
        resource: &'static str,
        observed: u64,
        limit: u64,
    },
}

#[derive(Debug, Clone, Copy)]
struct RootLimits {
    devices: u64,
    streams: u64,
    kernels: u64,
    ordered_events: u64,
    memory_payload_classes: u64,
    decoded_nodes: u64,
}

const ROOT_LIMITS: RootLimits = RootLimits {
    devices: 65_536,
    streams: 262_144,
    kernels: 1_000_000,
    ordered_events: 1_000_000,
    memory_payload_classes: 64,
    decoded_nodes: 2_000_000,
};

#[derive(Debug, Clone, Copy)]
struct ThreadblockLimits {
    warps: u64,
    instructions: u64,
    addresses: u64,
    lane_values: u64,
    lane_values_per_address: u64,
    decoded_nodes: u64,
}

const THREADBLOCK_LIMITS: ThreadblockLimits = ThreadblockLimits {
    warps: 65_536,
    // A single SM86 CTA can legitimately exceed one million dynamic warp
    // records (for example, an eight-warp fixed-latency probe). Keep a finite
    // decoded-resource ceiling, but do not reject a live bounded trace merely
    // because it crosses the old arbitrary 1M boundary.
    instructions: 2_000_000,
    addresses: 1_000_000,
    lane_values: 8_000_000,
    lane_values_per_address: 32,
    decoded_nodes: 2_000_000,
};

#[derive(Debug, Clone, Copy)]
enum WireValue<'a> {
    Varint(u64),
    Fixed64,
    LengthDelimited(&'a [u8]),
    Group,
    Fixed32,
}

#[derive(Debug, Clone, Copy)]
struct WireField<'a> {
    number: u32,
    value: WireValue<'a>,
}

const KNOWN_FIELD_WIRE_TYPE_MISMATCH: &str = "known protobuf field has incompatible wire type";
const UINT32_VARINT_OUT_OF_RANGE: &str = "protobuf uint32 field exceeds uint32 range";
const INT32_VARINT_OUT_OF_RANGE: &str = "protobuf int32 field is not a valid sign-extended value";
const DUPLICATE_MAP_KEY: &str = "protobuf map contains a duplicate key";
const DUPLICATE_MAP_KEY_FIELD: &str = "protobuf map entry contains a duplicate key field";
const DUPLICATE_MAP_VALUE_FIELD: &str = "protobuf map entry contains a duplicate value field";
const SIGN_EXTENDED_INT32_MIN: u64 = 0xffff_ffff_8000_0000;

fn require_varint(field: WireField<'_>) -> Result<(), WirePreflightError> {
    match field.value {
        WireValue::Varint(_) => Ok(()),
        _ => Err(WirePreflightError::Malformed(
            KNOWN_FIELD_WIRE_TYPE_MISMATCH,
        )),
    }
}

fn require_varint_value(field: WireField<'_>) -> Result<u64, WirePreflightError> {
    match field.value {
        WireValue::Varint(value) => Ok(value),
        _ => Err(WirePreflightError::Malformed(
            KNOWN_FIELD_WIRE_TYPE_MISMATCH,
        )),
    }
}

fn require_uint32(field: WireField<'_>) -> Result<(), WirePreflightError> {
    match field.value {
        WireValue::Varint(value) if u32::try_from(value).is_ok() => Ok(()),
        WireValue::Varint(_) => Err(WirePreflightError::Malformed(UINT32_VARINT_OUT_OF_RANGE)),
        _ => Err(WirePreflightError::Malformed(
            KNOWN_FIELD_WIRE_TYPE_MISMATCH,
        )),
    }
}

fn require_int32(field: WireField<'_>) -> Result<(), WirePreflightError> {
    // Protobuf's generic int32 decoder truncates a wider varint before the
    // signed conversion. Trace IDs, dimensions, widths, strides, and enum
    // fields are identity-bearing contract values, so admitting that wire form
    // would allow a malformed producer value to alias a different valid trace.
    // This is an application-level anti-truncation policy, intentionally
    // stricter than generic protobuf compatibility.
    require_int32_value(field).map(|_| ())
}

fn require_int32_value(field: WireField<'_>) -> Result<i32, WirePreflightError> {
    let value = require_varint_value(field)?;
    validate_int32_value(value)
}

fn validate_int32_value(value: u64) -> Result<i32, WirePreflightError> {
    if i32::try_from(value).is_ok() || value >= SIGN_EXTENDED_INT32_MIN {
        Ok(value as i32)
    } else {
        Err(WirePreflightError::Malformed(INT32_VARINT_OUT_OF_RANGE))
    }
}

fn require_shared_memory_int32(field: WireField<'_>) -> Result<(), WirePreflightError> {
    match field.value {
        // Protobuf int32 encodes negative values as sign-extended ten-byte
        // varints. Preserve those for the post-decode nonnegative semantic
        // check, but reject the over-wide middle range that can truncate to a
        // valid nonnegative shared-memory size.
        WireValue::Varint(value)
            if u32::try_from(value).is_ok() || value >= SIGN_EXTENDED_INT32_MIN =>
        {
            Ok(())
        }
        WireValue::Varint(_) => Err(WirePreflightError::Malformed(UINT32_VARINT_OUT_OF_RANGE)),
        _ => Err(WirePreflightError::Malformed(
            KNOWN_FIELD_WIRE_TYPE_MISMATCH,
        )),
    }
}

fn require_length_delimited(field: WireField<'_>) -> Result<&[u8], WirePreflightError> {
    match field.value {
        WireValue::LengthDelimited(payload) => Ok(payload),
        _ => Err(WirePreflightError::Malformed(
            KNOWN_FIELD_WIRE_TYPE_MISMATCH,
        )),
    }
}

fn require_repeated_varint(field: WireField<'_>) -> Result<Option<&[u8]>, WirePreflightError> {
    match field.value {
        WireValue::Varint(_) => Ok(None),
        WireValue::LengthDelimited(payload) => Ok(Some(payload)),
        _ => Err(WirePreflightError::Malformed(
            KNOWN_FIELD_WIRE_TYPE_MISMATCH,
        )),
    }
}

struct WireCursor<'a> {
    raw: &'a [u8],
    offset: usize,
}

impl<'a> WireCursor<'a> {
    fn new(raw: &'a [u8]) -> Self {
        Self { raw, offset: 0 }
    }

    fn is_empty(&self) -> bool {
        self.offset == self.raw.len()
    }

    fn read_varint(&mut self) -> Result<u64, WirePreflightError> {
        let mut value = 0_u64;
        for byte_index in 0..10_u32 {
            let byte = *self
                .raw
                .get(self.offset)
                .ok_or(WirePreflightError::Malformed("truncated varint"))?;
            self.offset += 1;
            if byte_index == 9 && byte > 1 {
                return Err(WirePreflightError::Malformed("varint exceeds uint64"));
            }
            value |= u64::from(byte & 0x7f) << (byte_index * 7);
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(WirePreflightError::Malformed("unterminated varint"))
    }

    fn read_key(&mut self) -> Result<(u32, u8), WirePreflightError> {
        let key = self.read_varint()?;
        let field_number = key >> 3;
        if field_number == 0 || field_number > 0x1fff_ffff {
            return Err(WirePreflightError::Malformed(
                "invalid protobuf field number",
            ));
        }
        let wire_type = (key & 0x07) as u8;
        if wire_type > 5 {
            return Err(WirePreflightError::Malformed("invalid protobuf wire type"));
        }
        Ok((field_number as u32, wire_type))
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], WirePreflightError> {
        let end = self
            .offset
            .checked_add(length)
            .filter(|end| *end <= self.raw.len())
            .ok_or(WirePreflightError::Malformed(
                "length-delimited field exceeds enclosing message",
            ))?;
        let value = &self.raw[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn skip_group(&mut self, expected_field: u32, depth: u32) -> Result<(), WirePreflightError> {
        if depth > MAX_GROUP_NESTING {
            return Err(WirePreflightError::Malformed(
                "group nesting exceeds preflight limit",
            ));
        }
        loop {
            if self.is_empty() {
                return Err(WirePreflightError::Malformed("unterminated group"));
            }
            let (field_number, wire_type) = self.read_key()?;
            if wire_type == 4 {
                if field_number != expected_field {
                    return Err(WirePreflightError::Malformed(
                        "group end field does not match group start",
                    ));
                }
                return Ok(());
            }
            self.skip_value(field_number, wire_type, depth)?;
        }
    }

    fn skip_value(
        &mut self,
        field_number: u32,
        wire_type: u8,
        depth: u32,
    ) -> Result<WireValue<'a>, WirePreflightError> {
        match wire_type {
            0 => Ok(WireValue::Varint(self.read_varint()?)),
            1 => {
                self.take(8)?;
                Ok(WireValue::Fixed64)
            }
            2 => {
                let length = usize::try_from(self.read_varint()?)
                    .map_err(|_| WirePreflightError::Malformed("field length exceeds usize"))?;
                Ok(WireValue::LengthDelimited(self.take(length)?))
            }
            3 => {
                self.skip_group(field_number, depth + 1)?;
                Ok(WireValue::Group)
            }
            4 => Err(WirePreflightError::Malformed("unexpected group end")),
            5 => {
                self.take(4)?;
                Ok(WireValue::Fixed32)
            }
            _ => Err(WirePreflightError::Malformed("invalid protobuf wire type")),
        }
    }

    fn next_field(&mut self) -> Result<Option<WireField<'a>>, WirePreflightError> {
        if self.is_empty() {
            return Ok(None);
        }
        let (number, wire_type) = self.read_key()?;
        let value = self.skip_value(number, wire_type, 0)?;
        Ok(Some(WireField { number, value }))
    }
}

fn bump(
    count: &mut u64,
    amount: u64,
    limit: u64,
    resource: &'static str,
) -> Result<(), WirePreflightError> {
    let observed = count.checked_add(amount).unwrap_or(u64::MAX);
    if observed > limit {
        return Err(WirePreflightError::ResourceLimit {
            resource,
            observed,
            limit,
        });
    }
    *count = observed;
    Ok(())
}

#[derive(Default)]
struct RootCounts {
    devices: u64,
    streams: u64,
    kernels: u64,
    ordered_events: u64,
    memory_payload_classes: u64,
    decoded_nodes: u64,
}

fn bump_root_node(counts: &mut RootCounts, limits: RootLimits) -> Result<(), WirePreflightError> {
    bump(
        &mut counts.decoded_nodes,
        1,
        limits.decoded_nodes,
        "root decoded nodes",
    )
}

fn bump_memory_payload_class(
    counts: &mut RootCounts,
    limits: RootLimits,
) -> Result<(), WirePreflightError> {
    bump(
        &mut counts.memory_payload_classes,
        1,
        limits.memory_payload_classes,
        "memory_payload_classes",
    )
}

fn scan_packed_root_classes(
    raw: &[u8],
    counts: &mut RootCounts,
    limits: RootLimits,
) -> Result<(), WirePreflightError> {
    let mut cursor = WireCursor::new(raw);
    while !cursor.is_empty() {
        validate_int32_value(cursor.read_varint()?)?;
        bump_memory_payload_class(counts, limits)?;
    }
    Ok(())
}

fn scan_trace(
    raw: &[u8],
    counts: &mut RootCounts,
    limits: RootLimits,
) -> Result<(), WirePreflightError> {
    let mut cursor = WireCursor::new(raw);
    // Prost implements protobuf maps with last-entry-wins semantics. Trace
    // maps are identity-bearing, so retain bounded key sets and reject an
    // encoded duplicate before generated decoding can hide it.
    let mut device_keys = BTreeSet::new();
    while let Some(field) = cursor.next_field()? {
        match field.number {
            1 | 3 | 7 => {
                require_length_delimited(field)?;
            }
            2 | 4 => require_int32(field)?,
            5 => require_varint(field)?,
            6 => {
                let map_entry = require_length_delimited(field)?;
                bump(&mut counts.devices, 1, limits.devices, "gpu devices")?;
                bump_root_node(counts, limits)?;
                let key = scan_device_map_entry(map_entry, counts, limits)?;
                if !device_keys.insert(key) {
                    return Err(WirePreflightError::Malformed(DUPLICATE_MAP_KEY));
                }
            }
            8 => match require_repeated_varint(field)? {
                None => {
                    require_int32(field)?;
                    bump_memory_payload_class(counts, limits)?;
                }
                Some(packed) => scan_packed_root_classes(packed, counts, limits)?,
            },
            _ => {}
        }
    }
    Ok(())
}

fn scan_device_map_entry(
    raw: &[u8],
    counts: &mut RootCounts,
    limits: RootLimits,
) -> Result<u64, WirePreflightError> {
    let mut cursor = WireCursor::new(raw);
    let mut key = None;
    let mut value_seen = false;
    while let Some(field) = cursor.next_field()? {
        match field.number {
            1 => {
                if key.replace(require_varint_value(field)?).is_some() {
                    return Err(WirePreflightError::Malformed(DUPLICATE_MAP_KEY_FIELD));
                }
            }
            2 => {
                if value_seen {
                    return Err(WirePreflightError::Malformed(DUPLICATE_MAP_VALUE_FIELD));
                }
                value_seen = true;
                let device = require_length_delimited(field)?;
                bump_root_node(counts, limits)?;
                scan_device(device, counts, limits)?;
            }
            _ => {}
        }
    }
    Ok(key.unwrap_or_default())
}

fn scan_device(
    raw: &[u8],
    counts: &mut RootCounts,
    limits: RootLimits,
) -> Result<(), WirePreflightError> {
    let mut cursor = WireCursor::new(raw);
    // Stream keys are part of the launch identity; duplicate encoded entries
    // must not collapse into one generated-map value.
    let mut stream_keys = BTreeSet::new();
    while let Some(field) = cursor.next_field()? {
        match field.number {
            1 => require_uint32(field)?,
            2 => {
                let map_entry = require_length_delimited(field)?;
                bump(&mut counts.streams, 1, limits.streams, "CUDA streams")?;
                bump_root_node(counts, limits)?;
                let key = scan_stream_map_entry(map_entry, counts, limits)?;
                if !stream_keys.insert(key) {
                    return Err(WirePreflightError::Malformed(DUPLICATE_MAP_KEY));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn scan_stream_map_entry(
    raw: &[u8],
    counts: &mut RootCounts,
    limits: RootLimits,
) -> Result<u64, WirePreflightError> {
    let mut cursor = WireCursor::new(raw);
    let mut key = None;
    let mut value_seen = false;
    while let Some(field) = cursor.next_field()? {
        match field.number {
            1 => {
                if key.replace(require_varint_value(field)?).is_some() {
                    return Err(WirePreflightError::Malformed(DUPLICATE_MAP_KEY_FIELD));
                }
            }
            2 => {
                if value_seen {
                    return Err(WirePreflightError::Malformed(DUPLICATE_MAP_VALUE_FIELD));
                }
                value_seen = true;
                let stream = require_length_delimited(field)?;
                bump_root_node(counts, limits)?;
                scan_stream(stream, counts, limits)?;
            }
            _ => {}
        }
    }
    Ok(key.unwrap_or_default())
}

fn scan_stream(
    raw: &[u8],
    counts: &mut RootCounts,
    limits: RootLimits,
) -> Result<(), WirePreflightError> {
    let mut cursor = WireCursor::new(raw);
    while let Some(field) = cursor.next_field()? {
        match field.number {
            1 => require_varint(field)?,
            2 => {
                let kernel = require_length_delimited(field)?;
                bump(&mut counts.kernels, 1, limits.kernels, "kernel launches")?;
                bump_root_node(counts, limits)?;
                scan_kernel(kernel, counts, limits)?;
            }
            3 => {
                require_length_delimited(field)?;
                bump(
                    &mut counts.ordered_events,
                    1,
                    limits.ordered_events,
                    "ordered CUDA events",
                )?;
                bump_root_node(counts, limits)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn scan_kernel(
    raw: &[u8],
    counts: &mut RootCounts,
    limits: RootLimits,
) -> Result<(), WirePreflightError> {
    let mut cursor = WireCursor::new(raw);
    let mut split_fields_seen = [false; 2];
    let mut launch_metadata_seen = [false; 2];
    while let Some(field) = cursor.next_field()? {
        match field.number {
            1 | 3 | 5 => require_int32(field)?,
            6 | 7 => require_varint(field)?,
            // Field 4 is a legacy int32, but valid shared-memory sizes are
            // nonnegative and participate in the uint32 split-field contract.
            // Rejecting an over-wide raw varint here prevents generated decoders
            // from truncating it before the aggregate/split equality check.
            4 => require_shared_memory_int32(field)?,
            2 => {
                require_length_delimited(field)?;
            }
            8..=10 => {
                let dimensions = require_length_delimited(field)?;
                bump_root_node(counts, limits)?;
                scan_dim3d(dimensions)?;
            }
            11..=12 => {
                require_uint32(field)?;
                let index = (field.number - 11) as usize;
                if split_fields_seen[index] {
                    return Err(WirePreflightError::Malformed(
                        "duplicate shared-memory split protobuf field",
                    ));
                }
                split_fields_seen[index] = true;
            }
            13 => {
                require_int32(field)?;
                if launch_metadata_seen[0] {
                    return Err(WirePreflightError::Malformed(
                        "duplicate shared-memory carveout protobuf field",
                    ));
                }
                launch_metadata_seen[0] = true;
            }
            14 => {
                require_uint32(field)?;
                if launch_metadata_seen[1] {
                    return Err(WirePreflightError::Malformed(
                        "duplicate driver occupancy protobuf field",
                    ));
                }
                launch_metadata_seen[1] = true;
            }
            _ => {}
        }
    }
    Ok(())
}

fn scan_dim3d(raw: &[u8]) -> Result<(), WirePreflightError> {
    let mut cursor = WireCursor::new(raw);
    while let Some(field) = cursor.next_field()? {
        if matches!(field.number, 1..=3) {
            require_int32(field)?;
        }
    }
    Ok(())
}

pub(crate) fn preflight_trace(raw: &[u8]) -> Result<(), WirePreflightError> {
    scan_trace(raw, &mut RootCounts::default(), ROOT_LIMITS)
}

#[derive(Default)]
struct ThreadblockCounts {
    warps: u64,
    instructions: u64,
    addresses: u64,
    lane_values: u64,
    decoded_nodes: u64,
}

fn bump_threadblock_node(
    counts: &mut ThreadblockCounts,
    limits: ThreadblockLimits,
) -> Result<(), WirePreflightError> {
    bump(
        &mut counts.decoded_nodes,
        1,
        limits.decoded_nodes,
        "threadblock decoded nodes",
    )
}

fn scan_threadblock(
    raw: &[u8],
    counts: &mut ThreadblockCounts,
    limits: ThreadblockLimits,
) -> Result<(), WirePreflightError> {
    let mut cursor = WireCursor::new(raw);
    // Warp map keys select the replay lane grouping and therefore cannot use
    // the generic last-entry-wins map behavior.
    let mut warp_keys = BTreeSet::new();
    while let Some(field) = cursor.next_field()? {
        match field.number {
            1 => {
                let block_id = require_length_delimited(field)?;
                bump_threadblock_node(counts, limits)?;
                scan_dim3d(block_id)?;
            }
            2 => {
                let map_entry = require_length_delimited(field)?;
                bump(&mut counts.warps, 1, limits.warps, "threadblock warps")?;
                bump_threadblock_node(counts, limits)?;
                let key = scan_warp_map_entry(map_entry, counts, limits)?;
                if !warp_keys.insert(key) {
                    return Err(WirePreflightError::Malformed(DUPLICATE_MAP_KEY));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn scan_warp_map_entry(
    raw: &[u8],
    counts: &mut ThreadblockCounts,
    limits: ThreadblockLimits,
) -> Result<i32, WirePreflightError> {
    let mut cursor = WireCursor::new(raw);
    let mut key = None;
    let mut value_seen = false;
    while let Some(field) = cursor.next_field()? {
        match field.number {
            1 => {
                if key.replace(require_int32_value(field)?).is_some() {
                    return Err(WirePreflightError::Malformed(DUPLICATE_MAP_KEY_FIELD));
                }
            }
            2 => {
                if value_seen {
                    return Err(WirePreflightError::Malformed(DUPLICATE_MAP_VALUE_FIELD));
                }
                value_seen = true;
                let warp = require_length_delimited(field)?;
                bump_threadblock_node(counts, limits)?;
                scan_warp(warp, counts, limits)?;
            }
            _ => {}
        }
    }
    Ok(key.unwrap_or_default())
}

fn scan_warp(
    raw: &[u8],
    counts: &mut ThreadblockCounts,
    limits: ThreadblockLimits,
) -> Result<(), WirePreflightError> {
    let mut cursor = WireCursor::new(raw);
    while let Some(field) = cursor.next_field()? {
        match field.number {
            1 => require_int32(field)?,
            2 => {
                let instruction = require_length_delimited(field)?;
                bump(
                    &mut counts.instructions,
                    1,
                    limits.instructions,
                    "threadblock instructions",
                )?;
                bump_threadblock_node(counts, limits)?;
                scan_instruction(instruction, counts, limits)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn scan_instruction(
    raw: &[u8],
    counts: &mut ThreadblockCounts,
    limits: ThreadblockLimits,
) -> Result<(), WirePreflightError> {
    let mut cursor = WireCursor::new(raw);
    let mut instrumentation_order_seen = false;
    while let Some(field) = cursor.next_field()? {
        match field.number {
            1..=3 => require_uint32(field)?,
            4 | 6..=10 | 12..=13 => require_int32(field)?,
            11 => require_varint(field)?,
            14 => {
                require_varint(field)?;
                if instrumentation_order_seen {
                    return Err(WirePreflightError::Malformed(
                        "duplicate instrumentation_order protobuf field",
                    ));
                }
                instrumentation_order_seen = true;
            }
            5 => {
                let address = require_length_delimited(field)?;
                bump(
                    &mut counts.addresses,
                    1,
                    limits.addresses,
                    "threadblock address records",
                )?;
                bump_threadblock_node(counts, limits)?;
                scan_address(address, counts, limits)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn bump_lane_value(
    address_lane_values: &mut u64,
    counts: &mut ThreadblockCounts,
    limits: ThreadblockLimits,
) -> Result<(), WirePreflightError> {
    bump(
        address_lane_values,
        1,
        limits.lane_values_per_address,
        "lane values per address record",
    )?;
    bump(
        &mut counts.lane_values,
        1,
        limits.lane_values,
        "threadblock lane values",
    )
}

fn scan_packed_lane_values(
    raw: &[u8],
    address_lane_values: &mut u64,
    counts: &mut ThreadblockCounts,
    limits: ThreadblockLimits,
) -> Result<(), WirePreflightError> {
    let mut cursor = WireCursor::new(raw);
    while !cursor.is_empty() {
        cursor.read_varint()?;
        bump_lane_value(address_lane_values, counts, limits)?;
    }
    Ok(())
}

fn scan_address(
    raw: &[u8],
    counts: &mut ThreadblockCounts,
    limits: ThreadblockLimits,
) -> Result<(), WirePreflightError> {
    let mut cursor = WireCursor::new(raw);
    let mut address_lane_values = 0_u64;
    while let Some(field) = cursor.next_field()? {
        match field.number {
            2 | 7 => require_uint32(field)?,
            1 | 3 | 5 | 8 => require_int32(field)?,
            4 => require_varint(field)?,
            9 => require_uint32(field)?,
            6 => match require_repeated_varint(field)? {
                None => bump_lane_value(&mut address_lane_values, counts, limits)?,
                Some(packed) => {
                    scan_packed_lane_values(packed, &mut address_lane_values, counts, limits)?;
                }
            },
            _ => {}
        }
    }
    Ok(())
}

pub(crate) fn preflight_threadblock(raw: &[u8]) -> Result<(), WirePreflightError> {
    scan_threadblock(raw, &mut ThreadblockCounts::default(), THREADBLOCK_LIMITS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn varint(mut value: u64) -> Vec<u8> {
        let mut encoded = Vec::new();
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            encoded.push(byte);
            if value == 0 {
                return encoded;
            }
        }
    }

    fn message_field(number: u8, payload: &[u8]) -> Vec<u8> {
        let mut encoded = vec![number << 3 | 2];
        encoded.extend(varint(payload.len() as u64));
        encoded.extend(payload);
        encoded
    }

    fn varint_field(number: u8, value: u64) -> Vec<u8> {
        let mut encoded = vec![number << 3];
        encoded.extend(varint(value));
        encoded
    }

    fn trace_with_kernel(kernel: &[u8]) -> Vec<u8> {
        let stream = message_field(2, kernel);
        let stream_entry = message_field(2, &stream);
        let device = message_field(2, &stream_entry);
        let device_entry = message_field(2, &device);
        message_field(6, &device_entry)
    }

    #[test]
    fn known_root_field_rejects_incompatible_wire_type() {
        assert_eq!(
            preflight_trace(&varint_field(1, 0)),
            Err(WirePreflightError::Malformed(
                KNOWN_FIELD_WIRE_TYPE_MISMATCH,
            ))
        );
    }

    #[test]
    fn known_nested_field_rejects_incompatible_wire_type() {
        let instruction = varint_field(5, 0);
        let warp = message_field(2, &instruction);
        let warp_entry = message_field(2, &warp);
        let threadblock = message_field(2, &warp_entry);
        assert_eq!(
            preflight_threadblock(&threadblock),
            Err(WirePreflightError::Malformed(
                KNOWN_FIELD_WIRE_TYPE_MISMATCH,
            ))
        );
    }

    #[test]
    fn duplicate_map_keys_are_rejected_before_prost_decode() {
        let device_entry = {
            let mut entry = varint_field(1, 0);
            entry.extend(message_field(2, &[]));
            entry
        };
        let mut duplicate_devices = message_field(6, &device_entry);
        duplicate_devices.extend(message_field(6, &device_entry));
        assert_eq!(
            preflight_trace(&duplicate_devices),
            Err(WirePreflightError::Malformed(DUPLICATE_MAP_KEY)),
        );

        let stream_entry = {
            let mut entry = varint_field(1, 0);
            entry.extend(message_field(2, &[]));
            entry
        };
        let mut device = message_field(2, &stream_entry);
        device.extend(message_field(2, &stream_entry));
        let mut device_entry = varint_field(1, 0);
        device_entry.extend(message_field(2, &device));
        assert_eq!(
            preflight_trace(&message_field(6, &device_entry)),
            Err(WirePreflightError::Malformed(DUPLICATE_MAP_KEY)),
        );

        let warp_entry = {
            let mut entry = varint_field(1, 0);
            entry.extend(message_field(2, &[]));
            entry
        };
        let mut duplicate_warps = message_field(2, &warp_entry);
        duplicate_warps.extend(message_field(2, &warp_entry));
        assert_eq!(
            preflight_threadblock(&duplicate_warps),
            Err(WirePreflightError::Malformed(DUPLICATE_MAP_KEY)),
        );
    }

    #[test]
    fn duplicate_map_key_fields_and_overwide_warp_keys_are_rejected() {
        let mut duplicate_key = varint_field(1, 0);
        duplicate_key.extend(varint_field(1, 0));
        duplicate_key.extend(message_field(2, &[]));
        assert_eq!(
            preflight_trace(&message_field(6, &duplicate_key)),
            Err(WirePreflightError::Malformed(DUPLICATE_MAP_KEY_FIELD)),
        );

        let mut overwide_key = varint_field(1, 1_u64 << 32);
        overwide_key.extend(message_field(2, &[]));
        assert_eq!(
            preflight_threadblock(&message_field(2, &overwide_key)),
            Err(WirePreflightError::Malformed(INT32_VARINT_OUT_OF_RANGE)),
        );
    }

    #[test]
    fn duplicate_map_value_fields_are_rejected_before_prost_decode() {
        let mut device_entry = varint_field(1, 0);
        device_entry.extend(message_field(2, &[]));
        device_entry.extend(message_field(2, &[]));
        assert_eq!(
            preflight_trace(&message_field(6, &device_entry)),
            Err(WirePreflightError::Malformed(DUPLICATE_MAP_VALUE_FIELD)),
        );

        let mut stream_entry = varint_field(1, 0);
        stream_entry.extend(message_field(2, &[]));
        stream_entry.extend(message_field(2, &[]));
        let mut device = varint_field(1, 0);
        device.extend(message_field(2, &stream_entry));
        let mut device_entry = varint_field(1, 0);
        device_entry.extend(message_field(2, &device));
        assert_eq!(
            preflight_trace(&message_field(6, &device_entry)),
            Err(WirePreflightError::Malformed(DUPLICATE_MAP_VALUE_FIELD)),
        );

        let mut warp_entry = varint_field(1, 0);
        warp_entry.extend(message_field(2, &[]));
        warp_entry.extend(message_field(2, &[]));
        assert_eq!(
            preflight_threadblock(&message_field(2, &warp_entry)),
            Err(WirePreflightError::Malformed(DUPLICATE_MAP_VALUE_FIELD)),
        );
    }

    #[test]
    fn embedded_warp_id_rejects_truncating_int32_varints() {
        assert_eq!(
            scan_warp(
                &varint_field(1, (1_u64 << 32) + 1),
                &mut ThreadblockCounts::default(),
                THREADBLOCK_LIMITS,
            ),
            Err(WirePreflightError::Malformed(INT32_VARINT_OUT_OF_RANGE)),
        );
    }

    #[test]
    fn repeated_memory_payload_enum_rejects_truncating_int32_varints() {
        let overwide = (1_u64 << 32) + 1;
        assert_eq!(
            preflight_trace(&varint_field(8, overwide)),
            Err(WirePreflightError::Malformed(INT32_VARINT_OUT_OF_RANGE)),
        );
        assert_eq!(
            preflight_trace(&message_field(8, &varint(overwide))),
            Err(WirePreflightError::Malformed(INT32_VARINT_OUT_OF_RANGE)),
        );
    }

    #[test]
    fn kernel_32_bit_fields_reject_overwide_varints_before_decode() {
        for field_number in [4, 11, 12, 14] {
            for value in [1_u64 << 32, (1_u64 << 32) + 4_096, 1_u64 << 63] {
                assert_eq!(
                    scan_kernel(
                        &varint_field(field_number, value),
                        &mut RootCounts::default(),
                        ROOT_LIMITS,
                    ),
                    Err(WirePreflightError::Malformed(UINT32_VARINT_OUT_OF_RANGE)),
                    "field {field_number} accepted raw varint {value}",
                );
            }
        }
        for field_number in [11, 12] {
            assert_eq!(
                scan_kernel(
                    &varint_field(field_number, u64::MAX),
                    &mut RootCounts::default(),
                    ROOT_LIMITS,
                ),
                Err(WirePreflightError::Malformed(UINT32_VARINT_OUT_OF_RANGE)),
            );
        }
        assert_eq!(
            scan_kernel(
                &varint_field(13, 1_u64 << 32),
                &mut RootCounts::default(),
                ROOT_LIMITS,
            ),
            Err(WirePreflightError::Malformed(INT32_VARINT_OUT_OF_RANGE)),
        );
        assert_eq!(
            scan_kernel(
                &varint_field(4, u64::MAX),
                &mut RootCounts::default(),
                ROOT_LIMITS,
            ),
            Ok(()),
        );
    }

    #[test]
    fn overwide_legacy_shared_memory_cannot_truncate_to_valid_value() {
        let kernel = varint_field(4, (1_u64 << 32) + 4_096);
        assert_eq!(
            preflight_trace(&trace_with_kernel(&kernel)),
            Err(WirePreflightError::Malformed(UINT32_VARINT_OUT_OF_RANGE)),
        );
    }

    #[test]
    fn uint32_fields_accept_max_boundary() {
        let mut kernel = varint_field(11, u64::from(u32::MAX));
        kernel.extend(varint_field(12, u64::from(u32::MAX)));
        assert_eq!(
            scan_kernel(&kernel, &mut RootCounts::default(), ROOT_LIMITS),
            Ok(()),
        );
        assert_eq!(
            scan_device(
                &varint_field(1, u64::from(u32::MAX)),
                &mut RootCounts::default(),
                ROOT_LIMITS,
            ),
            Ok(()),
        );

        for field_number in 1..=3 {
            assert_eq!(
                scan_instruction(
                    &varint_field(field_number, u64::from(u32::MAX)),
                    &mut ThreadblockCounts::default(),
                    THREADBLOCK_LIMITS,
                ),
                Ok(()),
            );
        }
        for field_number in [2, 7] {
            assert_eq!(
                scan_address(
                    &varint_field(field_number, u64::from(u32::MAX)),
                    &mut ThreadblockCounts::default(),
                    THREADBLOCK_LIMITS,
                ),
                Ok(()),
            );
        }
    }

    #[test]
    fn non_kernel_uint32_fields_reject_overwide_varints() {
        let overwide = 1_u64 << 32;
        assert_eq!(
            scan_device(
                &varint_field(1, overwide),
                &mut RootCounts::default(),
                ROOT_LIMITS,
            ),
            Err(WirePreflightError::Malformed(UINT32_VARINT_OUT_OF_RANGE)),
        );
        for field_number in 1..=3 {
            assert_eq!(
                scan_instruction(
                    &varint_field(field_number, overwide),
                    &mut ThreadblockCounts::default(),
                    THREADBLOCK_LIMITS,
                ),
                Err(WirePreflightError::Malformed(UINT32_VARINT_OUT_OF_RANGE)),
            );
        }
        for field_number in [2, 7] {
            assert_eq!(
                scan_address(
                    &varint_field(field_number, overwide),
                    &mut ThreadblockCounts::default(),
                    THREADBLOCK_LIMITS,
                ),
                Err(WirePreflightError::Malformed(UINT32_VARINT_OUT_OF_RANGE)),
            );
        }
    }

    #[test]
    fn signed_int32_fields_reject_truncating_middle_varints() {
        let overwide = (1_u64 << 32) + 1;
        let mut trace = varint_field(2, overwide);
        trace.extend(varint_field(4, overwide));
        assert_eq!(
            scan_trace(&trace, &mut RootCounts::default(), ROOT_LIMITS),
            Err(WirePreflightError::Malformed(INT32_VARINT_OUT_OF_RANGE)),
        );
        assert_eq!(
            scan_instruction(
                &varint_field(4, overwide),
                &mut ThreadblockCounts::default(),
                THREADBLOCK_LIMITS,
            ),
            Err(WirePreflightError::Malformed(INT32_VARINT_OUT_OF_RANGE)),
        );
        assert_eq!(
            scan_dim3d(&varint_field(1, overwide)),
            Err(WirePreflightError::Malformed(INT32_VARINT_OUT_OF_RANGE)),
        );
        assert_eq!(
            scan_address(
                &varint_field(5, overwide),
                &mut ThreadblockCounts::default(),
                THREADBLOCK_LIMITS,
            ),
            Err(WirePreflightError::Malformed(INT32_VARINT_OUT_OF_RANGE)),
        );
    }

    #[test]
    fn address_lane_mask_rejects_overwide_uint32_before_decode() {
        assert_eq!(
            scan_address(
                &varint_field(9, (1_u64 << 32) + 1),
                &mut ThreadblockCounts::default(),
                THREADBLOCK_LIMITS,
            ),
            Err(WirePreflightError::Malformed(UINT32_VARINT_OUT_OF_RANGE)),
        );
    }

    #[test]
    fn instrumentation_order_accepts_full_uint64_and_rejects_duplicate_or_wrong_wire() {
        assert_eq!(
            scan_instruction(
                &varint_field(14, u64::MAX),
                &mut ThreadblockCounts::default(),
                THREADBLOCK_LIMITS,
            ),
            Ok(()),
        );

        let mut duplicate = varint_field(14, 0);
        duplicate.extend(varint_field(14, 1));
        assert_eq!(
            scan_instruction(
                &duplicate,
                &mut ThreadblockCounts::default(),
                THREADBLOCK_LIMITS,
            ),
            Err(WirePreflightError::Malformed(
                "duplicate instrumentation_order protobuf field",
            )),
        );
        assert_eq!(
            scan_instruction(
                &message_field(14, &[]),
                &mut ThreadblockCounts::default(),
                THREADBLOCK_LIMITS,
            ),
            Err(WirePreflightError::Malformed(
                KNOWN_FIELD_WIRE_TYPE_MISMATCH,
            )),
        );
    }

    #[test]
    fn split_fields_reject_duplicates_and_wrong_wire_types() {
        let mut duplicate = varint_field(11, 0);
        duplicate.extend(varint_field(11, 1));
        assert_eq!(
            scan_kernel(&duplicate, &mut RootCounts::default(), ROOT_LIMITS),
            Err(WirePreflightError::Malformed(
                "duplicate shared-memory split protobuf field",
            )),
        );
        assert_eq!(
            scan_kernel(
                &message_field(12, &[]),
                &mut RootCounts::default(),
                ROOT_LIMITS,
            ),
            Err(WirePreflightError::Malformed(
                KNOWN_FIELD_WIRE_TYPE_MISMATCH,
            )),
        );
    }

    #[test]
    fn repeated_empty_kernel_messages_hit_root_node_budget() {
        let mut stream = Vec::new();
        for _ in 0..4 {
            stream.extend(message_field(2, &[]));
        }
        let stream_entry = message_field(2, &stream);
        let device = message_field(2, &stream_entry);
        let device_entry = message_field(2, &device);
        let trace = message_field(6, &device_entry);
        let limits = RootLimits {
            devices: 8,
            streams: 8,
            kernels: 3,
            ordered_events: 8,
            memory_payload_classes: 8,
            decoded_nodes: 32,
        };
        assert_eq!(
            scan_trace(&trace, &mut RootCounts::default(), limits),
            Err(WirePreflightError::ResourceLimit {
                resource: "kernel launches",
                observed: 4,
                limit: 3,
            })
        );
    }

    #[test]
    fn repeated_empty_instruction_messages_hit_per_file_budget() {
        let mut warp = Vec::new();
        for _ in 0..4 {
            warp.extend(message_field(2, &[]));
        }
        let warp_entry = message_field(2, &warp);
        let threadblock = message_field(2, &warp_entry);
        let limits = ThreadblockLimits {
            warps: 8,
            instructions: 3,
            addresses: 8,
            lane_values: 64,
            lane_values_per_address: 32,
            decoded_nodes: 32,
        };
        assert_eq!(
            scan_threadblock(&threadblock, &mut ThreadblockCounts::default(), limits,),
            Err(WirePreflightError::ResourceLimit {
                resource: "threadblock instructions",
                observed: 4,
                limit: 3,
            })
        );
    }

    #[test]
    fn address_lane_vector_is_bounded_before_decode() {
        let packed = vec![0_u8; 33];
        let address = message_field(6, &packed);
        let instruction = message_field(5, &address);
        let warp = message_field(2, &instruction);
        let warp_entry = message_field(2, &warp);
        let threadblock = message_field(2, &warp_entry);
        assert_eq!(
            preflight_threadblock(&threadblock),
            Err(WirePreflightError::ResourceLimit {
                resource: "lane values per address record",
                observed: 33,
                limit: 32,
            })
        );
    }
}
