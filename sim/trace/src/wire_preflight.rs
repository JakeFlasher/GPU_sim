//! Allocation-free protobuf wire preflight for repeated decoded structures.
//!
//! Raw artifact byte limits do not bound decoded heap amplification: a repeated
//! empty submessage occupies two wire bytes but still constructs a Rust value.
//! These implementation-owned ceilings are checked before Prost sees the bytes.

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
    instructions: 1_000_000,
    addresses: 1_000_000,
    lane_values: 8_000_000,
    lane_values_per_address: 32,
    decoded_nodes: 2_000_000,
};

#[derive(Debug, Clone, Copy)]
enum WireValue<'a> {
    Varint,
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

fn require_varint(field: WireField<'_>) -> Result<(), WirePreflightError> {
    match field.value {
        WireValue::Varint => Ok(()),
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
        WireValue::Varint => Ok(None),
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
            0 => {
                self.read_varint()?;
                Ok(WireValue::Varint)
            }
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
        cursor.read_varint()?;
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
    while let Some(field) = cursor.next_field()? {
        match field.number {
            1 | 3 | 7 => {
                require_length_delimited(field)?;
            }
            2 | 4 | 5 => require_varint(field)?,
            6 => {
                let map_entry = require_length_delimited(field)?;
                bump(&mut counts.devices, 1, limits.devices, "gpu devices")?;
                bump_root_node(counts, limits)?;
                scan_device_map_entry(map_entry, counts, limits)?;
            }
            8 => match require_repeated_varint(field)? {
                None => bump_memory_payload_class(counts, limits)?,
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
) -> Result<(), WirePreflightError> {
    let mut cursor = WireCursor::new(raw);
    while let Some(field) = cursor.next_field()? {
        match field.number {
            1 => require_varint(field)?,
            2 => {
                let device = require_length_delimited(field)?;
                bump_root_node(counts, limits)?;
                scan_device(device, counts, limits)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn scan_device(
    raw: &[u8],
    counts: &mut RootCounts,
    limits: RootLimits,
) -> Result<(), WirePreflightError> {
    let mut cursor = WireCursor::new(raw);
    while let Some(field) = cursor.next_field()? {
        match field.number {
            1 => require_varint(field)?,
            2 => {
                let map_entry = require_length_delimited(field)?;
                bump(&mut counts.streams, 1, limits.streams, "CUDA streams")?;
                bump_root_node(counts, limits)?;
                scan_stream_map_entry(map_entry, counts, limits)?;
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
) -> Result<(), WirePreflightError> {
    let mut cursor = WireCursor::new(raw);
    while let Some(field) = cursor.next_field()? {
        match field.number {
            1 => require_varint(field)?,
            2 => {
                let stream = require_length_delimited(field)?;
                bump_root_node(counts, limits)?;
                scan_stream(stream, counts, limits)?;
            }
            _ => {}
        }
    }
    Ok(())
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
    while let Some(field) = cursor.next_field()? {
        match field.number {
            1 | 3..=7 => require_varint(field)?,
            2 => {
                require_length_delimited(field)?;
            }
            8..=10 => {
                let dimensions = require_length_delimited(field)?;
                bump_root_node(counts, limits)?;
                scan_dim3d(dimensions)?;
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
            require_varint(field)?;
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
                scan_warp_map_entry(map_entry, counts, limits)?;
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
) -> Result<(), WirePreflightError> {
    let mut cursor = WireCursor::new(raw);
    while let Some(field) = cursor.next_field()? {
        match field.number {
            1 => require_varint(field)?,
            2 => {
                let warp = require_length_delimited(field)?;
                bump_threadblock_node(counts, limits)?;
                scan_warp(warp, counts, limits)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn scan_warp(
    raw: &[u8],
    counts: &mut ThreadblockCounts,
    limits: ThreadblockLimits,
) -> Result<(), WirePreflightError> {
    let mut cursor = WireCursor::new(raw);
    while let Some(field) = cursor.next_field()? {
        match field.number {
            1 => require_varint(field)?,
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
    while let Some(field) = cursor.next_field()? {
        match field.number {
            1..=4 | 6..=13 => require_varint(field)?,
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
            1..=5 | 7..=8 => require_varint(field)?,
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

    fn varint(mut value: usize) -> Vec<u8> {
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
        encoded.extend(varint(payload.len()));
        encoded.extend(payload);
        encoded
    }

    fn varint_field(number: u8, value: usize) -> Vec<u8> {
        let mut encoded = vec![number << 3];
        encoded.extend(varint(value));
        encoded
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
