//! Per-lane address reconstruction for the three wire compression formats.
//!
//! Exact port of the canonical Python decoder
//! `tools/nvbit_tracer/import_trace_contract.py::_reconstruct_lane_vector`,
//! FOLLOWED BY the canonical per-lane positional check the Python gate runs
//! after reconstruction (`lane_position_mismatch`: slot i is sentinel iff
//! bit i of `active_mask & predicate_mask` is clear). The two
//! implementations MUST stay in lockstep: the Python side is the producer
//! decoder, and this side is what the simulator actually replays. Replay
//! evidence comes from the complete captured workload corpus rather than a
//! hand-authored golden fixture.
//!
//! Compressed reconstruction uses checked i64 arithmetic in both runtimes.
//! Any record whose reconstruction leaves that domain is a typed
//! [`LaneError::AddressOverflow`] (never a wrap, never a panic); producer-
//! conforming GPU virtual addresses occupy at most 57 bits.

use crate::pb;

pub const LANES_PER_WARP: usize = 32;
/// Mirrors `LANE_INACTIVE_SENTINEL` in tracer_tool.cu / import_trace_contract.py.
pub const LANE_INACTIVE_SENTINEL: i64 = -1;

pub const FORMAT_LIST_ALL: i32 = 0;
pub const FORMAT_BASE_STRIDE: i32 = 1;
pub const FORMAT_BASE_DELTA: i32 = 2;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum LaneError {
    #[error("list_all record has {got} entries; expected exactly {LANES_PER_WARP}")]
    ListAllLengthInvalid { got: usize },
    #[error(
        "lane {lane}: {} but and_mask bit is {}",
        if *slot_is_sentinel { "reconstructs to LANE_INACTIVE_SENTINEL" } else { "reconstructs to a real address" },
        if *slot_is_sentinel { "set (expected a real address)" } else { "clear (expected the sentinel)" }
    )]
    LanePositionMismatch { lane: usize, slot_is_sentinel: bool },
    #[error(
        "base_stride record has non-empty addrs (len={got}); base_stride encodes lanes via \
         base_address+stride, addrs must be empty"
    )]
    BaseStrideAddrsNonEmpty { got: usize },
    #[error(
        "base_stride record with popcount(and_mask)=1 is structurally impossible: the producer's \
         base_stride_compress requires at least two adjacent active lanes"
    )]
    BaseStrideSingleLane,
    #[error("base_stride record but active lanes are not a contiguous run starting at {first}")]
    BaseStrideNonContiguous { first: usize },
    #[error("base_delta record has {got} delta entries; expected {expected} (popcount-1, or 0)")]
    BaseDeltaCountInvalid { got: usize, expected: usize },
    #[error(
        "reconstruction leaves the i64 domain at lane {lane} (base {base:#x}); real GPU VAs \
         are ≤57-bit — corrupt record"
    )]
    AddressOverflow { lane: usize, base: u64 },
    #[error(
        "lane {lane} reconstructs to invalid negative address {address}; only -1 is the inactive sentinel"
    )]
    NegativeAddress { lane: usize, address: i64 },
    #[error(
        "unknown compression_format={0}; supported: list_all(0), base_stride(1), base_delta(2)"
    )]
    UnknownFormat(i32),
}

/// Active lane indices implied by `and_mask` (`active_mask & predicate_mask`),
/// in lane-index order. Allocation-free.
pub fn active_lane_indices(and_mask: u32) -> impl Iterator<Item = usize> {
    (0..LANES_PER_WARP).filter(move |i| (and_mask >> i) & 1 == 1)
}

/// The canonical per-lane positional check (Python `lane_position_mismatch`):
/// slot i must be sentinel exactly when bit i of `and_mask` is clear.
fn check_lane_positions(vector: &[i64; LANES_PER_WARP], and_mask: u32) -> Result<(), LaneError> {
    for (lane, &slot) in vector.iter().enumerate() {
        let bit_set = (and_mask >> lane) & 1 == 1;
        let slot_is_sentinel = slot == LANE_INACTIVE_SENTINEL;
        if bit_set == slot_is_sentinel {
            return Err(LaneError::LanePositionMismatch {
                lane,
                slot_is_sentinel,
            });
        }
    }
    Ok(())
}

/// Reconstruct the 32-entry per-lane address vector from any compression
/// format and validate it positionally against `and_mask`. Active lanes
/// carry the address the producer encoded; inactive lanes carry
/// [`LANE_INACTIVE_SENTINEL`].
pub fn reconstruct_lane_vector(
    addr: &pb::Address,
    and_mask: u32,
) -> Result<[i64; LANES_PER_WARP], LaneError> {
    let n_active = and_mask.count_ones() as usize;

    let vector = match addr.compression_format {
        FORMAT_LIST_ALL => {
            if addr.addrs.len() != LANES_PER_WARP {
                return Err(LaneError::ListAllLengthInvalid {
                    got: addr.addrs.len(),
                });
            }
            let mut vector = [LANE_INACTIVE_SENTINEL; LANES_PER_WARP];
            vector.copy_from_slice(&addr.addrs);
            vector
        }
        FORMAT_BASE_STRIDE => {
            if !addr.addrs.is_empty() {
                return Err(LaneError::BaseStrideAddrsNonEmpty {
                    got: addr.addrs.len(),
                });
            }
            let mut vector = [LANE_INACTIVE_SENTINEL; LANES_PER_WARP];
            if n_active == 0 {
                return Ok(vector); // nothing to check positionally: all clear, all sentinel
            }
            if n_active == 1 {
                return Err(LaneError::BaseStrideSingleLane);
            }
            let first = and_mask.trailing_zeros() as usize;
            // Contiguous run check without allocation: mask >> first must be
            // exactly n_active low ones.
            let run = (and_mask >> first) as u64;
            if run != (1u64 << n_active) - 1 {
                return Err(LaneError::BaseStrideNonContiguous { first });
            }
            let base =
                i64::try_from(addr.base_address).map_err(|_| LaneError::AddressOverflow {
                    lane: first,
                    base: addr.base_address,
                })?;
            for k in 0..n_active {
                let lane = first + k;
                let offset = (k as i64).checked_mul(i64::from(addr.stride)).ok_or(
                    LaneError::AddressOverflow {
                        lane,
                        base: addr.base_address,
                    },
                )?;
                vector[lane] = base.checked_add(offset).ok_or(LaneError::AddressOverflow {
                    lane,
                    base: addr.base_address,
                })?;
            }
            vector
        }
        FORMAT_BASE_DELTA => {
            let expected = n_active.saturating_sub(1);
            if addr.addrs.len() != expected {
                return Err(LaneError::BaseDeltaCountInvalid {
                    got: addr.addrs.len(),
                    expected,
                });
            }
            let mut vector = [LANE_INACTIVE_SENTINEL; LANES_PER_WARP];
            if n_active == 0 {
                return Ok(vector);
            }
            let mut lanes = active_lane_indices(and_mask);
            let first = lanes.next().expect("n_active > 0");
            let mut current =
                i64::try_from(addr.base_address).map_err(|_| LaneError::AddressOverflow {
                    lane: first,
                    base: addr.base_address,
                })?;
            vector[first] = current;
            for (k, lane) in lanes.enumerate() {
                current = current
                    .checked_add(addr.addrs[k])
                    .ok_or(LaneError::AddressOverflow {
                        lane,
                        base: addr.base_address,
                    })?;
                vector[lane] = current;
            }
            vector
        }
        other => return Err(LaneError::UnknownFormat(other)),
    };

    for (lane, &address) in vector.iter().enumerate() {
        if (and_mask >> lane) & 1 == 1 && address < 0 && address != LANE_INACTIVE_SENTINEL {
            return Err(LaneError::NegativeAddress { lane, address });
        }
    }
    check_lane_positions(&vector, and_mask)?;
    Ok(vector)
}
