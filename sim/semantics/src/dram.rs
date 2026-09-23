//! Exact state of the modeled aggregate DRAM read server, not a DRAM-chip model.
//!
//! The native server uses fixed-point `cycle * bytes_per_sec` units. Quotient
//! and remainder retain that u128 frontier using ordinary u64 wire fields;
//! ceiling it to a cycle would introduce extra delay on the next request.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DramBoundaryState {
    /// No bandwidth server: the hierarchy still applies its latency ladder.
    LatencyOnly,
    Carded {
        bytes_per_sec: u64,
        sm_clock_hz: u64,
        free_cycle: u64,
        /// Numerator of the remaining cycle fraction; denominator is B/s.
        free_fraction: u64,
    },
}

impl DramBoundaryState {
    pub fn from_fixed_point(
        bytes_per_sec: u64,
        sm_clock_hz: u64,
        free_fp: u128,
    ) -> Result<Self, DramStateError> {
        if bytes_per_sec == 0 || sm_clock_hz == 0 {
            return Err(DramStateError::InvalidRate);
        }
        let divisor = u128::from(bytes_per_sec);
        let state = Self::Carded {
            bytes_per_sec,
            sm_clock_hz,
            free_cycle: u64::try_from(free_fp / divisor)
                .map_err(|_| DramStateError::CycleOverflow)?,
            // The remainder is strictly below a u64 divisor.
            free_fraction: (free_fp % divisor) as u64,
        };
        state.validate()?;
        Ok(state)
    }

    pub fn validate(self) -> Result<(), DramStateError> {
        if let Self::Carded {
            bytes_per_sec,
            sm_clock_hz,
            free_cycle,
            free_fraction,
        } = self
        {
            if bytes_per_sec == 0 || sm_clock_hz == 0 {
                return Err(DramStateError::InvalidRate);
            }
            if free_fraction >= bytes_per_sec {
                return Err(DramStateError::NoncanonicalFraction);
            }
            // Native service must have a representable ceil-to-cycle result.
            if free_cycle == u64::MAX && free_fraction != 0 {
                return Err(DramStateError::CycleOverflow);
            }
        }
        Ok(())
    }

    /// Reserve a read's bytes without mutating the entry state. Returns the
    /// exact next service state and this request's bandwidth completion cycle.
    /// Latency-only mode returns None: it imposes no bandwidth constraint.
    /// The caller must still combine the service completion with the hierarchy
    /// latency and merged-sector frontiers, never replace them with this time.
    pub fn preview_read(
        self,
        arrival: u64,
        bytes: u64,
    ) -> Result<Option<(Self, u64)>, DramStateError> {
        self.validate()?;
        let Self::Carded {
            bytes_per_sec,
            sm_clock_hz,
            free_cycle,
            free_fraction,
        } = self
        else {
            return Ok(None);
        };
        let rate = u128::from(bytes_per_sec);
        let frontier = u128::from(free_cycle) * rate + u128::from(free_fraction);
        let start = frontier.max(u128::from(arrival) * rate);
        let done = start
            .checked_add(u128::from(bytes) * u128::from(sm_clock_hz))
            .ok_or(DramStateError::FixedPointOverflow)?;
        let next = Self::from_fixed_point(bytes_per_sec, sm_clock_hz, done)?;
        let ready =
            u64::try_from(done.div_ceil(rate)).map_err(|_| DramStateError::CycleOverflow)?;
        Ok(Some((next, ready)))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DramStateError {
    #[error("typed DRAM rate or clock is zero or outside the native u64 domain")]
    InvalidRate,
    #[error("typed DRAM fractional numerator must be less than bytes_per_sec")]
    NoncanonicalFraction,
    #[error("typed DRAM completion exceeds the u64 cycle domain")]
    CycleOverflow,
    #[error("typed DRAM service exceeds the u128 fixed-point domain")]
    FixedPointOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractional_service_roundtrip_does_not_round_each_request() {
        let mut state = DramBoundaryState::from_fixed_point(100, 1, 0)
            .expect("valid carded DRAM rate should construct");
        let mut ready_times = Vec::new();
        for _ in 0..4 {
            // Exercise the actual JSON Value path used by typed reference deltas.
            state = serde_json::from_value(
                serde_json::to_value(state).expect("DRAM state should serialize"),
            )
            .expect("serialized DRAM state should deserialize");
            let (next, ready) = state
                .preview_read(0, 32)
                .expect("valid DRAM read should preview")
                .expect("carded DRAM should return a service result");
            state = next;
            ready_times.push(ready);
        }
        assert_eq!(ready_times, [1, 1, 1, 2]);
        assert_eq!(
            state,
            DramBoundaryState::from_fixed_point(100, 1, 128)
                .expect("valid fractional frontier should construct")
        );
        // Idle time replaces, rather than adds to, the existing frontier.
        assert_eq!(
            state
                .preview_read(10, 32)
                .expect("valid DRAM read should preview")
                .expect("carded DRAM should return a service result")
                .1,
            11
        );
        assert_eq!(
            DramBoundaryState::LatencyOnly.preview_read(10, 32),
            Ok(None)
        );
    }

    #[test]
    fn wide_frontier_and_limits_are_lossless_and_checked() {
        let wide = DramBoundaryState::from_fixed_point(u64::MAX, 1, u128::from(u64::MAX).pow(2))
            .expect("wide but representable frontier should construct");
        let json = serde_json::to_value(wide).expect("DRAM state should serialize");
        assert_eq!(
            serde_json::from_value::<DramBoundaryState>(json)
                .expect("serialized DRAM state should deserialize"),
            wide
        );
        assert_eq!(
            wide.preview_read(0, 0)
                .expect("zero-byte DRAM read should preview")
                .expect("carded DRAM should return a service result")
                .1,
            u64::MAX
        );
        assert_eq!(wide.preview_read(0, 1), Err(DramStateError::CycleOverflow));
        assert_eq!(
            DramBoundaryState::from_fixed_point(0, 1, 0),
            Err(DramStateError::InvalidRate)
        );
        assert_eq!(
            DramBoundaryState::from_fixed_point(1, 0, 0),
            Err(DramStateError::InvalidRate)
        );
        assert_eq!(
            DramBoundaryState::from_fixed_point(1, 1, u128::MAX),
            Err(DramStateError::CycleOverflow)
        );
        let bad_fraction = DramBoundaryState::Carded {
            bytes_per_sec: 10,
            sm_clock_hz: 1,
            free_cycle: 0,
            free_fraction: 10,
        };
        assert_eq!(
            bad_fraction.validate(),
            Err(DramStateError::NoncanonicalFraction)
        );
        // Every wire field is required; a mode is not inferred from missing data.
        assert!(
            serde_json::from_str::<DramBoundaryState>(
                r#"{"kind":"carded","bytes_per_sec":10,"sm_clock_hz":1,"free_cycle":0}"#
            )
            .is_err()
        );
    }
}
