//! The observations an exact or accelerated run promises to preserve.
//!
//! This contract is intentionally separate from the event representation. A
//! simulator is only required to bulk-update state and counters that influence
//! the selected observations; callers must not infer stronger accuracy from a
//! successful event reduction.

use serde::{Deserialize, Serialize};

pub const OBSERVATION_SCHEMA_VERSION: u32 = 1;

/// How finely an observation must be preserved by an accelerated executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationGranularity {
    /// The observation is outside the run's declared contract.
    Omitted,
    /// Preserve the value at declared boundaries (launch, external event, and
    /// completion), without promising every intermediate cycle.
    Boundary,
    /// Preserve the observation at every modeled cycle or an exactly
    /// equivalent per-cycle integral where the contract permits one.
    PerCycle,
}

impl ObservationGranularity {
    pub const fn requires_per_cycle(self) -> bool {
        matches!(self, Self::PerCycle)
    }

    pub const fn is_declared(self) -> bool {
        !matches!(self, Self::Omitted)
    }
}

/// A versioned, explicit accuracy contract for one simulation run.
///
/// The contract is model-relative: it describes the simulator state and
/// observations that an accelerated path must preserve. It is not a claim that
/// the selected observations match physical silicon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationContract {
    pub schema_version: u32,
    pub issue_order: ObservationGranularity,
    pub completion_order: ObservationGranularity,
    pub memory_transactions: ObservationGranularity,
    pub synchronization: ObservationGranularity,
    pub resource_state: ObservationGranularity,
    pub counters: ObservationGranularity,
    pub power: ObservationGranularity,
}

impl ObservationContract {
    /// Strict contract for the current detailed SM86 model. Power is omitted
    /// because the model does not expose a validated power transition system.
    pub const fn sm86_cycle_exact() -> Self {
        Self {
            schema_version: OBSERVATION_SCHEMA_VERSION,
            issue_order: ObservationGranularity::PerCycle,
            completion_order: ObservationGranularity::PerCycle,
            memory_transactions: ObservationGranularity::PerCycle,
            synchronization: ObservationGranularity::PerCycle,
            resource_state: ObservationGranularity::PerCycle,
            counters: ObservationGranularity::PerCycle,
            power: ObservationGranularity::Omitted,
        }
    }

    /// Boundary-only contract for semantic or functional replay. This cannot
    /// be used to make a cycle-accuracy claim.
    pub const fn boundary_replay() -> Self {
        Self {
            schema_version: OBSERVATION_SCHEMA_VERSION,
            issue_order: ObservationGranularity::Boundary,
            completion_order: ObservationGranularity::Boundary,
            memory_transactions: ObservationGranularity::Boundary,
            synchronization: ObservationGranularity::Boundary,
            resource_state: ObservationGranularity::Boundary,
            counters: ObservationGranularity::Boundary,
            power: ObservationGranularity::Omitted,
        }
    }

    pub const fn requires_per_cycle_state(self) -> bool {
        self.issue_order.requires_per_cycle()
            || self.completion_order.requires_per_cycle()
            || self.memory_transactions.requires_per_cycle()
            || self.synchronization.requires_per_cycle()
            || self.resource_state.requires_per_cycle()
            || self.counters.requires_per_cycle()
            || self.power.requires_per_cycle()
    }

    pub const fn has_timing_observation(self) -> bool {
        self.issue_order.is_declared() || self.completion_order.is_declared()
    }

    pub fn validate(self) -> Result<(), ObservationContractError> {
        if self.schema_version != OBSERVATION_SCHEMA_VERSION {
            return Err(ObservationContractError::UnsupportedVersion {
                found: self.schema_version,
                expected: OBSERVATION_SCHEMA_VERSION,
            });
        }
        if !self.has_timing_observation() {
            return Err(ObservationContractError::MissingTimingObservation);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ObservationContractError {
    #[error("observation contract schema {found} is unsupported; expected {expected}")]
    UnsupportedVersion { found: u32, expected: u32 },
    #[error("observation contract must declare issue or completion timing")]
    MissingTimingObservation,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_sm86_contract_declares_cycle_observations() {
        let contract = ObservationContract::sm86_cycle_exact();
        assert_eq!(contract.validate(), Ok(()));
        assert!(contract.requires_per_cycle_state());
        assert!(!contract.power.is_declared());
    }

    #[test]
    fn boundary_contract_round_trips_and_is_not_cycle_exact() {
        let contract = ObservationContract::boundary_replay();
        let encoded = serde_json::to_string(&contract).expect("test fixture should be valid");
        let decoded: ObservationContract =
            serde_json::from_str(&encoded).expect("test fixture should be valid");
        assert_eq!(decoded, contract);
        assert!(!contract.requires_per_cycle_state());
        assert_eq!(contract.validate(), Ok(()));
    }

    #[test]
    fn invalid_version_and_missing_timing_are_rejected() {
        let mut invalid = ObservationContract::boundary_replay();
        invalid.schema_version += 1;
        assert!(matches!(
            invalid.validate(),
            Err(ObservationContractError::UnsupportedVersion { .. })
        ));

        invalid = ObservationContract {
            schema_version: OBSERVATION_SCHEMA_VERSION,
            issue_order: ObservationGranularity::Omitted,
            completion_order: ObservationGranularity::Omitted,
            memory_transactions: ObservationGranularity::Omitted,
            synchronization: ObservationGranularity::Omitted,
            resource_state: ObservationGranularity::Omitted,
            counters: ObservationGranularity::Omitted,
            power: ObservationGranularity::Omitted,
        };
        assert_eq!(
            invalid.validate(),
            Err(ObservationContractError::MissingTimingObservation)
        );
    }
}
