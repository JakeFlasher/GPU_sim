//! Canonical SM boundary state used by bounded differential runs.
//!
//! The projection is intentionally explicit about coverage. It is useful for
//! finding state drift between detailed and accelerated executors, but it is not
//! a certificate until every state field that can affect the selected model is
//! included.

use serde::{Deserialize, Serialize};

pub const BOUNDARY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WarpBoundaryState {
    pub warp_slot: i32,
    pub next_instruction: u64,
    pub front_end_fetch_cursor: u64,
    pub instruction_buffer: Vec<(u64, u64)>,
    pub eligible: u64,
    pub imc_wait_until: u64,
    pub barrier_clear: [u64; 6],
    pub barrier_pending: [u32; 6],
    pub block: u64,
    pub arrival: u64,
    pub at_block_barrier: bool,
    pub cta_barrier_ready: u64,
    pub barrier_pipeline_ready: u64,
    pub branch_pipeline_ready: u64,
    pub block_barrier_dispatch: Option<u64>,
    pub block_barrier_arrival: Option<u64>,
    pub done: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockBoundaryState {
    pub unfinished: u64,
    pub waiting: u64,
    pub warp_count: u32,
    pub shared_memory_bytes: u64,
    pub warp_slots: Vec<i32>,
    pub first_warp: u64,
    pub admitted_warps: u64,
    pub barrier_generation: u64,
    pub barrier_first_dispatch: Option<u64>,
    pub barrier_first_arrival: Option<u64>,
    pub barrier_rearm_ready: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundarySnapshot {
    /// Current schema is SM86-specific (four SMSPs and six dependency
    /// barriers); a future gfx942 projection must use a separate version or
    /// explicitly typed geometry rather than silently reusing these widths.
    pub schema_version: u32,
    pub cycle: u64,
    pub issued: u64,
    pub resident_warps: u32,
    pub resident_shared_memory_bytes: u64,
    pub unfinished_warps: u64,
    pub next_block: u64,
    pub hold: [Option<u64>; 4],
    pub subcore_warps: Vec<Vec<i32>>,
    pub outstanding_responses: u64,
    /// Canonical physical L1 state for the owning SM, when a cache is
    /// attached. `None` is the flat-memory mode.
    pub l1_state: Option<String>,
    pub instruction_front_end_state: Option<String>,
    pub memory_issue_pipeline_state: Option<String>,
    pub scalar_execution_pipeline_state: Option<String>,
    pub async_copy_state: Vec<Option<String>>,
    pub barrier_pipe_ready: u64,
    pub tensor_pipe_ready: [u64; 4],
    pub ldgsts_destination_pipe_ready: u64,
    pub warp_states: Vec<WarpBoundaryState>,
    pub block_states: Vec<BlockBoundaryState>,
    /// Fields omitted from this projection are part of the schema contract.
    /// A non-empty list prevents callers from treating this as a complete
    /// state certificate.
    pub omitted_state: Vec<String>,
}

impl BoundarySnapshot {
    pub const fn schema_version() -> u32 {
        BOUNDARY_SCHEMA_VERSION
    }

    pub const fn is_complete(&self) -> bool {
        self.omitted_state.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_is_incomplete_until_all_state_is_represented() {
        let snapshot = BoundarySnapshot {
            schema_version: BOUNDARY_SCHEMA_VERSION,
            cycle: 0,
            issued: 0,
            resident_warps: 0,
            resident_shared_memory_bytes: 0,
            unfinished_warps: 0,
            next_block: 0,
            hold: [None; 4],
            subcore_warps: Vec::new(),
            outstanding_responses: 0,
            l1_state: None,
            instruction_front_end_state: None,
            memory_issue_pipeline_state: None,
            scalar_execution_pipeline_state: None,
            async_copy_state: Vec::new(),
            barrier_pipe_ready: 0,
            tensor_pipe_ready: [0; 4],
            ldgsts_destination_pipe_ready: 0,
            warp_states: Vec::new(),
            block_states: Vec::new(),
            omitted_state: vec!["cache_state".to_owned()],
        };
        assert!(!snapshot.is_complete());
        assert!(serde_json::to_value(snapshot).is_ok());
    }
}
