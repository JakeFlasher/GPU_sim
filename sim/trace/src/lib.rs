//! sim-trace: reader for NVBit dynamic traces (the simulator's primary input).
//!
//! Owns: the committed prost bindings for the tracer's protobuf schema
//! ([`pb`]), per-lane address reconstruction in lockstep with the Python
//! admission gate ([`lanes`]), and the trace-directory walker + the
//! reader-level wire contract ([`reader`]).
//!
//! Rule 1 (targets-never-features) note: this crate consumes NVBit traces
//! only. NCU label artifacts are prediction targets and never appear here.

pub mod exec_info;
pub mod lanes;
pub mod pb;
pub mod reader;

pub use exec_info::{ExecInfo, ExecInfoError, KernelStaticInfo};
pub use lanes::{LANE_INACTIVE_SENTINEL, LANES_PER_WARP, LaneError, reconstruct_lane_vector};
pub use reader::{
    AddressAccess, AllocationRegistry, KernelLoc, MemoryAccess, MemoryAccessMeta, TraceDir,
    TraceError, TraceSamplingSidecar, block_launch_index, check_allocation_identity, kernel_grid,
    memory_access, read_threadblock, visit_memory_access,
};
