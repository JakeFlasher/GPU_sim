//! Committed prost bindings for the NVBit tracer's `dynamic_trace` protobuf
//! schema (`tools/nvbit_tracer/traces_enhanced/dynamic_trace/*.proto`).
//!
//! Regenerate with `sim/trace/regen_pb.sh` (pinned protoc 34.1) whenever the
//! .proto schema changes, and commit the regenerated file in the same commit
//! as the schema change.

#[allow(clippy::doc_lazy_continuation, clippy::doc_overindented_list_items)]
mod dynamic_trace;

pub use dynamic_trace::*;
