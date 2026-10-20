//! Trace-directory reader for the NVBit tracer's on-disk layout:
//!
//! ```text
//! <trace_dir>/dynamic_trace.pb                    # dynamic_trace.Trace
//! <trace_dir>/threadblocks/device_<d>/stream_<s>/kernel_<k>/*.pb
//!                                                 # dynamic_trace.threadblock
//! <trace_dir>/extra_info/allocations.json         # allocation-identity registry
//! <trace_dir>/extra_info/enhanced_execution_info.json # static launch metadata
//! <trace_dir>/trace_sampling.json                 # sampling-provenance sidecar
//! ```
//!
//! [`TraceDir::open_direct`] reads ordinary replay inputs and ignores campaign
//! provenance. With the explicit `campaign` feature, [`TraceDir::open`] retains
//! the historical sealed-campaign behavior for the opt-in campaign skill. Both
//! loaders validate protobuf and simulator semantics before exposing replay
//! inputs.

// Child modules declared by the original public surface.

// Lexical sections share this parent module scope, preserving private fields and impl visibility.
include!("reader/error.rs");
include!("reader/snapshot.rs");
include!("reader/compact.rs");
include!("reader/sampling.rs");
include!("reader/preflight.rs");
